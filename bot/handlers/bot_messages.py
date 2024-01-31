import asyncio
import os
import sys
import logging
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor

from aiogram import Router, F, Bot
from aiogram.types import Message, FSInputFile
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import default_state

import bot.keyboards.Reply_keyboards as Reply_keyboards
from bot.utils.states import Style, Cycle, Cycle_command
from bot.models.StyleTransfer import StyleTransferClass
from bot.models.CycleGAN import CycleGANClass
from bot.middlewares.group_of_photo import photos


MAX_THREADS = 2

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt='[%(asctime)s] #%(levelname)-8s %(filename)s:'
        '%(lineno)d - %(name)s:%(funcName)s - %(message)s'
)
stdout = logging.StreamHandler(sys.stdout)

stdout.setFormatter(formatter)
logger.addHandler(stdout)

model1 = StyleTransferClass()
model2 = CycleGANClass()


pool = ThreadPoolExecutor(max_workers=MAX_THREADS)
router = Router()

router.message.middleware(photos())

# список запросов пользователей
PROCESSES = {}


@router.message(default_state, F.text.lower() == "отмена")
async def cancel_no_state(message: Message, state: FSMContext):
    await message.answer(
        "Отменять нечего, выбери следующее дествие",
        reply_markup=Reply_keyboards.main_kb
    )


@router.message(F.text.lower() == "отмена")
async def cancel(message: Message, state: FSMContext):

    current_state = await state.get_state()
    logger.debug("Cancelling state %r", current_state)
    await state.clear()
    future = PROCESSES.pop(message.from_user.id, None)

    if future is not None:
        future.cancel()
        logger.debug("Future cancelling requested")
        try:
            await future
        except asyncio.CancelledError:
            logger.debug("Future cancelled")
        except Exception:
            logger.exception(
                "Got unexpected exception when awaiting cancelled inference")
        if future.cancelled():
            await message.answer(
                "Операция отменена, выбери следующее дествие",
                reply_markup=Reply_keyboards.main_kb
            )
            return
        elif future.done():
            await message.answer(
                "Операция уже выполнена, выбери следующее дествие",
                reply_markup=Reply_keyboards.main_kb
            )
            return
    await message.answer(
                "Операция отменена, выбери следующее дествие",
                reply_markup=Reply_keyboards.main_kb
             )


@router.message(F.text.lower() == "перенос стиля с одной картинки на другую")
async def transfering(message: Message, state: FSMContext):
    await state.set_state(Style.pic1)
    await message.answer(
        "Отправь мне первое изображение, стиль которого необходимо изменить",
        reply_markup=Reply_keyboards.cancel_kb
    )


@router.message(Style.pic1, F.photo, ~F.media_group_id)
async def get_pic1(message: Message, state: FSMContext):

    await state.update_data(pic1=message.photo[-1].file_id)
    await state.set_state(Style.pic2)
    await message.answer(
        "Теперь отправь второе изображение, с которого будем переносить стиль"
    )


@router.message(Style.pic1, ~F.photo | F.media_group_id)
async def incorrect_input1(message: Message, state: FSMContext):
    await message.answer("Отправь одно изображение! ('.jpg')")
    logger.warning("File with unsupported mime_type or"
                    "no file or got more than one file")



@router.message(Style.pic2, F.photo, ~F.media_group_id)
async def get_pic2(message: Message, state: FSMContext, bot: Bot):
    
    await state.update_data(pic2=message.photo[-1].file_id)
    data = await state.get_data()
    await state.set_state(Style.processing)
    await message.answer("Нейросеть начинает работать!\nПримерное время ожидания:"
                         " 5 минут\nДля отмены операции нажмите на соотвествующую кнопку",
                         reply_markup=Reply_keyboards.cancel_kb
    )
    
    content_img_file = await bot.get_file(data['pic1'])
    file_path1 = content_img_file.file_path
    style_img_file = await bot.get_file(data['pic2'])
    file_path2 = style_img_file.file_path
        
    tmp = TemporaryDirectory()
    ph1 = os.path.join(tmp.name, 'pic1.jpg')
    ph2 = os.path.join(tmp.name, 'pic2.jpg')
    await bot.download_file(file_path1, ph1)
    await bot.download_file(file_path2, ph2)
    
    if await state.get_state() != Style.processing.state:
        logger.debug("The request superseded by another one "
                         "while downloading images.")
        return

    loop = asyncio.get_running_loop()
    logger.debug("run future")
    future = loop.run_in_executor(pool, model1.forward, ph1, ph2, tmp.name)
    cur_proc = PROCESSES.setdefault(message.from_user.id, future)
    if cur_proc is not future:
        logger.debug("The request superseded by other one.")
        return
    
    try:
        await future
        logger.debug("Future is done")
    except asyncio.CancelledError:
        logger.debug("Future is cancelled by other handler")
    except Exception:
        logger.exception("got unexpected exception when awaiting inference")
        await state.clear()
        tmp.cleanup()
    else:
        await message.answer_photo(
                FSInputFile(os.path.join(tmp.name, 'pic3.jpg')),
                caption="А вот и новый стиль!"
        )
        await message.answer(
                "Выбери следующее дествие",
                reply_markup=Reply_keyboards.main_kb
        )
        await state.clear()
        tmp.cleanup()
    finally:
        tmp.cleanup()
        if PROCESSES.get(message.from_user.id) is future:
            PROCESSES.pop(message.from_user.id, None)


@router.message(Style.pic2, ~F.photo | F.media_group_id)
async def incorrect_input2(message: Message, state: FSMContext):
    await message.answer("Отправь одно изображение! ('.jpg')")
    logger.warning("File with unsupported mime_type or"
                    "no file or got more than one file")


@router.message(F.text.lower() == "изменение стиля картинки на стиль картин ван гога")
async def transfering(message: Message, state: FSMContext):
    await state.set_state(Cycle.pic)
    await message.answer(
        "Отправь изображение",
        reply_markup=Reply_keyboards.cancel_kb
    )


@router.message(Cycle.pic, F.photo, ~F.media_group_id)
async def get_pic(message: Message, state: FSMContext, bot: Bot):
    
    await state.update_data(pic=message.photo[-1].file_id)
    data = await state.get_data()
    await state.set_state(Cycle.processing)

    await message.answer("Нейросеть начинает работать!\nПримерное время ожидания:"
                         " 15 секунд\nДля отмены операции нажми на соотвествующую кнопку",
                         reply_markup=Reply_keyboards.cancel_kb                 
    )

    content_img_file = await bot.get_file(data['pic'])
    file_path = content_img_file.file_path

    tmp = TemporaryDirectory()
    ph = os.path.join(tmp.name, 'pic.jpg')
    await bot.download_file(file_path, ph)
    logger.debug("images downloaded")
    if await state.get_state() != Cycle.processing.state:
        logger.debug("The request superseded by another one "
                         "while downloading image.")
        return
   
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(pool, model2.forward, ph, tmp.name)
    logger.debug("run future")
    cur_proc = PROCESSES.setdefault(message.from_user.id, future)
    if cur_proc is not future:
        logger.debug("The request superseded by other one.")
        return
    
    try:
        await future
        logger.debug("Future is done")
    except asyncio.CancelledError:
        logger.debug("Future is cancelled by other handler")
    except Exception:
        logger.exception("got unexpected exception when awaiting inference")
        await state.clear()
        tmp.cleanup()
    else:
        await message.answer_photo(
                FSInputFile(os.path.join(tmp.name, 'pic4.jpg')),
                caption="Твое изображение теперь картина Ван Гога!"
        )
        await message.answer(
                "Выбери следующее дествие",
                reply_markup=Reply_keyboards.main_kb
        )
        await state.clear()
        tmp.cleanup()
    finally:
        tmp.cleanup()
        if PROCESSES.get(message.from_user.id) is future:
            PROCESSES.pop(message.from_user.id, None)
    
    

@router.message(Cycle.pic, ~F.photo | F.media_group_id)
async def incorrect_input2(message: Message, state: FSMContext):
    await message.answer("Отправь одно изображение! ('.jpg')")
    logger.warning("File with unsupported mime_type or"
                    "no file or got more than one file")


@router.message(F.text.lower() == "/transfer_style")
async def transfer_style(message: Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is not None:

        logger.debug("Cancelling state %r", current_state)
        await state.clear()
        future = PROCESSES.pop(message.from_user.id, None)
        if future is not None:
            future.cancel()
            logger.debug("Future cancelling requested")
    
    await state.set_state(Cycle_command.pic)
    await message.answer(
        "Отправь изображение",
        reply_markup=Reply_keyboards.cancel_kb
    )


@router.message(Cycle_command.pic, F.photo, ~F.media_group_id)
async def get_pic_command(message: Message, state: FSMContext, bot: Bot):
   
    await state.update_data(pic=message.photo[-1].file_id)
    data = await state.get_data()
    await state.set_state(Cycle_command.processing)

    await message.answer("Нейросеть начинает работать!\nПримерное время ожидания:"
                         " 1 минута\nДля отмены операции нажми на соотвествующую кнопку",
                         reply_markup=Reply_keyboards.cancel_kb
    )
    content_img_file = await bot.get_file(data['pic'])
    file_path = content_img_file.file_path
    
    tmp = TemporaryDirectory()
    ph = os.path.join(tmp.name, 'pic.jpg')
    await bot.download_file(file_path, ph)

    if await state.get_state() != Cycle_command.processing.state:
        logger.debug("The request superseded by another one "
                         "while downloading image.")
        return
   
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(pool, model2.forward, ph, tmp.name)
    cur_proc = PROCESSES.setdefault(message.from_user.id, future)
    if cur_proc is not future:
        logger.debug("The request superseded by other one.")
        return
    
    try:
        await future
        logger.debug("Future is done")
    except asyncio.CancelledError:
        logger.debug("Future is cancelled by other handler")
    except Exception:
        logger.exception("Got unexpected exception when awaiting inference")
        await state.clear()
        tmp.cleanup()
    else:
        await message.answer_photo(
                FSInputFile(os.path.join(tmp.name, 'pic4.jpg')),
        )
        await message.answer(
                "Выбери следующее дествие",
                reply_markup=Reply_keyboards.main_kb
        )
        await state.clear()
        tmp.cleanup()
    finally:
        tmp.cleanup()
        if PROCESSES.get(message.from_user.id) is future:
            PROCESSES.pop(message.from_user.id, None)



@router.message(Cycle_command.pic, ~F.photo | F.media_group_id)
async def incorrect_input2(message: Message, state: FSMContext):
    await message.answer("Отправь одно изображение! ('.jpg')")
    logger.warning("File with unsupported mime_type or"
                    "no file or got more than one file")


@router.message()
async def echo(message: Message):
    await message.answer(f"Я тебя не понимаю!")