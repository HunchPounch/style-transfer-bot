import logging
import sys

from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext

import bot.keyboards.Reply_keyboards as Reply_keyboards

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt='[%(asctime)s] #%(levelname)-8s %(filename)s:'
        '%(lineno)d - %(name)s:%(funcName)s - %(message)s'
)
stdout = logging.StreamHandler(sys.stdout)

stdout.setFormatter(formatter)
logger.addHandler(stdout)


router = Router()

@router.message(Command("start"))
async def start(message: Message, state: FSMContext):

    current_state = await state.get_state()
    
    if current_state is None:
        await message.answer(f"<b>Привет, {message.from_user.first_name}</b>"
                         "\nЯ могу переносить стили изображений c помощью нейросетей!"
                         "\nВыбери одно из следующих действий",
                          reply_markup=Reply_keyboards.main_kb)
        return

    if (current_state == 'Cycle:processing') | (current_state == 'Cycle_command:processing') | (current_state == 'Style:processing'):
        await message.answer("Извини, предыдущая операция все еще выполняется, для ее отмены нажми 'Отмена'",
                          reply_markup=Reply_keyboards.cancel_kb)
        return
    
    logger.debug("Cancelling state %r for start", current_state)
    await state.clear()
    await message.answer(f"<b>Привет, {message.from_user.first_name}</b>"
                         "\nЯ могу переносить стили изображений при помощи нейросетей!"
                         "\nВыбери одно из следующих действий",
                          reply_markup=Reply_keyboards.main_kb)