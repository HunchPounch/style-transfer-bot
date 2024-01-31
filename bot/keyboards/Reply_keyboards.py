from aiogram.types import (
    ReplyKeyboardMarkup,
    KeyboardButton
)

main_kb = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="Перенос стиля с одной картинки на другую")
        ],
        [
            KeyboardButton(text="Изменение стиля картинки на стиль картин Ван Гога")
        ]
    ],
    resize_keyboard=True,
    one_time_keyboard=True,
    input_field_placeholder="Выберите действие из меню",
    selective=True
)

cancel_kb = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text="Отмена")
        ]   
    ],
    resize_keyboard=True,
    one_time_keyboard=True,
    input_field_placeholder="Прекратить операцию?",
    selective=True
)