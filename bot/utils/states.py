from aiogram.fsm.state import StatesGroup, State

class Style(StatesGroup):
    pic1 = State()
    pic2 = State()
    processing = State()

class Cycle(StatesGroup):
    pic = State()
    processing = State()

class Cycle_command(StatesGroup):
    pic = State()
    processing = State()