import asyncio
import logging

from aiogram import Bot, Dispatcher

from bot.handlers import bot_messages, user_commands
from bot.config_reader import config
from bot.middlewares.antiflood import AntiFloodMiddleware



async def main():

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] #%(levelname)-8s %(filename)s:'
                '%(lineno)d - %(name)s - %(message)s'
        )
    
    bot = Bot(config.tg_bot_token.get_secret_value(), parse_mode="HTML")
    dp = Dispatcher()
    dp.message.middleware(AntiFloodMiddleware())
    dp.include_routers(
        user_commands.router,
        bot_messages.router
    )
    
    await bot.delete_webhook(drop_pending_updates=True)
    logging.info("Starting bot")
    await dp.start_polling(bot)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.error("Bot stopped!")