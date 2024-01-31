import asyncio
from typing import Callable, Any, Awaitable, Dict

from cachetools import TTLCache
from aiogram import BaseMiddleware
from aiogram.types import Message

class AntiFloodMiddleware(BaseMiddleware):

    def __init__(self, time_limit: int=1) -> None:
        self.limit = TTLCache(maxsize=10_000, ttl=time_limit)

    
    async def __call__(
            self,
            handler: Callable[[Message, Dict[str, Any]], Awaitable[Any]],
            event: Message,
            data: Dict[str, Any]
    ) -> Any:
        if event.chat.id in self.limit:
            return
        else:
            self.limit[event.chat.id] = None
        
        return await handler(event, data)