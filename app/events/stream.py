from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[asyncio.Queue] = []

    async def publish(self, event_json: str) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event_json)
            except asyncio.QueueFull:
                logger.debug("Subscriber queue full — dropping event for slow consumer")

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(queue)
        except ValueError:
            pass

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


event_bus = EventBus()
