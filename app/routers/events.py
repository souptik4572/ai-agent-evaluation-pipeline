from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.events.stream import event_bus
from app.models.schemas import PipelineEvent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/events", tags=["Events"])


@router.get("/stream")
async def event_stream():
    """
    Server-Sent Events stream. Sends a keepalive comment every 30 seconds
    to prevent proxies from closing idle connections.
    """
    queue = event_bus.subscribe()
    logger.debug("SSE client connected (subscribers=%d)", event_bus.subscriber_count)

    async def generate():
        try:
            while True:
                try:
                    event_data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {event_data}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            event_bus.unsubscribe(queue)
            logger.debug("SSE client disconnected (subscribers=%d)", event_bus.subscriber_count)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/test")
async def publish_test_event():
    """Publish a test event to the SSE bus."""
    event = PipelineEvent(
        event_type="conversation_ingested",
        timestamp=datetime.now(timezone.utc),
        data={"message": "test event from /events/test"},
        conversation_id="test",
        agent_version="test",
    )
    try:
        await event_bus.publish(event.model_dump_json())
    except Exception:
        pass
    return {"published": True, "event": event.model_dump()}
