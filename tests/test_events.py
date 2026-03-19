"""
Tests for the SSE EventBus and event publishing integration.

Test 1: test_event_bus_publish_subscribe
Test 2: test_event_bus_multiple_subscribers
Test 3: test_event_bus_full_queue_does_not_block
Test 4: test_sse_endpoint_returns_event_stream
Test 5: test_evaluation_publishes_event
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from app.events.stream import EventBus, event_bus
from app.models.schemas import PipelineEvent


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_event(event_type: str = "conversation_ingested") -> PipelineEvent:
    return PipelineEvent(
        event_type=event_type,
        timestamp=datetime.now(timezone.utc),
        data={"test": True},
        conversation_id="conv_test",
        agent_version="v1.0.0",
    )


# ── Test 1: basic publish / subscribe / unsubscribe ───────────────────────────

@pytest.mark.asyncio
async def test_event_bus_publish_subscribe():
    """Subscribe, receive an event, unsubscribe, verify no further events received."""
    bus = EventBus()
    queue = bus.subscribe()

    event = _make_event("conversation_ingested")
    await bus.publish(event.model_dump_json())

    # Event should be in the queue within 1 second
    received = await asyncio.wait_for(queue.get(), timeout=1.0)
    data = json.loads(received)
    assert data["event_type"] == "conversation_ingested"

    # Unsubscribe and publish another event — queue should stay empty
    bus.unsubscribe(queue)
    await bus.publish(_make_event("alert_fired").model_dump_json())

    # Queue must be empty (no new events delivered after unsubscribe)
    assert queue.empty()


# ── Test 2: multiple subscribers each receive the event ───────────────────────

@pytest.mark.asyncio
async def test_event_bus_multiple_subscribers():
    """Two subscribers both receive the same published event."""
    bus = EventBus()
    q1 = bus.subscribe()
    q2 = bus.subscribe()

    event = _make_event("evaluation_completed")
    await bus.publish(event.model_dump_json())

    data1 = json.loads(await asyncio.wait_for(q1.get(), timeout=1.0))
    data2 = json.loads(await asyncio.wait_for(q2.get(), timeout=1.0))

    assert data1["event_type"] == "evaluation_completed"
    assert data2["event_type"] == "evaluation_completed"

    bus.unsubscribe(q1)
    bus.unsubscribe(q2)


# ── Test 3: full queue drops events without blocking the publisher ─────────────

@pytest.mark.asyncio
async def test_event_bus_full_queue_does_not_block():
    """Publishing more events than maxsize does not block or raise; earliest events kept."""
    bus = EventBus()

    # Manually create a tiny queue (maxsize=2) and inject it
    tiny_queue: asyncio.Queue = asyncio.Queue(maxsize=2)
    bus._subscribers.append(tiny_queue)

    # Publish 5 events — should not raise even though queue fills after 2
    for i in range(5):
        await bus.publish(_make_event("alert_fired").model_dump_json())

    # Queue has exactly 2 events (maxsize); no exception raised
    assert tiny_queue.qsize() == 2

    bus.unsubscribe(tiny_queue)


# ── Test 4: SSE endpoint returns text/event-stream ────────────────────────────

@pytest.mark.asyncio
async def test_sse_endpoint_returns_event_stream(client):
    """
    GET /api/v1/events/stream responds with text/event-stream content-type
    and delivers a 'conversation_ingested' event when a conversation is posted.
    """
    # Use a short-lived streaming request — collect the first chunk only
    collected: list[str] = []

    async def _consume():
        async with client.stream("GET", "/api/v1/events/stream") as resp:
            assert resp.headers["content-type"].startswith("text/event-stream")
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    collected.append(line)
                    break  # got one event, stop

    # Start consumer in background
    consumer = asyncio.create_task(_consume())
    await asyncio.sleep(0.1)  # let the consumer connect

    # Post a conversation to trigger a conversation_ingested event
    await client.post("/api/v1/conversations", json={
        "conversation_id": "conv_sse_test",
        "agent_version": "v1.0.0",
        "turns": [
            {"turn_id": 1, "role": "user", "content": "Hello", "timestamp": "2024-01-15T10:00:00Z"},
        ],
    })

    # Wait for consumer to finish (up to 3 seconds)
    try:
        await asyncio.wait_for(consumer, timeout=3.0)
    except asyncio.TimeoutError:
        consumer.cancel()

    # At least one SSE data line received
    assert len(collected) >= 1
    payload = json.loads(collected[0].removeprefix("data:").strip())
    assert payload["event_type"] == "conversation_ingested"


# ── Test 5: evaluation endpoint publishes an evaluation_completed event ────────

@pytest.mark.asyncio
async def test_evaluation_publishes_event(client):
    """
    POST /evaluate/{id} causes an 'evaluation_completed' event to be published
    to the event bus with correct conversation_id and a valid overall_score.
    """
    # Subscribe to the module-level singleton event_bus
    queue = event_bus.subscribe()

    try:
        # Ingest a conversation
        resp = await client.post("/api/v1/conversations", json={
            "conversation_id": "conv_eval_event_test",
            "agent_version": "v1.0.0",
            "turns": [
                {"turn_id": 1, "role": "user", "content": "Book a flight to NYC",
                 "timestamp": "2024-01-15T10:00:00Z"},
                {"turn_id": 2, "role": "assistant", "content": "I'll search for flights.",
                 "timestamp": "2024-01-15T10:00:01Z",
                 "tool_calls": [{
                     "tool_name": "flight_search",
                     "parameters": {"destination": "NYC", "date_range": "2024-01-22/2024-01-28"},
                     "result": {"status": "success", "flights": []},
                     "latency_ms": 200,
                 }]},
            ],
            "feedback": {"user_rating": 4},
            "metadata": {"total_latency_ms": 500, "mission_completed": True},
        })
        assert resp.status_code == 201

        # Run evaluation
        eval_resp = await client.post("/api/v1/evaluations/evaluate/conv_eval_event_test")
        assert eval_resp.status_code == 200

        # Poll queue for evaluation_completed event (timeout 2s)
        eval_event_data: dict | None = None
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if not queue.empty():
                raw = queue.get_nowait()
                data = json.loads(raw)
                if data.get("event_type") == "evaluation_completed":
                    eval_event_data = data
                    break
            await asyncio.sleep(0.05)

        assert eval_event_data is not None, "No evaluation_completed event received"
        assert eval_event_data["conversation_id"] == "conv_eval_event_test"
        assert 0.0 <= eval_event_data["data"]["overall_score"] <= 1.0

    finally:
        event_bus.unsubscribe(queue)
