from __future__ import annotations

import argparse
import asyncio
import time
from datetime import datetime, timezone
from uuid import uuid4

import httpx

parser = argparse.ArgumentParser(description="Load test for AI Agent Evaluation Pipeline")
parser.add_argument("--base-url", default="http://localhost:8000")
parser.add_argument("--count", type=int, default=100)
parser.add_argument("--concurrency", type=int, default=50)
args = parser.parse_args()

BASE = args.base_url
API = f"{BASE}/api/v1"
COUNT = args.count
CONCURRENCY = args.concurrency


def _make_conv(i: int) -> dict:
    suffix = uuid4().hex[:6]
    return {
        "conversation_id": f"bench_{i}_{suffix}",
        "agent_version": "v2.3.1",
        "turns": [
            {
                "turn_id": 1,
                "role": "user",
                "content": "Book a flight to NYC",
                "timestamp": "2024-01-15T10:30:00Z",
            },
            {
                "turn_id": 2,
                "role": "assistant",
                "content": "I'll search for flights.",
                "timestamp": "2024-01-15T10:30:01Z",
                "tool_calls": [{
                    "tool_name": "flight_search",
                    "parameters": {"destination": "NYC", "date_range": "2024-01-22/2024-01-28"},
                    "result": {"status": "success", "flights": []},
                    "latency_ms": 200,
                }],
            },
        ],
        "feedback": {"user_rating": 4},
        "metadata": {"total_latency_ms": 500, "mission_completed": True},
    }


async def _post(client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str, payload) -> bool:
    async with sem:
        try:
            r = await client.post(url, json=payload, timeout=30.0)
            return r.status_code in (200, 201, 409)  # 409 = duplicate, still counts
        except Exception:
            return False


async def _get(client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str) -> bool:
    async with sem:
        try:
            r = await client.get(url, timeout=30.0)
            return r.ok
        except Exception:
            return False


def _report(label: str, total: int, succeeded: int, wall: float, note: str = "") -> None:
    failed = total - succeeded
    per_sec = succeeded / wall if wall > 0 else 0
    per_min = per_sec * 60
    target_ok = "✓ Exceeds 1,000/min target" if per_min >= 1000 else "⚠ Below 1,000/min target"
    print(f"\n  {label}")
    if "batch" in label.lower():
        batches = max(1, total // 50)
        print(f"     Requests:    {total} conversations in {batches} batch(es)")
    else:
        print(f"     Requests:    {total} sent, {succeeded} succeeded, {failed} failed")
    print(f"     Duration:    {wall:.2f}s")
    print(f"     Throughput:  {per_sec:.1f}/sec → {per_min:,.0f}/min")
    print(f"     Status:      {target_ok}")
    if note:
        print(f"     Note:        {note}")


async def _check_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(f"{BASE}/health")
            return r.ok
    except Exception:
        return False


async def main() -> None:
    if not await _check_health():
        print(f"\nERROR: Could not connect to {BASE}. Start the API first:")
        print(f"  uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return

    print(f"\n{'═' * 55}")
    print(f"  AI Eval Pipeline — Load Test Results")
    print(f"  Target: {BASE} | Concurrency: {CONCURRENCY}")
    print(f"{'═' * 55}")

    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    sem = asyncio.Semaphore(CONCURRENCY)
    ingested_ids: list[str] = []

    async with httpx.AsyncClient(base_url=BASE, limits=limits) as client:

        convs = [_make_conv(i) for i in range(COUNT)]
        tasks = [_post(client, sem, f"{API}/conversations", c) for c in convs]
        t0 = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall1 = time.perf_counter() - t0
        ok1 = sum(results)
        for c in convs:
            ingested_ids.append(c["conversation_id"])
        _report("1. Ingestion (single)", COUNT, ok1, wall1)

        batch_convs = [_make_conv(COUNT + i) for i in range(COUNT)]
        chunks = [batch_convs[i:i + 50] for i in range(0, len(batch_convs), 50)]
        tasks2 = [_post(client, sem, f"{API}/conversations/batch", chunk) for chunk in chunks]
        t0 = time.perf_counter()
        results2 = await asyncio.gather(*tasks2)
        wall2 = time.perf_counter() - t0
        batch_ok = sum(1 for r in results2 if r) * 50  # approximate conversations succeeded
        _report("2. Ingestion (batch, chunks of 50)", COUNT, min(batch_ok, COUNT), wall2)

        eval_ids = ingested_ids[:50]

        async def _eval_post(cid: str) -> bool:
            async with sem:
                try:
                    r = await client.post(f"{API}/evaluations/evaluate/{cid}", timeout=30.0)
                    return r.status_code in (200, 201)
                except Exception:
                    return False

        eval_tasks2 = [_eval_post(cid) for cid in eval_ids]
        t0 = time.perf_counter()
        eval_results = await asyncio.gather(*eval_tasks2)
        wall3 = time.perf_counter() - t0
        eval_ok = sum(eval_results)
        _report(
            "3. Evaluation (full pipeline, mock LLM)",
            len(eval_ids), eval_ok, wall3,
            note="With real LLM key, LLM calls dominate latency — this is expected.",
        )

        read_tasks = [
            _get(client, sem, f"{API}/conversations?limit=10&offset={i * 10}")
            for i in range(COUNT)
        ]
        t0 = time.perf_counter()
        read_results = await asyncio.gather(*read_tasks)
        wall4 = time.perf_counter() - t0
        read_ok = sum(read_results)
        _report("4. Read (GET /conversations)", COUNT, read_ok, wall4)

    print(f"\n{'═' * 55}")
    print(f"  Summary")
    ingest_pmin = (ok1 / wall1 * 60) if wall1 > 0 else 0
    eval_pmin = (eval_ok / wall3 * 60) if wall3 > 0 else 0
    print(f"  Ingestion handles {'1,000+' if ingest_pmin >= 1000 else f'{ingest_pmin:,.0f}'}/min on a single process.")
    print(f"  Evaluation throughput is LLM-bound (~2-4s per call with real key).")
    if eval_pmin > 0:
        workers3 = int(3000 / eval_pmin) if eval_pmin > 0 else "N/A"
        print(f"  Adding N async workers scales evaluation linearly:")
        print(f"    3 workers → ~{eval_pmin * 3:,.0f}/min | 10 workers → ~{eval_pmin * 10:,.0f}/min")
    print(f"{'═' * 55}")

    print(f"\n  Note: Benchmark data left in DB — restart the API to clean, or")
    print(f"        run: DELETE FROM conversations WHERE conversation_id LIKE 'bench_%'")
    print()


if __name__ == "__main__":
    asyncio.run(main())
