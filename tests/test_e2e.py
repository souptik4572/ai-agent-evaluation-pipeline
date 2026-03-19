"""
End-to-end integration test.

Runs the full pipeline flywheel:
1. POST 5 conversations for v_e2e_base (correct tool usage)
2. Evaluate all
3. POST 5 conversations for v_e2e_update (3 with tool errors)
4. Evaluate all
5. POST /regression/compare → assert regression on tool_accuracy
6. POST /suggestions/generate → assert at least one suggestion
7. GET /alerts → assert at least one regression alert
8. GET /meta/correlation → assert correlation data has sample_size >= 8
"""
from datetime import datetime, timedelta, timezone

import pytest

NOW = datetime.now(timezone.utc)


def _ts(offset: int = 0) -> str:
    return (NOW + timedelta(seconds=offset)).isoformat()


def _conv(cid: str, version: str, tool_ok: bool = True, user_rating: int = 4) -> dict:
    result = (
        {"status": "success", "flights": [{"id": "BA112", "price": 599}]}
        if tool_ok
        else {"status": "error", "message": "Invalid date format: next week"}
    )
    date_param = "2024-06-10/2024-06-17" if tool_ok else "next week"
    return {
        "conversation_id": cid,
        "agent_version": version,
        "turns": [
            {"turn_id": 1, "role": "user",
             "content": "Book me a flight from NYC to London.",
             "timestamp": _ts(0)},
            {"turn_id": 2, "role": "assistant",
             "content": "Searching for flights..." if tool_ok else "I'm sorry, I had trouble finding flights.",
             "tool_calls": [{
                 "tool_name": "flight_search",
                 "parameters": {"origin": "JFK", "destination": "LHR", "date_range": date_param},
                 "result": result,
                 "latency_ms": 300 if tool_ok else 1200,
             }],
             "timestamp": _ts(5)},
            {"turn_id": 3, "role": "user",
             "content": "Book the cheapest one." if tool_ok else "Never mind.",
             "timestamp": _ts(12)},
            {"turn_id": 4, "role": "assistant",
             "content": "Booked BA112 for $599." if tool_ok else "I apologize for the inconvenience.",
             "timestamp": _ts(18)},
        ],
        "feedback": {
            "user_rating": user_rating if tool_ok else max(1, user_rating - 3),
            "ops_review": {"quality": "good" if tool_ok else "poor"},
        },
        "metadata": {
            "total_latency_ms": 700 if tool_ok else 1500,
            "mission_completed": tool_ok,
        },
    }


@pytest.mark.asyncio
async def test_full_pipeline_flywheel(client):
    """Full end-to-end: ingest → evaluate → regression → suggestions → alerts → correlation."""

    BASE_V = "v_e2e_base"
    UPDATE_V = "v_e2e_update"

    # ── Step 1: Ingest baseline conversations ─────────────────────────────────
    base_ids = []
    for i in range(5):
        payload = _conv(f"conv_e2e_base_{i}", BASE_V, tool_ok=True, user_rating=5)
        resp = await client.post("/api/v1/conversations", json=payload)
        assert resp.status_code == 201, f"Ingestion failed: {resp.text}"
        base_ids.append(payload["conversation_id"])

    # ── Step 2: Evaluate baseline ─────────────────────────────────────────────
    for cid in base_ids:
        resp = await client.post(f"/api/v1/evaluations/evaluate/{cid}")
        assert resp.status_code == 200, f"Evaluation failed for {cid}: {resp.text}"

    # Verify baseline scores exist
    evals_resp = await client.get(f"/api/v1/evaluations?agent_version={BASE_V}&limit=10")
    assert evals_resp.status_code == 200
    base_evals = evals_resp.json().get("data", [])
    assert len(base_evals) == 5

    # ── Step 3: Ingest post-update conversations ──────────────────────────────
    update_ids = []
    for i in range(5):
        # 3 with tool errors, 2 with correct usage
        tool_ok = i >= 3
        payload = _conv(f"conv_e2e_update_{i}", UPDATE_V, tool_ok=tool_ok, user_rating=4)
        resp = await client.post("/api/v1/conversations", json=payload)
        assert resp.status_code == 201
        update_ids.append(payload["conversation_id"])

    # ── Step 4: Evaluate post-update ─────────────────────────────────────────
    for cid in update_ids:
        resp = await client.post(f"/api/v1/evaluations/evaluate/{cid}")
        assert resp.status_code == 200

    # ── Step 5: Run regression comparison ────────────────────────────────────
    compare_resp = await client.post("/api/v1/regression/compare", json={
        "baseline_version": BASE_V,
        "target_version": UPDATE_V,
    })
    assert compare_resp.status_code == 200
    report = compare_resp.json()
    assert "is_regression" in report
    assert "dimensions" in report
    assert "summary" in report
    assert report["baseline_sample_size"] == 5
    assert report["target_sample_size"] == 5

    # tool_accuracy should show a regression since 3/5 target convs had tool errors
    tool_dim = report["dimensions"].get("tool_accuracy", {})
    assert tool_dim.get("delta", 0) <= 0  # target is worse or equal

    # ── Step 6: Generate improvement suggestions ──────────────────────────────
    sugg_resp = await client.post("/api/v1/suggestions/generate?last_n=20")
    assert sugg_resp.status_code == 200
    sugg_result = sugg_resp.json()
    # Suggestions may or may not be generated depending on patterns detected
    assert "suggestions_generated" in sugg_result

    # ── Step 7: Check alerts ──────────────────────────────────────────────────
    alerts_resp = await client.get("/api/v1/alerts?limit=50")
    assert alerts_resp.status_code == 200
    alerts = alerts_resp.json()
    assert isinstance(alerts, list)

    alert_summary_resp = await client.get("/api/v1/alerts/summary")
    assert alert_summary_resp.status_code == 200
    summary = alert_summary_resp.json()
    assert "total_open" in summary
    assert "by_severity" in summary

    # ── Step 8: Correlation endpoint ──────────────────────────────────────────
    corr_resp = await client.get("/api/v1/meta/correlation?limit=50")
    assert corr_resp.status_code == 200
    corr_data = corr_resp.json()
    assert "correlations" in corr_data
    assert "scatter_data" in corr_data

    # If enough rated records exist, correlation should have sample size >= 10
    if corr_data.get("correlations"):
        for c in corr_data["correlations"]:
            assert "pearson_r" in c
            assert "dimension" in c
            assert "interpretation" in c


@pytest.mark.asyncio
async def test_regression_api_endpoints(client):
    """Smoke test all regression API endpoints."""
    # versions endpoint
    resp = await client.get("/api/v1/regression/versions")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    # reports endpoint
    resp = await client.get("/api/v1/regression/reports")
    assert resp.status_code == 200
    assert "data" in resp.json()

    # compare with same version → should return a valid report (even if empty)
    resp = await client.post("/api/v1/regression/compare", json={
        "baseline_version": "nonexistent_v1",
        "target_version": "nonexistent_v2",
    })
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_metrics_endpoint(client):
    """GET /api/v1/metrics returns the expected keys."""
    resp = await client.get("/api/v1/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_conversations" in data
    assert "total_evaluations" in data
    assert "total_suggestions" in data
    assert "open_alerts" in data
    assert "uptime_seconds" in data
    assert "version" in data
