import argparse
from datetime import datetime, timedelta, timezone

import requests

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://localhost:8000")
args = parser.parse_args()

API = f"{args.base_url}/api/v1"
NOW = datetime.now(timezone.utc)


def ts(offset_seconds: int = 0) -> str:
    return (NOW + timedelta(seconds=offset_seconds)).isoformat()


def post(url: str, data) -> dict:
    r = requests.post(url, json=data, timeout=60)
    if not r.ok:
        print(f"  ⚠  POST {url} → {r.status_code}: {r.text[:200]}")
    return r.json() if r.ok else {}


def get(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, params=params, timeout=30)
    return r.json() if r.ok else {}


# ── Conversation builders ───────────────────────────────────────────────────────

def _flight_conv(
    conv_id: str,
    agent_version: str,
    date_range: str,
    user_rating: int,
    mission_completed: bool,
    latency_ms: int = 750,
    ops_notes: str | None = None,
) -> dict:
    """Flight booking conversation — success or failure depending on date format."""
    result_body = (
        {"status": "success", "flights": [{"id": "BA112", "price": 599}]}
        if mission_completed
        else {"status": "error", "message": f"Invalid date format: {date_range}"}
    )
    return {
        "conversation_id": conv_id,
        "agent_version": agent_version,
        "turns": [
            {"turn_id": 1, "role": "user",
             "content": "I need to book a round-trip flight from NYC to London next week.",
             "timestamp": ts(0)},
            {"turn_id": 2, "role": "assistant",
             "content": "I'll search for available round-trip flights from NYC to London.",
             "tool_calls": [{
                 "tool_name": "flight_search",
                 "parameters": {"origin": "JFK", "destination": "LHR", "date_range": date_range},
                 "result": result_body,
                 "latency_ms": 310,
             }],
             "timestamp": ts(5)},
            {"turn_id": 3, "role": "user",
             "content": "Great, book the cheapest option.",
             "timestamp": ts(12)},
            {"turn_id": 4, "role": "assistant",
             "content": (
                 "I've booked flight BA112 for $599. Confirmation: XK-4821."
                 if mission_completed
                 else "I'm sorry, I encountered an error with the date format. "
                      "Could you provide dates in YYYY-MM-DD format?"
             ),
             "timestamp": ts(18)},
        ],
        "feedback": {
            "user_rating": user_rating,
            "ops_review": {
                "quality": "good" if user_rating >= 3 else "poor",
                "notes": ops_notes,
            },
            "annotations": [{
                "type": "tool_accuracy",
                "label": "correct" if mission_completed else "incorrect",
                "annotator_id": "ann_001",
                "confidence": 0.92,
            }],
        },
        "metadata": {"total_latency_ms": latency_ms, "mission_completed": mission_completed},
    }


def _hotel_conv(conv_id: str, agent_version: str, user_rating: int = 5, latency_ms: int = 820) -> dict:
    """Hotel booking conversation — high quality."""
    return {
        "conversation_id": conv_id,
        "agent_version": agent_version,
        "turns": [
            {"turn_id": 1, "role": "user",
             "content": "Find me a hotel in Paris from 2024-03-10 to 2024-03-15, max $200/night.",
             "timestamp": ts(0)},
            {"turn_id": 2, "role": "assistant",
             "content": "I'll search for hotels in Paris within your budget.",
             "tool_calls": [{
                 "tool_name": "hotel_search",
                 "parameters": {"city": "Paris", "check_in": "2024-03-10",
                                "check_out": "2024-03-15", "max_price": 200},
                 "result": {"status": "success", "hotels": [{"name": "Hotel du Louvre", "price": 185}]},
                 "latency_ms": 280,
             }],
             "timestamp": ts(6)},
            {"turn_id": 3, "role": "user", "content": "Book Hotel du Louvre.", "timestamp": ts(14)},
            {"turn_id": 4, "role": "assistant",
             "content": "Hotel du Louvre booked for $185/night. Confirmation: HTL-9932.",
             "timestamp": ts(20)},
        ],
        "feedback": {
            "user_rating": user_rating,
            "ops_review": {"quality": "good"},
            "annotations": [{
                "type": "tool_accuracy",
                "label": "correct",
                "annotator_id": "ann_001",
                "confidence": 0.95,
            }],
        },
        "metadata": {"total_latency_ms": latency_ms, "mission_completed": True},
    }


def _car_conv(conv_id: str, agent_version: str, user_rating: int = 4, latency_ms: int = 700) -> dict:
    """Car rental conversation — high quality."""
    return {
        "conversation_id": conv_id,
        "agent_version": agent_version,
        "turns": [
            {"turn_id": 1, "role": "user",
             "content": "I need a car rental in Miami from 2024-04-05 to 2024-04-10.",
             "timestamp": ts(0)},
            {"turn_id": 2, "role": "assistant",
             "content": "Searching for car rentals in Miami for those dates.",
             "tool_calls": [{
                 "tool_name": "car_rental_search",
                 "parameters": {"location": "Miami", "pickup_date": "2024-04-05",
                                "return_date": "2024-04-10"},
                 "result": {"status": "success", "cars": [{"model": "Toyota Camry", "price_per_day": 65}]},
                 "latency_ms": 210,
             }],
             "timestamp": ts(5)},
            {"turn_id": 3, "role": "user", "content": "Book the Toyota Camry.", "timestamp": ts(11)},
            {"turn_id": 4, "role": "assistant",
             "content": "Toyota Camry booked at $65/day. Confirmation: CAR-5571.",
             "timestamp": ts(17)},
        ],
        "feedback": {
            "user_rating": user_rating,
            "ops_review": {"quality": "good"},
            "annotations": [{
                "type": "tool_accuracy",
                "label": "correct",
                "annotator_id": "ann_001",
                "confidence": 0.90,
            }],
        },
        "metadata": {"total_latency_ms": latency_ms, "mission_completed": True},
    }


def _context_loss_conv(conv_id: str, agent_version: str, forget_prefs: bool) -> dict:
    """Long multi-turn conversation that either retains or forgets user preferences."""
    prefs_msg = "I prefer window seats and my budget is under $400. I must depart after 6pm."
    remembered = "Found a window seat flight departing at 7:30pm for $385 — within your $400 budget!"
    forgotten = "Here are some flights. Flight AA202 departs at 8am and costs $520."

    turns = [
        {"turn_id": 1, "role": "user",
         "content": f"I need flights from Boston to Chicago. {prefs_msg}", "timestamp": ts(0)},
        {"turn_id": 2, "role": "assistant",
         "content": "Understood! I'll keep your preferences in mind — window seat, under $400, departing after 6pm.",
         "tool_calls": [{"tool_name": "flight_search",
                         "parameters": {"origin": "BOS", "destination": "ORD",
                                        "date_range": "2024-05-15/2024-05-22"},
                         "result": {"status": "success", "flights": []}, "latency_ms": 300}],
         "timestamp": ts(5)},
        {"turn_id": 3, "role": "user", "content": "What about a direct flight?", "timestamp": ts(15)},
        {"turn_id": 4, "role": "assistant", "content": "Searching for direct flights now.", "timestamp": ts(20)},
        {"turn_id": 5, "role": "user", "content": "And can you check for upgrades?", "timestamp": ts(30)},
        {"turn_id": 6, "role": "assistant", "content": "Let me check upgrade availability.", "timestamp": ts(35)},
        {"turn_id": 7, "role": "user", "content": "OK, what's the best option you found?", "timestamp": ts(50)},
        {"turn_id": 8, "role": "assistant",
         "content": forgotten if forget_prefs else remembered,
         "timestamp": ts(55)},
    ]

    return {
        "conversation_id": conv_id,
        "agent_version": agent_version,
        "turns": turns,
        "feedback": {
            "user_rating": 2 if forget_prefs else 5,
            "ops_review": {
                "quality": "poor" if forget_prefs else "good",
                "notes": "Agent forgot stated preferences" if forget_prefs else None,
            },
            "annotations": [{
                "type": "coherence",
                "label": "poor" if forget_prefs else "good",
                "annotator_id": "ann_002",
                "confidence": 0.88,
            }],
        },
        "metadata": {
            "total_latency_ms": 1100 if forget_prefs else 780,
            "mission_completed": not forget_prefs,
        },
    }


def _annotator_disagreement_conv(conv_id: str, agent_version: str, agree: bool) -> dict:
    """Travel insurance recommendation — annotators agree or disagree on quality."""
    annotations = (
        [{"type": "response_quality", "label": "good", "annotator_id": f"ann_{100+i}", "confidence": 0.9}
         for i in range(3)]
        if agree
        else [
            {"type": "response_quality", "label": "good", "annotator_id": "ann_201", "confidence": 0.70},
            {"type": "response_quality", "label": "good", "annotator_id": "ann_202", "confidence": 0.65},
            {"type": "response_quality", "label": "poor", "annotator_id": "ann_203", "confidence": 0.80},
        ]
    )
    return {
        "conversation_id": conv_id,
        "agent_version": agent_version,
        "turns": [
            {"turn_id": 1, "role": "user",
             "content": "Can you recommend travel insurance for my trip to Japan?",
             "timestamp": ts(0)},
            {"turn_id": 2, "role": "assistant",
             "content": "For Japan travel, I'd recommend comprehensive coverage including medical evacuation. "
                        "Plans typically range from $50-150 for a two-week trip.",
             "timestamp": ts(4)},
        ],
        "feedback": {
            "user_rating": 3,
            "ops_review": {"quality": "good"},
            "annotations": annotations,
        },
        "metadata": {"total_latency_ms": 550, "mission_completed": True},
    }


# ── v2.3.0 — Baseline (high quality) ───────────────────────────────────────────

baseline_convs: list[dict] = []
for i in range(5):
    baseline_convs.append(_flight_conv(
        f"conv_v230_flight_{i+1:02d}", "v2.3.0",
        date_range=f"2024-06-{10+i:02d}/2024-06-{17+i:02d}",
        user_rating=5, mission_completed=True, latency_ms=680,
    ))
for i in range(5):
    baseline_convs.append(_hotel_conv(f"conv_v230_hotel_{i+1:02d}", "v2.3.0", user_rating=5))
for i in range(5):
    baseline_convs.append(_car_conv(f"conv_v230_car_{i+1:02d}", "v2.3.0", user_rating=4))


# ── v2.3.1 — Regression (accuracy drops sharply) ───────────────────────────────
# Prompt update introduced ambiguous date handling — 15 bad conversations.

BAD_DATES = ["next Monday", "Jan 22-28", "1/22/2024", "next week", "Monday to Friday"]
v231_flight_convs: list[dict] = []
for i, bad_date in enumerate(BAD_DATES):
    v231_flight_convs.append(_flight_conv(
        f"conv_v231_flight_{i+1:02d}", "v2.3.1",
        date_range=bad_date,
        user_rating=1, mission_completed=False, latency_ms=1400,
        ops_notes="poor - wrong date format caused tool failure",
    ))

v231_context_convs: list[dict] = []
for i in range(5):
    v231_context_convs.append(_context_loss_conv(
        f"conv_v231_context_{i+1:02d}", "v2.3.1", forget_prefs=True,
    ))

v231_disagree_convs: list[dict] = []
for i in range(5):
    v231_disagree_convs.append(_annotator_disagreement_conv(
        f"conv_v231_disagree_{i+1:02d}", "v2.3.1", agree=False,
    ))

regression_convs = v231_flight_convs + v231_context_convs + v231_disagree_convs


# ── v2.3.2 — Recovery (accuracy recovers strongly) ─────────────────────────────
# Date format bug fixed in prompt. Context window handling improved.
# All tool calls succeed; user ratings are back to 5.

recovery_convs: list[dict] = []
for i in range(5):
    recovery_convs.append(_flight_conv(
        f"conv_v232_flight_{i+1:02d}", "v2.3.2",
        date_range=f"2024-07-{10+i:02d}/2024-07-{17+i:02d}",
        user_rating=5, mission_completed=True, latency_ms=380,
        ops_notes="excellent - date format fix confirmed working",
    ))
for i in range(5):
    recovery_convs.append(_hotel_conv(
        f"conv_v232_hotel_{i+1:02d}", "v2.3.2", user_rating=5, latency_ms=410,
    ))
for i in range(5):
    recovery_convs.append(_context_loss_conv(
        f"conv_v232_context_{i+1:02d}", "v2.3.2", forget_prefs=False,
    ))


# ── Execution ───────────────────────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"  AI Agent Evaluation Pipeline — Demo Seed")
print(f"  API: {API}")
print(f"{'='*65}")

# ── STEP 1: Seed v2.3.0 ────────────────────────────────────────────────────────
print(f"\nSTEP 1: Seeding {len(baseline_convs)} baseline conversations (v2.3.0)...")
baseline_ids = []
for conv in baseline_convs:
    result = post(f"{API}/conversations", conv)
    cid = result.get("conversation_id", conv["conversation_id"])
    baseline_ids.append(cid)
print(f"  ✓ Ingested {len(baseline_ids)} conversations")

# ── STEP 2: Evaluate v2.3.0 ───────────────────────────────────────────────────
print(f"\nSTEP 2: Evaluating {len(baseline_ids)} baseline conversations (v2.3.0)...")
r = requests.post(f"{API}/evaluations/evaluate/batch", json=baseline_ids, timeout=180)
if r.ok:
    print(f"  ✓ Evaluated {r.json().get('evaluated', 0)} conversations")
else:
    print(f"  ⚠  Batch eval failed ({r.status_code}) — falling back to individual evals")
    for cid in baseline_ids:
        requests.post(f"{API}/evaluations/evaluate/{cid}", timeout=60)

evals_v230 = get(f"{API}/evaluations", {"agent_version": "v2.3.0", "limit": 20})
v230_scores = [e["scores"]["overall"] for e in evals_v230.get("data", []) if "scores" in e]
v230_mean = sum(v230_scores) / len(v230_scores) if v230_scores else 0.0
print(f"  → v2.3.0 mean overall score: {v230_mean:.3f}")

# ── STEP 3: Seed v2.3.1 ────────────────────────────────────────────────────────
print(f"\nSTEP 3: Seeding {len(regression_convs)} regression conversations (v2.3.1)...")
regression_ids = []
for conv in regression_convs:
    result = post(f"{API}/conversations", conv)
    cid = result.get("conversation_id", conv["conversation_id"])
    regression_ids.append(cid)
print(f"  ✓ Ingested {len(regression_ids)} conversations")
print(f"    ├─ {len(v231_flight_convs)} with tool date-format regression")
print(f"    ├─ {len(v231_context_convs)} with context-loss (long conversations)")
print(f"    └─ {len(v231_disagree_convs)} with annotator disagreement")

# ── STEP 4: Evaluate v2.3.1 ───────────────────────────────────────────────────
print(f"\nSTEP 4: Evaluating {len(regression_ids)} regression conversations (v2.3.1)...")
r = requests.post(f"{API}/evaluations/evaluate/batch", json=regression_ids, timeout=180)
if r.ok:
    print(f"  ✓ Evaluated {r.json().get('evaluated', 0)} conversations")
else:
    print(f"  ⚠  Batch eval failed ({r.status_code}) — falling back to individual evals")
    for cid in regression_ids:
        requests.post(f"{API}/evaluations/evaluate/{cid}", timeout=60)

evals_v231 = get(f"{API}/evaluations", {"agent_version": "v2.3.1", "limit": 30})
v231_scores = [e["scores"]["overall"] for e in evals_v231.get("data", []) if "scores" in e]
v231_mean = sum(v231_scores) / len(v231_scores) if v231_scores else 0.0
print(f"  → v2.3.1 mean overall score: {v231_mean:.3f}")

# ── STEP 5: Regression check v2.3.0 vs v2.3.1 ─────────────────────────────────
print(f"\nSTEP 5: Regression comparison (v2.3.0 → v2.3.1)...")
reg_result = post(f"{API}/regression/compare", {
    "baseline_version": "v2.3.0",
    "target_version": "v2.3.1",
})
if reg_result:
    is_reg = reg_result.get("is_regression", False)
    severity = reg_result.get("severity", "none")
    regressions = reg_result.get("regressions_detected", [])
    print(f"  {'⚠  REGRESSION DETECTED' if is_reg else '✓ No regression'}: severity={severity}")
    print(f"  Affected dimensions: {regressions}")
    if "dimensions" in reg_result:
        for dim, data in reg_result["dimensions"].items():
            delta_pct = data.get("delta_pct", 0)
            arrow = "↓" if delta_pct < 0 else "↑"
            print(f"    {dim}: {data.get('baseline_mean', 0):.3f} → "
                  f"{data.get('target_mean', 0):.3f} ({arrow}{abs(delta_pct):.1f}%)")
else:
    print("  ⚠  Regression compare failed (not enough data?)")

# ── STEP 6: Seed v2.3.2 ────────────────────────────────────────────────────────
print(f"\nSTEP 6: Seeding {len(recovery_convs)} recovery conversations (v2.3.2)...")
recovery_ids = []
for conv in recovery_convs:
    result = post(f"{API}/conversations", conv)
    cid = result.get("conversation_id", conv["conversation_id"])
    recovery_ids.append(cid)
print(f"  ✓ Ingested {len(recovery_ids)} conversations")

# ── STEP 7: Evaluate v2.3.2 ───────────────────────────────────────────────────
print(f"\nSTEP 7: Evaluating {len(recovery_ids)} recovery conversations (v2.3.2)...")
r = requests.post(f"{API}/evaluations/evaluate/batch", json=recovery_ids, timeout=180)
if r.ok:
    print(f"  ✓ Evaluated {r.json().get('evaluated', 0)} conversations")
else:
    print(f"  ⚠  Batch eval failed ({r.status_code}) — falling back to individual evals")
    for cid in recovery_ids:
        requests.post(f"{API}/evaluations/evaluate/{cid}", timeout=60)

evals_v232 = get(f"{API}/evaluations", {"agent_version": "v2.3.2", "limit": 30})
v232_scores = [e["scores"]["overall"] for e in evals_v232.get("data", []) if "scores" in e]
v232_mean = sum(v232_scores) / len(v232_scores) if v232_scores else 0.0
print(f"  → v2.3.2 mean overall score: {v232_mean:.3f}")

# ── STEP 8: Recovery check v2.3.1 vs v2.3.2 ───────────────────────────────────
print(f"\nSTEP 8: Regression comparison (v2.3.1 → v2.3.2)...")
rec_result = post(f"{API}/regression/compare", {
    "baseline_version": "v2.3.1",
    "target_version": "v2.3.2",
})
if rec_result:
    is_reg = rec_result.get("is_regression", False)
    severity = rec_result.get("severity", "none")
    print(f"  {'⚠  Still a regression' if is_reg else '✓ Recovery confirmed'}: severity={severity}")
    if "dimensions" in rec_result:
        for dim, data in rec_result["dimensions"].items():
            delta_pct = data.get("delta_pct", 0)
            arrow = "↓" if delta_pct < 0 else "↑"
            print(f"    {dim}: {data.get('baseline_mean', 0):.3f} → "
                  f"{data.get('target_mean', 0):.3f} ({arrow}{abs(delta_pct):.1f}%)")
else:
    print("  ⚠  Recovery compare failed (not enough data?)")

# ── STEP 9: Generate improvement suggestions ───────────────────────────────────
print(f"\nSTEP 9: Generating improvement suggestions...")
r = requests.post(f"{API}/suggestions/generate?last_n=50", timeout=300)
suggestions_generated = 0
if r.ok:
    sugg_result = r.json()
    patterns = sugg_result.get("patterns_found", 0)
    suggestions_generated = sugg_result.get("suggestions_generated", 0)
    print(f"  ✓ Patterns found: {patterns}")
    print(f"  ✓ Suggestions generated: {suggestions_generated}")
    for s in sugg_result.get("suggestions", [])[:3]:
        stype = s.get("type", "?").upper()
        target = s.get("target", "n/a")
        conf = s.get("confidence", 0)
        text = s.get("suggestion", "")[:120]
        print(f"\n    [{stype}] {target} (confidence={conf:.2f})")
        print(f"    → {text}...")
else:
    print(f"  ⚠  Suggestion generation error: {r.status_code}")

# ── STEP 10: Seed meta-evaluation calibration data ────────────────────────────
print(f"\nSTEP 10: Seeding meta-evaluation calibration data...")
calib_count = 0
# v2.3.0 baselines — high human scores
for cid in baseline_ids[:5]:
    r = requests.post(f"{API}/meta/calibrate", json={
        "conversation_id": cid,
        "human_scores": {"overall": 0.90, "response_quality": 0.88,
                         "tool_accuracy": 0.95, "coherence": 0.92},
    }, timeout=30)
    if r.ok:
        calib_count += 1

# v2.3.1 regressions — low human scores
for cid in regression_ids[:5]:
    r = requests.post(f"{API}/meta/calibrate", json={
        "conversation_id": cid,
        "human_scores": {"overall": 0.45, "response_quality": 0.50,
                         "tool_accuracy": 0.30, "coherence": 0.48},
    }, timeout=30)
    if r.ok:
        calib_count += 1

# v2.3.2 recovery — high human scores, better than baseline
for cid in recovery_ids[:5]:
    r = requests.post(f"{API}/meta/calibrate", json={
        "conversation_id": cid,
        "human_scores": {"overall": 0.95, "response_quality": 0.94,
                         "tool_accuracy": 0.98, "coherence": 0.96},
    }, timeout=30)
    if r.ok:
        calib_count += 1

print(f"  ✓ Created {calib_count} calibration records")

# ── Summary ─────────────────────────────────────────────────────────────────────
alerts_summary = get(f"{API}/alerts/summary")
open_alerts = alerts_summary.get("total_open", 0)
by_severity = alerts_summary.get("by_severity", {})

all_conversations = baseline_convs + regression_convs + recovery_convs

delta_230_231 = ((v231_mean - v230_mean) / v230_mean * 100) if v230_mean > 0 else 0
delta_231_232 = ((v232_mean - v231_mean) / v231_mean * 100) if v231_mean > 0 else 0

print(f"\n{'='*65}")
print(f"  SEED COMPLETE")
print(f"{'='*65}")
print(f"  Seeded {len(all_conversations)} conversations across 3 versions:")
print(f"    ├─ {len(baseline_convs):2d}  v2.3.0  baseline   (score: {v230_mean:.3f})")
print(f"    ├─ {len(regression_convs):2d}  v2.3.1  regression (score: {v231_mean:.3f}  {delta_230_231:+.1f}%)")
print(f"    └─ {len(recovery_convs):2d}  v2.3.2  recovery   (score: {v232_mean:.3f}  {delta_231_232:+.1f}%)")
print(f"")
print(f"  v2.3.1 breakdown:")
print(f"    ├─ {len(v231_flight_convs)} with tool date-format regression")
print(f"    ├─ {len(v231_context_convs)} with context-loss across long turns")
print(f"    └─ {len(v231_disagree_convs)} with annotator disagreement")
print(f"")
print(f"  Suggestions generated : {suggestions_generated}")
print(f"  Open alerts           : {open_alerts}  {by_severity}")
print(f"")
print(f"  Ready for demo:")
print(f"    API docs   → {args.base_url}/docs")
print(f"    Dashboard  → http://localhost:8501")
print(f"{'='*65}\n")
