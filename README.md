# AI Agent Evaluation Pipeline

A pipeline for evaluating multi-turn AI agent conversations. It ingests conversation logs along with feedback signals, runs them through four categories of evaluators, detects regressions between agent versions, fires alerts when things go wrong, integrates human annotations, and — the part I found most interesting to build — automatically generates concrete improvement suggestions for both prompts and tool schemas based on observed failure patterns.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                 │
│  Overview │ Conversations │ Evaluations │ Suggestions │
│  Meta-Eval │ Alerts │ Regression                     │
└─────────────────────┬───────────────────────────────┘
                      │ HTTP
┌─────────────────────▼───────────────────────────────┐
│                   FastAPI Backend                     │
│                                                      │
│  ┌──────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │ Ingestion│  │ Evaluation │  │  Self-Update     │  │
│  │ Layer    │  │ Pipeline   │  │  Engine          │  │
│  │          │  │            │  │                  │  │
│  │ REST API │  │ Heuristic  │  │ Pattern Detector │  │
│  │ Batch +  │  │ Tool Call  │  │ Prompt Suggester │  │
│  │ Realtime │  │ Coherence  │  │ Tool Suggester   │  │
│  │          │  │ LLM-Judge  │  │                  │  │
│  └────┬─────┘  └─────┬──────┘  └────────┬─────────┘  │
│       │              │                   │            │
│  ┌────▼──────────────▼───────────────────▼─────────┐ │
│  │              SQLite / PostgreSQL                  │ │
│  └──────────────────────────────────────────────────┘ │
│                                                      │
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │  Regression  │  │  Alerting   │  │  Meta-Eval  │ │
│  │  Detector    │  │  System     │  │  Calibrator │ │
│  │  Comparator  │  │  Webhook    │  │  + Correl.  │ │
│  └──────────────┘  └─────────────┘  └─────────────┘ │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │  Structured Logging + Request Tracing        │   │
│  │  (X-Request-ID | JSON logs | /metrics)       │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| API Framework | FastAPI + Pydantic v2 | Auto-generates OpenAPI docs, async-native, and the request validation is quite solid |
| Database | SQLite + SQLAlchemy async | No setup needed for local development; changing `DATABASE_URL` is all it takes to move to Postgres |
| LLM | OpenAI-compatible API | Powers the LLM-as-Judge evaluator and suggestion generation; falls back gracefully if no key is set |
| Task execution | asyncio.gather | LLM calls are I/O-bound, so running all four evaluators concurrently made more sense than adding Celery |
| UI | Streamlit | Fast to iterate on; talks directly to the FastAPI endpoints |
| Container | Docker + docker-compose | Two services (api + ui) sharing a volume for the SQLite file |
| Testing | pytest + pytest-asyncio | In-memory SQLite keeps tests isolated and fast |

---

## Quick Start

### Option A — Docker (recommended)

```bash
git clone <repo> && cd ai-agent-evaluation-pipeline

cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

docker-compose up --build

# In a separate terminal, seed some demo data
pip install requests
python seed_data.py
```

### Option B — Local development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY in .env

# Terminal 1 — API
uvicorn app.main:app --reload

# Terminal 2 — Streamlit dashboard
pip install -r requirements-ui.txt
streamlit run streamlit_app/app.py

# Terminal 3 — seed demo data
python seed_data.py
```

Once both services are up:
- Swagger docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501

---

## Performance Benchmarks

To run the load test against a locally running instance:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000

# In another terminal
python benchmarks/load_test.py --count 100
```

Numbers from a single-process run on an M1 MacBook (roughly similar on a 4-core cloud VM):

| Operation | Throughput | 1,000/min target |
|-----------|-----------|-----------------|
| Ingestion (single) | ~2,500/min | ✓ |
| Ingestion (batch) | ~6,700/min | ✓ |
| Evaluation (mock LLM) | ~370/min | ⚠ LLM-bound |
| Read queries | ~5,800/min | ✓ |

Evaluation throughput is largely bottlenecked by LLM API latency, which is typically 2–4 seconds per call. With real LLM calls that number will be lower, but it scales quite linearly with async workers — 3 Celery workers gets you to around 1,100/min. See [What I'd Do With More Time](#what-id-do-with-more-time) for the production-oriented approach.

---

## Demo Walkthrough

After running `docker-compose up` and `python seed_data.py`, here is what you should see across the dashboard pages:

1. **Overview page** — a regression alert banner at the top, a version comparison card showing `v2.3.0 avg 0.91` vs `v2.3.1 avg 0.74` (a drop of ~18.7%), and the eval-to-user-rating correlation sitting around `r ≈ 0.82`
2. **Alerts page** — two open alerts: the regression detection and an elevated tool failure rate; you can acknowledge and resolve them from here
3. **Regression page** — pick `v2.3.0` as baseline and `v2.3.1` as target to see the dimension-by-dimension bar chart, issue rate changes, and severity classification
4. **Suggestions page** — auto-generated fixes, for instance: `[TOOL] flight_search — add YYYY-MM-DD date format to parameter description` with a confidence of 0.87
5. **Meta-eval page** — scatter plot of auto-eval score vs user rating with a trend line and Pearson r, plus per-dimension calibration drift for each evaluator

The full loop looks something like this:
```
Agent outputs → Evaluations → Regression detected → Alert fired →
Pattern analyzed → Improvement suggested → Evaluator calibrated
```

---

## API Reference

Full interactive docs are available at `/docs`. A quick reference of the main endpoints:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/conversations` | Ingest a conversation |
| POST | `/api/v1/conversations/batch` | Batch ingest |
| POST | `/api/v1/evaluations/evaluate/{id}` | Run all evaluators on a conversation |
| POST | `/api/v1/evaluations/evaluate/batch` | Batch evaluate |
| POST | `/api/v1/regression/compare` | Compare two agent versions |
| POST | `/api/v1/regression/auto-check/{version}` | Auto-find nearest baseline and compare |
| GET  | `/api/v1/regression/versions` | Version timeline with mean scores |
| GET  | `/api/v1/regression/reports` | Stored regression reports |
| GET  | `/api/v1/alerts` | List alerts (filterable by status, type, severity) |
| PATCH| `/api/v1/alerts/{id}` | Acknowledge or resolve an alert |
| POST | `/api/v1/alerts/webhook/configure` | Set a webhook URL for alert delivery |
| GET  | `/api/v1/alerts/summary` | Open alert counts by severity |
| POST | `/api/v1/suggestions/generate` | Detect failure patterns and generate suggestions |
| PATCH| `/api/v1/suggestions/{id}` | Accept, reject, or mark a suggestion as implemented |
| POST | `/api/v1/meta/calibrate` | Submit human scores for calibration |
| GET  | `/api/v1/meta/correlation` | Pearson r between auto scores and user ratings |
| GET  | `/api/v1/meta/drift` | Evaluator drift analysis over a rolling window |
| GET  | `/api/v1/metrics` | Live pipeline metrics and per-evaluator latency stats |
| GET  | `/api/v1/events/stream` | SSE stream of real-time pipeline events |
| POST | `/api/v1/events/test` | Publish a test event to verify the SSE connection |

---

## Real-Time Event Stream

The pipeline pushes events over Server-Sent Events (SSE) as things happen — ingestion, evaluation, regression detection, alerts, and suggestions all publish to the stream:

```bash
curl -N http://localhost:8000/api/v1/events/stream
```

A few example events:

```
data: {"event_type": "evaluation_completed", "timestamp": "2024-01-15T10:30:05Z", "data": {"evaluation_id": "eval_abc123", "overall_score": 0.73, "issues_count": 2}, "conversation_id": "conv_xyz", "agent_version": "v2.3.1"}

data: {"event_type": "regression_detected", "timestamp": "2024-01-15T10:30:06Z", "data": {"severity": "critical", "baseline_version": "v2.3.0", "target_version": "v2.3.1", "regressions": ["tool_accuracy"]}, "agent_version": "v2.3.1"}

data: {"event_type": "alert_fired", "timestamp": "2024-01-15T10:30:06Z", "data": {"alert_id": "alert_001", "type": "regression", "severity": "critical", "title": "Regression detected in v2.3.1"}}
```

The event bus is in-process and has no external dependencies, so it works out of the box. For production use with multiple processes, swapping it out for Redis Pub/Sub or Kafka would be the natural next step.

---

## Architecture Decisions

**Regression detection**
Welch's t-test (using scipy if available, falling back to a simple threshold comparison otherwise) is used to compare score distributions between versions. A regression is only flagged when there is both a meaningful delta (above 5%) and statistical significance — flagging everything as a regression tends to desensitise people to alerts fairly quickly. Severity reflects how large the drop actually is, not just whether one occurred.

**Alert deduplication**
At most one alert of each type can be open at a time. If a quality-drop alert is already open, a second check will not create another one — the existing alert has to be resolved first. Without this, high-frequency evaluation workloads would generate a lot of noise.

**Correlation analysis**
Pearson r is computed in NumPy after normalising user ratings to a 0–1 scale. The "best dimension" field in the response tells you which evaluator dimension tracks most closely with how users actually rate the agent — useful for deciding how much weight to give each dimension in the overall score.

**Structured logging and request tracing**
Each request is assigned a `uuid4` request ID via `ContextVar`. Every log line emitted during that request — evaluator calls, LLM calls, DB writes — carries the same ID. This makes it straightforward to reconstruct the full trace for a single request by filtering on one field.

**Why SQLite?**
It removes all infrastructure setup for local development and testing. Switching to Postgres requires changing exactly one environment variable: `DATABASE_URL=postgresql+asyncpg://...`.

**Why asyncio instead of a task queue?**
All four evaluators are kicked off concurrently with `asyncio.gather`. Since each one is waiting on either an LLM API call or a DB read, the total wall time for the pipeline is roughly equal to the slowest single evaluator — not the sum of all four. A task queue like Celery would add operational complexity without helping here.

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

What is covered:
- `test_ingestion.py` — conversation ingestion and retrieval
- `test_evaluators.py` — all four evaluators
- `test_feedback.py` — annotation aggregation and confidence-based routing
- `test_self_update.py` — pattern detection and suggestion generation
- `test_meta_eval.py` — calibration, drift detection, and correlation
- `test_regression.py` — regression detection across 5 scenarios including edge cases
- `test_alerts.py` — alert creation, deduplication, and status transitions
- `test_e2e.py` — full pipeline end-to-end test covering the complete flywheel
- `test_events.py` — EventBus unit tests and SSE endpoint integration

---

## What I'd Do With More Time

1. **Postgres + Redis** — SQLite works fine for a single process but will struggle with concurrent writes. Postgres for the DB and Redis-backed Celery for evaluation workers would sort that out.
2. **Durable event streaming** — the in-process SSE bus is convenient but doesn't survive restarts and can't fan out across multiple processes. A Kafka consumer would be the right replacement at scale.
3. **A/B test integration** — automatically split traffic between agent versions and run significance tests on the resulting evaluation scores, rather than relying on manual regression comparisons.
4. **Prompt versioning** — track which version of the system prompt was active for each agent version, so generated suggestions can be tied to the specific lines that need changing.
5. **Evaluator fine-tuning** — use the accumulated human annotation data to adjust LLM-as-Judge rubric weights, rather than keeping them fixed.
6. **Prometheus + Grafana** — the `/metrics` endpoint already exposes evaluator latency stats; exporting those in Prometheus format would let ops teams monitor the pipeline alongside everything else.
7. **RBAC** — right now anyone with API access can do anything. Separating ops access (acknowledge alerts) from ML engineer access (modify evaluator config) would matter in a team setting.
