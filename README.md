# AI Agent Evaluation Pipeline

An automated pipeline for continuously evaluating and improving multi-turn AI agent conversations. It ingests conversation logs with full feedback signals, runs them through four categories of evaluators, detects regressions between agent versions, fires alerts when quality drops, and — the most important part — automatically generates concrete improvement suggestions for both prompts and tool schemas based on observed failure patterns.

---

## Live Demo

The project is fully deployed on **Render** and ready to explore without any local setup.

| Service | URL |
|---------|-----|
| **Streamlit Dashboard (UI)** | https://ai-agent-evaluation-pipeline-ui.onrender.com |
| **FastAPI Backend (Swagger Docs)** | https://ai-agent-evaluation-pipeline.onrender.com/docs |
| **REST API Base URL** | https://ai-agent-evaluation-pipeline.onrender.com/api/v1 |

> **Note on cold starts** — Render's free tier spins services down after 15 minutes of inactivity. If the page takes 20–30 seconds to load on the first visit, that is expected. It will be fast after the initial wake-up.

Both services are backed by a **Render-hosted PostgreSQL database**, so all data persists across deployments and there is no need to re-seed after updates.

---

## Table of Contents

- [Live Demo](#live-demo)
- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Evaluation Framework](#evaluation-framework)
- [Feedback Integration](#feedback-integration)
- [Self-Updating Mechanism](#self-updating-mechanism)
- [Meta-Evaluation (Improving the Evals Themselves)](#meta-evaluation-improving-the-evals-themselves)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [API Reference](#api-reference)
- [Real-Time Event Stream](#real-time-event-stream)
- [Sample Data Schemas](#sample-data-schemas)
- [Performance Benchmarks](#performance-benchmarks)
- [Running Tests](#running-tests)
- [Demo Walkthrough](#demo-walkthrough)
- [Architecture Decisions](#architecture-decisions)
- [Scaling Strategy](#scaling-strategy)
- [What I'd Do With More Time](#what-id-do-with-more-time)

---

## Overview

Modern AI agents handle complex, multi-turn interactions at scale. To keep them reliable, teams need infrastructure that can:

- Detect quality regressions **before** they start affecting real users
- Align automated evaluation scores with how users actually rate responses
- Identify failure patterns across both prompts and tool usage
- Generate actionable improvement suggestions automatically
- Scale comfortably to production-level throughput (~1,000+ conversations/minute)

This pipeline addresses all of the above. It is designed to be the evaluation backbone for any AI agent system that uses multi-turn conversations and tool calls.

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
│  │                    PostgreSQL                       │ │
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

## How It Works

The full evaluation flywheel looks like this:

```
Agent outputs
    → Conversation ingested (single or batch)
    → All 4 evaluators run concurrently
    → Regression check triggered automatically
    → Alert fired if regression or quality drop detected
    → Failure patterns analysed
    → Improvement suggestions generated
    → Human annotations fed back in
    → Evaluator calibrated against human ground truth
    → Better evaluations next cycle
```

Each conversation goes through the full pipeline in one API call. The results — scores, issues, suggestions — are stored and queryable via REST or visible in the Streamlit dashboard.

---

## Evaluation Framework

The pipeline runs four evaluators in parallel for every conversation. All four run concurrently using `asyncio.gather`, so the total time is approximately equal to the slowest single evaluator rather than the sum of all four.

### 1. LLM-as-Judge

Uses an LLM (GPT-4.1-mini by default) to score response quality across three dimensions:

| Dimension | What it measures |
|-----------|-----------------|
| Response Quality | Helpfulness, clarity, accuracy of the assistant's replies |
| Coherence | Whether context is maintained and responses are consistent across turns |
| Overall | Combined quality score normalised to 0–1 |

Falls back gracefully if no API key is set (scores a neutral value and logs a warning).

### 2. Tool Call Evaluator

Specifically designed for agents that make tool calls. It checks:

| Check | Description |
|-------|-------------|
| **Selection accuracy** | Was the correct tool chosen for the task at hand? |
| **Parameter accuracy** | Were the right parameters extracted from the conversation context? |
| **Hallucinated parameters** | Did the model invent parameter values that were not mentioned? |
| **Execution success** | Did the tool call actually succeed at runtime? |

This evaluator is the most useful for catching prompt regressions — for example, after a prompt update causes the agent to start passing dates in the wrong format to `flight_search`.

### 3. Multi-Turn Coherence Evaluator

Checks whether the conversation holds together across multiple turns:

| Check | Description |
|-------|-------------|
| **Context maintenance** | Does the agent remember information mentioned in earlier turns? |
| **Contradiction detection** | Does the agent contradict something it said previously? |
| **Reference resolution** | Does the agent correctly handle pronouns and implicit references to prior context? |

This evaluator is particularly useful for catching "context loss" issues — for example, when agents forget user preferences mentioned in turn 1 after the conversation grows beyond 5 turns.

### 4. Heuristic Evaluator

Rule-based checks that do not require an LLM:

| Check | Description |
|-------|-------------|
| **Latency** | Was the response time within the configured threshold (default: 1000ms)? |
| **Mission completion** | Was `mission_completed` flagged as true in the conversation metadata? |
| **Format compliance** | Are required fields present and correctly formatted? |

---

## Feedback Integration

The pipeline ingests three types of feedback signals alongside each conversation:

| Signal | Source | How it is used |
|--------|--------|---------------|
| **User ratings** | 1–5 star rating from end users | Normalised to 0–1 and correlated against automated scores |
| **Ops review** | Quality notes from operations teams | Stored and searchable; feeds into the meta-evaluation layer |
| **Human annotations** | Labels from one or more annotators | Aggregated and weighted before influencing evaluator calibration |

### Handling Annotator Disagreement

When multiple annotators label the same conversation differently, the pipeline uses the following logic:

- **High agreement** (above the configured threshold, default 70%): The majority label is used with high confidence
- **Disagreement** (below threshold): Confidence score is reduced and the case is flagged for review
- **Confidence-based routing**:
  - Score above 0.85 → auto-labelled, no human needed
  - Score between 0.60 and 0.85 → routed to a human reviewer
  - Score below 0.60 → flagged as low-confidence and held back

This prevents noisy annotations from corrupting evaluator calibration.

---

## Self-Updating Mechanism

The pipeline does not just report scores — it actively identifies what needs to be fixed. This is handled by two components:

### Pattern Detector

Looks at the last N evaluations and groups failure modes:
- Repeated tool parameter errors (e.g., wrong date format)
- Low coherence scores concentrated in longer conversations
- High hallucination rates for specific tools

### Prompt Suggester

For each detected failure pattern, generates a concrete suggestion:

```json
{
  "type": "prompt",
  "target": "flight_search",
  "suggestion": "Add explicit date format instruction: 'Always use YYYY-MM-DD format for date parameters'",
  "rationale": "15% of flight_search calls have failed due to incorrect date formats since v2.3.1",
  "confidence": 0.87
}
```

### Tool Suggester

For tool-related failures, suggests schema improvements:

```json
{
  "type": "tool",
  "target": "flight_search.date_range",
  "suggestion": "Update parameter description to specify ISO 8601 format (YYYY-MM-DD/YYYY-MM-DD)",
  "rationale": "Parameter description is ambiguous about date format; agents are inferring different formats",
  "confidence": 0.82
}
```

Suggestions have a lifecycle: `pending` → `accepted` / `rejected` → `implemented`. This allows teams to track which suggestions were acted on and measure their impact in subsequent evaluations.

---

## Meta-Evaluation (Improving the Evals Themselves)

This is what makes the pipeline self-improving over time. The meta-evaluation layer measures how well the automated evaluators are actually performing.

### Calibration

When human annotations are submitted, the system compares them against the automated scores for the same conversation. If there is consistent divergence — for example, the LLM-judge always rates a certain conversation type higher than humans do — a calibration record is stored.

```
POST /api/v1/meta/calibrate
→ Submits human scores for a conversation
→ Computes delta between auto and human scores
→ Stores calibration record for drift tracking
```

### Drift Detection

Checks whether automated evaluator scores are drifting away from human ground truth over time. Returns per-dimension drift values for each evaluator so teams can see which dimensions are becoming unreliable.

### Correlation Analysis

Computes Pearson r between automated evaluation scores and user ratings. A high correlation means the pipeline is measuring what users actually care about. The `best_dimension` field identifies which evaluator dimension tracks most closely with user satisfaction.

The flywheel this creates:
```
Agent outputs → Evaluations → Human feedback → Better evaluators → Better evaluations
```

---

## Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| API Framework | FastAPI + Pydantic v2 | Auto-generates OpenAPI docs, async-native, strong request validation |
| Database | PostgreSQL + SQLAlchemy async | Persistent, production-grade storage. Used both locally (via external Render connection) and in the deployed environment (via Render's internal connection). SQLite is still supported locally for quick offline testing by changing one env variable. |
| LLM | OpenAI-compatible API (GPT-4.1-mini) | Powers LLM-as-Judge and suggestion generation; graceful fallback if key is absent |
| Task execution | asyncio.gather | All 4 evaluators are I/O-bound; running them concurrently keeps latency low |
| UI | Streamlit | Quick to iterate; talks directly to the FastAPI endpoints |
| Container | Docker + docker-compose | Two services (api + ui) sharing a volume for the SQLite file |
| Testing | pytest + pytest-asyncio | In-memory SQLite keeps tests isolated and fast |
| Stats | scipy / numpy | Welch's t-test for regression detection; Pearson r for correlation |

---

## Project Structure

```
ai-agent-evaluation-pipeline/
├── app/
│   ├── main.py                    # FastAPI app setup, router registration
│   ├── config.py                  # Settings loaded from .env
│   ├── database.py                # SQLAlchemy async engine + session factory
│   ├── models/
│   │   ├── schemas.py             # Pydantic request/response models
│   │   └── db_models.py           # SQLAlchemy ORM models
│   ├── routers/
│   │   ├── conversations.py       # Ingestion endpoints
│   │   ├── evaluations.py         # Evaluation trigger endpoints
│   │   ├── regression.py          # Version comparison endpoints
│   │   ├── alerts.py              # Alert management endpoints
│   │   ├── suggestions.py         # Improvement suggestions endpoints
│   │   ├── meta.py                # Meta-evaluation endpoints
│   │   └── events.py              # SSE stream endpoint
│   ├── evaluators/
│   │   ├── base.py                # Abstract evaluator base class
│   │   ├── heuristic.py           # Rule-based checks (latency, format)
│   │   ├── tool_call.py           # Tool selection + parameter accuracy
│   │   ├── coherence.py           # Multi-turn context consistency
│   │   ├── llm_judge.py           # LLM-as-Judge scoring
│   │   └── pipeline.py            # Orchestrates all 4 evaluators concurrently
│   ├── feedback/
│   │   ├── aggregator.py          # Aggregates annotations, handles disagreement
│   │   └── routing.py             # Confidence-based routing logic
│   ├── self_update/
│   │   ├── pattern_detector.py    # Groups failures into patterns
│   │   ├── prompt_suggester.py    # Generates prompt improvement suggestions
│   │   └── tool_suggester.py      # Generates tool schema improvement suggestions
│   ├── meta_eval/
│   │   ├── calibrator.py          # Compares auto scores against human annotations
│   │   └── drift_detector.py      # Tracks evaluator accuracy over time
│   ├── regression/
│   │   ├── detector.py            # Statistical significance testing (Welch's t-test)
│   │   └── comparator.py          # Version-to-version comparison logic
│   ├── alerting/
│   │   └── alerts.py              # Alert creation, deduplication, webhook dispatch
│   ├── analytics/
│   │   └── correlation.py         # Pearson r between auto scores and user ratings
│   ├── events/
│   │   └── stream.py              # In-process SSE event bus
│   ├── middleware/
│   │   └── logging.py             # Request ID injection, structured JSON logging
│   └── services/
│       ├── evaluation_service.py  # Coordinates the full evaluation flow
│       └── llm_client.py          # Thin wrapper around OpenAI-compatible API
├── streamlit_app/
│   ├── app.py                     # Entry point for the Streamlit dashboard
│   ├── utils.py                   # Shared helpers (API calls, formatting)
│   └── pages/
│       ├── 01_conversations.py    # Conversation browser
│       ├── 02_evaluations.py      # Evaluation results viewer
│       ├── 03_suggestions.py      # Improvement suggestions
│       ├── 04_regression.py       # Version comparison charts
│       ├── 05_alerts.py           # Alert management
│       └── 06_meta_eval.py        # Meta-evaluation and calibration
├── tests/
│   ├── conftest.py                # Shared fixtures (in-memory DB, test client)
│   ├── test_ingestion.py
│   ├── test_evaluators.py
│   ├── test_feedback.py
│   ├── test_self_update.py
│   ├── test_regression.py
│   ├── test_alerts.py
│   ├── test_meta_eval.py
│   ├── test_e2e.py
│   └── test_events.py
├── benchmarks/
│   └── load_test.py               # Throughput test (configurable count)
├── seed_data.py                   # Populates the DB with realistic demo conversations
├── Dockerfile.api
├── Dockerfile.ui
├── docker-compose.yml
├── requirements.txt
├── requirements-ui.txt
└── .env.example
```

---

## Quick Start

### Option A — Docker (Recommended)

```bash
git clone <repo-url> && cd ai-agent-evaluation-pipeline

cp .env.example .env
# Open .env and set:
#   OPENAI_API_KEY=sk-...
#   DATABASE_URL=postgresql+asyncpg://<user>:<password>@<host>/<db>

docker-compose up --build

# In a separate terminal, load demo data
pip install requests
python seed_data.py
```

### Option B — Local Development

```bash
# Create and activate a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install API dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Set the following in .env:
#   OPENAI_API_KEY=sk-...
#   DATABASE_URL=postgresql+asyncpg://<user>:<password>@<host>/<db>
#
# If you do not have a Postgres instance handy, you can use SQLite for local testing:
#   DATABASE_URL=sqlite+aiosqlite:///./eval_pipeline.db

# Terminal 1 — Start the API server
uvicorn app.main:app --reload

# Terminal 2 — Start the Streamlit dashboard
pip install -r requirements-ui.txt
streamlit run streamlit_app/app.py

# Terminal 3 — Load demo data
python seed_data.py
```

Once both services are running:

| Service | URL |
|---------|-----|
| Swagger API docs | http://localhost:8000/docs |
| Streamlit dashboard | http://localhost:8501 |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(empty)* | Required for LLM-as-Judge and suggestion generation. Pipeline works without it but LLM-powered features will be skipped. |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL is used in production and recommended locally. For quick offline testing you can switch to `sqlite+aiosqlite:///./eval_pipeline.db` without any other changes. |
| `LLM_MODEL` | `gpt-4.1-mini` | Any OpenAI-compatible model identifier |
| `LOG_LEVEL` | `INFO` | Standard Python log levels |
| `LATENCY_THRESHOLD_MS` | `1000` | Latency above this value triggers a heuristic warning |
| `ANNOTATOR_AGREEMENT_THRESHOLD` | `0.7` | Minimum agreement ratio before a disagreement is flagged |
| `CONFIDENCE_AUTO_LABEL_THRESHOLD` | `0.85` | Confidence above this → auto-label, no human needed |
| `CONFIDENCE_HUMAN_REVIEW_THRESHOLD` | `0.6` | Confidence below this → flagged as low-confidence |
| `REGRESSION_CHECK_EVAL_THRESHOLD` | `5` | Auto-trigger a regression check after every N evaluations |

---

## API Reference

Full interactive documentation is available at `/docs` once the server is running. Quick reference below.

### Conversations

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/conversations` | Ingest a single conversation |
| POST | `/api/v1/conversations/batch` | Batch ingest multiple conversations |
| GET  | `/api/v1/conversations` | List conversations (`limit`, `offset`, `agent_version`, `has_feedback`) |
| GET  | `/api/v1/conversations/{conversation_id}` | Get a specific conversation |

### Evaluations

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/evaluations/evaluate/{conversation_id}` | Run all evaluators on one conversation |
| POST | `/api/v1/evaluations/evaluate/batch` | Batch evaluate multiple conversations |
| GET  | `/api/v1/evaluations` | List evaluations (`limit`, `offset`, `agent_version`, `min_score`, `max_score`) |
| GET  | `/api/v1/evaluations/{evaluation_id}` | Get a specific evaluation |

### Regression

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/regression/compare` | Compare evaluation scores between two agent versions |
| POST | `/api/v1/regression/auto-check/{version}` | Auto-find nearest baseline and run comparison |
| GET  | `/api/v1/regression/versions` | All known agent versions with summary stats |
| GET  | `/api/v1/regression/reports` | List stored regression reports (`limit`, `offset`, `is_regression`) |

### Alerts

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/api/v1/alerts` | List alerts (`status`, `type`, `severity`) |
| GET  | `/api/v1/alerts/summary` | Open alert counts grouped by severity |
| GET  | `/api/v1/alerts/{alert_id}` | Get a specific alert |
| PATCH| `/api/v1/alerts/{alert_id}` | Update alert status (`open` / `acknowledged` / `resolved`) |
| POST | `/api/v1/alerts/webhook/configure` | Configure a webhook URL for alert notifications |

### Suggestions

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/suggestions/generate` | Detect failure patterns and generate improvement suggestions |
| GET  | `/api/v1/suggestions` | List suggestions (`type`, `status`, `min_confidence`) |
| GET  | `/api/v1/suggestions/summary` | Suggestions grouped by type and status |
| GET  | `/api/v1/suggestions/{suggestion_id}` | Get a specific suggestion |
| PATCH| `/api/v1/suggestions/{suggestion_id}` | Update suggestion status (`accepted` / `rejected` / `implemented`) |

### Meta-Evaluation

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/meta/calibrate` | Submit human scores to calibrate against automated scores |
| GET  | `/api/v1/meta/calibration` | Summary of recent calibration records |
| GET  | `/api/v1/meta/correlation` | Pearson r between auto scores and user ratings |
| GET  | `/api/v1/meta/drift` | Evaluator drift across all dimensions |
| GET  | `/api/v1/meta/drift/{evaluator_name}` | Drift for a specific evaluator |

### Events and Metrics

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/api/v1/events/stream` | SSE stream of real-time pipeline events |
| POST | `/api/v1/events/test` | Publish a test event to verify SSE connection |
| GET  | `/api/v1/metrics` | Live pipeline metrics and per-evaluator latency stats |

---

## Real-Time Event Stream

The pipeline pushes events over Server-Sent Events (SSE) as things happen — ingestion, evaluation, regression detection, alerts, and suggestions all publish to the stream.

```bash
curl -N http://localhost:8000/api/v1/events/stream
```

Example events:

```
data: {"event_type": "evaluation_completed", "timestamp": "2024-01-15T10:30:05Z", "data": {"evaluation_id": "eval_abc123", "overall_score": 0.73, "issues_count": 2}, "conversation_id": "conv_xyz", "agent_version": "v2.3.1"}

data: {"event_type": "regression_detected", "timestamp": "2024-01-15T10:30:06Z", "data": {"severity": "critical", "baseline_version": "v2.3.0", "target_version": "v2.3.1", "regressions": ["tool_accuracy"]}, "agent_version": "v2.3.1"}

data: {"event_type": "alert_fired", "timestamp": "2024-01-15T10:30:06Z", "data": {"alert_id": "alert_001", "type": "regression", "severity": "critical", "title": "Regression detected in v2.3.1"}}
```

The event bus is in-process with no external dependencies, so it works out of the box. For production use with multiple processes, swapping it out for Redis Pub/Sub or Kafka would be the natural next step.

---

## Sample Data Schemas

### Conversation Input

```json
{
  "conversation_id": "conv_abc123",
  "agent_version": "v2.3.1",
  "turns": [
    {
      "turn_id": 1,
      "role": "user",
      "content": "I need to book a flight to NYC next week",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    {
      "turn_id": 2,
      "role": "assistant",
      "content": "I'd be happy to help you book a flight to NYC...",
      "tool_calls": [
        {
          "tool_name": "flight_search",
          "parameters": {
            "destination": "NYC",
            "date_range": "2024-01-22/2024-01-28"
          },
          "result": {"status": "success", "flights": ["..."]},
          "latency_ms": 450
        }
      ],
      "timestamp": "2024-01-15T10:30:02Z"
    }
  ],
  "feedback": {
    "user_rating": 4,
    "ops_review": {
      "quality": "good",
      "notes": "Correct tool usage"
    },
    "annotations": [
      {
        "type": "tool_accuracy",
        "label": "correct",
        "annotator_id": "ann_001"
      }
    ]
  },
  "metadata": {
    "total_latency_ms": 1200,
    "mission_completed": true
  }
}
```

### Evaluation Output

```json
{
  "evaluation_id": "eval_xyz789",
  "conversation_id": "conv_abc123",
  "scores": {
    "overall": 0.87,
    "response_quality": 0.90,
    "tool_accuracy": 0.95,
    "coherence": 0.85
  },
  "tool_evaluation": {
    "selection_accuracy": 1.0,
    "parameter_accuracy": 0.95,
    "hallucinated_params": [],
    "execution_success": true
  },
  "issues_detected": [
    {
      "type": "latency",
      "severity": "warning",
      "description": "Response latency 1200ms exceeds 1000ms target"
    }
  ],
  "improvement_suggestions": [
    {
      "type": "prompt",
      "suggestion": "Add explicit date format instruction",
      "rationale": "Reduce date inference errors",
      "confidence": 0.72
    }
  ]
}
```

---

## Performance Benchmarks

To run the load test against a locally running instance:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000

# In a separate terminal
python benchmarks/load_test.py --count 100
```

Numbers from a single-process run on an M1 MacBook (roughly similar on a 4-core cloud VM):

| Operation | Throughput | 1,000/min target |
|-----------|-----------|-----------------|
| Ingestion (single) | ~2,500/min | ✓ |
| Ingestion (batch) | ~6,700/min | ✓ |
| Evaluation (mock LLM) | ~370/min | ⚠ LLM-bound |
| Read queries | ~5,800/min | ✓ |

Evaluation throughput is bottlenecked by LLM API latency (typically 2–4 seconds per call). With real LLM calls that number will be lower, but it scales linearly with async workers — 3 Celery workers gets to approximately 1,100/min. See [What I'd Do With More Time](#what-id-do-with-more-time) for the production-oriented approach.

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

Test coverage by file:

| File | What is tested |
|------|---------------|
| `test_ingestion.py` | Conversation ingestion and retrieval |
| `test_evaluators.py` | All four evaluators individually |
| `test_feedback.py` | Annotation aggregation and confidence-based routing |
| `test_self_update.py` | Pattern detection and suggestion generation |
| `test_regression.py` | Regression detection across 5 scenarios including edge cases |
| `test_alerts.py` | Alert creation, deduplication, and status transitions |
| `test_meta_eval.py` | Calibration, drift detection, and correlation |
| `test_e2e.py` | Full pipeline end-to-end test covering the complete flywheel |
| `test_events.py` | EventBus unit tests and SSE endpoint integration |

All tests use an in-memory SQLite database so they run without any external setup and do not touch the development database.

---

## Demo Walkthrough

The live dashboard is available at https://ai-agent-evaluation-pipeline-ui.onrender.com — all demo data is already seeded. If running locally, first run `docker-compose up` and `python seed_data.py`. Here is what you should see across the dashboard pages:

1. **Overview page** — a regression alert banner at the top, a version comparison card showing `v2.3.0 avg 0.91` vs `v2.3.1 avg 0.74` (a drop of ~18.7%), and the eval-to-user-rating correlation around `r ≈ 0.82`
2. **Alerts page** — two open alerts: the regression detection and an elevated tool failure rate; both can be acknowledged and resolved from here
3. **Regression page** — select `v2.3.0` as baseline and `v2.3.1` as target to see the dimension-by-dimension bar chart, issue rate changes, and severity classification
4. **Suggestions page** — auto-generated fixes, for example: `[TOOL] flight_search — add YYYY-MM-DD date format to parameter description` with confidence 0.87
5. **Meta-eval page** — scatter plot of auto-eval score vs user rating with a trend line and Pearson r, plus per-dimension calibration drift for each evaluator

---

## Architecture Decisions

**Regression detection**
Welch's t-test (via scipy if available, with a simple threshold comparison as fallback) is used to compare score distributions between versions. A regression is flagged only when there is both a meaningful delta (above 5%) and statistical significance. Severity reflects how large the drop actually is, not just whether one occurred.

**Alert deduplication**
At most one alert of each type can be open at a time. If a quality-drop alert is already open, a second check will not create another one — the existing alert must be resolved first. Without this, high-frequency evaluation workloads would generate a lot of noise.

**Correlation analysis**
Pearson r is computed in NumPy after normalising user ratings to a 0–1 scale. The `best_dimension` field in the response identifies which evaluator dimension tracks most closely with how users actually rate the agent — useful for deciding how much weight to assign to each dimension in the overall score.

**Structured logging and request tracing**
Each request is assigned a `uuid4` request ID via `ContextVar`. Every log line emitted during that request — evaluator calls, LLM calls, DB writes — carries the same ID. This makes it straightforward to reconstruct the full trace for a single request by filtering on one field.

**PostgreSQL as the primary database**
The deployed environment runs on a Render-hosted PostgreSQL instance. The SQLAlchemy async layer is database-agnostic — switching between Postgres and SQLite requires changing exactly one environment variable (`DATABASE_URL`). SQLite is still used in the test suite (in-memory) and is available as a fallback for quick offline local runs, but PostgreSQL is the default for both local and production use.

**Why asyncio instead of a task queue?**
All four evaluators are kicked off concurrently with `asyncio.gather`. Since each one is waiting on either an LLM API call or a DB read, the total wall time is roughly equal to the slowest single evaluator rather than the sum of all four. A task queue like Celery would add operational overhead without helping in this case.

---

## Scaling Strategy

### Current State (Single Process)

The current setup handles the required ~1,000 conversations/minute target for ingestion and read queries comfortably. Evaluation throughput is LLM-bound, not compute-bound.

### 10x Scale (10,000 conversations/minute)

- **PostgreSQL is already in place** — handles concurrent writes properly and supports connection pooling; the database layer is ready for this level without changes
- **Celery + Redis** for the evaluation worker pool — allows evaluation jobs to be distributed across multiple processes without code changes
- **Read replicas** for the query-heavy dashboard endpoints

### 100x Scale (100,000 conversations/minute)

- **Kafka** for the ingestion layer — decouple ingestion from evaluation so spikes in conversation volume do not block the evaluation pipeline
- **Horizontal scaling** of evaluation workers — add more Celery workers behind a load balancer
- **Kafka** to replace the in-process SSE event bus — durable, fan-out capable, survives process restarts
- **Separate services** — split the ingestion API, evaluation workers, and dashboard backend into independently scalable services
- **Caching** for regression reports and correlation computations — these are expensive to recompute on every request at high volume

---

## What I'd Do With More Time

1. **Redis + Celery worker pool** — Postgres is already in place for persistent storage. The next bottleneck for evaluation throughput is the single-process async loop. Adding Redis-backed Celery workers would allow evaluation jobs to be distributed across multiple processes and scale linearly.
2. **Durable event streaming** — the in-process SSE bus is convenient but does not survive restarts and cannot fan out across multiple processes. Kafka would be the right replacement at scale.
3. **A/B test integration** — automatically split traffic between agent versions and run significance tests on evaluation scores, rather than relying on manual regression comparisons.
4. **Prompt versioning** — track which version of the system prompt was active for each agent version, so generated suggestions can be tied to specific lines that need changing.
5. **Evaluator fine-tuning** — use accumulated human annotation data to adjust LLM-as-Judge rubric weights, rather than keeping them fixed.
6. **Prometheus + Grafana** — the `/metrics` endpoint already exposes evaluator latency stats; exporting those in Prometheus format would allow ops teams to monitor the pipeline alongside everything else.
7. **RBAC** — right now anyone with API access can do everything. Separating ops access (acknowledge alerts) from ML engineer access (modify evaluator config) would matter in a team setting.
