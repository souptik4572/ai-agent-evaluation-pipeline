import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import create_tables, get_db
from app.middleware.logging import (
    RequestTracingMiddleware,
    configure_json_logging,
    get_evaluator_duration_stats,
    get_uptime_seconds,
)

configure_json_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — creating DB tables...")
    await create_tables()
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="AI Agent Evaluation Pipeline",
    version="0.1.0",
    description=(
        "Automated evaluation pipeline for multi-turn AI agent conversations. "
        "Runs LLM-as-Judge, Tool Call, Coherence, and Heuristic evaluators. "
        "Integrates human annotations and auto-generates improvement suggestions."
    ),
    lifespan=lifespan,
)

app.add_middleware(RequestTracingMiddleware)

# Routers are imported lazily to avoid circular imports at module load time.
from app.routers.conversations import router as conversations_router  # noqa: E402
from app.routers.evaluations import router as evaluations_router      # noqa: E402
from app.routers.suggestions import router as suggestions_router      # noqa: E402
from app.routers.meta import router as meta_router                    # noqa: E402
from app.routers.regression import router as regression_router        # noqa: E402
from app.routers.alerts import router as alerts_router                # noqa: E402
from app.routers.events import router as events_router                # noqa: E402

app.include_router(conversations_router, prefix="/api/v1")
app.include_router(evaluations_router, prefix="/api/v1")
app.include_router(suggestions_router, prefix="/api/v1")
app.include_router(meta_router, prefix="/api/v1")
app.include_router(regression_router, prefix="/api/v1")
app.include_router(alerts_router, prefix="/api/v1")
app.include_router(events_router, prefix="/api/v1")


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.get("/api/v1/metrics", tags=["Observability"])
async def metrics(db: AsyncSession = Depends(get_db)):
    """Lightweight observability endpoint with pipeline stats."""
    from app.models.db_models import AlertRecord, Conversation, Evaluation, Suggestion

    total_conversations = (await db.execute(select(func.count(Conversation.id)))).scalar() or 0
    total_evaluations = (await db.execute(select(func.count(Evaluation.id)))).scalar() or 0
    total_suggestions = (await db.execute(select(func.count(Suggestion.id)))).scalar() or 0
    open_alerts = (
        await db.execute(select(func.count(AlertRecord.id)).where(AlertRecord.status == "open"))
    ).scalar() or 0

    return {
        "total_conversations": total_conversations,
        "total_evaluations": total_evaluations,
        "total_suggestions": total_suggestions,
        "open_alerts": open_alerts,
        "evaluator_durations": get_evaluator_duration_stats(),
        "uptime_seconds": get_uptime_seconds(),
        "version": "0.1.0",
    }
