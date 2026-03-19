import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.events.stream import event_bus
from app.models.db_models import Evaluation, Suggestion
from app.models.schemas import ImprovementSuggestion, PipelineEvent, SuggestionStatusUpdate
from app.self_update.pattern_detector import PatternDetector
from app.self_update.prompt_suggester import PromptSuggester
from app.self_update.tool_suggester import ToolSuggester

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/suggestions", tags=["Suggestions"])

_detector = PatternDetector()
_prompt_suggester = PromptSuggester()
_tool_suggester = ToolSuggester()


def _db_to_schema(s: Suggestion) -> dict:
    return {
        "suggestion_id": s.suggestion_id,
        "evaluation_id": s.evaluation_id,
        "type": s.type,
        "target": s.target,
        "suggestion": s.suggestion,
        "rationale": s.rationale,
        "confidence": s.confidence,
        "expected_impact": s.expected_impact,
        "status": s.status,
        "created_at": s.created_at,
    }


@router.post("/generate")
async def generate_suggestions(
    last_n: int = Query(100, ge=1, le=1000, description="Analyse the most recent N evaluations"),
    db: AsyncSession = Depends(get_db),
):
    """Detect failure patterns and generate improvement suggestions."""
    stmt = select(Evaluation).order_by(Evaluation.created_at.desc()).limit(last_n)
    result = await db.execute(stmt)
    evals_db = result.scalars().all()

    if not evals_db:
        return {"patterns_found": 0, "suggestions_generated": 0, "suggestions": []}

    # Build lightweight EvaluationResult objects
    from app.models.schemas import EvaluationResult, EvaluationScores, IssueDetected, ToolEvaluation
    eval_results: list[EvaluationResult] = []
    for ev in evals_db:
        scores = EvaluationScores(**ev.scores)
        tool_eval = ToolEvaluation(**ev.tool_evaluation) if ev.tool_evaluation else None
        issues = [IssueDetected(**i) for i in (ev.issues or [])]
        eval_results.append(EvaluationResult(
            evaluation_id=ev.evaluation_id,
            conversation_id=ev.conversation_id,
            agent_version=ev.agent_version,
            scores=scores,
            tool_evaluation=tool_eval,
            issues_detected=issues,
            improvement_suggestions=[],
            evaluator_details=ev.evaluator_details or {},
            created_at=ev.created_at,
        ))

    patterns = await _detector.detect_failure_patterns(eval_results)
    logger.info("Detected %d patterns from %d evaluations", len(patterns), len(eval_results))

    prompt_suggestions = await _prompt_suggester.generate_suggestions(patterns)
    tool_suggestions = await _tool_suggester.generate_suggestions(patterns)
    all_suggestions = prompt_suggestions + tool_suggestions

    stored: list[dict] = []
    for sugg in all_suggestions:
        eval_id = evals_db[0].evaluation_id if evals_db else "batch"
        db_sugg = Suggestion(
            suggestion_id=f"sugg_{uuid.uuid4().hex[:12]}",
            evaluation_id=eval_id,
            type=sugg.type,
            target=sugg.target,
            suggestion=sugg.suggestion,
            rationale=sugg.rationale,
            confidence=sugg.confidence,
            expected_impact=sugg.expected_impact,
            status="pending",
        )
        db.add(db_sugg)
        stored.append(_db_to_schema(db_sugg))

    await db.commit()

    for sugg_row, sugg_schema in zip(stored, all_suggestions):
        try:
            await event_bus.publish(PipelineEvent(
                event_type="suggestion_generated",
                timestamp=datetime.now(timezone.utc),
                data={
                    "suggestion_id": sugg_row.get("suggestion_id", "unknown"),
                    "type": sugg_schema.type,
                    "target": sugg_schema.target,
                    "confidence": sugg_schema.confidence,
                },
            ).model_dump_json())
        except Exception:
            pass

    return {
        "patterns_found": len(patterns),
        "suggestions_generated": len(stored),
        "suggestions": stored,
    }


@router.get("/summary")
async def suggestions_summary(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Suggestion))
    all_s = result.scalars().all()

    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for s in all_s:
        by_type[s.type] = by_type.get(s.type, 0) + 1
        by_status[s.status] = by_status.get(s.status, 0) + 1

    top5 = sorted(
        [s for s in all_s if s.status == "pending"],
        key=lambda x: -x.confidence,
    )[:5]

    return {
        "total": len(all_s),
        "by_type": by_type,
        "by_status": by_status,
        "top_pending": [_db_to_schema(s) for s in top5],
    }


@router.get("/{suggestion_id}")
async def get_suggestion(suggestion_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Suggestion).where(Suggestion.suggestion_id == suggestion_id)
    )
    s = result.scalar_one_or_none()
    if not s:
        raise HTTPException(status_code=404, detail=f"Suggestion '{suggestion_id}' not found.")
    return _db_to_schema(s)


@router.patch("/{suggestion_id}")
async def update_suggestion_status(
    suggestion_id: str,
    payload: SuggestionStatusUpdate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Suggestion).where(Suggestion.suggestion_id == suggestion_id)
    )
    s = result.scalar_one_or_none()
    if not s:
        raise HTTPException(status_code=404, detail=f"Suggestion '{suggestion_id}' not found.")
    s.status = payload.status
    await db.commit()
    await db.refresh(s)
    return _db_to_schema(s)


@router.get("")
async def list_suggestions(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    type: str | None = None,
    status: str | None = None,
    min_confidence: float | None = Query(None, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Suggestion).offset(offset).limit(limit).order_by(Suggestion.confidence.desc())
    if type:
        stmt = stmt.where(Suggestion.type == type)
    if status:
        stmt = stmt.where(Suggestion.status == status)

    result = await db.execute(stmt)
    suggestions = result.scalars().all()

    items = [_db_to_schema(s) for s in suggestions if min_confidence is None or s.confidence >= min_confidence]
    return {"data": items, "meta": {"count": len(items), "offset": offset, "limit": limit}}
