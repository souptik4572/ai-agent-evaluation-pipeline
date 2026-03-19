import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.events.stream import event_bus
from app.models.db_models import Conversation, Evaluation
from app.models.schemas import ConversationCreate, EvaluationResult, PipelineEvent
from app.services.evaluation_service import run_evaluation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluations", tags=["Evaluations"])


def _db_eval_to_result(ev: Evaluation) -> EvaluationResult:
    from datetime import datetime, timezone
    from app.models.schemas import EvaluationScores, IssueDetected, ImprovementSuggestion, ToolEvaluation

    scores = EvaluationScores(**ev.scores)
    tool_eval = ToolEvaluation(**ev.tool_evaluation) if ev.tool_evaluation else None
    issues = [IssueDetected(**i) for i in (ev.issues or [])]
    suggestions = [ImprovementSuggestion(**s) for s in (ev.suggestions or [])]
    return EvaluationResult(
        evaluation_id=ev.evaluation_id,
        conversation_id=ev.conversation_id,
        agent_version=ev.agent_version,
        scores=scores,
        tool_evaluation=tool_eval,
        issues_detected=issues,
        improvement_suggestions=suggestions,
        evaluator_details=ev.evaluator_details or {},
        annotation_agreement=ev.annotation_agreement,
        routing_decision=ev.routing_decision,
        created_at=ev.created_at,
    )


async def _fetch_and_build_conversation(conversation_id: str, db: AsyncSession) -> ConversationCreate:
    result = await db.execute(
        select(Conversation).where(Conversation.conversation_id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")

    from app.models.schemas import Feedback, ConversationMetadata, Turn
    turns = [Turn(**t) for t in conv.data.get("turns", [])]
    feedback = Feedback(**conv.feedback_data) if conv.feedback_data else None
    metadata = ConversationMetadata(**conv.metadata_data) if conv.metadata_data else None
    return ConversationCreate(
        conversation_id=conv.conversation_id,
        agent_version=conv.agent_version,
        turns=turns,
        feedback=feedback,
        metadata=metadata,
    )


@router.post("/evaluate/batch", tags=["Evaluations"])
async def batch_evaluate(conversation_ids: list[str], db: AsyncSession = Depends(get_db)):
    eval_ids: list[str] = []
    for conv_id in conversation_ids:
        try:
            conversation = await _fetch_and_build_conversation(conv_id, db)
            eval_result = await run_evaluation(conversation)
            db_eval = Evaluation(
                evaluation_id=eval_result.evaluation_id,
                conversation_id=eval_result.conversation_id,
                agent_version=eval_result.agent_version,
                scores=eval_result.scores.model_dump(),
                tool_evaluation=eval_result.tool_evaluation.model_dump() if eval_result.tool_evaluation else None,
                issues=[i.model_dump() for i in eval_result.issues_detected],
                suggestions=[s.model_dump() for s in eval_result.improvement_suggestions],
                evaluator_details=eval_result.evaluator_details,
                annotation_agreement=eval_result.annotation_agreement,
                routing_decision=eval_result.routing_decision,
            )
            db.add(db_eval)
            eval_ids.append(eval_result.evaluation_id)
        except Exception as e:
            logger.error("Failed to evaluate %s: %s", conv_id, e)
    await db.commit()

    try:
        from app.alerting.alerts import AlertManager
        alert_mgr = AlertManager()
        await alert_mgr.check_quality_drop(db)
        await alert_mgr.check_tool_failure_rate(db)
        await alert_mgr.check_annotator_disagreement(db)
    except Exception as exc:
        logger.error("Post-batch alert check failed: %s", exc)

    return {"evaluated": len(eval_ids), "evaluation_ids": eval_ids}


@router.post("/evaluate/{conversation_id}", response_model=EvaluationResult, tags=["Evaluations"])
async def evaluate_conversation(conversation_id: str, db: AsyncSession = Depends(get_db)):
    conversation = await _fetch_and_build_conversation(conversation_id, db)
    eval_result = await run_evaluation(conversation)

    db_eval = Evaluation(
        evaluation_id=eval_result.evaluation_id,
        conversation_id=eval_result.conversation_id,
        agent_version=eval_result.agent_version,
        scores=eval_result.scores.model_dump(),
        tool_evaluation=eval_result.tool_evaluation.model_dump() if eval_result.tool_evaluation else None,
        issues=[i.model_dump() for i in eval_result.issues_detected],
        suggestions=[s.model_dump() for s in eval_result.improvement_suggestions],
        evaluator_details=eval_result.evaluator_details,
        annotation_agreement=eval_result.annotation_agreement,
        routing_decision=eval_result.routing_decision,
    )
    db.add(db_eval)
    await db.commit()
    await db.refresh(db_eval)
    logger.info("Stored evaluation %s for conversation %s", eval_result.evaluation_id, conversation_id)

    try:
        await event_bus.publish(PipelineEvent(
            event_type="evaluation_completed",
            timestamp=datetime.now(timezone.utc),
            data={
                "evaluation_id": eval_result.evaluation_id,
                "overall_score": eval_result.scores.overall,
                "issues_count": len(eval_result.issues_detected),
            },
            conversation_id=eval_result.conversation_id,
            agent_version=eval_result.agent_version,
        ).model_dump_json())
    except Exception:
        pass

    await _maybe_trigger_regression_check(eval_result.agent_version, db)

    return eval_result


async def _maybe_trigger_regression_check(agent_version: str, db: AsyncSession) -> None:
    """Trigger auto regression check when a version's eval count hits the threshold."""
    count_result = await db.execute(
        select(func.count(Evaluation.id)).where(Evaluation.agent_version == agent_version)
    )
    count = count_result.scalar() or 0
    threshold = settings.regression_check_eval_threshold
    if count == threshold or (count > threshold and count % threshold == 0):
        try:
            from app.regression.comparator import VersionComparator
            from app.alerting.alerts import AlertManager
            comparator = VersionComparator()
            alert_mgr = AlertManager()
            report = await comparator.auto_check(agent_version, db)
            if report and report.is_regression:
                related_id = f"auto_{report.baseline_version}_vs_{report.target_version}"
                await alert_mgr.create_regression_alert(report, related_id, db)
                logger.warning(
                    "Auto-regression check: %s regression detected vs %s (severity=%s)",
                    agent_version, report.baseline_version, report.severity,
                )
            elif report:
                logger.info(
                    "Auto-regression check: %s vs %s — no regression (severity=%s)",
                    agent_version, report.baseline_version, report.severity,
                )
        except Exception as exc:
            logger.error("Auto regression check failed for %s: %s", agent_version, exc)


@router.get("/{evaluation_id}", response_model=EvaluationResult)
async def get_evaluation(evaluation_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Evaluation).where(Evaluation.evaluation_id == evaluation_id)
    )
    ev = result.scalar_one_or_none()
    if not ev:
        raise HTTPException(status_code=404, detail=f"Evaluation '{evaluation_id}' not found.")
    return _db_eval_to_result(ev)


@router.get("")
async def list_evaluations(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    agent_version: str | None = None,
    min_score: float | None = Query(None, ge=0.0, le=1.0),
    max_score: float | None = Query(None, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Evaluation).offset(offset).limit(limit).order_by(Evaluation.created_at.desc())
    if agent_version:
        stmt = stmt.where(Evaluation.agent_version == agent_version)

    result = await db.execute(stmt)
    evals = result.scalars().all()

    items = []
    for ev in evals:
        er = _db_eval_to_result(ev)
        if min_score is not None and er.scores.overall < min_score:
            continue
        if max_score is not None and er.scores.overall > max_score:
            continue
        items.append(er)

    return {"data": items, "meta": {"count": len(items), "offset": offset, "limit": limit}}
