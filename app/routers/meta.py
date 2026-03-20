import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.analytics.correlation import compute_correlation
from app.database import get_db
from app.meta_eval.calibrator import EvaluatorCalibrator
from app.meta_eval.drift_detector import DriftDetector
from app.models.db_models import Conversation, Evaluation, MetaEvalRecord
from app.models.schemas import CalibrationRequest, CorrelationResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/meta", tags=["Meta-Evaluation"])

_calibrator = EvaluatorCalibrator()
_drift_detector = DriftDetector()


@router.post("/calibrate")
async def calibrate_evaluator(
    payload: CalibrationRequest,
    conversation_id: str = Query(..., description="ID of the conversation to calibrate against"),
    db: AsyncSession = Depends(get_db),
):
    """Compare human scores against the stored auto-evaluation scores for a conversation."""
    result = await db.execute(
        select(Evaluation).where(Evaluation.conversation_id == conversation_id)
        .order_by(Evaluation.created_at.desc())
    )
    ev = result.scalars().first()
    if not ev:
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation found for conversation '{conversation_id}'.",
        )

    auto_scores = ev.scores  # dict: {overall, response_quality, tool_accuracy, coherence}
    report = await _calibrator.calibrate(
        conversation_id=conversation_id,
        human_scores=payload.human_scores,
        auto_scores=auto_scores,
        db=db,
    )
    return report


@router.get("/calibration")
async def calibration_summary(
    limit: int = Query(50, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    """Summary of recent calibration records."""
    result = await db.execute(
        select(MetaEvalRecord).order_by(MetaEvalRecord.created_at.desc()).limit(limit)
    )
    records = result.scalars().all()

    by_evaluator: dict[str, list[float]] = {}
    for r in records:
        by_evaluator.setdefault(r.evaluator_name, []).append(
            abs(r.human_score - r.auto_score)
        )

    summary = {}
    for name, diffs in by_evaluator.items():
        avg_diff = sum(diffs) / len(diffs)
        summary[name] = {
            "record_count": len(diffs),
            "avg_diff": round(avg_diff, 4),
            "alignment_rate": round(sum(1 for d in diffs if d < 0.15) / len(diffs), 4),
        }

    return {"summary": summary, "total_records": len(records)}


@router.get("/drift/{evaluator_name}")
async def drift_for_evaluator(
    evaluator_name: str,
    window_days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    return await _drift_detector.detect_drift(evaluator_name, window_days=window_days, db=db)


@router.get("/drift")
async def drift_all_evaluators(
    window_days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    return await _drift_detector.detect_all_drift(window_days=window_days, db=db)


@router.get("/correlation", response_model=CorrelationResponse)
async def get_correlation(
    limit: int = Query(200, ge=10, le=1000),
    db: AsyncSession = Depends(get_db),
):
    """
    Compute Pearson correlation between automated evaluation scores and user ratings.
    Joins evaluations with their conversation's feedback.user_rating.
    """
    # Load recent evaluations
    eval_result = await db.execute(
        select(Evaluation).order_by(Evaluation.created_at.desc()).limit(limit)
    )
    evals = eval_result.scalars().all()

    # Build a map of conversation_id -> user_rating
    conv_ids = list({ev.conversation_id for ev in evals})
    if not conv_ids:
        return CorrelationResponse(correlations=[], best_dimension=None, scatter_data=[])

    conv_result = await db.execute(
        select(Conversation).where(Conversation.conversation_id.in_(conv_ids))
    )
    conversations = conv_result.scalars().all()
    rating_map: dict[str, int | None] = {}
    for conv in conversations:
        feedback = conv.feedback_data or {}
        rating_map[conv.conversation_id] = feedback.get("user_rating")

    # Assemble records for correlation
    records = []
    for ev in evals:
        user_rating = rating_map.get(ev.conversation_id)
        if user_rating is not None:
            records.append({
                "conversation_id": ev.conversation_id,
                "scores": ev.scores or {},
                "user_rating": user_rating,
            })

    return compute_correlation(records)
