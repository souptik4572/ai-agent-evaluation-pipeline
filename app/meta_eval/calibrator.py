import logging
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import MetaEvalRecord

logger = logging.getLogger(__name__)


class EvaluatorCalibrator:
    async def calibrate(
        self,
        conversation_id: str,
        human_scores: dict[str, float],
        auto_scores: dict[str, float],
        db: AsyncSession,
    ) -> dict:
        dimensions: dict[str, dict] = {}
        records_to_store: list[tuple[str, float, float]] = []

        for dim, human_score in human_scores.items():
            auto_score = auto_scores.get(dim)
            if auto_score is None:
                continue
            diff = abs(human_score - auto_score)
            if diff < 0.15:
                status = "aligned"
            elif diff < 0.30:
                status = "minor_drift"
            else:
                status = "major_drift"

            dimensions[dim] = {
                "human": round(human_score, 4),
                "auto": round(auto_score, 4),
                "diff": round(diff, 4),
                "status": status,
            }
            records_to_store.append((dim, human_score, auto_score))

        # Store meta-eval records
        for evaluator_dim, human_score, auto_score in records_to_store:
            agreement = abs(human_score - auto_score) < 0.15
            record = MetaEvalRecord(
                evaluator_name=evaluator_dim,
                human_score=human_score,
                auto_score=auto_score,
                agreement=agreement,
                conversation_id=conversation_id,
            )
            db.add(record)
        await db.commit()

        # Compute overall alignment
        all_diffs = [d["diff"] for d in dimensions.values()]
        overall_alignment = 1.0 - (sum(all_diffs) / len(all_diffs)) if all_diffs else 1.0

        # Generate recommendation
        worst_dim = max(dimensions.items(), key=lambda x: x[1]["diff"], default=(None, None))
        if worst_dim[0] and worst_dim[1]["diff"] >= 0.15:
            dim_name = worst_dim[0]
            human_val = worst_dim[1]["human"]
            auto_val = worst_dim[1]["auto"]
            direction = "under-scoring" if auto_val < human_val else "over-scoring"
            recommendation = (
                f"LLM-as-Judge is {direction} '{dim_name}' "
                f"(human={human_val:.2f}, auto={auto_val:.2f}, diff={worst_dim[1]['diff']:.2f}). "
                f"Consider adjusting the rubric weights or adding more specific criteria for this dimension."
            )
        else:
            recommendation = "All evaluator dimensions are well-aligned with human scores."

        return {
            "conversation_id": conversation_id,
            "dimensions": dimensions,
            "overall_alignment": round(overall_alignment, 4),
            "recommendation": recommendation,
        }
