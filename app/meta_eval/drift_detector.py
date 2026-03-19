import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import MetaEvalRecord

logger = logging.getLogger(__name__)


class DriftDetector:
    async def detect_drift(
        self,
        evaluator_name: str,
        window_days: int = 7,
        db: AsyncSession = None,
    ) -> dict:
        cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
        stmt = (
            select(MetaEvalRecord)
            .where(MetaEvalRecord.evaluator_name == evaluator_name)
            .where(MetaEvalRecord.created_at >= cutoff)
            .order_by(MetaEvalRecord.created_at)
        )
        result = await db.execute(stmt)
        records = result.scalars().all()

        if not records:
            return {
                "evaluator": evaluator_name,
                "window_days": window_days,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "trend": "insufficient_data",
                "blind_spots": [],
                "recommendation": f"No meta-eval records found for '{evaluator_name}' in the last {window_days} days.",
                "record_count": 0,
            }

        n = len(records)
        accurate = [r for r in records if abs(r.human_score - r.auto_score) < 0.15]
        accuracy = len(accurate) / n

        auto_poor = [r for r in records if r.auto_score < 0.5]
        human_poor = [r for r in records if r.human_score < 0.5]
        true_positives = [r for r in auto_poor if r.human_score < 0.5]

        precision = len(true_positives) / len(auto_poor) if auto_poor else 1.0
        recall = len(true_positives) / len(human_poor) if human_poor else 1.0

        half = n // 2
        first_acc = sum(1 for r in records[:half] if abs(r.human_score - r.auto_score) < 0.15) / max(half, 1)
        second_acc = sum(1 for r in records[half:] if abs(r.human_score - r.auto_score) < 0.15) / max(n - half, 1)

        if second_acc > first_acc + 0.05:
            trend = "improving"
        elif second_acc < first_acc - 0.05:
            trend = "degrading"
        else:
            trend = "stable"

        missed = [r for r in records if r.human_score < 0.5 and r.auto_score >= 0.5]
        blind_spots: list[str] = []
        if missed:
            blind_spots.append("low_quality_conversations_missed_by_auto_eval")
        if any(abs(r.human_score - r.auto_score) > 0.3 for r in records):
            blind_spots.append("high_variance_scoring_cases")

        if trend == "degrading":
            recommendation = (
                f"Evaluator '{evaluator_name}' accuracy is degrading (first half: {first_acc:.2f}, "
                f"second half: {second_acc:.2f}). Review recent rubric or model changes."
            )
        elif recall < 0.7:
            recommendation = (
                f"Evaluator '{evaluator_name}' has low recall ({recall:.2f}) — missing many poor conversations "
                f"flagged by humans. Consider lowering the threshold for flagging issues."
            )
        elif precision < 0.7:
            recommendation = (
                f"Evaluator '{evaluator_name}' has low precision ({precision:.2f}) — flagging too many "
                f"false positives. Review rubric strictness."
            )
        else:
            recommendation = (
                f"Evaluator '{evaluator_name}' is performing well "
                f"(accuracy={accuracy:.2f}, precision={precision:.2f}, recall={recall:.2f})."
            )

        return {
            "evaluator": evaluator_name,
            "window_days": window_days,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "trend": trend,
            "blind_spots": blind_spots,
            "recommendation": recommendation,
            "record_count": n,
        }

    async def detect_all_drift(self, window_days: int = 7, db: AsyncSession = None) -> list[dict]:
        stmt = select(MetaEvalRecord.evaluator_name).distinct()
        result = await db.execute(stmt)
        evaluator_names = [row[0] for row in result.fetchall()]

        results = []
        for name in evaluator_names:
            drift = await self.detect_drift(name, window_days=window_days, db=db)
            results.append(drift)
        return results
