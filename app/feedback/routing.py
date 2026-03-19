import logging

from app.config import settings
from app.models.schemas import Annotation, EvaluationResult

logger = logging.getLogger(__name__)


class ConfidenceRouter:
    def route(self, evaluation: EvaluationResult, annotations: list[Annotation] | None) -> dict:
        score = evaluation.scores.overall
        auto_threshold = settings.confidence_auto_label_threshold  # 0.85
        low_threshold = 1 - auto_threshold  # 0.15

        has_critical = any(
            issue.severity == "critical" for issue in evaluation.issues_detected
        )

        if has_critical:
            return {
                "routing_decision": "human_review",
                "auto_label": None,
                "confidence": score,
                "reason": "Critical issue detected — requires human review.",
            }

        if annotations:
            from app.feedback.aggregator import AnnotationAggregator
            agg = AnnotationAggregator()
            agreement = agg.compute_agreement(annotations)
            if agreement.get("disagreements"):
                return {
                    "routing_decision": "tiebreaker",
                    "auto_label": None,
                    "confidence": score,
                    "reason": f"Annotator disagreement on: {[d['type'] for d in agreement['disagreements']]}",
                }

        if score >= auto_threshold:
            return {
                "routing_decision": "auto_label",
                "auto_label": "good",
                "confidence": score,
                "reason": "High confidence auto-evaluation score.",
            }
        elif score <= low_threshold:
            return {
                "routing_decision": "auto_label",
                "auto_label": "poor",
                "confidence": 1 - score,
                "reason": "Low evaluation score — auto-labelled as poor.",
            }
        else:
            return {
                "routing_decision": "human_review",
                "auto_label": None,
                "confidence": score,
                "reason": f"Score {score:.2f} is in uncertain range ({low_threshold:.2f}-{auto_threshold:.2f}).",
            }
