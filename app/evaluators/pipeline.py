import asyncio
import logging
import uuid
from datetime import datetime, timezone

from app.evaluators.coherence import CoherenceEvaluator
from app.evaluators.heuristic import HeuristicEvaluator
from app.evaluators.llm_judge import LLMJudgeEvaluator
from app.evaluators.tool_call import ToolCallEvaluator
from app.models.schemas import (
    EvaluationResult,
    EvaluationScores,
    ImprovementSuggestion,
    IssueDetected,
    ToolEvaluation,
    ConversationCreate,
)

logger = logging.getLogger(__name__)

_WEIGHTS = {
    "heuristic": 0.15,
    "tool_call": 0.25,
    "coherence": 0.25,
    "llm_judge": 0.35,
}


class EvaluationPipeline:
    def __init__(self):
        self.evaluators = [
            HeuristicEvaluator(),
            ToolCallEvaluator(),
            CoherenceEvaluator(),
            LLMJudgeEvaluator(),
        ]

    async def run(self, conversation: ConversationCreate) -> EvaluationResult:
        logger.info("Running evaluation pipeline for conversation %s", conversation.conversation_id)

        results = await asyncio.gather(
            *[ev.evaluate(conversation) for ev in self.evaluators],
            return_exceptions=True,
        )

        evaluator_details: dict = {}
        raw_scores: dict[str, float] = {}
        all_issues: list[dict] = []

        for ev, result in zip(self.evaluators, results):
            if isinstance(result, Exception):
                logger.error("Evaluator %s raised: %s", ev.name, result)
                evaluator_details[ev.name] = {"error": str(result), "score": 0.5}
                raw_scores[ev.name] = 0.5
            else:
                evaluator_details[ev.name] = result.get("details", {})
                evaluator_details[ev.name]["score"] = result["score"]
                raw_scores[ev.name] = result["score"]
                all_issues.extend(result.get("issues", []))

        overall = sum(raw_scores.get(k, 0) * w for k, w in _WEIGHTS.items())
        tool_det = evaluator_details.get("tool_call", {})
        coherence_det = evaluator_details.get("coherence", {})
        llm_det = evaluator_details.get("llm_judge", {})

        scores = EvaluationScores(
            overall=round(overall, 4),
            response_quality=round(float(llm_det.get("response_quality", raw_scores.get("llm_judge", 0.7))), 4),
            tool_accuracy=round(float(tool_det.get("selection_accuracy", raw_scores.get("tool_call", 1.0))), 4),
            coherence=round(float(raw_scores.get("coherence", 1.0)), 4),
        )

        tool_eval: ToolEvaluation | None = None
        if tool_det.get("tool_call_count", 0) > 0:
            tool_eval = ToolEvaluation(
                selection_accuracy=float(tool_det.get("selection_accuracy", 1.0)),
                parameter_accuracy=float(tool_det.get("parameter_accuracy", 1.0)),
                hallucinated_params=tool_det.get("hallucinated_params", []),
                execution_success=bool(tool_det.get("execution_success", True)),
            )

        seen_issues: set[str] = set()
        unique_issues: list[IssueDetected] = []
        for iss in all_issues:
            key = f"{iss.get('type')}:{iss.get('description', '')[:60]}"
            if key not in seen_issues:
                seen_issues.add(key)
                unique_issues.append(IssueDetected(
                    type=iss["type"],
                    severity=iss.get("severity", "info"),
                    description=iss.get("description", ""),
                ))

        evaluation_id = f"eval_{uuid.uuid4().hex[:12]}"
        return EvaluationResult(
            evaluation_id=evaluation_id,
            conversation_id=conversation.conversation_id,
            agent_version=conversation.agent_version,
            scores=scores,
            tool_evaluation=tool_eval,
            issues_detected=unique_issues,
            improvement_suggestions=[],
            evaluator_details=evaluator_details,
            created_at=datetime.now(timezone.utc),
        )
