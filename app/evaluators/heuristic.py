import logging
from datetime import datetime

from app.config import settings
from app.evaluators.base import BaseEvaluator
from app.middleware.logging import timed_evaluator
from app.models.schemas import ConversationCreate

logger = logging.getLogger(__name__)


class HeuristicEvaluator(BaseEvaluator):
    @property
    def name(self) -> str:
        return "heuristic"

    @timed_evaluator
    async def evaluate(self, conversation: ConversationCreate) -> dict:
        checks: dict[str, float] = {}
        issues: list[dict] = []

        threshold = settings.latency_threshold_ms
        latency = conversation.metadata.total_latency_ms if conversation.metadata else None
        if latency is None:
            checks["latency"] = 1.0
        elif latency <= threshold:
            checks["latency"] = 1.0
        elif latency <= threshold * 2:
            checks["latency"] = 0.5
            issues.append({
                "type": "latency",
                "severity": "warning",
                "description": f"Total latency {latency}ms exceeds threshold {threshold}ms.",
            })
        else:
            checks["latency"] = 0.0
            issues.append({
                "type": "latency",
                "severity": "critical",
                "description": f"Total latency {latency}ms is >2x threshold {threshold}ms.",
            })

        missing_content = [
            t.turn_id for t in conversation.turns
            if t.role == "assistant" and not t.content.strip()
        ]
        if missing_content:
            checks["required_fields"] = 0.5
            issues.append({
                "type": "missing_content",
                "severity": "warning",
                "description": f"Assistant turns with empty content: {missing_content}",
            })
        else:
            checks["required_fields"] = 1.0

        tool_result_ok = True
        for turn in conversation.turns:
            if not turn.tool_calls:
                continue
            for tc in turn.tool_calls:
                if tc.result is None:
                    tool_result_ok = False
                    issues.append({
                        "type": "missing_tool_result",
                        "severity": "warning",
                        "description": f"Tool call '{tc.tool_name}' in turn {turn.turn_id} has no result.",
                    })
        checks["tool_results"] = 1.0 if tool_result_ok else 0.5

        format_ok = True
        for turn in conversation.turns:
            if not turn.tool_calls:
                continue
            for tc in turn.tool_calls:
                if not tc.parameters:
                    format_ok = False
                    issues.append({
                        "type": "empty_tool_params",
                        "severity": "info",
                        "description": f"Tool '{tc.tool_name}' called with no parameters in turn {turn.turn_id}.",
                    })
        checks["format_compliance"] = 1.0 if format_ok else 0.7

        mission_completed = (
            conversation.metadata.mission_completed
            if conversation.metadata and conversation.metadata.mission_completed is not None
            else None
        )
        if mission_completed is False:
            checks["mission_completion"] = 0.0
            issues.append({
                "type": "mission_not_completed",
                "severity": "warning",
                "description": "Mission was not completed according to metadata.",
            })
        elif mission_completed is True:
            checks["mission_completion"] = 1.0
        else:
            checks["mission_completion"] = 0.75

        score = sum(checks.values()) / len(checks)
        return {
            "score": round(score, 4),
            "details": {"checks": checks},
            "issues": issues,
        }
