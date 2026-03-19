import logging
import re
from datetime import datetime

from app.evaluators.base import BaseEvaluator
from app.middleware.logging import timed_evaluator
from app.models.schemas import ConversationCreate, ToolCall

logger = logging.getLogger(__name__)

_TOOL_INTENTS: dict[str, list[str]] = {
    "flight_search": ["flight", "fly", "plane", "airline", "airport", "travel", "book"],
    "hotel_search": ["hotel", "room", "accommodation", "stay", "lodging"],
    "car_rental": ["car", "rental", "rent", "drive", "vehicle"],
    "weather": ["weather", "forecast", "rain", "sunny", "temperature"],
    "calendar": ["schedule", "calendar", "appointment", "event", "meeting"],
    "search": ["search", "find", "look up", "query", "information"],
    "send_email": ["email", "send", "message", "mail"],
}


def _date_valid(value: str) -> bool:
    """Return True if value looks like a valid date or date-range string."""
    patterns = [
        r"^\d{4}-\d{2}-\d{2}$",
        r"^\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}$",
    ]
    return any(re.match(p, str(value)) for p in patterns)


def _value_in_context(value: str, context_text: str) -> bool:
    """Check if a string value appears in the conversation context."""
    val = str(value).lower().strip()
    return val in context_text.lower()


class ToolCallEvaluator(BaseEvaluator):
    @property
    def name(self) -> str:
        return "tool_call"

    @timed_evaluator
    async def evaluate(self, conversation: ConversationCreate) -> dict:
        all_tool_calls: list[tuple[int, ToolCall]] = []
        for turn in conversation.turns:
            if turn.tool_calls:
                for tc in turn.tool_calls:
                    all_tool_calls.append((turn.turn_id, tc))

        if not all_tool_calls:
            return {
                "score": 1.0,
                "details": {
                    "selection_accuracy": 1.0,
                    "parameter_accuracy": 1.0,
                    "execution_success": True,
                    "hallucinated_params": [],
                    "tool_call_count": 0,
                },
                "issues": [],
            }

        context_text = " ".join(turn.content for turn in conversation.turns)
        user_text = " ".join(
            t.content for t in conversation.turns if t.role == "user"
        ).lower()

        issues: list[dict] = []
        selection_scores: list[float] = []
        param_scores: list[float] = []
        exec_results: list[bool] = []
        hallucinated_params: list[str] = []

        for turn_id, tc in all_tool_calls:
            tool_lower = tc.tool_name.lower()
            intents = _TOOL_INTENTS.get(tool_lower, [])
            if not intents:
                selection_scores.append(0.8)
            else:
                matched = any(kw in user_text for kw in intents)
                selection_scores.append(1.0 if matched else 0.3)
                if not matched:
                    issues.append({
                        "type": "tool_selection_mismatch",
                        "severity": "warning",
                        "description": (
                            f"Tool '{tc.tool_name}' in turn {turn_id} may not match user intent. "
                            f"Expected keywords: {intents}"
                        ),
                    })

            if tc.parameters:
                param_ok_count = 0
                for param_name, param_value in tc.parameters.items():
                    val_str = str(param_value)
                    traceable = _value_in_context(val_str, context_text)

                    if "date" in param_name.lower() and not _date_valid(val_str):
                        traceable = False
                        issues.append({
                            "type": "tool_parameter_format_error",
                            "severity": "warning",
                            "description": (
                                f"Parameter '{param_name}' in tool '{tc.tool_name}' (turn {turn_id}) "
                                f"has invalid date format: '{val_str}'. Expected YYYY-MM-DD or YYYY-MM-DD/YYYY-MM-DD."
                            ),
                        })
                        key = f"{tc.tool_name}.{param_name}"
                        if key not in hallucinated_params:
                            hallucinated_params.append(key)

                    if traceable:
                        param_ok_count += 1
                    else:
                        key = f"{tc.tool_name}.{param_name}"
                        if key not in hallucinated_params and "date" not in param_name.lower():
                            hallucinated_params.append(key)

                param_scores.append(param_ok_count / len(tc.parameters))
            else:
                param_scores.append(1.0)

            if tc.result is not None:
                success = tc.result.get("status") == "success" or "error" not in tc.result
                exec_results.append(success)
                if not success:
                    issues.append({
                        "type": "tool_execution_failure",
                        "severity": "critical",
                        "description": (
                            f"Tool '{tc.tool_name}' in turn {turn_id} returned failure status. "
                            f"Result: {tc.result}"
                        ),
                    })
            else:
                exec_results.append(True)

        selection_acc = sum(selection_scores) / len(selection_scores) if selection_scores else 1.0
        param_acc = sum(param_scores) / len(param_scores) if param_scores else 1.0
        exec_rate = sum(1 for r in exec_results if r) / len(exec_results) if exec_results else 1.0
        exec_success = exec_rate >= 0.9

        score = (selection_acc * 0.35 + param_acc * 0.40 + exec_rate * 0.25)

        return {
            "score": round(score, 4),
            "details": {
                "selection_accuracy": round(selection_acc, 4),
                "parameter_accuracy": round(param_acc, 4),
                "execution_success": exec_success,
                "execution_rate": round(exec_rate, 4),
                "hallucinated_params": hallucinated_params,
                "tool_call_count": len(all_tool_calls),
            },
            "issues": issues,
        }
