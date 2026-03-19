import logging

from app.evaluators.base import BaseEvaluator
from app.middleware.logging import timed_evaluator
from app.models.schemas import ConversationCreate
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_RUBRIC = """\
You are an expert AI quality evaluator. Score this AI agent conversation on these dimensions.
Each score is 0.0 to 1.0.

1. response_quality: Is the response helpful, complete, and well-structured?
   - 1.0: Perfectly addresses user's need, comprehensive, well-organized
   - 0.7: Addresses the need but missing minor details
   - 0.4: Partially addresses the need, significant gaps
   - 0.0: Completely unhelpful or off-topic

2. helpfulness: Does the response move the user toward their goal?
   - 1.0: Directly advances the user's stated goal
   - 0.5: Somewhat helpful but requires user to do more work
   - 0.0: Does not help at all

3. factuality: Are all claims accurate? Are tool results interpreted correctly?
   - 1.0: All claims verifiable and accurate
   - 0.5: Mostly accurate with minor issues
   - 0.0: Contains fabricated information

4. tone: Is the tone appropriate, professional, and empathetic?
   - 1.0: Perfect tone for the context
   - 0.5: Acceptable but could be better
   - 0.0: Inappropriate tone

Respond with ONLY this JSON structure, no other text:
{"response_quality": 0.0, "helpfulness": 0.0, "factuality": 0.0, "tone": 0.0, "issues": [], "reasoning": ""}
"""

_WEIGHTS = {
    "response_quality": 0.35,
    "helpfulness": 0.30,
    "factuality": 0.25,
    "tone": 0.10,
}


class LLMJudgeEvaluator(BaseEvaluator):
    def __init__(self):
        self.llm_client = LLMClient()

    @property
    def name(self) -> str:
        return "llm_judge"

    @timed_evaluator
    async def evaluate(self, conversation: ConversationCreate) -> dict:
        lines: list[str] = []
        for turn in conversation.turns:
            lines.append(f"[{turn.role.upper()} - Turn {turn.turn_id}]: {turn.content}")
            if turn.tool_calls:
                for tc in turn.tool_calls:
                    lines.append(
                        f"  [TOOL CALL] {tc.tool_name}({tc.parameters}) -> {tc.result}"
                    )
        conversation_text = "\n".join(lines)

        result = await self.llm_client.evaluate_with_rubric(conversation_text, _RUBRIC)

        rq = float(result.get("response_quality", 0.7))
        h = float(result.get("helpfulness", 0.7))
        f = float(result.get("factuality", 0.8))
        t = float(result.get("tone", 0.75))

        overall = (
            rq * _WEIGHTS["response_quality"]
            + h * _WEIGHTS["helpfulness"]
            + f * _WEIGHTS["factuality"]
            + t * _WEIGHTS["tone"]
        )

        issues: list[dict] = []
        for issue in result.get("issues", []):
            if isinstance(issue, str):
                issues.append({"type": "llm_flagged", "severity": "warning", "description": issue})
            elif isinstance(issue, dict):
                issues.append({
                    "type": issue.get("type", "llm_flagged"),
                    "severity": issue.get("severity", "warning"),
                    "description": issue.get("description", str(issue)),
                })

        return {
            "score": round(overall, 4),
            "details": {
                "response_quality": rq,
                "helpfulness": h,
                "factuality": f,
                "tone": t,
                "reasoning": result.get("reasoning", ""),
                "mock": result.get("mock", False),
            },
            "issues": issues,
        }
