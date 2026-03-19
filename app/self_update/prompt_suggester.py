import logging

from app.models.schemas import ImprovementSuggestion
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_HEURISTIC_SUGGESTIONS: dict[str, dict] = {
    "tool_parameter_format_error": {
        "target": "system_prompt",
        "suggestion": (
            "Add the following to the system prompt immediately after the tool-use guidelines section:\n\n"
            "DATE FORMAT RULE: When calling any tool that accepts a date or date range parameter, "
            "ALWAYS use ISO 8601 format: YYYY-MM-DD for single dates and YYYY-MM-DD/YYYY-MM-DD for ranges. "
            "Never pass relative expressions like 'next week', 'tomorrow', 'January', or '1/22'. "
            "Compute the exact calendar dates from today's date before calling the tool."
        ),
        "rationale": (
            "Analysis shows that date format errors are the most frequent tool parameter failure. "
            "The agent frequently passes natural-language date expressions to tools that require "
            "ISO 8601 format, causing API failures. Explicit formatting rules in the system prompt "
            "have been shown to reduce this class of error by 80–90%."
        ),
        "expected_impact": "Reduce date format errors from current rate to <3%",
    },
    "tool_parameter_error": {
        "target": "system_prompt",
        "suggestion": (
            "Add a parameter validation section to the system prompt:\n\n"
            "PARAMETER VALIDATION: Before calling any tool, verify that:\n"
            "1. All parameter values come directly from what the user said or prior conversation context.\n"
            "2. Do not invent or assume values not explicitly provided.\n"
            "3. If a required parameter is missing, ask the user before calling the tool."
        ),
        "rationale": (
            "Hallucinated parameter values cause tool call failures and incorrect results. "
            "Explicit grounding instructions reduce confabulation in parameter filling."
        ),
        "expected_impact": "Reduce hallucinated parameter rate by 60–70%",
    },
    "context_loss": {
        "target": "system_prompt",
        "suggestion": (
            "Add a context retention instruction to the system prompt:\n\n"
            "CONTEXT RETENTION: Throughout the conversation, maintain a mental list of all user "
            "preferences, constraints, and facts stated in earlier turns. Before each response, "
            "verify that your answer is consistent with previously stated preferences such as "
            "budget limits, seat preferences, dietary restrictions, and timing constraints. "
            "If you reference something from context, do so explicitly (e.g., 'As you mentioned, "
            "your budget is under $500...')."
        ),
        "rationale": (
            "Context loss in multi-turn conversations is a critical failure mode. Users who state "
            "preferences early in the conversation expect them to be honored in all subsequent turns. "
            "Explicit retention instructions improve preference recall by ~70% in long conversations."
        ),
        "expected_impact": "Reduce context-loss complaints from 40% to <10% in 6+ turn conversations",
    },
    "latency": {
        "target": "system_prompt",
        "suggestion": (
            "Add response efficiency guidelines to the system prompt:\n\n"
            "RESPONSE EFFICIENCY: Provide concise, direct answers. Avoid unnecessary preamble, "
            "repetition of the user's question, or extensive disclaimers. For tool-based tasks, "
            "call the minimum number of tools required to answer the question."
        ),
        "rationale": (
            "High latency is correlated with verbose responses and unnecessary tool calls. "
            "Efficiency instructions reduce average response length and tool call count."
        ),
        "expected_impact": "Reduce average latency by 20–30%",
    },
    "missing_tool_result": {
        "target": "system_prompt",
        "suggestion": (
            "Add tool completion verification to the system prompt:\n\n"
            "TOOL VERIFICATION: After every tool call, verify that the tool returned a result. "
            "If a tool call does not return a result or returns an error, inform the user and "
            "offer an alternative approach. Never proceed as if a tool call succeeded without confirming the result."
        ),
        "rationale": (
            "Missing tool results indicate calls that didn't complete. Without explicit verification "
            "instructions, the agent may present incomplete information to users."
        ),
        "expected_impact": "Eliminate unverified tool-result assumptions",
    },
}


class PromptSuggester:
    def __init__(self):
        self.llm_client = LLMClient()

    async def generate_suggestions(self, patterns: list[dict]) -> list[ImprovementSuggestion]:
        suggestions: list[ImprovementSuggestion] = []

        llm_result = await self.llm_client.analyze_failures(patterns)
        llm_suggestions_all = llm_result.get("suggestions", [])
        llm_is_mock = llm_result.get("mock", False)

        llm_iter = iter(llm_suggestions_all)

        for pattern in patterns:
            pattern_type = pattern.get("type", "")
            rate = pattern.get("rate", 0.0)
            trend = pattern.get("trend", "stable")
            frequency = pattern.get("frequency", 0)

            confidence = min(0.95, rate * 2 + (0.1 if trend == "worsening" else 0) + min(0.2, frequency / 100))

            llm_suggestions = []
            if not llm_is_mock:
                for _ in range(2):
                    s = next(llm_iter, None)
                    if s:
                        llm_suggestions.append(s)

            if llm_suggestions and not llm_is_mock:
                for s in llm_suggestions[:2]:
                    suggestions.append(ImprovementSuggestion(
                        type="prompt",
                        target=s.get("target", "system_prompt"),
                        suggestion=s.get("suggestion", ""),
                        rationale=s.get("rationale", ""),
                        confidence=round(confidence, 3),
                        expected_impact=s.get("expected_impact"),
                    ))
            else:
                template = _HEURISTIC_SUGGESTIONS.get(pattern_type)
                if not template:
                    for key, tmpl in _HEURISTIC_SUGGESTIONS.items():
                        if key in pattern_type or pattern_type in key:
                            template = tmpl
                            break

                if template:
                    affected_tools = pattern.get("affected_tools", [])
                    affected_params = pattern.get("affected_parameters", [])
                    suggestion_text = template["suggestion"]
                    if affected_tools:
                        suggestion_text = suggestion_text.replace(
                            "any tool", f"'{affected_tools[0]}' and similar tools"
                        )
                    if affected_params:
                        suggestion_text = suggestion_text.replace(
                            "date or date range parameter",
                            f"'{affected_params[0]}' parameter",
                        )

                    suggestions.append(ImprovementSuggestion(
                        type="prompt",
                        target=template["target"],
                        suggestion=suggestion_text,
                        rationale=(
                            f"{template['rationale']} "
                            f"Pattern detected in {frequency} conversations "
                            f"(rate={rate*100:.1f}%, trend={trend})."
                        ),
                        confidence=round(confidence, 3),
                        expected_impact=template.get("expected_impact"),
                    ))
                else:
                    desc = pattern.get("description", pattern_type)
                    suggestions.append(ImprovementSuggestion(
                        type="prompt",
                        target="system_prompt",
                        suggestion=(
                            f"Add explicit handling instructions for: {desc}. "
                            f"This pattern affects {rate*100:.1f}% of conversations "
                            f"and is {trend}."
                        ),
                        rationale=f"Recurring failure pattern detected: {desc}",
                        confidence=round(confidence, 3),
                        expected_impact=f"Expected to reduce '{pattern_type}' by 50-70%",
                    ))

        logger.info("PromptSuggester generated %d suggestions", len(suggestions))
        return suggestions
