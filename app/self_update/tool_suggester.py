import logging
import re

from app.models.schemas import ImprovementSuggestion
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ToolSuggester:
    def __init__(self):
        self.llm_client = LLMClient()

    async def generate_suggestions(self, patterns: list[dict]) -> list[ImprovementSuggestion]:
        suggestions: list[ImprovementSuggestion] = []

        tool_patterns = [
            p for p in patterns
            if "tool" in p.get("type", "") or p.get("affected_tools")
        ]

        llm_result = await self.llm_client.analyze_failures(tool_patterns) if tool_patterns else {"suggestions": [], "mock": True}
        llm_is_mock = llm_result.get("mock", False)
        llm_suggestions_all = llm_result.get("suggestions", [])
        llm_iter = iter(llm_suggestions_all)

        for pattern in tool_patterns:
            affected_tools = pattern.get("affected_tools", [])
            affected_params = pattern.get("affected_parameters", [])
            rate = pattern.get("rate", 0.0)
            trend = pattern.get("trend", "stable")
            frequency = pattern.get("frequency", 0)
            confidence = min(0.95, rate * 2 + (0.1 if trend == "worsening" else 0) + min(0.2, frequency / 100))

            for tool_name in affected_tools[:2]:
                for param in affected_params[:2]:
                    root_cause = self._classify_root_cause(pattern, tool_name, param)
                    sugg = self._build_suggestion(root_cause, tool_name, param, rate, trend, frequency, confidence)
                    if sugg:
                        suggestions.append(sugg)

            llm_s = next(llm_iter, None)
            if llm_s and not llm_is_mock:
                if llm_s.get("type") == "tool":
                    suggestions.append(ImprovementSuggestion(
                        type="tool",
                        target=llm_s.get("target"),
                        suggestion=llm_s.get("suggestion", ""),
                        rationale=llm_s.get("rationale", ""),
                        confidence=round(confidence, 3),
                        expected_impact=llm_s.get("expected_impact"),
                    ))

        logger.info("ToolSuggester generated %d suggestions", len(suggestions))
        return suggestions

    def _classify_root_cause(self, pattern: dict, tool_name: str, param: str) -> str:
        desc = pattern.get("description", "").lower()
        if "format" in desc or "date" in param.lower():
            return "format_error"
        if "hallucinate" in desc or "made up" in desc or "invented" in desc:
            return "hallucination_prone"
        if "missing" in desc:
            return "missing_parameter"
        return "schema_issue"

    def _build_suggestion(
        self,
        root_cause: str,
        tool_name: str,
        param: str,
        rate: float,
        trend: str,
        frequency: int,
        confidence: float,
    ) -> ImprovementSuggestion | None:
        target = f"{tool_name}.{param}"

        if root_cause == "format_error" and "date" in param.lower():
            return ImprovementSuggestion(
                type="tool",
                target=target,
                suggestion=(
                    f"Change the '{param}' parameter description in the '{tool_name}' tool schema from "
                    f"its current vague form to:\n"
                    f"'Date range in ISO 8601 format: YYYY-MM-DD for single dates, "
                    f"YYYY-MM-DD/YYYY-MM-DD for ranges. Both start and end dates are required. "
                    f"Example: 2024-01-22/2024-01-28. Do NOT use relative expressions like \"next week\".' "
                    f"\nAlso add JSON Schema validation: "
                    f'\"pattern\": \"^\\\\d{{4}}-\\\\d{{2}}-\\\\d{{2}}(/\\\\d{{4}}-\\\\d{{2}}-\\\\d{{2}})?$\"'
                ),
                rationale=(
                    f"The current '{param}' description is ambiguous, causing the model to pass "
                    f"natural-language dates in {rate*100:.1f}% of '{tool_name}' calls. "
                    f"Adding a clear format specification with an example and JSON Schema validation "
                    f"will enforce correct formatting at both the model and API boundary levels."
                ),
                confidence=round(confidence, 3),
                expected_impact=f"Reduce '{tool_name}' date format errors from {rate*100:.1f}% to <3%",
            )

        elif root_cause == "hallucination_prone":
            return ImprovementSuggestion(
                type="tool",
                target=target,
                suggestion=(
                    f"For the '{param}' parameter in '{tool_name}': "
                    f"If the set of valid values is bounded, change the type from 'string' to 'enum' "
                    f"listing all valid options. If unbounded, add a 'description' field that explicitly says: "
                    f"'ONLY use values explicitly provided by the user. Do not infer or guess this value.' "
                    f"Also consider adding 'nullable: true' and requiring the model to ask the user "
                    f"if the value is not present in the conversation."
                ),
                rationale=(
                    f"'{param}' values in '{tool_name}' are being hallucinated in {rate*100:.1f}% of calls. "
                    f"Enum constraints or explicit grounding instructions prevent the model from inventing values."
                ),
                confidence=round(confidence, 3),
                expected_impact=f"Eliminate hallucinated '{param}' values in '{tool_name}'",
            )

        elif root_cause == "missing_parameter":
            return ImprovementSuggestion(
                type="tool",
                target=tool_name,
                suggestion=(
                    f"Review whether '{tool_name}' needs an additional parameter to cover the missing-data case. "
                    f"If '{param}' is optional, mark it as 'required: false' and document the default behavior. "
                    f"If it is required, add validation and a user-facing error message when it is absent."
                ),
                rationale=(
                    f"Missing '{param}' values in '{tool_name}' calls indicate either an under-specified "
                    f"schema or the model not knowing when the parameter is needed."
                ),
                confidence=round(confidence, 3),
                expected_impact=f"Reduce '{tool_name}' call failures due to missing '{param}'",
            )

        else:  # schema_issue
            return ImprovementSuggestion(
                type="tool",
                target=target,
                suggestion=(
                    f"Improve the description of '{param}' in '{tool_name}'. "
                    f"Current description is likely too brief or ambiguous. "
                    f"Add: (1) the expected data type, (2) a concrete example, "
                    f"(3) any format constraints, and (4) what to do if the value is unknown."
                ),
                rationale=(
                    f"Vague parameter descriptions lead to model uncertainty about the correct input format, "
                    f"contributing to the {rate*100:.1f}% error rate on '{tool_name}.{param}'."
                ),
                confidence=round(confidence, 3),
                expected_impact=f"Reduce '{tool_name}.{param}' errors by 40-60%",
            )
