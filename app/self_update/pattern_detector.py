import logging
import uuid
from collections import defaultdict

from app.models.schemas import EvaluationResult

logger = logging.getLogger(__name__)


class PatternDetector:
    async def detect_failure_patterns(self, evaluations: list[EvaluationResult]) -> list[dict]:
        if not evaluations:
            return []

        total = len(evaluations)
        issue_groups: dict[str, list[dict]] = defaultdict(list)
        tool_failures: dict[str, dict] = defaultdict(lambda: {"count": 0, "params": defaultdict(int), "conv_ids": []})
        coherence_failures: list[dict] = []

        for ev in evaluations:
            for issue in ev.issues_detected:
                issue_groups[issue.type].append({
                    "evaluation_id": ev.evaluation_id,
                    "conversation_id": ev.conversation_id,
                    "agent_version": ev.agent_version,
                    "description": issue.description,
                    "severity": issue.severity,
                })

            if ev.tool_evaluation:
                for hp in ev.tool_evaluation.hallucinated_params:
                    parts = hp.split(".")
                    tool_name = parts[0] if parts else "unknown"
                    param = parts[1] if len(parts) > 1 else "unknown"
                    tool_failures[tool_name]["count"] += 1
                    tool_failures[tool_name]["params"][param] += 1
                    if ev.conversation_id not in tool_failures[tool_name]["conv_ids"]:
                        tool_failures[tool_name]["conv_ids"].append(ev.conversation_id)

            if ev.scores.coherence < 0.6:
                coherence_failures.append({
                    "conversation_id": ev.conversation_id,
                    "turn_count": len([
                        k for k in ev.evaluator_details.get("coherence", {}).get("sub_scores", {})
                    ]),
                    "score": ev.scores.coherence,
                })

        patterns: list[dict] = []

        for issue_type, occurrences in issue_groups.items():
            frequency = len(occurrences)
            rate = frequency / total
            if rate < 0.05:
                continue

            affected_versions = list({o["agent_version"] for o in occurrences})
            sample_ids = [o["conversation_id"] for o in occurrences[:3]]

            half = len(evaluations) // 2
            first_half_ids = {ev.evaluation_id for ev in evaluations[:half]}
            second_half_ids = {ev.evaluation_id for ev in evaluations[half:]}
            first_count = sum(1 for o in occurrences if o["evaluation_id"] in first_half_ids)
            second_count = sum(1 for o in occurrences if o["evaluation_id"] in second_half_ids)
            first_rate = first_count / max(half, 1)
            second_rate = second_count / max(total - half, 1)

            if second_rate > first_rate * 1.2:
                trend = "worsening"
            elif second_rate < first_rate * 0.8:
                trend = "improving"
            else:
                trend = "stable"

            severity = "high" if rate > 0.2 or any(o["severity"] == "critical" for o in occurrences) else "medium"

            patterns.append({
                "pattern_id": f"pat_{uuid.uuid4().hex[:8]}",
                "type": issue_type,
                "description": (
                    f"{issue_type.replace('_', ' ').title()} detected in "
                    f"{frequency}/{total} ({rate*100:.1f}%) conversations. "
                    f"Example: {occurrences[0]['description']}"
                ),
                "frequency": frequency,
                "rate": round(rate, 4),
                "trend": trend,
                "affected_versions": affected_versions,
                "affected_tools": _extract_tools(occurrences),
                "affected_parameters": _extract_params(occurrences),
                "sample_conversation_ids": sample_ids,
                "severity": severity,
            })

        for tool_name, data in tool_failures.items():
            frequency = data["count"]
            rate = frequency / total
            if rate < 0.05:
                continue
            top_params = sorted(data["params"].items(), key=lambda x: -x[1])
            param_names = [p for p, _ in top_params]
            patterns.append({
                "pattern_id": f"pat_{uuid.uuid4().hex[:8]}",
                "type": "tool_parameter_error",
                "description": (
                    f"Tool '{tool_name}' called with hallucinated/invalid parameters "
                    f"in {frequency}/{total} ({rate*100:.1f}%) conversations. "
                    f"Most affected parameters: {param_names}"
                ),
                "frequency": frequency,
                "rate": round(rate, 4),
                "trend": "stable",
                "affected_versions": list({ev.agent_version for ev in evaluations}),
                "affected_tools": [tool_name],
                "affected_parameters": param_names,
                "sample_conversation_ids": data["conv_ids"][:3],
                "severity": "high" if rate > 0.1 else "medium",
            })

        def _sort_key(p: dict) -> float:
            sev_weight = {"high": 2.0, "medium": 1.0, "low": 0.5}.get(p.get("severity", "medium"), 1.0)
            return p["rate"] * sev_weight

        patterns.sort(key=_sort_key, reverse=True)
        logger.info("Detected %d significant failure patterns from %d evaluations", len(patterns), total)
        return patterns


def _extract_tools(occurrences: list[dict]) -> list[str]:
    tools: set[str] = set()
    for o in occurrences:
        desc = o.get("description", "")
        import re
        found = re.findall(r"['\"](\w+_\w+)['\"]", desc)
        tools.update(found)
    return list(tools)


def _extract_params(occurrences: list[dict]) -> list[str]:
    params: set[str] = set()
    for o in occurrences:
        desc = o.get("description", "").lower()
        import re
        found = re.findall(r"parameter[s]?\s+['\"]?(\w+)['\"]?", desc)
        params.update(found)
    return list(params)
