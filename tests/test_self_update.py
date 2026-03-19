import pytest

from app.self_update.pattern_detector import PatternDetector
from app.self_update.prompt_suggester import PromptSuggester
from app.self_update.tool_suggester import ToolSuggester
from app.models.schemas import (
    EvaluationResult,
    EvaluationScores,
    IssueDetected,
    ToolEvaluation,
)
from datetime import datetime, timezone


def _make_eval(conv_id: str, has_date_error: bool = False, has_context_loss: bool = False) -> EvaluationResult:
    issues = []
    hallucinated = []
    if has_date_error:
        issues.append(IssueDetected(
            type="tool_parameter_format_error",
            severity="warning",
            description="Parameter 'date_range' in tool 'flight_search' has invalid date format: 'next week'.",
        ))
        hallucinated = ["flight_search.date_range"]
    if has_context_loss:
        issues.append(IssueDetected(
            type="context_loss",
            severity="critical",
            description="Agent forgot user preferences in final turn.",
        ))
    tool_eval = ToolEvaluation(
        selection_accuracy=0.9,
        parameter_accuracy=0.5 if has_date_error else 1.0,
        hallucinated_params=hallucinated,
        execution_success=not has_date_error,
    ) if has_date_error or not has_context_loss else None

    score = 0.4 if (has_date_error or has_context_loss) else 0.9
    return EvaluationResult(
        evaluation_id=f"eval_{conv_id}",
        conversation_id=conv_id,
        agent_version="v2.3.1",
        scores=EvaluationScores(overall=score, response_quality=score, tool_accuracy=score, coherence=score),
        tool_evaluation=tool_eval,
        issues_detected=issues,
        improvement_suggestions=[],
        evaluator_details={},
        created_at=datetime.now(timezone.utc),
    )


@pytest.mark.asyncio
async def test_pattern_detector_finds_date_format_errors():
    evals = [_make_eval(f"conv_{i:03d}", has_date_error=(i < 4)) for i in range(20)]
    detector = PatternDetector()
    patterns = await detector.detect_failure_patterns(evals)
    # 4/20 = 20% rate — should be found
    types = [p["type"] for p in patterns]
    assert "tool_parameter_format_error" in types or "tool_parameter_error" in types


@pytest.mark.asyncio
async def test_prompt_suggester_generates_specific_suggestion():
    pattern = {
        "pattern_id": "pat_test001",
        "type": "tool_parameter_format_error",
        "description": "flight_search called with invalid date format in 20% of conversations.",
        "frequency": 4,
        "rate": 0.20,
        "trend": "worsening",
        "affected_versions": ["v2.3.1"],
        "affected_tools": ["flight_search"],
        "affected_parameters": ["date_range"],
        "sample_conversation_ids": ["conv_001"],
        "severity": "high",
    }
    suggester = PromptSuggester()
    suggestions = await suggester.generate_suggestions([pattern])
    assert len(suggestions) > 0
    # Must mention the specific tool name
    combined = " ".join(s.suggestion + s.rationale for s in suggestions)
    assert "flight_search" in combined or "date" in combined.lower()
    # Must NOT be a one-liner generic suggestion
    assert len(suggestions[0].suggestion) > 50


@pytest.mark.asyncio
async def test_tool_suggester_generates_suggestion_with_expected_impact():
    pattern = {
        "pattern_id": "pat_test002",
        "type": "tool_parameter_format_error",
        "description": "flight_search.date_range has invalid format in 15% of calls.",
        "frequency": 3,
        "rate": 0.15,
        "trend": "stable",
        "affected_versions": ["v2.3.1"],
        "affected_tools": ["flight_search"],
        "affected_parameters": ["date_range"],
        "sample_conversation_ids": ["conv_001"],
        "severity": "medium",
    }
    suggester = ToolSuggester()
    suggestions = await suggester.generate_suggestions([pattern])
    assert len(suggestions) > 0
    assert suggestions[0].expected_impact is not None
    assert suggestions[0].type == "tool"
    assert "flight_search" in (suggestions[0].target or "")
