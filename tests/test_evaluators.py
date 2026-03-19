import pytest

from app.evaluators.coherence import CoherenceEvaluator
from app.evaluators.heuristic import HeuristicEvaluator
from app.evaluators.pipeline import EvaluationPipeline
from app.evaluators.tool_call import ToolCallEvaluator


@pytest.mark.asyncio
async def test_heuristic_catches_high_latency(sample_conversation_with_tool_error):
    ev = HeuristicEvaluator()
    result = await ev.evaluate(sample_conversation_with_tool_error)
    assert result["score"] < 1.0
    latency_issues = [i for i in result["issues"] if i["type"] == "latency"]
    assert len(latency_issues) > 0


@pytest.mark.asyncio
async def test_heuristic_passes_valid_conversation(sample_conversation):
    ev = HeuristicEvaluator()
    result = await ev.evaluate(sample_conversation)
    assert result["score"] > 0.7
    latency_issues = [i for i in result["issues"] if i["type"] == "latency"]
    assert len(latency_issues) == 0


@pytest.mark.asyncio
async def test_tool_call_detects_bad_date_format(sample_conversation_with_tool_error):
    ev = ToolCallEvaluator()
    result = await ev.evaluate(sample_conversation_with_tool_error)
    # "next week" is not a valid date format — should be flagged
    assert len(result["details"]["hallucinated_params"]) > 0 or result["score"] < 1.0


@pytest.mark.asyncio
async def test_tool_call_scores_high_for_correct_usage(sample_conversation):
    ev = ToolCallEvaluator()
    result = await ev.evaluate(sample_conversation)
    # 2024-01-22/2024-01-29 is valid ISO format
    assert result["details"]["hallucinated_params"] == [] or result["score"] > 0.5
    assert result["details"]["tool_call_count"] == 1


@pytest.mark.asyncio
async def test_coherence_detects_context_loss(sample_long_conversation):
    ev = CoherenceEvaluator()
    result = await ev.evaluate(sample_long_conversation)
    # 6-turn conversation where preferences are ignored in final turn
    context_issues = [
        i for i in result["issues"]
        if i["type"] in ("context_loss", "contradiction")
    ]
    assert len(context_issues) > 0 or result["score"] < 0.9


@pytest.mark.asyncio
async def test_full_pipeline_returns_structured_result(sample_conversation):
    pipeline = EvaluationPipeline()
    result = await pipeline.run(sample_conversation)
    assert result.evaluation_id.startswith("eval_")
    assert result.conversation_id == "conv_test_001"
    assert 0.0 <= result.scores.overall <= 1.0
    assert isinstance(result.issues_detected, list)
    assert isinstance(result.improvement_suggestions, list)
    assert "heuristic" in result.evaluator_details
    assert "llm_judge" in result.evaluator_details
