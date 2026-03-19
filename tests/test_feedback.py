import pytest

from app.feedback.aggregator import AnnotationAggregator
from app.feedback.routing import ConfidenceRouter
from app.models.schemas import Annotation, EvaluationResult, EvaluationScores, IssueDetected


def _make_eval(overall: float, has_critical: bool = False) -> EvaluationResult:
    from datetime import datetime, timezone
    issues = []
    if has_critical:
        issues.append(IssueDetected(type="critical_test", severity="critical", description="Test critical issue"))
    return EvaluationResult(
        evaluation_id="eval_test",
        conversation_id="conv_test",
        agent_version="v1",
        scores=EvaluationScores(overall=overall, response_quality=overall, tool_accuracy=overall, coherence=overall),
        issues_detected=issues,
        improvement_suggestions=[],
        evaluator_details={},
        created_at=datetime.now(timezone.utc),
    )


def test_annotation_aggregator_unanimous():
    anns = [
        Annotation(type="tool_accuracy", label="correct", annotator_id="ann_001", confidence=0.9),
        Annotation(type="tool_accuracy", label="correct", annotator_id="ann_002", confidence=0.85),
        Annotation(type="tool_accuracy", label="correct", annotator_id="ann_003", confidence=0.95),
    ]
    agg = AnnotationAggregator()
    result = agg.compute_agreement(anns)
    assert result["by_type"]["tool_accuracy"]["agreement"] == 1.0
    assert result["overall_agreement"] == 1.0
    assert len(result["disagreements"]) == 0


def test_annotation_aggregator_detects_disagreement():
    anns = [
        Annotation(type="response_quality", label="good", annotator_id="ann_201", confidence=0.75),
        Annotation(type="response_quality", label="good", annotator_id="ann_202", confidence=0.65),
        Annotation(type="response_quality", label="poor", annotator_id="ann_203", confidence=0.80),
    ]
    agg = AnnotationAggregator()
    result = agg.compute_agreement(anns)
    assert result["by_type"]["response_quality"]["agreement"] < 1.0
    assert len(result["disagreements"]) > 0


def test_confidence_router_auto_labels_good():
    router = ConfidenceRouter()
    ev = _make_eval(0.92)
    decision = router.route(ev, None)
    assert decision["routing_decision"] == "auto_label"
    assert decision["auto_label"] == "good"


def test_confidence_router_auto_labels_poor():
    router = ConfidenceRouter()
    ev = _make_eval(0.08)
    decision = router.route(ev, None)
    assert decision["routing_decision"] == "auto_label"
    assert decision["auto_label"] == "poor"


def test_confidence_router_human_review_uncertain():
    router = ConfidenceRouter()
    ev = _make_eval(0.65)
    decision = router.route(ev, None)
    assert decision["routing_decision"] == "human_review"


def test_confidence_router_human_review_critical_issue():
    router = ConfidenceRouter()
    ev = _make_eval(0.92, has_critical=True)
    decision = router.route(ev, None)
    assert decision["routing_decision"] == "human_review"


def test_confidence_router_tiebreaker_on_disagreement():
    router = ConfidenceRouter()
    ev = _make_eval(0.90)
    anns = [
        Annotation(type="response_quality", label="good", annotator_id="ann_1", confidence=0.8),
        Annotation(type="response_quality", label="poor", annotator_id="ann_2", confidence=0.8),
    ]
    decision = router.route(ev, anns)
    assert decision["routing_decision"] == "tiebreaker"
