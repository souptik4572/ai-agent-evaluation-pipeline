"""
Tests for version-over-version regression detection.

Covers:
- test_detects_regression_when_scores_drop
- test_no_regression_when_scores_stable
- test_auto_check_finds_previous_version
- test_insufficient_data_returns_none
- test_critical_severity_on_large_drop
- test_issue_rate_change_detected
"""
from datetime import datetime, timezone

import pytest

from app.models.db_models import Evaluation
from app.regression.comparator import VersionComparator
from app.regression.detector import RegressionDetector


def _make_eval(conv_id: str, version: str, scores: dict) -> Evaluation:
    return Evaluation(
        evaluation_id=f"eval_{conv_id}",
        conversation_id=conv_id,
        agent_version=version,
        scores=scores,
        issues=[],
        suggestions=[],
        evaluator_details={},
        created_at=datetime.now(timezone.utc),
    )


def _good_scores(overall: float = 0.90) -> dict:
    return {
        "overall": overall,
        "response_quality": overall,
        "tool_accuracy": overall,
        "coherence": overall,
    }


def _bad_scores(overall: float = 0.60) -> dict:
    return {
        "overall": overall,
        "response_quality": overall,
        "tool_accuracy": overall,
        "coherence": overall,
    }


async def _seed_evals(db_session, version: str, scores_list: list[dict]) -> None:
    for i, scores in enumerate(scores_list):
        ev = _make_eval(f"conv_{version}_{i}", version, scores)
        db_session.add(ev)
    await db_session.commit()


@pytest.mark.asyncio
async def test_detects_regression_when_scores_drop(db_session):
    """10 evals for v2.3.0 (avg 0.90) and 10 for v2.3.1 (avg 0.68) → regression."""
    await _seed_evals(db_session, "v2.3.0", [_good_scores(0.90)] * 10)
    await _seed_evals(db_session, "v2.3.1-reg", [_bad_scores(0.68)] * 10)

    detector = RegressionDetector()
    report = await detector.compare("v2.3.0", "v2.3.1-reg", db_session)

    assert report.is_regression is True
    assert report.severity in ("minor", "major", "critical")
    assert "overall" in report.regressions_detected
    assert report.baseline_sample_size == 10
    assert report.target_sample_size == 10
    assert report.dimensions["overall"].delta < 0


@pytest.mark.asyncio
async def test_no_regression_when_scores_stable(db_session):
    """Both versions at ~0.85 → no regression."""
    await _seed_evals(db_session, "v2.4.0", [_good_scores(0.85)] * 8)
    await _seed_evals(db_session, "v2.4.1", [_good_scores(0.86)] * 8)

    detector = RegressionDetector()
    report = await detector.compare("v2.4.0", "v2.4.1", db_session)

    assert report.is_regression is False
    assert report.severity == "none"
    assert len(report.regressions_detected) == 0


@pytest.mark.asyncio
async def test_critical_severity_on_large_drop(db_session):
    """Overall drops more than 15% → severity should be critical."""
    await _seed_evals(db_session, "v3.0.0", [_good_scores(0.95)] * 10)
    # Drop overall by ~22% and tool_accuracy by ~25%
    bad = [{"overall": 0.73, "response_quality": 0.75, "tool_accuracy": 0.70, "coherence": 0.72}] * 10
    await _seed_evals(db_session, "v3.0.1-critical", bad)

    detector = RegressionDetector()
    report = await detector.compare("v3.0.0", "v3.0.1-critical", db_session)

    assert report.is_regression is True
    assert report.severity == "critical"


@pytest.mark.asyncio
async def test_insufficient_data_returns_none(db_session):
    """Only 3 evals for target → auto_check returns None."""
    await _seed_evals(db_session, "v5.0.0", [_good_scores(0.90)] * 8)
    await _seed_evals(db_session, "v5.0.1-sparse", [_good_scores(0.85)] * 3)

    comparator = VersionComparator()
    result = await comparator.auto_check("v5.0.1-sparse", db_session)

    assert result is None


@pytest.mark.asyncio
async def test_auto_check_finds_previous_version(db_session):
    """auto_check on v6.0.1 should compare against v6.0.0 (the prior version)."""
    await _seed_evals(db_session, "v6.0.0", [_good_scores(0.90)] * 6)
    await _seed_evals(db_session, "v6.0.1", [_bad_scores(0.65)] * 6)

    comparator = VersionComparator()
    report = await comparator.auto_check("v6.0.1", db_session)

    assert report is not None
    assert report.baseline_version == "v6.0.0"
    assert report.target_version == "v6.0.1"
    assert report.is_regression is True


@pytest.mark.asyncio
async def test_issue_rate_change_detected(db_session):
    """Evaluations with elevated tool issues in target → is_elevated flag set."""
    # Baseline: no tool issues
    for i in range(8):
        ev = _make_eval(f"conv_v7_base_{i}", "v7.0.0", _good_scores(0.90))
        ev.issues = []
        db_session.add(ev)

    # Target: 6 of 8 have a tool issue
    for i in range(8):
        ev = _make_eval(f"conv_v7_target_{i}", "v7.0.1", _bad_scores(0.78))
        ev.issues = [{"type": "tool_error", "severity": "warning", "description": "Bad format"}] if i < 6 else []
        db_session.add(ev)

    await db_session.commit()

    detector = RegressionDetector()
    report = await detector.compare("v7.0.0", "v7.0.1", db_session)

    assert "tool_error" in report.issue_rate_changes
    tool_change = report.issue_rate_changes["tool_error"]
    assert tool_change.target_rate > tool_change.baseline_rate
    assert tool_change.is_elevated is True


@pytest.mark.asyncio
async def test_regression_summary_message_contains_versions(db_session):
    """Summary string should reference both versions."""
    await _seed_evals(db_session, "v8.0.0", [_good_scores(0.88)] * 6)
    await _seed_evals(db_session, "v8.0.1", [_bad_scores(0.60)] * 6)

    detector = RegressionDetector()
    report = await detector.compare("v8.0.0", "v8.0.1", db_session)

    assert "v8.0.0" in report.summary
    assert "v8.0.1" in report.summary
