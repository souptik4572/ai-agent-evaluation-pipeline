import pytest
from datetime import datetime, timezone

from app.analytics.correlation import compute_correlation
from app.meta_eval.calibrator import EvaluatorCalibrator
from app.meta_eval.drift_detector import DriftDetector
from app.models.db_models import MetaEvalRecord


@pytest.mark.asyncio
async def test_calibrator_detects_drift(db_session):
    calibrator = EvaluatorCalibrator()
    report = await calibrator.calibrate(
        conversation_id="conv_calib_test",
        human_scores={"response_quality": 0.9, "tool_accuracy": 0.8},
        auto_scores={"response_quality": 0.6, "tool_accuracy": 0.75},
        db=db_session,
    )
    assert "response_quality" in report["dimensions"]
    rq = report["dimensions"]["response_quality"]
    assert rq["status"] in ("minor_drift", "major_drift")
    assert rq["diff"] == pytest.approx(0.3, abs=0.01)
    assert "under-scoring" in report["recommendation"] or "over-scoring" in report["recommendation"]


@pytest.mark.asyncio
async def test_calibrator_aligned(db_session):
    calibrator = EvaluatorCalibrator()
    report = await calibrator.calibrate(
        conversation_id="conv_calib_aligned",
        human_scores={"response_quality": 0.85},
        auto_scores={"response_quality": 0.83},
        db=db_session,
    )
    rq = report["dimensions"]["response_quality"]
    assert rq["status"] == "aligned"
    assert report["overall_alignment"] > 0.85


@pytest.mark.asyncio
async def test_drift_detector_computes_accuracy(db_session):
    # Seed some MetaEvalRecords
    for i in range(10):
        human = 0.8
        auto = 0.8 if i < 8 else 0.3  # 2 misses
        record = MetaEvalRecord(
            evaluator_name="llm_judge",
            human_score=human,
            auto_score=auto,
            agreement=abs(human - auto) < 0.15,
            conversation_id=f"conv_drift_{i}",
            created_at=datetime.now(timezone.utc),
        )
        db_session.add(record)
    await db_session.commit()

    detector = DriftDetector()
    result = await detector.detect_drift("llm_judge", window_days=30, db=db_session)
    assert result["accuracy"] is not None
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["trend"] in ("stable", "improving", "degrading")
    assert "recommendation" in result


# ── Correlation tests ─────────────────────────────────────────────────────────

def test_perfect_positive_correlation():
    """Auto scores [0.2, 0.4, 0.6, 0.8, 1.0] vs user ratings [1, 2, 3, 4, 5] → r ≈ 1.0."""
    records = [
        {
            "conversation_id": f"conv_{i}",
            "scores": {
                "overall": s,
                "response_quality": s,
                "tool_accuracy": s,
                "coherence": s,
            },
            "user_rating": r,
        }
        for i, (s, r) in enumerate(zip([0.2, 0.4, 0.6, 0.8, 1.0], [1, 2, 3, 4, 5]))
    ]
    result = compute_correlation(records)
    assert len(result.correlations) == 4
    overall_corr = next(c for c in result.correlations if c.dimension == "overall")
    assert overall_corr.pearson_r > 0.95
    assert overall_corr.sample_size == 5
    assert "strong" in overall_corr.interpretation


def test_no_correlation_with_random_like_data():
    """Constant auto scores paired with varied ratings → r ≈ 0."""
    records = [
        {
            "conversation_id": f"conv_{i}",
            "scores": {"overall": 0.75, "response_quality": 0.75, "tool_accuracy": 0.75, "coherence": 0.75},
            "user_rating": r,
        }
        for i, r in enumerate([1, 5, 2, 4, 3, 1, 5, 2, 4, 3])
    ]
    result = compute_correlation(records)
    overall_corr = next(c for c in result.correlations if c.dimension == "overall")
    assert abs(overall_corr.pearson_r) < 0.1


def test_correlation_requires_minimum_samples():
    """With only 1 sample, correlations should be empty."""
    records = [{"conversation_id": "c1", "scores": {"overall": 0.8, "response_quality": 0.8,
                                                      "tool_accuracy": 0.8, "coherence": 0.8},
                "user_rating": 4}]
    result = compute_correlation(records)
    assert len(result.correlations) == 0
    assert result.best_dimension is None


def test_correlation_ignores_records_without_rating():
    """Records without user_rating should be excluded from correlation."""
    records = [
        {"conversation_id": "c1", "scores": {"overall": 0.9, "response_quality": 0.9, "tool_accuracy": 0.9, "coherence": 0.9}, "user_rating": None},
        {"conversation_id": "c2", "scores": {"overall": 0.5, "response_quality": 0.5, "tool_accuracy": 0.5, "coherence": 0.5}, "user_rating": 2},
        {"conversation_id": "c3", "scores": {"overall": 0.8, "response_quality": 0.8, "tool_accuracy": 0.8, "coherence": 0.8}, "user_rating": 4},
    ]
    result = compute_correlation(records)
    # Only 2 valid records — just enough
    assert all(c.sample_size == 2 for c in result.correlations)


def test_correlation_best_dimension_is_highest_abs_r():
    """best_dimension should be the dimension with highest |pearson_r|."""
    records = [
        {
            "conversation_id": f"c{i}",
            "scores": {
                "overall": s,
                "response_quality": s * 0.9,  # slightly weaker correlation
                "tool_accuracy": s,
                "coherence": 0.7,  # flat — no correlation
            },
            "user_rating": r,
        }
        for i, (s, r) in enumerate(zip([0.2, 0.4, 0.6, 0.8, 1.0], [1, 2, 3, 4, 5]))
    ]
    result = compute_correlation(records)
    assert result.best_dimension in ("overall", "tool_accuracy")


def test_correlation_scatter_data_contains_all_rated_records():
    """scatter_data should contain one entry per rated record."""
    records = [
        {"conversation_id": f"c{i}", "scores": {"overall": 0.8, "response_quality": 0.8,
                                                  "tool_accuracy": 0.8, "coherence": 0.8},
         "user_rating": 4}
        for i in range(7)
    ]
    result = compute_correlation(records)
    assert len(result.scatter_data) == 7
