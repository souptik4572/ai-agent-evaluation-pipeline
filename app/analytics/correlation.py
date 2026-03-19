"""
Compute correlation between automated evaluation scores and user feedback signals.

Pearson r is computed manually (numpy) to keep it dependency-light.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.models.schemas import CorrelationResponse, FeedbackCorrelation

_DIMENSIONS = ["overall", "response_quality", "tool_accuracy", "coherence"]


def _pearson_r(x: list[float], y: list[float]) -> float:
    """Compute Pearson r manually."""
    if len(x) < 2:
        return 0.0
    ax = np.array(x, dtype=float)
    ay = np.array(y, dtype=float)
    x_mean = ax.mean()
    y_mean = ay.mean()
    num = float(((ax - x_mean) * (ay - y_mean)).sum())
    denom = float(np.sqrt(((ax - x_mean) ** 2).sum() * ((ay - y_mean) ** 2).sum()))
    if denom == 0:
        return 0.0
    return round(num / denom, 4)


def _interpret(r: float) -> str:
    abs_r = abs(r)
    direction = "positive" if r >= 0 else "negative"
    if abs_r >= 0.7:
        return f"strong {direction}"
    if abs_r >= 0.4:
        return f"moderate {direction}"
    return "weak"


def compute_correlation(evaluations_with_feedback: list[dict]) -> CorrelationResponse:
    """
    Compute Pearson r between each evaluation dimension and normalized user_rating.

    Each item in evaluations_with_feedback should have:
      - "scores": dict with dimension keys
      - "user_rating": int 1-5
      - "conversation_id": str
    """
    valid = [
        e for e in evaluations_with_feedback
        if e.get("user_rating") is not None and e.get("scores")
    ]

    correlations: list[FeedbackCorrelation] = []
    scatter_data: list[dict[str, Any]] = []
    best_dim: str | None = None
    best_r: float = 0.0

    if len(valid) < 2:
        return CorrelationResponse(
            correlations=[],
            best_dimension=None,
            scatter_data=[],
        )

    # Build normalized ratings list once
    norm_ratings = [(e["user_rating"] - 1) / 4.0 for e in valid]

    for dim in _DIMENSIONS:
        dim_scores = [float(e["scores"].get(dim, 0.0)) for e in valid]
        r = _pearson_r(dim_scores, norm_ratings)
        correlations.append(FeedbackCorrelation(
            dimension=dim,
            pearson_r=r,
            sample_size=len(valid),
            interpretation=_interpret(r),
        ))
        if abs(r) > abs(best_r):
            best_r = r
            best_dim = dim

    # Build scatter data (overall score vs user_rating for the UI)
    for e in valid:
        scatter_data.append({
            "conversation_id": e.get("conversation_id", ""),
            "user_rating": e["user_rating"],
            "auto_score": float(e["scores"].get("overall", 0.0)),
            "response_quality": float(e["scores"].get("response_quality", 0.0)),
            "tool_accuracy": float(e["scores"].get("tool_accuracy", 0.0)),
            "coherence": float(e["scores"].get("coherence", 0.0)),
        })

    return CorrelationResponse(
        correlations=correlations,
        best_dimension=best_dim,
        scatter_data=scatter_data,
    )
