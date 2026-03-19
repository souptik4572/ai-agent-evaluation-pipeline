from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import Evaluation
from app.models.schemas import DimensionComparison, IssueRateChange, RegressionReport

logger = logging.getLogger(__name__)

_DIMENSIONS = ["overall", "response_quality", "tool_accuracy", "coherence"]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def _welch_t_pvalue(a: list[float], b: list[float]) -> float | None:
    """Welch's t-test via scipy; returns None if scipy is unavailable."""
    try:
        from scipy import stats  # type: ignore
        if len(a) < 2 or len(b) < 2:
            return None
        _, p = stats.ttest_ind(a, b, equal_var=False)
        return float(p)
    except ImportError:
        return None


def _significance(delta: float, p_value: float | None, n_baseline: int, n_target: int) -> str:
    min_n = min(n_baseline, n_target)
    if p_value is not None:
        if p_value < 0.05:
            return "significant"
        if p_value < 0.15:
            return "marginal"
        return "not_significant"
    # Threshold fallback when scipy is unavailable
    if abs(delta) > 0.10 and min_n >= 5:
        return "significant"
    if abs(delta) > 0.05 and min_n >= 5:
        return "marginal"
    return "not_significant"


def _severity(regressions: list[str], dimensions: dict[str, DimensionComparison]) -> str:
    if not regressions:
        return "none"
    _empty = DimensionComparison(
        baseline_mean=0, target_mean=0, delta=0, delta_pct=0,
        is_regression=False, significance="not_significant",
    )
    overall_delta_pct = dimensions.get("overall", _empty).delta_pct
    tool_delta_pct = dimensions.get("tool_accuracy", _empty).delta_pct
    if overall_delta_pct < -15 or tool_delta_pct < -20:
        return "critical"
    for dim in regressions:
        if dimensions[dim].delta_pct < -10:
            return "major"
    return "minor"


def _summarize(
    baseline_version: str,
    target_version: str,
    regressions: list[str],
    dimensions: dict[str, DimensionComparison],
    severity: str,
    n_baseline: int,
    n_target: int,
) -> str:
    if not regressions:
        return (
            f"No regressions detected comparing {baseline_version} (n={n_baseline}) "
            f"→ {target_version} (n={n_target}). All dimensions stable."
        )
    parts = [f"{dim} {dimensions[dim].delta_pct:+.1f}%" for dim in regressions]
    return (
        f"{severity.upper()} regression detected in {target_version} vs {baseline_version} "
        f"(n={n_baseline}→{n_target}): {', '.join(parts)}."
    )


class RegressionDetector:
    async def compare(
        self,
        baseline_version: str,
        target_version: str,
        db: AsyncSession,
    ) -> RegressionReport:
        baseline_evals = await self._load_evals(baseline_version, db)
        target_evals = await self._load_evals(target_version, db)

        n_b = len(baseline_evals)
        n_t = len(target_evals)

        def _dim_scores(evals: list[Any], dim: str) -> list[float]:
            return [float(ev.scores.get(dim, 0.0)) for ev in evals if ev.scores]

        dimensions: dict[str, DimensionComparison] = {}
        regressions_detected: list[str] = []

        for dim in _DIMENSIONS:
            b_scores = _dim_scores(baseline_evals, dim)
            t_scores = _dim_scores(target_evals, dim)

            b_mean = _mean(b_scores)
            t_mean = _mean(t_scores)
            delta = t_mean - b_mean
            delta_pct = (delta / b_mean * 100) if b_mean > 0 else 0.0

            p_value = _welch_t_pvalue(b_scores, t_scores)
            sig = _significance(delta, p_value, n_b, n_t)

            is_regression = delta < -0.05 and sig in ("significant", "marginal")
            if is_regression:
                regressions_detected.append(dim)

            dimensions[dim] = DimensionComparison(
                baseline_mean=round(b_mean, 4),
                target_mean=round(t_mean, 4),
                delta=round(delta, 4),
                delta_pct=round(delta_pct, 2),
                is_regression=is_regression,
                significance=sig,
            )

        issue_rate_changes = await self._compare_issue_rates(baseline_evals, target_evals)
        severity = _severity(regressions_detected, dimensions)
        summary = _summarize(baseline_version, target_version, regressions_detected, dimensions, severity, n_b, n_t)

        return RegressionReport(
            baseline_version=baseline_version,
            target_version=target_version,
            baseline_sample_size=n_b,
            target_sample_size=n_t,
            dimensions=dimensions,
            issue_rate_changes=issue_rate_changes,
            regressions_detected=regressions_detected,
            is_regression=bool(regressions_detected),
            severity=severity,
            summary=summary,
        )

    async def _load_evals(self, version: str, db: AsyncSession) -> list[Any]:
        result = await db.execute(
            select(Evaluation).where(Evaluation.agent_version == version)
        )
        return result.scalars().all()

    async def _compare_issue_rates(
        self,
        baseline_evals: list[Any],
        target_evals: list[Any],
    ) -> dict[str, IssueRateChange]:
        def _issue_counts(evals: list[Any]) -> dict[str, int]:
            counts: dict[str, int] = {}
            for ev in evals:
                seen: set[str] = set()
                for issue in ev.issues or []:
                    issue_type = issue.get("type", "unknown")
                    if issue_type not in seen:
                        counts[issue_type] = counts.get(issue_type, 0) + 1
                        seen.add(issue_type)
            return counts

        n_b = max(len(baseline_evals), 1)
        n_t = max(len(target_evals), 1)
        b_counts = _issue_counts(baseline_evals)
        t_counts = _issue_counts(target_evals)

        result: dict[str, IssueRateChange] = {}
        for issue_type in set(b_counts) | set(t_counts):
            b_rate = b_counts.get(issue_type, 0) / n_b
            t_rate = t_counts.get(issue_type, 0) / n_t
            change_pct = ((t_rate - b_rate) / b_rate * 100) if b_rate > 0 else (100.0 if t_rate > 0 else 0.0)
            result[issue_type] = IssueRateChange(
                baseline_rate=round(b_rate, 4),
                target_rate=round(t_rate, 4),
                change_pct=round(change_pct, 2),
                is_elevated=t_rate > b_rate * 1.5 and t_rate > 0.05,
            )
        return result
