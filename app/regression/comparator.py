from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db_models import Conversation, Evaluation
from app.models.schemas import RegressionReport, VersionSummary
from app.regression.detector import RegressionDetector

logger = logging.getLogger(__name__)

MIN_EVALS = 5


def _parse_semver(v: str) -> tuple[int, ...]:
    cleaned = v.lstrip("v")
    parts = re.split(r"[.\-]", cleaned)
    result = []
    for p in parts:
        try:
            result.append(int(p))
        except ValueError:
            result.append(0)
    return tuple(result)


class VersionComparator:
    def __init__(self) -> None:
        self._detector = RegressionDetector()

    async def auto_check(self, version: str, db: AsyncSession) -> RegressionReport | None:
        """Find the nearest previous version and compare if both have enough evaluations."""
        all_versions = await self._get_all_versions_with_eval_count(db)

        if version not in all_versions or all_versions[version] < MIN_EVALS:
            logger.info(
                "auto_check skipped for %s: only %d evaluations (need %d)",
                version,
                all_versions.get(version, 0),
                MIN_EVALS,
            )
            return None

        sorted_versions = sorted(all_versions.keys(), key=_parse_semver)
        try:
            idx = sorted_versions.index(version)
        except ValueError:
            return None

        baseline_version: str | None = None
        for prev in reversed(sorted_versions[:idx]):
            if all_versions.get(prev, 0) >= MIN_EVALS:
                baseline_version = prev
                break

        if baseline_version is None:
            logger.info("No suitable baseline version found for %s", version)
            return None

        logger.info("Auto-checking %s vs baseline %s", version, baseline_version)
        return await self._detector.compare(baseline_version, version, db)

    async def get_version_timeline(self, db: AsyncSession) -> list[VersionSummary]:
        """Return all versions with summary stats, sorted by semver."""
        conv_result = await db.execute(
            select(Conversation.agent_version, func.count(Conversation.id).label("conv_count"))
            .group_by(Conversation.agent_version)
        )
        conv_counts: dict[str, int] = {row.agent_version: row.conv_count for row in conv_result}

        eval_result = await db.execute(
            select(Evaluation.agent_version, Evaluation.scores, Evaluation.created_at)
        )
        rows = eval_result.all()

        version_data: dict[str, dict[str, Any]] = {}
        for row in rows:
            v = row.agent_version
            if v not in version_data:
                version_data[v] = {"scores": [], "dates": []}
            overall = float(row.scores.get("overall", 0.0)) if row.scores else 0.0
            version_data[v]["scores"].append(overall)
            if row.created_at:
                version_data[v]["dates"].append(row.created_at)

        summaries: list[VersionSummary] = []
        for v in set(conv_counts) | set(version_data):
            scores = version_data.get(v, {}).get("scores", [])
            dates = version_data.get(v, {}).get("dates", [])
            mean_score = sum(scores) / len(scores) if scores else 0.0
            summaries.append(VersionSummary(
                version=v,
                conversation_count=conv_counts.get(v, 0),
                eval_count=len(scores),
                mean_overall_score=round(mean_score, 4),
                date_range={
                    "earliest": min(dates).isoformat() if dates else None,
                    "latest": max(dates).isoformat() if dates else None,
                },
            ))

        summaries.sort(key=lambda s: _parse_semver(s.version))
        return summaries

    async def _get_all_versions_with_eval_count(self, db: AsyncSession) -> dict[str, int]:
        result = await db.execute(
            select(Evaluation.agent_version, func.count(Evaluation.id).label("cnt"))
            .group_by(Evaluation.agent_version)
        )
        return {row.agent_version: row.cnt for row in result}
