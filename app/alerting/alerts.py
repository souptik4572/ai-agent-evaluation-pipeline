from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.events.stream import event_bus
from app.models.db_models import AlertRecord, Evaluation, WebhookConfigRecord
from app.models.schemas import Alert, PipelineEvent, RegressionReport

logger = logging.getLogger(__name__)

QUALITY_DROP_THRESHOLD = 0.6
TOOL_FAILURE_THRESHOLD = 0.10
DISAGREEMENT_THRESHOLD = 0.30


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AlertManager:
    async def create_regression_alert(
        self,
        report: RegressionReport,
        related_id: str,
        db: AsyncSession,
    ) -> AlertRecord:
        severity_map = {"critical": "critical", "major": "critical", "minor": "warning", "none": "info"}
        severity = severity_map.get(report.severity, "warning")
        title = f"Regression detected: {report.target_version} vs {report.baseline_version}"
        description = (
            f"{report.summary}\n\n"
            f"Affected dimensions: {', '.join(report.regressions_detected)}.\n"
            f"Samples: baseline={report.baseline_sample_size}, target={report.target_sample_size}."
        )
        return await self._create(
            alert_type="regression",
            severity=severity,
            title=title,
            description=description,
            related_entity_id=related_id,
            db=db,
        )

    async def check_quality_drop(self, db: AsyncSession) -> AlertRecord | None:
        result = await db.execute(
            select(Evaluation).order_by(Evaluation.created_at.desc()).limit(20)
        )
        evals = result.scalars().all()
        if len(evals) < 5:
            return None

        scores = [float(ev.scores.get("overall", 0.0)) for ev in evals if ev.scores]
        avg = sum(scores) / len(scores) if scores else 1.0
        if avg >= QUALITY_DROP_THRESHOLD:
            return None

        if await self._has_open_alert("quality_drop", db):
            return None

        return await self._create(
            alert_type="quality_drop",
            severity="warning",
            title=f"Quality drop detected: avg score {avg:.2f} (threshold {QUALITY_DROP_THRESHOLD})",
            description=(
                f"The rolling average overall score over the last {len(scores)} evaluations "
                f"is {avg:.3f}, below the threshold of {QUALITY_DROP_THRESHOLD}."
            ),
            related_entity_id=None,
            db=db,
        )

    async def check_tool_failure_rate(self, db: AsyncSession) -> AlertRecord | None:
        result = await db.execute(
            select(Evaluation).order_by(Evaluation.created_at.desc()).limit(50)
        )
        evals = result.scalars().all()
        if len(evals) < 10:
            return None

        failures = 0
        for ev in evals:
            for issue in ev.issues or []:
                if "tool" in issue.get("type", "").lower() and issue.get("severity") in ("critical", "warning"):
                    failures += 1
                    break

        rate = failures / len(evals)
        if rate <= TOOL_FAILURE_THRESHOLD:
            return None

        if await self._has_open_alert("high_failure_rate", db):
            return None

        return await self._create(
            alert_type="high_failure_rate",
            severity="warning",
            title=f"High tool failure rate: {rate:.1%} (threshold {TOOL_FAILURE_THRESHOLD:.0%})",
            description=(
                f"{failures} of {len(evals)} recent evaluations have tool-related issues "
                f"({rate:.1%}), exceeding the {TOOL_FAILURE_THRESHOLD:.0%} threshold."
            ),
            related_entity_id=None,
            db=db,
        )

    async def check_annotator_disagreement(self, db: AsyncSession) -> AlertRecord | None:
        result = await db.execute(
            select(Evaluation)
            .where(Evaluation.annotation_agreement.isnot(None))
            .order_by(Evaluation.created_at.desc())
            .limit(20)
        )
        evals = result.scalars().all()
        if len(evals) < 5:
            return None

        disagreements = sum(
            1 for ev in evals
            if ev.annotation_agreement
            and float(ev.annotation_agreement.get("overall_agreement", 1.0)) < 1.0
        )
        rate = disagreements / len(evals)
        if rate <= DISAGREEMENT_THRESHOLD:
            return None

        if await self._has_open_alert("annotator_conflict", db):
            return None

        return await self._create(
            alert_type="annotator_conflict",
            severity="info",
            title=f"High annotator disagreement: {rate:.1%} (threshold {DISAGREEMENT_THRESHOLD:.0%})",
            description=(
                f"{disagreements} of {len(evals)} recent annotated evaluations show annotator "
                f"disagreement ({rate:.1%}), exceeding the {DISAGREEMENT_THRESHOLD:.0%} threshold."
            ),
            related_entity_id=None,
            db=db,
        )

    async def _create(
        self,
        alert_type: str,
        severity: str,
        title: str,
        description: str,
        related_entity_id: str | None,
        db: AsyncSession,
    ) -> AlertRecord:
        alert_id = f"alert_{uuid.uuid4().hex[:12]}"
        record = AlertRecord(
            alert_id=alert_id,
            type=alert_type,
            severity=severity,
            title=title,
            description=description,
            related_entity_id=related_entity_id,
            status="open",
            created_at=_now(),
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        logger.warning("Alert created [%s] %s: %s", severity.upper(), alert_type, title)

        try:
            await event_bus.publish(PipelineEvent(
                event_type="alert_fired",
                timestamp=_now(),
                data={
                    "alert_id": record.alert_id,
                    "type": record.type,
                    "severity": record.severity,
                    "title": record.title,
                },
            ).model_dump_json())
        except Exception:
            pass

        await self._fire_webhook(record, db)
        return record

    async def _has_open_alert(self, alert_type: str, db: AsyncSession) -> bool:
        result = await db.execute(
            select(AlertRecord)
            .where(AlertRecord.type == alert_type, AlertRecord.status == "open")
            .limit(1)
        )
        return result.scalar_one_or_none() is not None

    async def _fire_webhook(self, record: AlertRecord, db: AsyncSession) -> None:
        result = await db.execute(
            select(WebhookConfigRecord).order_by(WebhookConfigRecord.id.desc()).limit(1)
        )
        config = result.scalar_one_or_none()
        if not config:
            return

        payload = {
            "alert_id": record.alert_id,
            "type": record.type,
            "severity": record.severity,
            "title": record.title,
            "description": record.description,
            "status": record.status,
            "created_at": record.created_at.isoformat() if record.created_at else None,
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(config.url, json=payload)
                logger.info("Webhook fired to %s: status=%d", config.url, resp.status_code)
        except Exception as exc:
            logger.warning("Webhook delivery failed to %s: %s", config.url, exc)
