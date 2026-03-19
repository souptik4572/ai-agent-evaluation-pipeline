import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.events.stream import event_bus
from app.models.db_models import RegressionReportRecord
from app.models.schemas import PipelineEvent, RegressionCompareRequest, RegressionReport, VersionSummary
from app.regression.comparator import VersionComparator
from app.regression.detector import RegressionDetector

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/regression", tags=["Regression"])

_detector = RegressionDetector()
_comparator = VersionComparator()


async def _store_report(report: RegressionReport, db: AsyncSession) -> str:
    report_id = f"reg_{uuid.uuid4().hex[:12]}"
    record = RegressionReportRecord(
        report_id=report_id,
        baseline_version=report.baseline_version,
        target_version=report.target_version,
        is_regression=report.is_regression,
        severity=report.severity,
        report_data=report.model_dump(),
    )
    db.add(record)
    await db.commit()
    return report_id


@router.post("/compare", response_model=RegressionReport)
async def compare_versions(
    payload: RegressionCompareRequest,
    db: AsyncSession = Depends(get_db),
):
    """Compare evaluation scores between two agent versions."""
    report = await _detector.compare(payload.baseline_version, payload.target_version, db)
    await _store_report(report, db)

    # Fire alerts if regression detected
    if report.is_regression:
        from app.alerting.alerts import AlertManager
        alert_mgr = AlertManager()
        report_id = f"reg_{report.baseline_version}_vs_{report.target_version}"
        await alert_mgr.create_regression_alert(report, report_id, db)

    if report.is_regression:
        try:
            await event_bus.publish(PipelineEvent(
                event_type="regression_detected",
                timestamp=datetime.now(timezone.utc),
                data={
                    "baseline_version": report.baseline_version,
                    "target_version": report.target_version,
                    "severity": report.severity,
                    "regressions": report.regressions_detected,
                },
                agent_version=report.target_version,
            ).model_dump_json())
        except Exception:
            pass

    logger.info(
        "Regression compare %s vs %s: is_regression=%s severity=%s",
        payload.baseline_version,
        payload.target_version,
        report.is_regression,
        report.severity,
    )
    return report


@router.post("/auto-check/{version}", response_model=RegressionReport | None)
async def auto_check_version(version: str, db: AsyncSession = Depends(get_db)):
    """Auto-find baseline version and run comparison if enough data exists."""
    report = await _comparator.auto_check(version, db)
    if report is None:
        return None

    await _store_report(report, db)

    if report.is_regression:
        from app.alerting.alerts import AlertManager
        alert_mgr = AlertManager()
        report_id = f"reg_{report.baseline_version}_vs_{report.target_version}"
        await alert_mgr.create_regression_alert(report, report_id, db)

    return report


@router.get("/versions", response_model=list[VersionSummary])
async def get_version_timeline(db: AsyncSession = Depends(get_db)):
    """Return all known versions with summary stats, sorted by semver."""
    return await _comparator.get_version_timeline(db)


@router.get("/reports")
async def list_regression_reports(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    is_regression: bool | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List stored regression reports with pagination."""
    stmt = (
        select(RegressionReportRecord)
        .order_by(RegressionReportRecord.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    if is_regression is not None:
        stmt = stmt.where(RegressionReportRecord.is_regression == is_regression)

    result = await db.execute(stmt)
    records = result.scalars().all()

    return {
        "data": [
            {
                "report_id": r.report_id,
                "baseline_version": r.baseline_version,
                "target_version": r.target_version,
                "is_regression": r.is_regression,
                "severity": r.severity,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "summary": r.report_data.get("summary", ""),
            }
            for r in records
        ],
        "meta": {"count": len(records), "offset": offset, "limit": limit},
    }
