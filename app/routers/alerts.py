import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.alerting.alerts import AlertManager
from app.database import get_db
from app.models.db_models import AlertRecord, WebhookConfigRecord
from app.models.schemas import Alert, AlertStatusUpdate, WebhookConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/alerts", tags=["Alerts"])

_alert_mgr = AlertManager()


def _record_to_alert(r: AlertRecord) -> Alert:
    return Alert(
        alert_id=r.alert_id,
        type=r.type,
        severity=r.severity,
        title=r.title,
        description=r.description,
        related_entity_id=r.related_entity_id,
        status=r.status,
        created_at=r.created_at,
        acknowledged_at=r.acknowledged_at,
        resolved_at=r.resolved_at,
    )


@router.get("/summary")
async def alert_summary(db: AsyncSession = Depends(get_db)):
    """Count of open alerts by severity."""
    result = await db.execute(
        select(AlertRecord).where(AlertRecord.status == "open")
    )
    open_alerts = result.scalars().all()
    counts: dict[str, int] = {"critical": 0, "warning": 0, "info": 0}
    for a in open_alerts:
        counts[a.severity] = counts.get(a.severity, 0) + 1
    return {
        "total_open": len(open_alerts),
        "by_severity": counts,
    }


@router.get("", response_model=list[Alert])
async def list_alerts(
    status: str | None = Query(None, description="Filter by status: open, acknowledged, resolved"),
    type: str | None = Query(None, description="Filter by type"),
    severity: str | None = Query(None, description="Filter by severity"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(AlertRecord).order_by(AlertRecord.created_at.desc()).offset(offset).limit(limit)
    if status:
        stmt = stmt.where(AlertRecord.status == status)
    if type:
        stmt = stmt.where(AlertRecord.type == type)
    if severity:
        stmt = stmt.where(AlertRecord.severity == severity)

    result = await db.execute(stmt)
    return [_record_to_alert(r) for r in result.scalars().all()]


@router.get("/{alert_id}", response_model=Alert)
async def get_alert(alert_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(AlertRecord).where(AlertRecord.alert_id == alert_id)
    )
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    return _record_to_alert(record)


@router.patch("/{alert_id}", response_model=Alert)
async def update_alert_status(
    alert_id: str,
    payload: AlertStatusUpdate,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(AlertRecord).where(AlertRecord.alert_id == alert_id)
    )
    record = result.scalar_one_or_none()
    if not record:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")

    now = datetime.now(timezone.utc)
    record.status = payload.status
    if payload.status == "acknowledged":
        record.acknowledged_at = now
    elif payload.status == "resolved":
        record.resolved_at = now

    await db.commit()
    await db.refresh(record)
    return _record_to_alert(record)


@router.post("/webhook/configure")
async def configure_webhook(payload: WebhookConfig, db: AsyncSession = Depends(get_db)):
    """Store a webhook URL to receive alert notifications."""
    config = WebhookConfigRecord(url=payload.url)
    db.add(config)
    await db.commit()
    logger.info("Webhook configured: %s", payload.url)
    return {"status": "configured", "url": payload.url}
