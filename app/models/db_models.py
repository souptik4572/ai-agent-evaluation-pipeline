from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    agent_version: Mapped[str] = mapped_column(String, nullable=False)
    data: Mapped[dict] = mapped_column(JSON, nullable=False)
    feedback_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metadata_data: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_id: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    conversation_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    agent_version: Mapped[str] = mapped_column(String, nullable=False)
    scores: Mapped[dict] = mapped_column(JSON, nullable=False)
    tool_evaluation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    issues: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    suggestions: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    evaluator_details: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    annotation_agreement: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    routing_decision: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Suggestion(Base):
    __tablename__ = "suggestions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    suggestion_id: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    evaluation_id: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(String, nullable=False)
    target: Mapped[str | None] = mapped_column(String, nullable=True)
    suggestion: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    expected_impact: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="pending", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class RegressionReportRecord(Base):
    __tablename__ = "regression_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    report_id: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    baseline_version: Mapped[str] = mapped_column(String, nullable=False)
    target_version: Mapped[str] = mapped_column(String, nullable=False)
    is_regression: Mapped[bool] = mapped_column(Boolean, index=True, nullable=False)
    severity: Mapped[str] = mapped_column(String, nullable=False)
    report_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class AlertRecord(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_id: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    type: Mapped[str] = mapped_column(String, index=True, nullable=False)
    severity: Mapped[str] = mapped_column(String, index=True, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    related_entity_id: Mapped[str | None] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, default="open", index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WebhookConfigRecord(Base):
    __tablename__ = "webhook_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class MetaEvalRecord(Base):
    __tablename__ = "meta_eval_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluator_name: Mapped[str] = mapped_column(String, index=True, nullable=False)
    human_score: Mapped[float] = mapped_column(Float, nullable=False)
    auto_score: Mapped[float] = mapped_column(Float, nullable=False)
    agreement: Mapped[bool] = mapped_column(Boolean, nullable=False)
    conversation_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
