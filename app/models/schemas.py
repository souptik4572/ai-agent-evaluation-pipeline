from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    tool_name: str
    parameters: dict[str, Any]
    result: dict[str, Any] | None = None
    latency_ms: int | None = None


class Turn(BaseModel):
    turn_id: int
    role: Literal["user", "assistant", "system"]
    content: str
    tool_calls: list[ToolCall] | None = None
    timestamp: datetime


class Annotation(BaseModel):
    type: str
    label: str
    annotator_id: str
    confidence: float | None = None


class OpsReview(BaseModel):
    quality: str
    notes: str | None = None


class Feedback(BaseModel):
    user_rating: int | None = Field(None, ge=1, le=5)
    ops_review: OpsReview | None = None
    annotations: list[Annotation] | None = None


class ConversationMetadata(BaseModel):
    total_latency_ms: int | None = None
    mission_completed: bool | None = None


class ConversationCreate(BaseModel):
    conversation_id: str
    agent_version: str
    turns: list[Turn]
    feedback: Feedback | None = None
    metadata: ConversationMetadata | None = None


class ConversationResponse(ConversationCreate):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class ToolEvaluation(BaseModel):
    selection_accuracy: float
    parameter_accuracy: float
    hallucinated_params: list[str]
    execution_success: bool


class IssueDetected(BaseModel):
    type: str
    severity: Literal["critical", "warning", "info"]
    description: str


class ImprovementSuggestion(BaseModel):
    type: Literal["prompt", "tool"]
    target: str | None = None
    suggestion: str
    rationale: str
    confidence: float
    expected_impact: str | None = None


class EvaluationScores(BaseModel):
    overall: float
    response_quality: float
    tool_accuracy: float
    coherence: float


class EvaluationResult(BaseModel):
    evaluation_id: str
    conversation_id: str
    agent_version: str
    scores: EvaluationScores
    tool_evaluation: ToolEvaluation | None = None
    issues_detected: list[IssueDetected]
    improvement_suggestions: list[ImprovementSuggestion]
    evaluator_details: dict[str, Any]
    annotation_agreement: dict | None = None
    routing_decision: dict | None = None
    created_at: datetime


class SuggestionStatusUpdate(BaseModel):
    status: Literal["pending", "accepted", "rejected", "implemented"]


class CalibrationRequest(BaseModel):
    human_scores: dict[str, float]


class DimensionComparison(BaseModel):
    baseline_mean: float
    target_mean: float
    delta: float
    delta_pct: float
    is_regression: bool
    significance: Literal["significant", "marginal", "not_significant"]


class IssueRateChange(BaseModel):
    baseline_rate: float
    target_rate: float
    change_pct: float
    is_elevated: bool


class RegressionReport(BaseModel):
    baseline_version: str
    target_version: str
    baseline_sample_size: int
    target_sample_size: int
    dimensions: dict[str, DimensionComparison]
    issue_rate_changes: dict[str, IssueRateChange]
    regressions_detected: list[str]
    is_regression: bool
    severity: Literal["none", "minor", "major", "critical"]
    summary: str


class VersionSummary(BaseModel):
    version: str
    conversation_count: int
    eval_count: int
    mean_overall_score: float
    date_range: dict[str, str | None]


class RegressionCompareRequest(BaseModel):
    baseline_version: str
    target_version: str


class Alert(BaseModel):
    alert_id: str
    type: Literal["regression", "quality_drop", "high_failure_rate", "annotator_conflict"]
    severity: Literal["info", "warning", "critical"]
    title: str
    description: str
    related_entity_id: str | None = None
    status: Literal["open", "acknowledged", "resolved"] = "open"
    created_at: datetime
    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None


class AlertStatusUpdate(BaseModel):
    status: Literal["acknowledged", "resolved"]


class WebhookConfig(BaseModel):
    url: str


class FeedbackCorrelation(BaseModel):
    dimension: str
    pearson_r: float
    sample_size: int
    interpretation: str


class CorrelationResponse(BaseModel):
    correlations: list[FeedbackCorrelation]
    best_dimension: str | None
    scatter_data: list[dict[str, Any]]


class PipelineEvent(BaseModel):
    event_type: Literal[
        "conversation_ingested",
        "evaluation_completed",
        "regression_detected",
        "alert_fired",
        "suggestion_generated",
    ]
    timestamp: datetime
    data: dict[str, Any]
    conversation_id: str | None = None
    agent_version: str | None = None
