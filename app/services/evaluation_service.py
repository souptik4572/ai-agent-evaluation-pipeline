import logging

from app.evaluators.pipeline import EvaluationPipeline
from app.feedback.aggregator import AnnotationAggregator
from app.feedback.routing import ConfidenceRouter
from app.models.schemas import ConversationCreate, EvaluationResult

logger = logging.getLogger(__name__)

_pipeline = EvaluationPipeline()
_aggregator = AnnotationAggregator()
_router = ConfidenceRouter()


async def run_evaluation(conversation: ConversationCreate) -> EvaluationResult:
    """Orchestrate evaluation + annotation aggregation + routing."""
    result = await _pipeline.run(conversation)

    annotations = (
        conversation.feedback.annotations
        if conversation.feedback and conversation.feedback.annotations
        else None
    )

    if annotations:
        agreement = _aggregator.compute_agreement(annotations)
        result.annotation_agreement = agreement
        logger.info(
            "Annotation agreement for %s: overall=%.2f kappa=%.2f",
            conversation.conversation_id,
            agreement["overall_agreement"],
            agreement["overall_kappa"],
        )
    else:
        result.annotation_agreement = None

    routing = _router.route(result, annotations)
    result.routing_decision = routing
    logger.info(
        "Routing decision for %s: %s",
        conversation.conversation_id,
        routing["routing_decision"],
    )

    return result
