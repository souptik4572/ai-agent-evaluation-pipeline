from abc import ABC, abstractmethod

from app.models.schemas import ConversationCreate


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique evaluator name."""
        ...

    @abstractmethod
    async def evaluate(self, conversation: ConversationCreate) -> dict:
        """
        Run evaluation on a conversation.
        Returns a dict with:
          - score: float (0.0 to 1.0)
          - details: dict (evaluator-specific details)
          - issues: list[dict] (any issues found)
        """
        ...
