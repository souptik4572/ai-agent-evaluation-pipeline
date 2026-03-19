import logging
import re

from app.evaluators.base import BaseEvaluator
from app.middleware.logging import timed_evaluator
from app.models.schemas import ConversationCreate
from app.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

# Pronouns and demonstratives that signal reference resolution
_REFERENCE_WORDS = re.compile(
    r"\b(this|that|these|those|it|they|them|the flight|the hotel|the car|the booking)\b",
    re.IGNORECASE,
)

# Preference markers
_PREFERENCE_MARKERS = [
    "prefer", "want", "need", "must", "always", "never", "allergic",
    "budget", "window seat", "aisle seat", "vegetarian", "vegan",
]


class CoherenceEvaluator(BaseEvaluator):
    def __init__(self):
        self.llm_client = LLMClient()

    @property
    def name(self) -> str:
        return "coherence"

    @timed_evaluator
    async def evaluate(self, conversation: ConversationCreate) -> dict:
        turns = conversation.turns
        assistant_turns = [t for t in turns if t.role == "assistant"]
        user_turns = [t for t in turns if t.role == "user"]

        issues: list[dict] = []
        sub_scores: dict[str, float] = {}

        if len(turns) > 2:
            context_refs = 0
            for i, turn in enumerate(assistant_turns[1:], 1):
                prior_text = " ".join(t.content for t in turns[:i * 2] if t.role == "user")
                if any(marker in turn.content.lower() for marker in _PREFERENCE_MARKERS):
                    context_refs += 1
                elif _REFERENCE_WORDS.search(turn.content):
                    context_refs += 1
            sub_scores["context_maintenance"] = min(
                1.0, context_refs / max(len(assistant_turns) - 1, 1)
            )
        else:
            sub_scores["context_maintenance"] = 1.0

        turn_dicts = [t.model_dump(mode="json") for t in turns if t.role == "assistant"]
        if len(turn_dicts) >= 2:
            coherence_result = await self.llm_client.check_coherence(turn_dicts)
            contradictions = coherence_result.get("contradictions", [])
            llm_coherence = coherence_result.get("coherence_score", 0.85)
            sub_scores["llm_contradiction_check"] = llm_coherence
            for c in contradictions:
                issues.append({
                    "type": "contradiction",
                    "severity": "critical",
                    "description": c.get("description", "Contradiction detected between assistant turns."),
                })
        else:
            sub_scores["llm_contradiction_check"] = 1.0

        ref_ok = True
        for turn in assistant_turns:
            if _REFERENCE_WORDS.search(turn.content):
                turn_idx = turns.index(turn)
                prior_content = " ".join(t.content for t in turns[:turn_idx])
                if len(prior_content.strip()) < 20:
                    ref_ok = False
                    issues.append({
                        "type": "dangling_reference",
                        "severity": "info",
                        "description": (
                            f"Turn {turn.turn_id}: assistant uses reference pronouns but prior context is thin."
                        ),
                    })
        sub_scores["reference_resolution"] = 1.0 if ref_ok else 0.6

        if len(turns) > 5:
            early_prefs: list[str] = []
            for turn in turns[:2]:
                for marker in _PREFERENCE_MARKERS:
                    if marker in turn.content.lower():
                        early_prefs.append(marker)

            if early_prefs:
                last_assistant = next(
                    (t for t in reversed(turns) if t.role == "assistant"), None
                )
                if last_assistant:
                    preserved = sum(
                        1 for p in early_prefs if p in last_assistant.content.lower()
                    )
                    pref_score = preserved / len(early_prefs) if early_prefs else 1.0
                    sub_scores["context_window_retention"] = pref_score
                    if pref_score < 0.5:
                        issues.append({
                            "type": "context_loss",
                            "severity": "critical",
                            "description": (
                                f"In a {len(turns)}-turn conversation, the agent appears to have forgotten "
                                f"early user preferences: {early_prefs}. "
                                f"Only {preserved}/{len(early_prefs)} preferences reflected in final response."
                            ),
                        })
                else:
                    sub_scores["context_window_retention"] = 1.0
            else:
                sub_scores["context_window_retention"] = 1.0
        else:
            sub_scores["context_window_retention"] = 1.0

        score = sum(sub_scores.values()) / len(sub_scores)
        return {
            "score": round(score, 4),
            "details": {"sub_scores": sub_scores},
            "issues": issues,
        }
