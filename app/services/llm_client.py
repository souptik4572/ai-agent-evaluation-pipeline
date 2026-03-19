import json
import logging
import random

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_MOCK_JUDGE_RESPONSE = {
    "response_quality": 0.72,
    "helpfulness": 0.68,
    "factuality": 0.80,
    "tone": 0.75,
    "issues": [],
    "reasoning": "Mock evaluation — no API key configured.",
    "mock": True,
}

_MOCK_COHERENCE_RESPONSE = {
    "contradictions": [],
    "coherence_score": 0.80,
    "reasoning": "Mock coherence check — no API key configured.",
    "mock": True,
}


class LLMClient:
    def __init__(self):
        self._has_key = bool(settings.openai_api_key)
        if self._has_key:
            self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_model

    async def evaluate_with_rubric(self, conversation_text: str, rubric: str) -> dict:
        """Send conversation + rubric to LLM, get structured JSON scores back."""
        if not self._has_key:
            logger.debug("No API key — returning mock judge scores.")
            result = _MOCK_JUDGE_RESPONSE.copy()
            result["response_quality"] = round(random.uniform(0.55, 0.95), 2)
            result["helpfulness"] = round(random.uniform(0.50, 0.95), 2)
            result["factuality"] = round(random.uniform(0.60, 0.98), 2)
            result["tone"] = round(random.uniform(0.65, 1.0), 2)
            return result

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert AI quality evaluator. "
                    "You MUST respond with ONLY valid JSON — no markdown, no preamble, no explanation. "
                    "Follow the rubric exactly."
                ),
            },
            {"role": "user", "content": f"{rubric}\n\n---CONVERSATION---\n{conversation_text}"},
        ]

        async def _call() -> str:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()

        try:
            return json.loads(await _call())
        except Exception as e:
            logger.warning("LLM judge parse error (%s) — retrying.", e)
            try:
                raw = await _call()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                return json.loads(raw.strip())
            except Exception as e2:
                logger.error("LLM judge failed twice: %s", e2)
                return {**_MOCK_JUDGE_RESPONSE, "error": str(e2)}

    async def analyze_failures(self, failure_data: list[dict]) -> dict:
        """Send aggregated failure data to LLM, get improvement suggestions."""
        if not self._has_key:
            return {
                "suggestions": [
                    {
                        "type": "prompt",
                        "target": "system_prompt",
                        "suggestion": (
                            "Add explicit date formatting instructions to the system prompt: "
                            "'ALWAYS format dates as YYYY-MM-DD when calling tools. "
                            "Never use relative date expressions such as \"next week\" or \"January\".' "
                            "Place this immediately after the tool usage guidelines section."
                        ),
                        "rationale": (
                            "Analysis of failure data shows date format errors are the most common issue, "
                            "occurring in approximately 15% of tool calls."
                        ),
                        "confidence": 0.82,
                        "expected_impact": "Reduce date format errors from ~15% to <3%",
                    }
                ],
                "mock": True,
            }

        failure_json = json.dumps(failure_data, indent=2)
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert prompt engineer. Given failure patterns in an AI agent, "
                            "suggest SPECIFIC prompt modifications to fix them. "
                            "Respond with ONLY valid JSON: "
                            "{\"suggestions\": [{\"type\": \"prompt\"|\"tool\", \"target\": \"...\", "
                            "\"suggestion\": \"...\", \"rationale\": \"...\", \"confidence\": 0.0-1.0, "
                            "\"expected_impact\": \"...\"}]}"
                        ),
                    },
                    {"role": "user", "content": f"Failure patterns:\n{failure_json}"},
                ],
                response_format={"type": "json_object"},
                max_tokens=1024,
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            logger.error("analyze_failures LLM call failed: %s", e)
            return {"suggestions": [], "error": str(e)}

    async def check_coherence(self, turns: list[dict]) -> dict:
        """Check multi-turn coherence by asking LLM to find contradictions."""
        if not self._has_key:
            result = _MOCK_COHERENCE_RESPONSE.copy()
            result["coherence_score"] = round(random.uniform(0.65, 0.95), 2)
            return result

        turns_text = "\n".join(
            f"[Turn {t.get('turn_id', i+1)} - {t.get('role','?').upper()}]: {t.get('content','')}"
            for i, t in enumerate(turns)
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert conversation analyst. "
                            "Analyze the following AI assistant conversation turns for contradictions or context loss. "
                            "Respond with ONLY valid JSON: "
                            "{\"contradictions\": [{\"turn_ids\": [1,3], \"description\": \"...\"}], "
                            "\"coherence_score\": 0.0-1.0, \"reasoning\": \"...\"}"
                        ),
                    },
                    {"role": "user", "content": turns_text},
                ],
                response_format={"type": "json_object"},
                max_tokens=512,
            )
            return json.loads(response.choices[0].message.content.strip())
        except Exception as e:
            logger.error("check_coherence LLM call failed: %s", e)
            return {**_MOCK_COHERENCE_RESPONSE, "error": str(e)}
