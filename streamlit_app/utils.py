import os

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1")
TIMEOUT = 15

# Root URL of the backend (strips /api/v1 suffix if present)
_API_ROOT = API_BASE.rstrip("/")
if _API_ROOT.endswith("/api/v1"):
    _API_ROOT = _API_ROOT[: -len("/api/v1")]
DOCS_URL = f"{_API_ROOT}/docs"


def render_api_docs_button() -> None:
    """Render a sidebar button that opens the Swagger API docs in a new tab."""
    st.sidebar.markdown("---")
    st.sidebar.link_button("📄 API Docs (Swagger)", DOCS_URL, use_container_width=True)


def _get(path: str, params: dict | None = None) -> dict | list:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _post(path: str, data: dict | None = None, params: dict | None = None) -> dict:
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _patch(path: str, data: dict) -> dict:
    try:
        r = requests.patch(f"{API_BASE}{path}", json=data, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def get_conversations(limit: int = 50, offset: int = 0, agent_version: str | None = None) -> dict:
    params = {"limit": limit, "offset": offset}
    if agent_version:
        params["agent_version"] = agent_version
    return _get("/conversations", params)


def get_conversation(conversation_id: str) -> dict:
    return _get(f"/conversations/{conversation_id}")


def get_evaluations(limit: int = 50, offset: int = 0, min_score: float | None = None) -> dict:
    params: dict = {"limit": limit, "offset": offset}
    if min_score is not None:
        params["min_score"] = min_score
    return _get("/evaluations", params)


def get_evaluation(eval_id: str) -> dict:
    return _get(f"/evaluations/{eval_id}")


def get_latest_evaluation_for_conversation(conversation_id: str) -> dict | None:
    result = _get("/evaluations", params={"conversation_id": conversation_id, "limit": 1})
    if isinstance(result, dict) and result.get("data"):
        return result["data"][0]
    return None


def evaluate_conversation(conversation_id: str) -> dict:
    return _post(f"/evaluations/evaluate/{conversation_id}")


def get_suggestions(status: str | None = None, suggestion_type: str | None = None, limit: int = 100) -> dict:
    params: dict = {"limit": limit}
    if status:
        params["status"] = status
    if suggestion_type:
        params["type"] = suggestion_type
    return _get("/suggestions", params)


def get_suggestions_summary() -> dict:
    return _get("/suggestions/summary")


def update_suggestion_status(suggestion_id: str, status: str) -> dict:
    return _patch(f"/suggestions/{suggestion_id}", {"status": status})


def generate_suggestions(last_n: int = 100) -> dict:
    return _post("/suggestions/generate", params={"last_n": last_n})


def get_meta_drift() -> list:
    return _get("/meta/drift")


def get_meta_calibration() -> dict:
    return _get("/meta/calibration")


def get_meta_correlation() -> dict:
    return _get("/meta/correlation")


# ── Alerts ─────────────────────────────────────────────────────────────────────

def get_alerts(status: str | None = None, alert_type: str | None = None,
               severity: str | None = None, limit: int = 100) -> list:
    params: dict = {"limit": limit}
    if status:
        params["status"] = status
    if alert_type:
        params["type"] = alert_type
    if severity:
        params["severity"] = severity
    result = _get("/alerts", params)
    return result if isinstance(result, list) else []


def get_alert_summary() -> dict:
    result = _get("/alerts/summary")
    return result if isinstance(result, dict) else {}


def update_alert_status(alert_id: str, status: str) -> dict:
    return _patch(f"/alerts/{alert_id}", {"status": status})


# ── Regression ─────────────────────────────────────────────────────────────────

def get_regression_versions() -> list:
    result = _get("/regression/versions")
    return result if isinstance(result, list) else []


def compare_versions(baseline: str, target: str) -> dict:
    return _post("/regression/compare", {"baseline_version": baseline, "target_version": target})


def auto_check_version(version: str) -> dict:
    return _post(f"/regression/auto-check/{version}")


def get_regression_reports(limit: int = 20, is_regression: bool | None = None) -> dict:
    params: dict = {"limit": limit}
    if is_regression is not None:
        params["is_regression"] = str(is_regression).lower()
    result = _get("/regression/reports", params)
    return result if isinstance(result, dict) else {}


# ── Metrics ────────────────────────────────────────────────────────────────────

def get_metrics() -> dict:
    result = _get("/metrics")
    return result if isinstance(result, dict) else {}


def health_check() -> dict:
    try:
        r = requests.get(f"{_API_ROOT}/health", timeout=5)
        return r.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}
