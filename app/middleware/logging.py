from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

_evaluator_durations: dict[str, list[float]] = {}
_app_start_time: float = time.time()


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        req_id = request_id_var.get("")
        if req_id:
            data["request_id"] = req_id
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data)


def configure_json_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        req_id = str(uuid.uuid4())
        token = request_id_var.set(req_id)
        start = time.perf_counter()

        try:
            response = await call_next(request)
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            request_id_var.reset(token)

        response.headers["X-Request-ID"] = req_id
        _log_request(req_id, request.method, str(request.url.path), response.status_code, duration_ms)
        return response


def _log_request(req_id: str, method: str, path: str, status: int, duration_ms: float) -> None:
    logging.getLogger("app.http").info(json.dumps({
        "request_id": req_id,
        "method": method,
        "path": path,
        "status_code": status,
        "duration_ms": duration_ms,
    }))


def timed_evaluator(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> dict:
        start = time.perf_counter()
        result = await func(self, *args, **kwargs)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        evaluator_name = getattr(self, "name", func.__qualname__)
        conv_id = args[0].conversation_id if args else "unknown"
        score = result.get("score", 0.0) if isinstance(result, dict) else 0.0

        _evaluator_durations.setdefault(evaluator_name, []).append(duration_ms)
        if len(_evaluator_durations[evaluator_name]) > 200:
            _evaluator_durations[evaluator_name] = _evaluator_durations[evaluator_name][-200:]

        logging.getLogger("app.evaluator").info(json.dumps({
            "request_id": request_id_var.get(""),
            "evaluator": evaluator_name,
            "conversation_id": conv_id,
            "score": score,
            "duration_ms": duration_ms,
        }))

        if isinstance(result, dict):
            result.setdefault("details", {})["duration_ms"] = duration_ms
        return result

    return wrapper


def get_evaluator_duration_stats() -> dict[str, dict[str, float]]:
    import numpy as np
    stats: dict[str, dict[str, float]] = {}
    for name, durations in _evaluator_durations.items():
        if not durations:
            continue
        arr = np.array(durations)
        stats[name] = {
            "avg_ms": round(float(arr.mean()), 1),
            "p95_ms": round(float(np.percentile(arr, 95)), 1),
        }
    return stats


def get_uptime_seconds() -> float:
    return round(time.time() - _app_start_time, 1)
