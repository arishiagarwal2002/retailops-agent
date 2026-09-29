"""Structured JSON logs: one line per event, trace_id on every line, no secrets or raw PII."""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")

_REDACT_KEYS = {"api_key", "x-api-key", "authorization", "password", "anthropic_api_key", "customer_id"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "trace_id": getattr(record, "trace_id", None) or trace_id_var.get(),
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update({k: ("[redacted]" if k.lower() in _REDACT_KEYS else v) for k, v in fields.items()})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    for noisy in ("httpx", "httpx2", "httpcore", "anthropic", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


_logger = logging.getLogger("retailops")


def log_event(event: str, level: int = logging.INFO, **fields) -> None:
    _logger.log(level, event, extra={"fields": fields})
