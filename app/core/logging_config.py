"""Logging setup, built on Python's standard `logging` module.

Code anywhere in the app logs the usual way:

    logger = logging.getLogger(__name__)
    logger.info("document.ingested", extra={"document_id": "...", "size_bytes": 123})

Where the output goes is decided here, by the handler and formatter. To send
logs to a file or a cloud service later, add or swap a handler; no calling
code changes. (See docs/adr/0002-standard-logging.md for why we don't wrap
logging in our own interface.)

Every line automatically includes the current request ID, so all the lines
from one request can be found together.
"""

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, Literal

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_STANDARD_ATTRIBUTES = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {
    "message",
    "asctime",
    "request_id",
}


_default_record_factory = logging.getLogRecordFactory()


def _record_with_request_id(*args: Any, **kwargs: Any) -> logging.LogRecord:
    """Create log records with the current request ID attached.

    Attaching it at creation (rather than in one handler) means every handler
    sees it: our console handler, pytest's log capture, a future cloud handler.
    """
    record = _default_record_factory(*args, **kwargs)
    record.request_id = request_id_var.get()
    return record


def _extra_fields(record: logging.LogRecord) -> dict[str, object]:
    return {k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRIBUTES}


class TextFormatter(logging.Formatter):
    """`2026-10-21 10:15:02 INFO    [ab12…] app.x: event key=value key=value`"""

    def format(self, record: logging.LogRecord) -> str:
        extras = "".join(f" {k}={v}" for k, v in _extra_fields(record).items())
        line = (
            f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')} {record.levelname:<7} "
            f"[{getattr(record, 'request_id', '-')}] {record.name}: "
            f"{record.getMessage()}{extras}"
        )
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class JsonFormatter(logging.Formatter):
    """One JSON object per line, ready for log collectors."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            **_extra_fields(record),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str, log_format: Literal["text", "json"]) -> None:
    """Install our handler on the root logger. Safe to call more than once."""
    logging.setLogRecordFactory(_record_with_request_id)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter() if log_format == "json" else TextFormatter())
    handler.set_name("documind")

    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if h.get_name() != "documind"]
    root.addHandler(handler)
    root.setLevel(level)

    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True

    logging.getLogger("uvicorn.access").disabled = True
