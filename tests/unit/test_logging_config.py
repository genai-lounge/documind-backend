import json
import logging
import sys

from app.core.logging_config import (
    JsonFormatter,
    TextFormatter,
    configure_logging,
    request_id_var,
)


def make_record(**extra: object) -> logging.LogRecord:
    configure_logging("INFO", "text")
    token = request_id_var.set("req-123")
    try:
        record = logging.getLogger("test").makeRecord(
            "test",
            logging.INFO,
            __file__,
            1,
            "something.happened",
            (),
            None,
            extra=extra,
        )
    finally:
        request_id_var.reset(token)
    return record


def test_records_carry_the_current_request_id() -> None:
    assert make_record().__dict__["request_id"] == "req-123"


def test_text_format_includes_request_id_and_extras() -> None:
    line = TextFormatter().format(make_record(size_bytes=42))

    assert "[req-123]" in line
    assert "something.happened size_bytes=42" in line


def test_json_format_is_one_parseable_object() -> None:
    payload = json.loads(JsonFormatter().format(make_record(size_bytes=42)))

    assert payload["message"] == "something.happened"
    assert payload["request_id"] == "req-123"
    assert payload["size_bytes"] == 42
    assert payload["level"] == "INFO"


def test_exceptions_are_included() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = make_record()
        record.exc_info = sys.exc_info()

    assert "ValueError: boom" in JsonFormatter().format(record)


def test_configuring_twice_keeps_a_single_handler() -> None:
    configure_logging("INFO", "text")
    configure_logging("INFO", "json")

    ours = [h for h in logging.getLogger().handlers if h.get_name() == "documind"]
    assert len(ours) == 1
    assert isinstance(ours[0].formatter, JsonFormatter)
