"""Integration tests: the Claude adapter, through the real Anthropic SDK, against
a simulated Anthropic API."""

import logging

import pytest

from app.domain.answers import CitedSpan, EvidencePassage
from app.domain.errors import AnswerRefused, LLMRateLimited, LLMUnavailable
from app.infrastructure.llm.claude_llm_provider import (
    MISCONFIGURED,
    REJECTED,
    SYSTEM_PROMPT,
)
from tests.fake_anthropic import API_KEY, FakeAnthropic, cited_text, error, message

EVIDENCE = [
    EvidencePassage("doc1:0", "finance.txt · page 1", "Revenue grew 12% in Q3."),
    EvidencePassage("doc2:3", "office.txt", "The office cat is named Biscuit."),
]

# --- The request ---


def test_sends_passages_as_search_results_then_the_question() -> None:
    api = FakeAnthropic()

    api.provider(effort="low").answer("How much did revenue grow?", EVIDENCE)

    [request] = api.requests
    assert request["model"] == "claude-haiku-5-5"
    assert request["system"] == SYSTEM_PROMPT
    assert request["output_config"] == {"effort": "low"}
    assert request["max_tokens"] == 16000
    content = request["messages"][0]["content"]
    assert content[0] == {
        "type": "search_result",
        "source": "doc1:0",
        "title": "finance.txt · page 1",
        "content": [{"type": "text", "text": "Revenue grew 12% in Q3."}],
        "citations": {"enabled": True},
    }
    assert content[1]["source"] == "doc2:3"
    assert content[-1] == {"type": "text", "text": "How much did revenue grow?"}
    assert api.headers[0]["x-api-key"] == API_KEY


def test_system_prompt_guards_against_instructions_inside_documents() -> None:
    assert "not instructions" in SYSTEM_PROMPT
    assert "don't contain enough information" in SYSTEM_PROMPT


def test_sends_no_sampling_parameters_or_prefill() -> None:
    api = FakeAnthropic()

    api.provider().answer("q", EVIDENCE)

    request = api.requests[0]
    assert not {"temperature", "top_p", "top_k"} & set(request)
    assert [m["role"] for m in request["messages"]] == ["user"]


# --- The response ---


def test_reads_text_and_citations_and_skips_thinking() -> None:
    api = FakeAnthropic(
        [
            message(
                [
                    {"type": "thinking", "thinking": "", "signature": "s"},
                    cited_text("Revenue grew 12%.", 0, "Revenue grew 12% in Q3."),
                    {
                        "type": "text",
                        "text": " Nothing else is known.",
                        "citations": None,
                    },
                ]
            )
        ]
    )

    answer = api.provider().answer("q", EVIDENCE)

    assert answer.model == "claude-haiku-5-5"
    assert [s.text for s in answer.segments] == [
        "Revenue grew 12%.",
        " Nothing else is known.",
    ]
    assert answer.segments[0].cited == (CitedSpan(0, "Revenue grew 12% in Q3."),)
    assert answer.segments[1].cited == ()


def test_refusal_becomes_answer_refused() -> None:
    api = FakeAnthropic(
        [
            message(
                [],
                stop_reason="refusal",
                stop_details={
                    "type": "refusal",
                    "category": "cyber",
                    "explanation": "x",
                },
            )
        ]
    )

    with pytest.raises(AnswerRefused):
        api.provider().answer("q", EVIDENCE)


def test_truncated_answer_is_unavailable() -> None:
    api = FakeAnthropic(
        [message([{"type": "text", "text": "Rev", "citations": None}], "max_tokens")]
    )

    with pytest.raises(LLMUnavailable, match="too long"):
        api.provider().answer("q", EVIDENCE)


# --- Errors ---


def test_rate_limit_keeps_retry_after() -> None:
    api = FakeAnthropic([error(429, "rate_limit_error", {"retry-after": "12"})])

    with pytest.raises(LLMRateLimited) as raised:
        api.provider().answer("q", EVIDENCE)

    assert raised.value.retry_after_seconds == 12.0


@pytest.mark.parametrize(
    ("status", "kind", "message_text"),
    [
        (401, "authentication_error", MISCONFIGURED),
        (403, "permission_error", MISCONFIGURED),
        (400, "invalid_request_error", REJECTED),
        (404, "not_found_error", REJECTED),
    ],
)
def test_client_errors(status: int, kind: str, message_text: str) -> None:
    api = FakeAnthropic([error(status, kind)])

    with pytest.raises(LLMUnavailable) as raised:
        api.provider().answer("q", EVIDENCE)

    assert raised.value.message == message_text
    assert not isinstance(raised.value, LLMRateLimited)


@pytest.mark.parametrize(
    ("status", "kind"), [(500, "api_error"), (529, "overloaded_error")]
)
def test_server_errors_are_unavailable(status: int, kind: str) -> None:
    with pytest.raises(LLMUnavailable):
        FakeAnthropic([error(status, kind)]).provider().answer("q", EVIDENCE)


def test_sdk_retries_transient_errors_before_giving_up() -> None:
    api = FakeAnthropic([error(529, "overloaded_error", {"retry-after": "0"})])

    answer = api.provider(max_retries=1).answer("q", EVIDENCE)

    assert len(api.requests) == 2
    assert answer.segments


def test_the_key_and_the_question_are_never_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    api = FakeAnthropic([error(401, "authentication_error")])

    with caplog.at_level(logging.INFO), pytest.raises(LLMUnavailable):
        api.provider().answer("secret merger plans", EVIDENCE)
    FakeAnthropic().provider().answer("secret merger plans", EVIDENCE)

    logged = " ".join(str(vars(r)) for r in caplog.records)
    assert API_KEY not in logged
    assert "secret merger plans" not in logged


def test_requires_an_api_key() -> None:
    from app.infrastructure.llm.claude_llm_provider import ClaudeLLMProvider

    with pytest.raises(ValueError):
        ClaudeLLMProvider(api_key="", model_name="claude-haiku-5-5")
