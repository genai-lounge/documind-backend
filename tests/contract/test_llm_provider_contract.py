"""Contract tests: rules every LLMProvider must follow.

Claude runs against the simulated Anthropic API; the real API is exercised
only by tests/live/ (opt-in).
"""

from collections.abc import Callable

import pytest

from app.application.ports import LLMProvider
from app.domain.answers import EvidencePassage, GeneratedAnswer
from app.domain.errors import LLMUnavailable
from tests.fake_anthropic import FakeAnthropic, error
from tests.fakes import FakeLLMProvider

PROVIDERS: dict[str, Callable[[], LLMProvider]] = {
    "FakeLLMProvider": FakeLLMProvider,
    "ClaudeLLMProvider": lambda: FakeAnthropic().provider(),
}
FAILING: dict[str, Callable[[], LLMProvider]] = {
    "FakeLLMProvider": lambda: FakeLLMProvider(error=LLMUnavailable("down")),
    "ClaudeLLMProvider": lambda: FakeAnthropic([error(500, "api_error")]).provider(),
}
EVIDENCE = [
    EvidencePassage("a:0", "a.txt", "Alpha passage."),
    EvidencePassage("b:0", "b.txt", "Beta passage."),
]


@pytest.mark.parametrize("name", list(PROVIDERS))
def test_has_a_model_name(name: str) -> None:
    assert PROVIDERS[name]().model_name


@pytest.mark.parametrize("name", list(PROVIDERS))
def test_returns_segments_citing_only_passages_that_were_sent(name: str) -> None:
    answer = PROVIDERS[name]().answer("question?", EVIDENCE)

    assert isinstance(answer, GeneratedAnswer)
    assert answer.segments
    for segment in answer.segments:
        for span in segment.cited:
            assert 0 <= span.passage_index < len(EVIDENCE)


@pytest.mark.parametrize("name", list(FAILING))
def test_failures_raise_llm_unavailable(name: str) -> None:
    with pytest.raises(LLMUnavailable):
        FAILING[name]().answer("question?", EVIDENCE)
