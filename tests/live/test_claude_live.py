"""Live tests against the real Claude API. Run with `make test-live`.

Skipped unless ANTHROPIC_API_KEY is set. Two small requests with made-up
passages on the configured model (Claude Haiku 5.5 by default: a fraction of a cent).
"""

import pytest

from app.application.ports import LLMProvider
from app.core.config import ConfigurationError, Settings
from app.domain.answers import EvidencePassage
from app.llm_providers import build_llm_provider

pytestmark = pytest.mark.live

EVIDENCE = [
    EvidencePassage(
        "finance:0",
        "finance.txt · page 1",
        "In the third quarter, revenue grew by 12% to $4M.",
    ),
    EvidencePassage("office:0", "office.txt", "The office cat is named Biscuit."),
]


def provider() -> LLMProvider:
    try:
        return build_llm_provider(Settings())
    except ConfigurationError as exc:
        pytest.skip(str(exc))


def test_answers_with_a_citation_to_the_right_passage() -> None:
    answer = provider().answer("How much did revenue grow in Q3?", EVIDENCE)

    cited = {span.passage_index for s in answer.segments for span in s.cited}
    assert 0 in cited
    assert 1 not in cited


def test_unanswerable_question_cites_nothing() -> None:
    answer = provider().answer("What is the CEO's favourite colour?", EVIDENCE)

    assert not [span for s in answer.segments for span in s.cited]
