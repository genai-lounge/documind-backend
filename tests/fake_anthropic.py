"""A simulated Anthropic Messages API, served in-process via httpx2.MockTransport.

The real Anthropic SDK builds the request and parses the response, so these
tests check our request shape and our handling of real-shaped responses
without any network, key, or cost.
"""

import json
from typing import Any

import anthropic
import httpx2

from app.infrastructure.llm.claude_llm_provider import ClaudeLLMProvider

API_KEY = "sk-ant-test-key"


def message(
    content: list[dict[str, Any]], stop_reason: str = "end_turn", **extra: Any
) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 40},
        **extra,
    }


def cited_text(text: str, index: int, quote: str) -> dict[str, Any]:
    return {
        "type": "text",
        "text": text,
        "citations": [
            {
                "type": "search_result_location",
                "cited_text": quote,
                "source": f"doc:{index}",
                "title": "t",
                "search_result_index": index,
                "start_block_index": 0,
                "end_block_index": 1,
            }
        ],
    }


class FakeAnthropic:
    """Replies with `responses` in order: a message dict, or (status, body, headers).

    When none are left, it answers citing the first search result it was sent.
    """

    def __init__(self, responses: list[Any] | None = None) -> None:
        self.responses = list(responses or [])
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx2.Headers] = []

    def provider(self, **kwargs: Any) -> ClaudeLLMProvider:
        defaults: dict[str, Any] = {"max_retries": 0}
        return ClaudeLLMProvider(
            api_key=API_KEY,
            model_name="claude-haiku-5-5",
            http_client=anthropic.DefaultHttpxClient(
                transport=httpx2.MockTransport(self._handle)
            ),
            **{**defaults, **kwargs},
        )

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        self.headers.append(request.headers)
        if self.responses:
            reply = self.responses.pop(0)
            if isinstance(reply, tuple):
                status, payload, headers = reply
                return httpx2.Response(status, json=payload, headers=headers)
            return httpx2.Response(200, json=reply)
        first = next(
            block
            for block in body["messages"][0]["content"]
            if block["type"] == "search_result"
        )
        quote = first["content"][0]["text"]
        return httpx2.Response(
            200,
            json=message(
                [
                    {"type": "thinking", "thinking": "", "signature": "sig"},
                    cited_text("The documents say so.", 0, quote),
                ]
            ),
        )


def error(
    status: int, kind: str, headers: dict[str, str] | None = None
) -> tuple[int, dict[str, Any], dict[str, str]]:
    return (
        status,
        {"type": "error", "error": {"type": kind, "message": f"simulated {kind}"}},
        headers or {},
    )
