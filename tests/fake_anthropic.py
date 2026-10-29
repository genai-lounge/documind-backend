"""A simulated Anthropic Messages API, served in-process via httpx2.MockTransport.

The real Anthropic SDK builds the request and parses the streamed response,
so these tests check our request shape and our handling of real-shaped
server-sent events without any network, key, or cost.
"""

import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class Interrupted:
    """Streams the start of `message`, then an SSE `error` event (status 200 already sent)."""

    message: dict[str, Any]
    kind: str = "overloaded_error"


def interrupted(message: dict[str, Any], kind: str = "overloaded_error") -> Interrupted:
    return Interrupted(message, kind)


def stream_events(message: dict[str, Any]) -> list[dict[str, Any]]:
    """The events Anthropic streams for `message`: citations before text, text in two parts."""
    start = {
        **message,
        "content": [],
        "stop_reason": None,
        "usage": {**message["usage"], "output_tokens": 0},
    }
    events: list[dict[str, Any]] = [{"type": "message_start", "message": start}]
    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            events.append(
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "text", "text": "", "citations": None},
                }
            )
            for citation in block.get("citations") or []:
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "citations_delta", "citation": citation},
                    }
                )
            half = len(block["text"]) // 2
            for part in (block["text"][:half], block["text"][half:]):
                if part:
                    events.append(
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "text_delta", "text": part},
                        }
                    )
        else:
            events.append(
                {"type": "content_block_start", "index": index, "content_block": block}
            )
        events.append({"type": "content_block_stop", "index": index})
    events.append(
        {
            "type": "message_delta",
            "delta": {
                "stop_reason": message["stop_reason"],
                "stop_sequence": None,
                "stop_details": message.get("stop_details"),
            },
            "usage": {"output_tokens": message["usage"]["output_tokens"]},
        }
    )
    events.append({"type": "message_stop"})
    return events


def _sse(events: list[dict[str, Any]]) -> httpx2.Response:
    body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return httpx2.Response(
        200, content=body.encode(), headers={"content-type": "text/event-stream"}
    )


class FakeAnthropic:
    """Replies with `responses` in order: a message dict (streamed), an
    `Interrupted` stream, or an HTTP error (status, body, headers).

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
            if isinstance(reply, Interrupted):
                failure = error(529, reply.kind)[1]
                return _sse([*stream_events(reply.message)[:3], {**failure}])
            return _sse(stream_events(reply))
        first = next(
            block
            for block in body["messages"][-1]["content"]
            if block["type"] == "search_result"
        )
        quote = first["content"][0]["text"]
        return _sse(
            stream_events(
                message(
                    [
                        {"type": "thinking", "thinking": "", "signature": "sig"},
                        cited_text("The documents say so.", 0, quote),
                    ]
                )
            )
        )


def error(
    status: int, kind: str, headers: dict[str, str] | None = None
) -> tuple[int, dict[str, Any], dict[str, str]]:
    return (
        status,
        {"type": "error", "error": {"type": kind, "message": f"simulated {kind}"}},
        headers or {},
    )
