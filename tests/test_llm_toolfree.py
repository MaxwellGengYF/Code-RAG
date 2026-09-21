"""Tool-free wire-format tests: an SDK-matched MockTransport captures the request.

Asserts every provider sends exactly system+user roles and NO ``tools`` field.

The transport module is DISCOVERED from the installed SDK (see
:func:`sdk_http`): the openai/anthropic clients type-check the ``http_client``
they are given, so a MockTransport must be built from the very module the SDK
passes to its own AsyncClient — ``httpx2`` for the repo's pinned SDKs, ``httpx``
for the older ones. Hardcoding either one makes this file fail on whichever
interpreter carries the other SDK (a stray ``uv run pytest`` that resolves the
system pytest is the observed case).
"""
from __future__ import annotations

import json

import httpx
import pytest

from rag.llm import create_llm
from rag.llm.config import ProviderConfig


def sdk_http():
    """The httpx-compatible module the INSTALLED openai/anthropic SDK is built on.

    anthropic >= 1.0 and openai >= 3.0 are built on ``httpx2`` (a fork with its
    own ``Response`` / ``MockTransport`` types); earlier releases use ``httpx``.
    Both vendored clients re-export the module they imported, so read it off the
    SDK instead of guessing: a transport from the wrong module is not merely
    ignored, the SDK raises on the client type.
    """
    for sdk in ("anthropic", "openai"):
        base_client = __import__(f"{sdk}._base_client", fromlist=["_base_client"])
        for name in ("httpx2", "httpx"):
            mod = getattr(base_client, name, None)
            if mod is not None and hasattr(mod, "MockTransport"):
                return mod
    return httpx  # pragma: no cover - both SDKs always ship their http module


#: the module both SDK-backed transports below must come from
HTTP = sdk_http()


def test_sdk_http_is_the_module_the_sdk_imported():
    """Pin the discovery: the MockTransport module must be the SDK's own.

    Guards against someone hardcoding `httpx` (or `httpx2`) again — the failure
    mode is a `No module named 'httpx2'` / client-type error on whichever
    interpreter carries the other generation of the SDKs.
    """
    import anthropic._base_client as base_client
    candidates = {getattr(base_client, n) for n in ("httpx", "httpx2")
                  if hasattr(base_client, n)}
    assert candidates, "the anthropic SDK must expose the http module it imported"
    assert HTTP in candidates, f"{HTTP.__name__} is not the module the SDK uses"
    assert hasattr(HTTP, "MockTransport") and hasattr(HTTP, "AsyncClient")


def make_transport(capture: dict) -> "HTTP.MockTransport":
    def handler(request) -> "HTTP.Response":
        capture["body"] = json.loads(request.content)
        capture["url"] = str(request.url)
        capture["headers"] = dict(request.headers)
        # Minimal chat.completions-shaped response; anthropic path overrode via
        # separate transport below.
        return HTTP.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello world response"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                          "total_tokens": 15},
            },
        )

    return HTTP.MockTransport(handler)


def anthropic_handler(capture: dict):
    def handler(request):
        capture["body"] = json.loads(request.content)
        capture["url"] = str(request.url)
        return HTTP.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "test-model",
                "content": [{"type": "text", "text": "hello world response"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    return handler


async def check_chat_payload(capture: dict, cfg_dict: dict):
    capture.clear()
    cfg = ProviderConfig.from_dict(cfg_dict)
    client = create_llm(cfg, http_client=HTTP.AsyncClient(
        transport=make_transport(capture)))
    result = await client.generate("You are a corpus builder.", "PAGE MARKDOWN HERE")
    assert result.text == "hello world response"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    body = capture["body"]
    assert "tools" not in body, f"tools field leaked: {body.keys()}"
    assert "tool_choice" not in body
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user"], roles
    assert body["messages"][0]["content"] == "You are a corpus builder."
    assert body["messages"][1]["content"] == "PAGE MARKDOWN HERE"
    return body


@pytest.mark.asyncio
async def test_openai_legacy_toolfree():
    capture: dict = {}
    body = await check_chat_payload(capture, {
        "model": "qwen3.8-flash", "type": "openai_legacy",
        "api_key": "sk-test", "thinking_effort": "low",
        "capabilities": ["thinking"],
    })
    # thinking effort passes through the SDK reasoning_effort field
    assert body.get("reasoning_effort") == "low"


@pytest.mark.asyncio
async def test_kimi_toolfree():
    capture: dict = {}
    body = await check_chat_payload(capture, {
        "model": "kimi-for-coding", "type": "kimi", "api_key": "sk-test",
        "thinking_effort": "max", "capabilities": ["thinking"], "max_tokens": 384000,
    })
    # the openai SDK merges extra_body into the request body top-level
    assert body.get("thinking", {}).get("type") == "enabled"
    assert body.get("thinking", {}).get("effort") == "max"
    # max_tokens normalized to max_completion_tokens and clamped to the safe bound
    assert body.get("max_completion_tokens") == 384000
    assert "max_tokens" not in body
    assert body.get("temperature") == 1.0
    assert body.get("prompt_cache_key")


@pytest.mark.asyncio
async def test_kimi_no_thinking_disables_extra_body():
    capture: dict = {}
    body = await check_chat_payload(capture, {
        "model": "kimi-k2", "type": "kimi", "api_key": "sk-test",
    })
    assert body.get("thinking", {}).get("type") == "disabled"
    assert body.get("temperature") == 0.6


@pytest.mark.asyncio
async def test_anthropic_toolfree():
    capture: dict = {}
    cfg = ProviderConfig.from_dict({
        "model": "qwen3.8-flash", "type": "anthropic", "api_key": "sk-test",
        "thinking_effort": "low", "capabilities": ["thinking"], "max_tokens": 131072,
    })
    # the SDK type-checks the http_client: it must come from the module the
    # installed anthropic release itself imported (httpx2 for 1.x, httpx for 0.x)
    client = create_llm(cfg, http_client=HTTP.AsyncClient(
        transport=HTTP.MockTransport(anthropic_handler(capture))))

    result = await client.generate("You are a corpus builder.", "PAGE MARKDOWN HERE")
    assert result.text == "hello world response"
    assert result.input_tokens == 10 and result.output_tokens == 5
    body = capture["body"]
    assert "tools" not in body
    assert "system" in body  # system is a top-level param, not a message
    assert body["system"] == "You are a corpus builder."
    assert body["messages"] == [{"role": "user", "content": "PAGE MARKDOWN HERE"}]
    # non-streaming max_tokens clamped to the SDK's 10-minute bound
    assert body["max_tokens"] == 21333
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
