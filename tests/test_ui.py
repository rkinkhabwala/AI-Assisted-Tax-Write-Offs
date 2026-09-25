"""Streamlit UI: the API client, the SSE parser, and the page itself (Streamlit AppTest)."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from writeoff.config import get_settings
from writeoff.ui import client as ui_client
from writeoff.ui.client import APIError, ServerEvent, WriteOffClient, parse_sse

APP = str(Path(__file__).resolve().parents[1] / "src" / "writeoff" / "ui" / "app.py")
SID = "11111111-1111-1111-1111-111111111111"
ANSWER = {
    "session_id": SID,
    "request_id": "22222222-2222-2222-2222-222222222222",
    "status": "complete",
    "answer": "**Short answer**: 50% is deductible [IRC § 274(n)(1)].",
    "sources": [
        {
            "citation": "IRC § 274(n)(1)",
            "title": "26 U.S.C. § 274",
            "url": "https://uscode.house.gov/",
            "text": "(1) In general ... 50 percent ...",
        }
    ],
    "verification": {
        "status": "revised",
        "supported": 5,
        "partially_supported": 1,
        "unsupported": 1,
        "unconfirmed": [],
    },
    "usage": {"turns": 3, "tool_calls": 2, "latency_ms": 900, "cost_usd": 0.1},
    "prompt_version": "1.0.0",
}


def sse(*events: tuple[str, dict[str, Any]]) -> str:
    body = ": keep-alive\n\n"
    return body + "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events)


def api_handler(requests: list[httpx.Request]) -> httpx.MockTransport:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == "/v1/sessions":
            return httpx.Response(201, json={"session_id": SID})
        if path.startswith("/v1/sessions/"):
            return httpx.Response(200, json={"session_id": SID})
        if path == "/v1/chat/stream":
            text = sse(
                ("progress", {"kind": "tool", "message": "Searching the tax law"}),
                ("answer", ANSWER),
            )
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        if path == "/v1/expenses/csv":
            return httpx.Response(
                422, json={"detail": {"refused": True, "message": "This file was not processed"}}
            )
        return httpx.Response(404, json={"detail": "not found"})

    return httpx.MockTransport(handle)


def test_parse_sse_skips_comments_and_joins_data() -> None:
    lines = [": keep-alive", "", "event: progress", 'data: {"a": 1}', "", 'data: {"b": 2}']
    assert list(parse_sse(lines)) == [
        ServerEvent("progress", {"a": 1}),
        ServerEvent("message", {"b": 2}),
    ]


def test_client_calls_the_api_with_the_token() -> None:
    requests: list[httpx.Request] = []
    c = WriteOffClient("http://api", "tok", transport=api_handler(requests))
    assert c.create_session("sole_prop", 2025)["session_id"] == SID
    events = list(c.stream_chat("q", SID))
    assert [e.event for e in events] == ["progress", "answer"]
    assert requests[0].headers["authorization"] == "Bearer tok"
    with pytest.raises(APIError, match="not processed") as info:
        c.upload_csv("x.csv", b"a,b\n", "sole_prop", 2025)
    assert info.value.status == 422


def test_client_turns_network_failures_into_api_errors() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    c = WriteOffClient("http://api", transport=httpx.MockTransport(refuse))
    with pytest.raises(APIError, match="can't reach"):
        c.create_session(None, 2025)
    with pytest.raises(APIError, match="can't reach"):
        list(c.stream_chat("q", SID))


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[httpx.Request]]:
    requests: list[httpx.Request] = []
    monkeypatch.setattr(
        ui_client.WriteOffClient,
        "from_env",
        classmethod(lambda cls: cls("http://api", transport=api_handler(requests))),
    )
    st.cache_resource.clear()
    yield requests
    st.cache_resource.clear()


def test_page_streams_an_answer_with_sources(fake_api: list[httpx.Request]) -> None:
    at = AppTest.from_file(APP, default_timeout=10).run()
    assert not at.exception
    assert at.sidebar.title[0].value == "WriteOff Assistant"
    at.sidebar.selectbox[0].select("S corporation").run()
    at.chat_input[0].set_value("Can I deduct a client lunch?").run()
    assert not at.exception
    assert any("50% is deductible" in m.value for m in at.markdown)
    assert any("IRC § 274(n)(1)" in m.value for m in at.markdown)  # Sources panel
    assert any("5 claim(s) supported, 2 narrowed" in c.value for c in at.caption)
    create = json.loads(fake_api[0].content)
    assert create == {"entity_type": "s_corp", "tax_year": 2025}


def test_page_shows_an_error_when_the_api_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(
        ui_client.WriteOffClient,
        "from_env",
        classmethod(lambda cls: cls("http://api", transport=httpx.MockTransport(refuse))),
    )
    st.cache_resource.clear()
    at = AppTest.from_file(APP, default_timeout=10).run()
    at.chat_input[0].set_value("hello").run()
    assert not at.exception
    assert "can't reach" in at.error[0].value
    st.cache_resource.clear()


def test_upload_tab_needs_an_entity_type(fake_api: list[httpx.Request]) -> None:
    at = AppTest.from_file(APP, default_timeout=10).run()
    assert any("entity type" in i.value for i in at.info)


def test_client_reads_url_and_token_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WRITEOFF_API_URL", "http://api.example:9000")
    monkeypatch.setenv("API_TOKEN", "from-settings")
    get_settings.cache_clear()
    try:
        c = WriteOffClient.from_env()
        assert str(c._http.base_url) == "http://api.example:9000"
        assert c._http.headers["authorization"] == "Bearer from-settings"
    finally:
        get_settings.cache_clear()
