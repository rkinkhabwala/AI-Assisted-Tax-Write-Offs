"""HTTP client the Streamlit UI uses to talk to the API (kept free of Streamlit so it can
be tested on its own)."""

import json
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from writeoff.config import get_settings


@dataclass(frozen=True, slots=True)
class ServerEvent:
    event: str
    data: dict[str, Any]


def parse_sse(lines: Iterable[str]) -> Iterator[ServerEvent]:
    """Server-sent events from response lines. Comments (keep-alives) are skipped."""
    event, data = "message", list[str]()
    for line in lines:
        if not line:
            if data:
                yield ServerEvent(event, json.loads("\n".join(data)))
            event, data = "message", []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data.append(line.removeprefix("data:").strip())
    if data:
        yield ServerEvent(event, json.loads("\n".join(data)))


class APIError(RuntimeError):
    def __init__(self, status: int, detail: Any) -> None:
        self.status = status
        self.detail = detail
        message = detail.get("message") if isinstance(detail, dict) else detail
        super().__init__(str(message))


@contextmanager
def _network_errors() -> Iterator[None]:
    try:
        yield
    except httpx.HTTPError as exc:
        raise APIError(0, f"can't reach the WriteOff API ({type(exc).__name__})") from exc


class WriteOffClient:
    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._http = httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout, transport=transport
        )

    def _call(self, send: Callable[[], httpx.Response]) -> Any:
        with _network_errors():
            response = send()
        return self._check(response)

    @classmethod
    def from_env(cls) -> "WriteOffClient":
        """WRITEOFF_API_URL and API_TOKEN from the environment or, when run locally
        (`make ui`), from `.env` through the project settings."""
        settings = get_settings()
        token = settings.api_token.get_secret_value() if settings.api_token else None
        return cls(settings.writeoff_api_url, token or None)

    def _check(self, response: httpx.Response) -> Any:
        if response.is_success:
            return response.json()
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise APIError(response.status_code, detail)

    def create_session(self, entity_type: str | None, tax_year: int | None) -> dict[str, Any]:
        body = {"entity_type": entity_type, "tax_year": tax_year}
        result: dict[str, Any] = self._call(lambda: self._http.post("/v1/sessions", json=body))
        return result

    def update_session(
        self, session_id: str, entity_type: str | None, tax_year: int | None
    ) -> dict[str, Any]:
        body = {"entity_type": entity_type, "tax_year": tax_year}
        result: dict[str, Any] = self._call(
            lambda: self._http.patch(f"/v1/sessions/{session_id}", json=body)
        )
        return result

    def stream_chat(self, question: str, session_id: str) -> Iterator[ServerEvent]:
        body = {"question": question, "session_id": session_id}
        with (
            _network_errors(),
            self._http.stream("POST", "/v1/chat/stream", json=body) as response,
        ):
            if not response.is_success:
                response.read()
                self._check(response)
            yield from parse_sse(response.iter_lines())

    def upload_csv(self, name: str, data: bytes, entity_type: str, tax_year: int) -> Any:
        return self._call(
            lambda: self._http.post(
                "/v1/expenses/csv",
                files={"file": (name, data, "text/csv")},
                data={"entity_type": entity_type, "tax_year": str(tax_year)},
            )
        )
