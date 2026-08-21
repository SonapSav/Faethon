"""Shared OpenRouter HTTP client.

All three legs of the pipeline (STT, LLM, TTS) go through one authenticated
client so retry, timeout, and cost accounting behave the same everywhere.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import httpx

from .. import disclosure

log = logging.getLogger(__name__)

BASE_URL = "https://openrouter.ai/api/v1"

# Retried with backoff: rate limits and transient upstream failures.
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3
BACKOFF_BASE = 0.6


class OpenRouterError(RuntimeError):
    """A request failed in a way retrying will not fix."""

    #: HTTP status, when the failure came back as one. None means the request
    #: never got an answer -- which is the difference between "no network" and
    #: "no credit", two problems with entirely different fixes.
    status: int | None = None


def _http_error(path: str, status: int, body: str) -> OpenRouterError:
    err = OpenRouterError(f"{path} -> {status}: {body}")
    err.status = status
    return err


class OpenRouterClient:
    def __init__(self, api_key: str, timeout: float = 60.0) -> None:
        if not api_key:
            raise OpenRouterError(
                "No OpenRouter API key. Copy .env.example to .env and set "
                "OPENROUTER_API_KEY (get one at https://openrouter.ai/keys)."
            )
        self._client = httpx.Client(
            base_url=BASE_URL,
            timeout=httpx.Timeout(timeout, connect=10.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                # Optional attribution headers OpenRouter uses for its rankings.
                "HTTP-Referer": "https://github.com/local/faethon",
                "X-Title": "Faethon",
            },
        )
        # Running total for the process, in USD.
        self.spent = 0.0

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenRouterClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _sleep_for(self, attempt: int, response: httpx.Response | None) -> float:
        if response is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after:
                try:
                    return min(float(retry_after), 30.0)
                except ValueError:
                    pass
        return BACKOFF_BASE * (2**attempt)

    def get_json(self, path: str) -> dict[str, Any]:
        """GET expecting a JSON response, with the same retry policy as POST."""
        return self._json_request("GET", path, None)

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST expecting a JSON response."""
        return self._json_request("POST", path, payload)

    def _json_request(
        self, method: str, path: str, payload: dict[str, Any] | None
    ) -> dict[str, Any]:
        last_error: str = ""
        for attempt in range(MAX_ATTEMPTS):
            try:
                self._disclose(path)
                r = self._client.request(method, path, json=payload)
            except httpx.RequestError as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt == MAX_ATTEMPTS - 1:
                    break
                time.sleep(self._sleep_for(attempt, None))
                continue

            if r.status_code in RETRY_STATUS and attempt < MAX_ATTEMPTS - 1:
                delay = self._sleep_for(attempt, r)
                log.warning("%s -> %s, retrying in %.1fs", path, r.status_code, delay)
                time.sleep(delay)
                continue

            if r.status_code >= 400:
                raise _http_error(path, r.status_code, r.text[:400])

            data = r.json()
            self._record_cost(data)
            return data

        raise OpenRouterError(f"{path} failed after {MAX_ATTEMPTS} attempts: {last_error}")

    @contextmanager
    def post_stream(self, path: str, payload: dict[str, Any]) -> Iterator[httpx.Response]:
        """POST expecting a streamed binary response (the TTS endpoint).

        Retries only on connection setup; once bytes are flowing a failure is
        surfaced to the caller rather than silently restarting playback.
        """
        last_error = ""
        for attempt in range(MAX_ATTEMPTS):
            try:
                self._disclose(path)
                with self._client.stream("POST", path, json=payload) as r:
                    if r.status_code in RETRY_STATUS and attempt < MAX_ATTEMPTS - 1:
                        r.read()
                        delay = self._sleep_for(attempt, r)
                        log.warning("%s -> %s, retrying in %.1fs", path, r.status_code, delay)
                        time.sleep(delay)
                        continue
                    if r.status_code >= 400:
                        body = r.read().decode(errors="replace")[:400]
                        raise _http_error(path, r.status_code, body)

                    # Past here the caller owns the stream. A failure while
                    # reading it must NOT be retried: the caller has already
                    # consumed part of the response, and restarting the request
                    # would silently truncate it instead of reporting the fault.
                    try:
                        yield r
                    except httpx.RequestError as e:
                        raise OpenRouterError(
                            f"{path} stream broke mid-response: {type(e).__name__}: {e}"
                        ) from e
                    return
            except httpx.RequestError as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt == MAX_ATTEMPTS - 1:
                    break
                time.sleep(self._sleep_for(attempt, None))

        raise OpenRouterError(f"{path} failed after {MAX_ATTEMPTS} attempts: {last_error}")

    def _disclose(self, path: str) -> None:
        """Note the request in the ledger, before it is made.

        Before rather than after, and per attempt rather than per success: a
        retried upload crossed the wire twice, and a request that times out
        still sent everything it was carrying. A ledger of what arrived safely
        would understate what left.
        """
        disclosure.LEDGER.record("openrouter.ai", path, disclosure.kind_for(path))

    def record_usage(self, usage: dict[str, Any] | None) -> None:
        """Add a response's cost to the running total.

        Public because streamed responses carry usage in a trailing SSE chunk
        rather than in a JSON body this client ever sees whole.
        """
        if isinstance(usage, dict):
            cost = usage.get("cost")
            if isinstance(cost, (int, float)):
                self.spent += float(cost)

    def _record_cost(self, data: dict[str, Any]) -> None:
        self.record_usage(data.get("usage"))
