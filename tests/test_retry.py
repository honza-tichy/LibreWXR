# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Regression tests for the shared HTTP retry helper.

``retry_get`` caught ``httpx.DecodeError``, which httpx does not define
(the class is ``DecodingError``).  An ``except`` clause is *evaluated*
whenever an exception reaches it, so the bad attribute lookup raised
``AttributeError`` from inside the handler and replaced whatever was in
flight.  That broke two things at once: the decode retry never ran, and
any ``CancelledError`` passing through — shutdown, or an enclosing
``asyncio.wait_for`` deadline — came back out as an unrelated error.
"""

import asyncio

import httpx
import pytest

from librewxr.data.retry import retry_get

pytestmark = pytest.mark.asyncio


class _ScriptedClient:
    """Minimal AsyncClient stand-in that replays a list of outcomes."""

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    async def get(self, url, **kwargs):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


async def test_decoding_error_is_retried():
    """A truncated body must be retried, not turned into AttributeError."""
    response = httpx.Response(200, content=b"ok")
    client = _ScriptedClient([
        httpx.DecodingError("truncated"),
        response,
    ])

    result = await retry_get(client, "https://example.test/x", retries=1, delay=0)

    assert result is response
    assert client.calls == 2


async def test_decoding_error_gives_up_after_retries():
    client = _ScriptedClient([
        httpx.DecodingError("truncated"),
        httpx.DecodingError("truncated"),
    ])

    result = await retry_get(client, "https://example.test/x", retries=1, delay=0)

    assert result is None
    assert client.calls == 2


async def test_cancellation_is_not_swallowed():
    """Cancelling a fetch must surface CancelledError, not AttributeError.

    This is the one that mattered in production: every ``wait_for``
    deadline in the fetch path cancels the inner coroutine, and the
    broken handler converted that cancellation into an error the caller
    was not prepared for.
    """

    class _HangingClient:
        async def get(self, url, **kwargs):
            await asyncio.sleep(30)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            retry_get(_HangingClient(), "https://example.test/x", retries=1),
            timeout=0.05,
        )


async def test_transport_error_is_retried():
    response = httpx.Response(200, content=b"ok")
    client = _ScriptedClient([
        httpx.ConnectError("refused"),
        response,
    ])

    result = await retry_get(client, "https://example.test/x", retries=1, delay=0)

    assert result is response
    assert client.calls == 2


async def test_http_status_response_is_returned_not_retried():
    """A server that answered is not a transport problem — hand it back."""
    response = httpx.Response(503, content=b"nope")
    client = _ScriptedClient([response])

    result = await retry_get(client, "https://example.test/x", retries=1, delay=0)

    assert result is response
    assert client.calls == 1
