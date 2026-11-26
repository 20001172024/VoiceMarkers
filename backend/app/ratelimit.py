"""A small in-memory rate limiter for the prediction endpoints.

Fixed one-minute windows per client IP. State lives in this process only, so
with several server processes each one counts separately; that is acceptable
for a single free-tier instance and keeps the demo dependency-free.

Behind proxies, the client IP is taken from `X-Forwarded-For` counting from the
*right*: each proxy appends the address it saw, so the rightmost entries are
written by trusted infrastructure while the leftmost ones are whatever the
client chose to send. Trusting the first entry would let anyone dodge the limit
by sending a fake header.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from fastapi import Request


class RateLimitExceeded(Exception):
    """Raised when a client exceeds its per-minute allowance."""

    def __init__(self, retry_after: int) -> None:
        super().__init__(f"Rate limit exceeded; retry after {retry_after} s")
        self.retry_after = retry_after


class RateLimiter:
    """Allow `limit` requests per client per `window_s` seconds (0 = unlimited)."""

    def __init__(
        self, limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._windows: dict[str, tuple[float, int]] = {}  # client -> (window start, count)
        self._lock = threading.Lock()

    def hit(self, client: str) -> None:
        """Count one request; raise `RateLimitExceeded` if over the limit."""
        if self.limit <= 0:
            return
        now = self._clock()
        with self._lock:
            start, count = self._windows.get(client, (now, 0))
            if now - start >= self.window_s:
                start, count = now, 0
            if count >= self.limit:
                raise RateLimitExceeded(retry_after=max(1, int(start + self.window_s - now)))
            self._windows[client] = (start, count + 1)
            if len(self._windows) > 10_000:  # drop stale clients so memory stays bounded
                self._windows = {
                    c: w for c, w in self._windows.items() if now - w[0] < self.window_s
                }


def client_id(request: Request, forwarded_hops: int = 0) -> str:
    """The client's IP: the `forwarded_hops`-th X-Forwarded-For entry from the right.

    With `forwarded_hops == 0`, or if the header has too few entries, the
    connection's peer address is used.
    """
    peer = request.client.host if request.client else "unknown"
    if forwarded_hops <= 0:
        return peer
    entries = [
        e.strip() for e in request.headers.get("x-forwarded-for", "").split(",") if e.strip()
    ]
    return entries[-forwarded_hops] if len(entries) >= forwarded_hops else peer
