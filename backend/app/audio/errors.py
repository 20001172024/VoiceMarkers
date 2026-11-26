"""Errors raised for audio that cannot be analysed, with user-facing messages."""

from __future__ import annotations


class AudioError(ValueError):
    """Audio that cannot be analysed.

    `code` is a stable machine-readable reason (used by the API in M4);
    the message is written for the person who made the recording.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
