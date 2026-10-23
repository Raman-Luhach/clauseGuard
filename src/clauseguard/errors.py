"""Errors that carry a stable code and a message that can be shown to an end user."""

from __future__ import annotations


class InputError(ValueError):
    """The user's input can't be analysed (empty, scanned PDF, wrong language, ...)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ModelNotAvailableError(RuntimeError):
    """No trained model exists for the requested module."""
