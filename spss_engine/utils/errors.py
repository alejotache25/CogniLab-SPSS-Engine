"""
Error types for the SPSS engine.

Two main error classes per the contract (CONSTRAINT 3.9 / 3.10):
  - SyntaxError (spss): stop-on-first-error.
  - SPSSRuntimeError: capture and continue.
"""

from __future__ import annotations
from typing import Optional


class SPSSSyntaxError(Exception):
    """Raised when the parser encounters a syntax error.

    Per CONSTRAINT 3.9, the executor stops on the first SyntaxError.
    """

    def __init__(self, message: str, line: Optional[int] = None,
                 command: Optional[str] = None) -> None:
        self.message: str = message
        self.line: Optional[int] = line
        self.command: Optional[str] = command
        parts: list[str] = []
        if line is not None:
            parts.append(f"line {line}")
        parts.append(message)
        if command:
            parts.append(f"(command: {command})")
        super().__init__(": ".join(parts))


class SPSSRuntimeError(Exception):
    """Raised when a command is syntactically valid but execution fails.

    Per CONSTRAINT 3.10, the executor captures this and continues.
    """

    def __init__(self, message: str, line: Optional[int] = None,
                 command: Optional[str] = None) -> None:
        self.message: str = message
        self.line: Optional[int] = line
        self.command: Optional[str] = command
        parts: list[str] = []
        if line is not None:
            parts.append(f"line {line}")
        parts.append(message)
        if command:
            parts.append(f"(command: {command})")
        super().__init__(": ".join(parts))


class SPSSWarning(Exception):
    """Non-fatal warning — captured in warnings[] without stopping execution."""

    def __init__(self, message: str) -> None:
        self.message: str = message
        super().__init__(message)