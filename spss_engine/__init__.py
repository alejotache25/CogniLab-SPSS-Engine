"""
spss_engine — A Python SPSS-compatible statistical engine for CogniLab In-Silico.

This package implements a subset of IBM SPSS Statistics syntax parsing and
execution, designed to run as a backend service for the CogniLab platform.
"""

__version__ = "0.1.0"
__author__ = "CogniLab In-Silico"

# Convenience imports for the most-used public APIs
from spss_engine.data.dataset import Dataset
from spss_engine.parser.lexer import Lexer
from spss_engine.parser.parser import Parser
from spss_engine.parser.command_registry import CommandRegistry, Executor

__all__ = [
    "Dataset",
    "Lexer",
    "Parser",
    "CommandRegistry",
    "Executor",
]