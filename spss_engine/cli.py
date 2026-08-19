"""
CLI entry point for the SPSS engine.

Reads JSON from stdin, processes SPSS syntax, and outputs JSON to stdout.

Input format:
  {"syntax": "...", "format": "json"}

Output format:
  {"success": true, "tables": [...], "errors": [], "warnings": []}

Exit code: 0 on success, 1 on error.
"""

from __future__ import annotations
import sys
import json
import logging
import math
from typing import Any, Dict, List, Optional

from spss_engine.data.dataset import Dataset
from spss_engine.parser.lexer import Lexer
from spss_engine.parser.parser import Parser
from spss_engine.parser.command_registry import CommandRegistry, Executor
from spss_engine.output.exporter_json import export_table
from spss_engine.utils.errors import SPSSSyntaxError, SPSSRuntimeError

logger = logging.getLogger(__name__)


def run_engine(syntax: str) -> Dict[str, Any]:
    """Run SPSS syntax through the engine and return JSON-serializable result.

    Args:
        syntax: SPSS syntax text.

    Returns:
        Dict with keys: success, tables, errors, warnings.
    """
    result: Dict[str, Any] = {
        "success": False,
        "tables": [],
        "errors": [],
        "warnings": [],
    }

    if not syntax or not syntax.strip():
        result["errors"].append({
            "type": "InputError",
            "message": "No syntax provided.",
        })
        return result

    try:
        # Parse
        parser = Parser()
        commands = parser.parse(syntax)

        # Set up registry and executor
        registry = CommandRegistry()
        executor = Executor(registry)

        # Execute
        executor.execute(commands)

        # Collect results
        result["errors"] = executor.errors
        result["warnings"] = executor.warnings

        # Export tables as JSON
        for table in executor.tables:
            result["tables"].append(export_table(table))

        # Determine success
        has_syntax_errors = any(
            e.get("type") == "SyntaxError" for e in executor.errors
        )
        result["success"] = not has_syntax_errors

    except SPSSSyntaxError as e:
        result["errors"].append({
            "type": "SyntaxError",
            "message": e.message,
            "line": e.line,
            "command": e.command,
        })
    except Exception as e:
        result["errors"].append({
            "type": "UnexpectedError",
            "message": str(e),
        })

    return result


def main() -> None:
    """CLI entry point: reads JSON from stdin, writes JSON to stdout."""
    # Read JSON from stdin
    try:
        input_data = sys.stdin.read()
        request = json.loads(input_data) if input_data.strip() else {}
    except json.JSONDecodeError as e:
        output = {
            "success": False,
            "tables": [],
            "errors": [{"type": "JSONParseError", "message": str(e)}],
            "warnings": [],
        }
        sys.stdout.write(json.dumps(output))
        sys.exit(1)

    syntax: str = request.get("syntax", "")
    fmt: str = request.get("format", "json")

    # Run the engine
    result = run_engine(syntax)

    if fmt == "json":
        sys.stdout.write(json.dumps(result, default=_json_default,
                                      ensure_ascii=False))
    else:
        # Default to JSON
        sys.stdout.write(json.dumps(result, default=_json_default,
                                      ensure_ascii=False))

    # Exit code
    if result["success"] and not result["errors"]:
        sys.exit(0)
    else:
        sys.exit(1)


def _json_default(obj: Any) -> Any:
    """Fallback JSON serializer."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return str(obj)


if __name__ == "__main__":
    main()