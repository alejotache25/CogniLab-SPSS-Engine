"""
Command Registry and Executor for the SPSS engine.

The CommandRegistry maps command names to handler functions.
The Executor processes a list of CommandNode objects sequentially:

  1. Immediate commands (MISSING VALUES, VARIABLE LABELS, etc.) execute
     right away — they modify dictionary metadata.
  2. Transform commands (COMPUTE, RECODE, SELECT IF) are enqueued as
     "pending" and executed on the next data pass.
  3. Data-reading commands (FREQUENCIES, DESCRIPTIVES, EXECUTE, etc.)
     trigger a data pass, which flushes pending transforms first.

CONSTRAINT 3.9: SyntaxError → stop-on-first-error.
CONSTRAINT 3.10: SPSSRuntimeError → capture and continue.
"""

from __future__ import annotations
import logging
from typing import Callable, Dict, List, Optional, Any

from spss_engine.parser.ast_nodes import CommandNode, SubcommandNode
from spss_engine.data.dataset import Dataset
from spss_engine.utils.errors import SPSSSyntaxError, SPSSRuntimeError

logger = logging.getLogger(__name__)

# Type for command handler functions
CommandHandler = Callable[[CommandNode, Dataset, "Executor"], Any]


class CommandRegistry:
    """Registry mapping command names to handler functions.

    Commands are categorized:
      - IMMEDIATE: execute without reading data (MISSING VALUES, etc.)
      - TRANSFORM: pending until next data pass (COMPUTE, RECODE, etc.)
      - DATA_READ: trigger a data pass (FREQUENCIES, EXECUTE, etc.)
    """

    # Command categories
    IMMEDIATE: set[str] = {
        "MISSING VALUES", "VARIABLE LABELS", "VALUE LABELS",
        "FORMATS", "STRING", "FILTER", "SPLIT FILE", "WEIGHT",
        "NUMERIC", "RENAME VARIABLES",
    }

    TRANSFORM: set[str] = {
        "COMPUTE", "RECODE", "SELECT IF", "IF", "DO IF",
        "COUNT", "LEAVE",
    }

    DATA_READ: set[str] = {
        "DATA LIST", "FREQUENCIES", "DESCRIPTIVES", "CROSSTABS", "LIST", "EXECUTE",
        "SORT CASES", "T-TEST", "ONEWAY", "CORRELATIONS", "REGRESSION",
        "RELIABILITY", "FACTOR", "MEANS", "EXAMINE", "NPAR TESTS",
        "SAVE", "AGGREGATE", "AUTORECODE", "RANK",
    }

    def __init__(self) -> None:
        self._handlers: Dict[str, CommandHandler] = {}
        self._register_defaults()

    def register(self, command_name: str, handler: CommandHandler) -> None:
        """Register a handler for a command name."""
        self._handlers[command_name.upper()] = handler

    def get_handler(self, command_name: str) -> Optional[CommandHandler]:
        """Get the handler for a command name."""
        return self._handlers.get(command_name.upper())

    def has_command(self, command_name: str) -> bool:
        """Check if a command is registered."""
        return command_name.upper() in self._handlers

    def category(self, command_name: str) -> str:
        """Get the category of a command: IMMEDIATE, TRANSFORM, DATA_READ, or UNKNOWN."""
        name = command_name.upper()
        if name in self.IMMEDIATE:
            return "IMMEDIATE"
        if name in self.TRANSFORM:
            return "TRANSFORM"
        if name in self.DATA_READ:
            return "DATA_READ"
        return "UNKNOWN"

    def _register_defaults(self) -> None:
        """Register default handlers for Phase 1 commands."""
        # Data-reading / setup commands
        self.register("DATA LIST", _handle_data_list)

        # Immediate commands
        self.register("MISSING VALUES", _handle_missing_values)
        self.register("VARIABLE LABELS", _handle_variable_labels)
        self.register("VALUE LABELS", _handle_value_labels)
        self.register("FORMATS", _handle_formats)
        self.register("STRING", _handle_string)
        self.register("FILTER", _handle_filter)
        self.register("SPLIT FILE", _handle_split_file)
        self.register("WEIGHT", _handle_weight)

        # Transform commands (pending — enqueued, not executed immediately)
        self.register("COMPUTE", _handle_compute)
        self.register("RECODE", _handle_recode)
        self.register("SELECT IF", _handle_select_if)

        # Data-reading commands
        self.register("EXECUTE", _handle_execute)
        self.register("LIST", _handle_list)
        self.register("SORT CASES", _handle_sort_cases)

        # Statistical procedures (Phase 2 — stub handlers that flush transforms)
        self.register("FREQUENCIES", _handle_frequencies)
        self.register("DESCRIPTIVES", _handle_descriptives)
        self.register("CROSSTABS", _handle_crosstabs)
        self.register("T-TEST", _handle_t_test)
        self.register("ONEWAY", _handle_oneway)
        self.register("CORRELATIONS", _handle_correlations)
        self.register("REGRESSION", _handle_regression)
        self.register("RELIABILITY", _handle_reliability)
        self.register("FACTOR", _handle_factor)
        self.register("MEANS", _handle_means)
        self.register("EXAMINE", _handle_examine)
        self.register("NPAR TESTS", _handle_npar_tests)


class Executor:
    """Executes a list of CommandNode objects against a Dataset.

    Key behaviors:
      - IMMEDIATE commands execute right away.
      - TRANSFORM commands are enqueued as pending.
      - DATA_READ commands flush pending transforms, then execute.
      - SyntaxError → stop immediately (CONSTRAINT 3.9).
      - SPSSRuntimeError → capture error, continue (CONSTRAINT 3.10).
    """

    def __init__(self, registry: CommandRegistry,
                 dataset: Optional[Dataset] = None) -> None:
        self.registry: CommandRegistry = registry
        self.dataset: Dataset = dataset or Dataset()
        self.pending_transforms: List[CommandNode] = []
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.tables: List[Any] = []
        self._stopped: bool = False

    def execute(self, commands: List[CommandNode]) -> None:
        """Execute a list of commands sequentially.

        Stops on first SyntaxError. Continues on SPSSRuntimeError.
        """
        for cmd in commands:
            if self._stopped:
                # Don't execute commands after a syntax error
                break
            self._execute_command(cmd)

    def _execute_command(self, cmd: CommandNode) -> None:
        """Execute a single command, handling errors per CONSTRAINT 3.9/3.10."""
        handler = self.registry.get_handler(cmd.name)
        if handler is None:
            # Unknown command — SyntaxError, stop execution
            err = SPSSSyntaxError(
                f"Unknown command: {cmd.name}",
                line=cmd.line,
                command=cmd.name,
            )
            self._record_syntax_error(err)
            self._stopped = True
            return

        category = self.registry.category(cmd.name)

        try:
            # For DATA_READ commands, flush pending transforms first
            if category == "DATA_READ":
                self._flush_transforms()

            handler(cmd, self.dataset, self)

        except SPSSSyntaxError as e:
            self._record_syntax_error(e)
            self._stopped = True
        except SPSSRuntimeError as e:
            self._record_runtime_error(e, cmd)
            # Continue execution
        except Exception as e:
            # Unexpected error — treat as runtime error
            self._record_runtime_error(
                SPSSRuntimeError(str(e), line=cmd.line, command=cmd.name),
                cmd,
            )

    def _flush_transforms(self) -> None:
        """Execute all pending transform commands."""
        if not self.pending_transforms:
            return
        for cmd in self.pending_transforms:
            handler = self.registry.get_handler(cmd.name)
            if handler is not None:
                try:
                    handler(cmd, self.dataset, self)
                except SPSSRuntimeError as e:
                    self._record_runtime_error(e, cmd)
                except SPSSSyntaxError as e:
                    self._record_syntax_error(e)
                    self._stopped = True
                    return
        self.pending_transforms.clear()

    def _record_syntax_error(self, err: SPSSSyntaxError) -> None:
        """Record a syntax error in the errors list."""
        self.errors.append({
            "type": "SyntaxError",
            "message": err.message,
            "line": err.line,
            "command": err.command,
        })
        logger.error("SyntaxError: %s (line %s)", err.message, err.line)

    def _record_runtime_error(self, err: SPSSRuntimeError,
                              cmd: CommandNode) -> None:
        """Record a runtime error in the errors list."""
        self.errors.append({
            "type": "SPSSRuntimeError",
            "message": err.message,
            "line": err.line,
            "command": cmd.name,
        })
        logger.error("RuntimeError: %s (line %s)", err.message, err.line)

    def add_warning(self, message: str) -> None:
        """Add a warning to the warnings list."""
        self.warnings.append(message)

    def add_table(self, table: Any) -> None:
        """Add a result table."""
        self.tables.append(table)

    @property
    def stopped(self) -> bool:
        return self._stopped


# ----------------------------------------------------------------------
# Default command handlers
# ----------------------------------------------------------------------

def _handle_missing_values(cmd: CommandNode, ds: Dataset,
                            exe: Executor) -> None:
    """MISSING VALUES varlist (values)."""
    for sc in cmd.subcommands:
        if sc.name == "VALUES":
            # Variables and missing values
            var_names = ds.get_varlist(sc.variables)
            missing_vals = sc.raw_tokens  # list of strings
            for vn in var_names:
                if not ds.has_variable(vn):
                    raise SPSSRuntimeError(f"Variable not found: {vn}")
                var = ds.get_variable(vn)
                mv = var.missing
                mv.discrete.clear()
                for mv_val in missing_vals:
                    try:
                        mv.discrete.append(float(mv_val))
                    except (ValueError, TypeError):
                        mv.discrete.append(mv_val)


def _handle_variable_labels(cmd: CommandNode, ds: Dataset,
                             exe: Executor) -> None:
    """VARIABLE LABELS var 'label' var 'label' ..."""
    for sc in cmd.subcommands:
        if sc.name == "LABEL" and sc.variables and sc.raw_tokens:
            var_name = sc.variables[0]
            label = sc.raw_tokens[0]
            if not ds.has_variable(var_name):
                raise SPSSRuntimeError(f"Variable not found: {var_name}")
            ds.get_variable(var_name).set_label(label)


def _handle_value_labels(cmd: CommandNode, ds: Dataset,
                          exe: Executor) -> None:
    """VALUE LABELS varlist val 'label' val 'label' ..."""
    for sc in cmd.subcommands:
        if sc.name == "LABELS":
            var_names = ds.get_varlist(sc.variables)
            pairs = sc.raw_tokens  # list of (value, label) tuples
            for vn in var_names:
                if not ds.has_variable(vn):
                    raise SPSSRuntimeError(f"Variable not found: {vn}")
                var = ds.get_variable(vn)
                for val_str, label in pairs:
                    try:
                        val: float | str = float(val_str)
                    except (ValueError, TypeError):
                        val = val_str
                    var.add_value_label(val, label)


def _handle_formats(cmd: CommandNode, ds: Dataset,
                     exe: Executor) -> None:
    """FORMATS var (format) var (format) ..."""
    for sc in cmd.subcommands:
        if sc.name == "FORMAT" and sc.variables:
            var_name = sc.variables[0]
            if not ds.has_variable(var_name):
                raise SPSSRuntimeError(f"Variable not found: {var_name}")
            fmt = sc.raw_tokens[0] if sc.raw_tokens else "F8.2"
            # Check for unsupported formats
            if fmt.upper().startswith("DOLLAR") or fmt.upper().startswith("DATE"):
                raise SPSSRuntimeError(
                    f"Format not supported in Phase 1: {fmt}")
            ds.get_variable(var_name).format = fmt


def _handle_string(cmd: CommandNode, ds: Dataset,
                    exe: Executor) -> None:
    """STRING varname (A10) ... — declare string variables."""
    for vd in cmd.var_defs:
        var = Variable(
            name=vd.name,
            var_type="string",
            width=vd.width,
            format=vd.format_spec or f"A{vd.width}",
        )
        ds.add_variable(var)
        # Add empty column to DataFrame
        if ds.df is not None and len(ds.df) > 0:
            ds.df[vd.name] = ""
        elif ds.df is not None:
            ds.df[vd.name] = pd.Series(dtype="object")


def _handle_filter(cmd: CommandNode, ds: Dataset,
                     exe: Executor) -> None:
    """FILTER BY var / FILTER OFF."""
    for sc in cmd.subcommands:
        if sc.name == "OFF":
            ds.clear_filter()
        elif sc.name == "BY" and sc.variables:
            ds.set_filter(sc.variables[0])


def _handle_split_file(cmd: CommandNode, ds: Dataset,
                        exe: Executor) -> None:
    """SPLIT FILE BY vars / SPLIT FILE OFF / SPLIT FILE LAYERED BY vars."""
    for sc in cmd.subcommands:
        if sc.name == "OFF":
            ds.clear_split()
        elif sc.name == "BY":
            layered = bool(sc.raw_tokens)
            ds.set_split(sc.variables, layered)


def _handle_weight(cmd: CommandNode, ds: Dataset,
                    exe: Executor) -> None:
    """WEIGHT BY var / WEIGHT OFF."""
    for sc in cmd.subcommands:
        if sc.name == "OFF":
            ds.clear_weight()
        elif sc.name == "BY" and sc.variables:
            ds.set_weight(sc.variables[0])


# ----------------------------------------------------------------------
# Transform handlers (pending — enqueued by executor)
# ----------------------------------------------------------------------

def _handle_compute(cmd: CommandNode, ds: Dataset,
                     exe: Executor) -> None:
    """COMPUTE target = expression."""
    from spss_engine.transforms.compute import execute_compute
    execute_compute(cmd, ds)


def _handle_recode(cmd: CommandNode, ds: Dataset,
                    exe: Executor) -> None:
    """RECODE varlist (val1=val2) ... INTO newvar."""
    from spss_engine.transforms.recode import execute_recode
    execute_recode(cmd, ds)


def _handle_select_if(cmd: CommandNode, ds: Dataset,
                       exe: Executor) -> None:
    """SELECT IF expression — filters cases permanently."""
    from spss_engine.transforms.select_if import execute_select_if
    execute_select_if(cmd, ds)


# ----------------------------------------------------------------------
# Data-reading command handlers
# ----------------------------------------------------------------------

def _handle_execute(cmd: CommandNode, ds: Dataset,
                     exe: Executor) -> None:
    """EXECUTE — flushes pending transforms, produces no output."""
    # Transforms are already flushed by the executor before calling this handler
    pass


def _handle_list(cmd: CommandNode, ds: Dataset,
                  exe: Executor) -> None:
    """LIST — lists cases. Phase 1 stub."""
    pass


def _handle_sort_cases(cmd: CommandNode, ds: Dataset,
                         exe: Executor) -> None:
    """SORT CASES BY var (A) var (D) ..."""
    for sc in cmd.subcommands:
        if sc.name == "BY" and sc.variables:
            var_names = sc.variables
            directions = sc.raw_tokens if sc.raw_tokens else []
            # Build sort keys
            if ds.df is not None and not ds.df.empty:
                sort_cols: list[str] = []
                ascending: list[bool] = []
                for i, vn in enumerate(var_names):
                    if not ds.has_variable(vn):
                        raise SPSSRuntimeError(f"Variable not found: {vn}")
                    sort_cols.append(vn)
                    if i < len(directions) and directions[i].upper() in ("D", "DESCENDING"):
                        ascending.append(False)
                    else:
                        ascending.append(True)
                ds.df.sort_values(by=sort_cols, ascending=ascending,
                                   inplace=True)
                ds.df.reset_index(drop=True, inplace=True)


def _handle_stub_procedure(cmd: CommandNode, ds: Dataset,
                            exe: Executor) -> None:
    """Stub handler for statistical procedures not yet implemented."""
    pass


def _handle_frequencies(cmd: CommandNode, ds: Dataset,
                         exe: Executor) -> None:
    """FREQUENCIES handler."""
    from spss_engine.procedures.frequencies import execute_frequencies
    df = ds.get_filtered_df()
    if len(df) == 0:
        exe.add_warning("No cases available after filtering")
    tables = execute_frequencies(cmd, ds)
    for t in tables:
        exe.add_table(t)


def _handle_descriptives(cmd: CommandNode, ds: Dataset,
                          exe: Executor) -> None:
    """DESCRIPTIVES handler."""
    from spss_engine.procedures.descriptives import execute_descriptives
    tables = execute_descriptives(cmd, ds)
    for t in tables:
        exe.add_table(t)


def _handle_crosstabs(cmd: CommandNode, ds: Dataset,
                       exe: Executor) -> None:
    """CROSSTABS handler."""
    from spss_engine.procedures.crosstabs import execute_crosstabs
    tables = execute_crosstabs(cmd, ds)
    for t in tables:
        exe.add_table(t)


# ----------------------------------------------------------------------
# Phase 4 procedure handlers
# ----------------------------------------------------------------------

def _handle_t_test(cmd: CommandNode, ds: Dataset,
                     exe: Executor) -> None:
    """T-TEST handler."""
    from spss_engine.procedures.t_test import execute_t_test
    execute_t_test(cmd, ds, exe)


def _handle_oneway(cmd: CommandNode, ds: Dataset,
                     exe: Executor) -> None:
    """ONEWAY handler."""
    from spss_engine.procedures.oneway import execute_oneway
    execute_oneway(cmd, ds, exe)


def _handle_correlations(cmd: CommandNode, ds: Dataset,
                            exe: Executor) -> None:
    """CORRELATIONS handler."""
    from spss_engine.procedures.correlations import execute_correlations
    execute_correlations(cmd, ds, exe)


def _handle_regression(cmd: CommandNode, ds: Dataset,
                          exe: Executor) -> None:
    """REGRESSION handler."""
    from spss_engine.procedures.regression import execute_regression
    execute_regression(cmd, ds, exe)


def _handle_reliability(cmd: CommandNode, ds: Dataset,
                          exe: Executor) -> None:
    """RELIABILITY handler."""
    from spss_engine.procedures.reliability import execute_reliability
    execute_reliability(cmd, ds, exe)


def _handle_factor(cmd: CommandNode, ds: Dataset,
                     exe: Executor) -> None:
    """FACTOR handler."""
    from spss_engine.procedures.factor import execute_factor
    execute_factor(cmd, ds, exe)


def _handle_means(cmd: CommandNode, ds: Dataset,
                    exe: Executor) -> None:
    """MEANS handler."""
    from spss_engine.procedures.means import execute_means
    execute_means(cmd, ds, exe)


def _handle_examine(cmd: CommandNode, ds: Dataset,
                      exe: Executor) -> None:
    """EXAMINE handler."""
    from spss_engine.procedures.examine import execute_examine
    execute_examine(cmd, ds, exe)


def _handle_npar_tests(cmd: CommandNode, ds: Dataset,
                         exe: Executor) -> None:
    """NPAR TESTS handler."""
    from spss_engine.procedures.npar import execute_npar_tests
    execute_npar_tests(cmd, ds, exe)


def _handle_data_list(cmd: CommandNode, ds: Dataset,
                       exe: Executor) -> None:
    """DATA LIST — load data into the dataset."""
    ds.load_from_data_list(cmd)


# Late imports for handler functions
from spss_engine.data.variable import Variable  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402