"""
Syntax Generator: converts experiment data (simulation_logs) into SPSS syntax.

Takes structured experiment data and generates appropriate SPSS syntax
for common statistical analyses.

Input: experiment data dict with variables and observations.
Output: SPSS syntax string.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def generate_syntax(experiment_data: Dict[str, Any],
                      analysis_type: str = "descriptive") -> str:
    """Generate SPSS syntax from experiment data.

    Args:
        experiment_data: Dict with keys:
            - variables: list of variable names
            - observations: list of dicts (cases) or list of lists
            - group_var: optional grouping variable name
        analysis_type: Type of analysis to generate syntax for.
            Options: "descriptive", "frequencies", "t_test",
            "oneway", "correlation", "regression", "reliability",
            "npar", "full"

    Returns:
        SPSS syntax string.
    """
    variables: List[str] = experiment_data.get("variables", [])
    observations: List[Any] = experiment_data.get("observations", [])
    group_var: Optional[str] = experiment_data.get("group_var")
    dep_vars: List[str] = [v for v in variables
                             if v != group_var] if group_var else variables

    if not variables or not observations:
        return ""

    # Build DATA LIST section
    syntax_lines: List[str] = []
    syntax_lines.append("DATA LIST LIST /")
    for var in variables:
        syntax_lines.append(f"  {var}")
    syntax_lines.append(".")
    syntax_lines.append("")

    # Build BEGIN DATA section
    syntax_lines.append("BEGIN DATA.")
    for obs in observations:
        if isinstance(obs, dict):
            vals = []
            for var in variables:
                v = obs.get(var, "")
                if v is None or v == "":
                    vals.append(".")
                else:
                    vals.append(str(v))
            syntax_lines.append(" ".join(vals))
        elif isinstance(obs, (list, tuple)):
            vals = []
            for v in obs:
                if v is None or v == "":
                    vals.append(".")
                else:
                    vals.append(str(v))
            syntax_lines.append(" ".join(vals))
    syntax_lines.append("END DATA.")
    syntax_lines.append("EXECUTE.")
    syntax_lines.append("")

    # Generate analysis-specific syntax
    atype = analysis_type.lower().strip()

    if atype == "descriptive":
        syntax_lines.extend(_gen_descriptives(dep_vars))
    elif atype == "frequencies":
        syntax_lines.extend(_gen_frequencies(variables))
    elif atype == "t_test":
        syntax_lines.extend(_gen_t_test(dep_vars, group_var))
    elif atype == "oneway":
        syntax_lines.extend(_gen_oneway(dep_vars, group_var))
    elif atype == "correlation":
        syntax_lines.extend(_gen_correlations(dep_vars))
    elif atype == "regression":
        syntax_lines.extend(_gen_regression(dep_vars, group_var))
    elif atype == "reliability":
        syntax_lines.extend(_gen_reliability(dep_vars))
    elif atype == "npar":
        syntax_lines.extend(_gen_npar(dep_vars, group_var))
    elif atype == "full":
        syntax_lines.extend(_gen_descriptives(dep_vars))
        syntax_lines.extend(_gen_frequencies(variables))
        if group_var:
            syntax_lines.extend(_gen_oneway(dep_vars, group_var))
        syntax_lines.extend(_gen_correlations(dep_vars))
    else:
        syntax_lines.extend(_gen_descriptives(dep_vars))

    return "\n".join(syntax_lines)


def _gen_descriptives(vars_list: List[str]) -> List[str]:
    """Generate DESCRIPTIVES syntax."""
    if not vars_list:
        return []
    var_str = " ".join(vars_list)
    return [
        f"DESCRIPTIVES VARIABLES={var_str}",
        "  /STATISTICS=MEAN STDDEV MIN MAX.",
        "",
    ]


def _gen_frequencies(vars_list: List[str]) -> List[str]:
    """Generate FREQUENCIES syntax."""
    if not vars_list:
        return []
    var_str = " ".join(vars_list)
    return [
        f"FREQUENCIES VARIABLES={var_str}",
        "  /STATISTICS=MEAN STDDEV MIN MAX.",
        "",
    ]


def _gen_t_test(dep_vars: List[str],
                  group_var: Optional[str]) -> List[str]:
    """Generate T-TEST syntax."""
    if not dep_vars or not group_var:
        return []
    var_str = " ".join(dep_vars)
    return [
        f"T-TEST GROUPS={group_var}(1,2)",
        f"  /VARIABLES={var_str}.",
        "",
    ]


def _gen_oneway(dep_vars: List[str],
                  group_var: Optional[str]) -> List[str]:
    """Generate ONEWAY syntax."""
    if not dep_vars or not group_var:
        return []
    var_str = " ".join(dep_vars)
    return [
        f"ONEWAY {var_str} BY {group_var}",
        "  /STATISTICS=DESCRIPTIVES HOMOGENEITY",
        "  /POSTHOC=TUKEY.",
        "",
    ]


def _gen_correlations(vars_list: List[str]) -> List[str]:
    """Generate CORRELATIONS syntax."""
    if len(vars_list) < 2:
        return []
    var_str = " ".join(vars_list)
    return [
        f"CORRELATIONS VARIABLES={var_str}",
        "  /MISSING=PAIRWISE",
        "  /PRINT=TWOTAIL.",
        "",
    ]


def _gen_regression(dep_vars: List[str],
                      group_var: Optional[str]) -> List[str]:
    """Generate REGRESSION syntax (first dep var as DV, rest as IVs)."""
    if len(dep_vars) < 2:
        return []
    dv = dep_vars[0]
    ivs = dep_vars[1:]
    iv_str = " ".join(ivs)
    return [
        "REGRESSION",
        f"  /DEPENDENT={dv}",
        f"  /METHOD=ENTER {iv_str}.",
        "",
    ]


def _gen_reliability(vars_list: List[str]) -> List[str]:
    """Generate RELIABILITY syntax."""
    if len(vars_list) < 2:
        return []
    var_str = " ".join(vars_list)
    return [
        "RELIABILITY",
        f"  /VARIABLES={var_str}",
        "  /SCALE(ALL) = ALL",
        "  /SUMMARY=TOTAL.",
        "",
    ]


def _gen_npar(dep_vars: List[str],
                group_var: Optional[str]) -> List[str]:
    """Generate NPAR TESTS syntax."""
    if not dep_vars:
        return []
    var_str = " ".join(dep_vars)
    lines = []
    if group_var:
        lines.append(f"NPAR TESTS")
        lines.append(f"  /M-W={var_str} BY {group_var}(1,2)")
        lines.append(f"  /KRUSKAL={var_str} BY {group_var}.")
    else:
        lines.append(f"NPAR TESTS")
        lines.append(f"  /CHISQUARE={var_str}.")
    lines.append("")
    return lines


def generate_from_simulation_logs(logs: List[Dict[str, Any]],
                                      variables: List[str],
                                      group_var: Optional[str] = None,
                                      analysis_type: str = "descriptive") -> str:
    """Generate SPSS syntax from simulation log data.

    Args:
        logs: List of simulation log entries (dicts with variable names as keys).
        variables: List of variable names to include.
        group_var: Optional grouping variable.
        analysis_type: Type of analysis.

    Returns:
        SPSS syntax string.
    """
    experiment_data = {
        "variables": variables,
        "observations": logs,
        "group_var": group_var,
    }
    return generate_syntax(experiment_data, analysis_type)