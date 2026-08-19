"""
Text exporter for PivotTable.

Renders a PivotTable as aligned tabular text, suitable for console/logs.

Format:
                    Frequency   Percent   Valid %   Cumulative %
1 Male                  42       35.0%      42.0%         42.0%
2 Female                58       48.3%      58.0%        100.0%
Missing                  20       16.7%
Total                  120      100.0%

Multiple row dimensions are nested with indentation.
Layer dimensions produce separate sections prefixed with [Layer: value].
"""

from __future__ import annotations
import math
from typing import List, Optional, Sequence

from spss_engine.output.pivot_table import (
    PivotTable, Dimension, DimensionPlace, CellText,
)


def _all_layer_combos(layer_dims: List[Dimension]) -> List[List[str]]:
    """Cartesian product of layer category labels."""
    if not layer_dims:
        return [[]]
    combos: List[List[str]] = [[]]
    for dim in layer_dims:
        labels = [c.display_text() for c in dim.categories if not c.hidden]
        new_combos: List[List[str]] = []
        for combo in combos:
            for lbl in labels:
                new_combos.append(combo + [lbl])
        combos = new_combos
    return combos


def _all_row_combos(row_dims: List[Dimension]) -> List[List[str]]:
    """Cartesian product of row category labels (row-major)."""
    return _all_layer_combos(row_dims)


def export_table(table: PivotTable, max_width: int = 100) -> str:
    """Render a PivotTable as aligned tabular text."""
    lines: List[str] = []
    if table.title:
        lines.append(table.title)
        lines.append("")

    row_dims = table.row_dimensions()
    col_dims = table.column_dimensions()
    layer_dims = table.layer_dimensions()

    if not row_dims and not col_dims:
        # No dimensions — just dump cells
        for key, cell in table.cells.items():
            lines.append(cell.display_text())
        return "\n".join(lines)

    col_labels: List[str] = []
    for cd in col_dims:
        col_labels.extend([c.display_text() for c in cd.categories
                           if not c.hidden])

    layer_combos = _all_layer_combos(layer_dims)
    row_combos = _all_row_combos(row_dims)

    n_row_dims = len(row_dims)
    # Width for row label columns: each row dim gets its own column
    row_label_widths: List[int] = []
    for rd in row_dims:
        w = max((len(c.display_text()) for c in rd.categories if not c.hidden),
                default=8)
        w = max(w, len(rd.name))
        row_label_widths.append(max(w, 6))

    # Width for data columns
    col_widths: List[int] = []
    for i, cl in enumerate(col_labels):
        max_cell = len(cl)
        for rc in row_combos:
            for lc in layer_combos:
                cell = table.get_cell(row_cats=rc, col_cats=[cl],
                                      layer_cats=lc)
                if cell is not None:
                    max_cell = max(max_cell, len(cell.display_text()))
        col_widths.append(max(max_cell, 8))

    for lc in layer_combos:
        if layer_dims and lc:
            layer_str = ", ".join(
                f"{ld.name}={v}" for ld, v in zip(layer_dims, lc)
            )
            lines.append(f"[Layer: {layer_str}]")
            lines.append("")

        # Header row
        header_parts: List[str] = []
        for i, rd in enumerate(row_dims):
            header_parts.append(rd.name.ljust(row_label_widths[i]))
        for j, cl in enumerate(col_labels):
            header_parts.append(cl.rjust(col_widths[j]))
        header = "  ".join(header_parts)
        lines.append(header)
        lines.append("-" * min(len(header), max_width))

        # Data rows
        for rc in row_combos:
            parts: List[str] = []
            for i, lbl in enumerate(rc):
                parts.append(lbl.ljust(row_label_widths[i]))
            for j, cl in enumerate(col_labels):
                cell = table.get_cell(row_cats=rc, col_cats=[cl],
                                       layer_cats=lc)
                if cell is not None:
                    parts.append(cell.display_text().rjust(col_widths[j]))
                else:
                    parts.append("".rjust(col_widths[j]))
            lines.append("  ".join(parts))
        if layer_dims and lc:
            lines.append("")

    return "\n".join(lines)


def export_tables(tables: Sequence[PivotTable]) -> str:
    """Render multiple PivotTables as text, separated by blank lines."""
    parts: List[str] = []
    for t in tables:
        parts.append(export_table(t))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"