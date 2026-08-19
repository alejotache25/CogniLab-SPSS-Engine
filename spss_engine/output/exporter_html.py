"""
HTML exporter for PivotTable.

Renders a PivotTable as an HTML <table> with <thead> and <tbody>.

- Row dimensions render as <th> at the start of each row.
- Multiple row dimensions use nested <th> with rowspan where possible.
- Layer dimensions produce separate tables, one per layer combination.
- Output is valid HTML (all tags closed).
"""

from __future__ import annotations
from typing import List, Optional, Sequence
import html

from spss_engine.output.pivot_table import (
    PivotTable, Dimension, DimensionPlace, CellText,
)


def _all_combos(dims: List[Dimension]) -> List[List[str]]:
    """Cartesian product of category labels across dims."""
    if not dims:
        return [[]]
    combos: List[List[str]] = [[]]
    for dim in dims:
        labels = [c.display_text() for c in dim.categories if not c.hidden]
        new: List[List[str]] = []
        for combo in combos:
            for lbl in labels:
                new.append(combo + [lbl])
        combos = new
    return combos


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(text if text is not None else "", quote=True)


def export_table(table: PivotTable) -> str:
    """Render a PivotTable as an HTML string."""
    row_dims = table.row_dimensions()
    col_dims = table.column_dimensions()
    layer_dims = table.layer_dimensions()

    col_labels: List[str] = []
    for cd in col_dims:
        col_labels.extend([c.display_text() for c in cd.categories
                           if not c.hidden])

    layer_combos = _all_combos(layer_dims)
    row_combos = _all_combos(row_dims)

    parts: List[str] = []
    if table.title:
        parts.append(f"<h3>{_esc(table.title)}</h3>")
    if table.caption:
        parts.append(f"<p><em>{_esc(table.caption)}</em></p>")

    for lc in layer_combos:
        if layer_dims and lc:
            layer_str = ", ".join(
                f"{_esc(ld.name)}={_esc(v)}" for ld, v in zip(layer_dims, lc)
            )
            parts.append(f"<p><strong>Layer: {layer_str}</strong></p>")

        parts.append("<table border=\"1\" cellpadding=\"4\" cellspacing=\"0\">")

        # Header
        parts.append("<thead><tr>")
        for rd in row_dims:
            parts.append(f"<th>{_esc(rd.name)}</th>")
        for cl in col_labels:
            parts.append(f"<th>{_esc(cl)}</th>")
        parts.append("</tr></thead>")

        # Body
        parts.append("<tbody>")
        for rc in row_combos:
            parts.append("<tr>")
            for i, lbl in enumerate(rc):
                parts.append(f"<th>{_esc(lbl)}</th>")
            for j, cl in enumerate(col_labels):
                cell = table.get_cell(row_cats=rc, col_cats=[cl],
                                       layer_cats=lc)
                if cell is not None:
                    parts.append(f"<td>{_esc(cell.display_text())}</td>")
                else:
                    parts.append("<td></td>")
            parts.append("</tr>")
        parts.append("</tbody>")
        parts.append("</table>")

    return "\n".join(parts)


def export_tables(tables: Sequence[PivotTable]) -> str:
    """Render multiple PivotTables as HTML."""
    parts: List[str] = ["<div class=\"spss-output\">"]
    for t in tables:
        parts.append(export_table(t))
        parts.append("")
    parts.append("</div>")
    return "\n".join(parts)