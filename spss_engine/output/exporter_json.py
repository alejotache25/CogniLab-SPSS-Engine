"""
JSON exporter for PivotTable.

Serializes a PivotTable (or list of PivotTables) to a structured JSON dict
suitable for consumption by a Next.js frontend.

Output structure:
{
  "tables": [
    {
      "title": "...",
      "caption": null | "...",
      "dimensions": [
        {"place": "row|column|layer", "name": "...",
         "categories": ["label1", "label2", ...]},
        ...
      ],
      "cells": [
        {"row": ["label"], "col": ["label"], "layer": [],
         "value": <number|null>, "text": "display text",
         "format": "Count" | null},
        ...
      ]
    }
  ]
}
"""

from __future__ import annotations
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Union

from spss_engine.output.pivot_table import (
    PivotTable, Dimension, Category, CellText, CellType,
    DimensionPlace, FormatSpec,
)


def _category_labels(dim: Dimension) -> List[str]:
    return [c.display_text() for c in dim.categories if not c.hidden]


def _serialize_cell(table: PivotTable,
                    key: tuple,
                    row_dims: List[Dimension],
                    col_dims: List[Dimension],
                    layer_dims: List[Dimension]) -> Dict[str, Any]:
    """Serialize a single cell at the given key."""
    cell = table.cells.get(key)
    if cell is None:
        return {}

    # Split key back into layer/row/col parts based on dim sizes
    n_layer = len(layer_dims)
    n_row = len(row_dims)
    n_col = len(col_dims)
    # Each dimension contributes one label in the key
    layer_labels = list(key[:n_layer]) if n_layer else []
    row_labels = list(key[n_layer:n_layer + n_row]) if n_row else []
    col_labels = list(key[n_layer + n_row:]) if n_col else []

    fmt_spec = table.cell_formats.get(key)
    fmt_name = fmt_spec.fmt.value if fmt_spec is not None else None

    entry: Dict[str, Any] = {
        "row": row_labels,
        "col": col_labels,
        "layer": layer_labels,
        "value": cell.raw_value(),
        "text": cell.display_text(),
        "format": fmt_name,
    }
    return entry


def export_table(table: PivotTable) -> Dict[str, Any]:
    """Serialize a single PivotTable to a JSON-compatible dict."""
    row_dims = table.row_dimensions()
    col_dims = table.column_dimensions()
    layer_dims = table.layer_dimensions()

    dims_out: List[Dict[str, Any]] = []
    for dim in (layer_dims + row_dims + col_dims):
        dims_out.append({
            "place": dim.place.value,
            "name": dim.name,
            "categories": _category_labels(dim),
        })

    cells_out: List[Dict[str, Any]] = []
    for key in table.cells:
        cells_out.append(
            _serialize_cell(table, key, row_dims, col_dims, layer_dims)
        )

    return {
        "title": table.title,
        "caption": table.caption,
        "dimensions": dims_out,
        "cells": cells_out,
        "notes": list(table.notes) if table.notes else [],
    }


def export_tables(tables: Union[PivotTable, Sequence[PivotTable]]
                  ) -> Dict[str, Any]:
    """Serialize one or more PivotTables to a JSON-compatible dict."""
    if isinstance(tables, PivotTable):
        table_list = [tables]
    else:
        table_list = list(tables)
    return {"tables": [export_table(t) for t in table_list]}


def to_json(tables: Union[PivotTable, Sequence[PivotTable]],
            indent: Optional[int] = 2) -> str:
    """Serialize PivotTable(s) to a JSON string."""
    return json.dumps(export_tables(tables), indent=indent,
                      default=_json_default, ensure_ascii=False)


def _json_default(obj: Any) -> Any:
    """Fallback JSON serializer for unusual types."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, CellText):
        return obj.raw_value()
    return str(obj)