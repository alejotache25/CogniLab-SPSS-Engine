"""
Pivot Table Engine for the SPSS engine.

Equivalent to spss.BasePivotTable from IBM SPSS. Each table has:
  - title (str)
  - dimensions: list of Dimension (row, column, layer)
  - cells: dict keyed by tuple of category-labels → CellText

A Dimension has a place ("row", "column", "layer"), a name, and an ordered
list of Category labels.

Cells are typed via CellText (Number, String, VarName, VarValue) with an
optional FormatSpec describing how to render the value (GeneralStat, Mean,
StdDev, Percent, Count, Sig, etc.).

simple_pivot_table() is a convenience builder for the common 1-row-dim +
1-col-dim case, mirroring the py_basepivot.txt API.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import math


# ----------------------------------------------------------------------
# CellText: typed cell content
# ----------------------------------------------------------------------

class CellType(Enum):
    NUMBER = "number"
    STRING = "string"
    VAR_NAME = "varname"
    VAR_VALUE = "varvalue"


@dataclass
class CellText:
    """A typed cell value.

    For NUMBER: value is a float (may be NaN).
    For STRING: value is a str.
    For VAR_NAME: value is the variable name string.
    For VAR_VALUE: value is the value, varname is the source variable.
    """
    cell_type: CellType
    value: Any = None
    varname: Optional[str] = None
    text: Optional[str] = None  # optional display override

    @classmethod
    def number(cls, value: float, text: Optional[str] = None) -> "CellText":
        return cls(cell_type=CellType.NUMBER, value=float(value), text=text)

    @classmethod
    def string(cls, value: str) -> "CellText":
        return cls(cell_type=CellType.STRING, value=str(value))

    @classmethod
    def var_name(cls, name: str) -> "CellText":
        return cls(cell_type=CellType.VAR_NAME, value=str(name))

    @classmethod
    def var_value(cls, varname: str, value: Any) -> "CellText":
        return cls(cell_type=CellType.VAR_VALUE, value=value, varname=varname)

    def display_text(self) -> str:
        """Human-readable text for this cell."""
        if self.text is not None:
            return self.text
        if self.cell_type == CellType.NUMBER:
            v = self.value
            if v is None:
                return ""
            try:
                f = float(v)
            except (TypeError, ValueError):
                return str(v)
            if math.isnan(f):
                return "."
            # Format: trim trailing zeros, keep reasonable precision
            if f == int(f) and abs(f) < 1e15:
                return f"{int(f)}"
            return f"{f:.4f}".rstrip("0").rstrip(".")
        if self.cell_type == CellType.STRING:
            return str(self.value) if self.value is not None else ""
        if self.cell_type == CellType.VAR_NAME:
            return str(self.value) if self.value is not None else ""
        if self.cell_type == CellType.VAR_VALUE:
            return str(self.value) if self.value is not None else ""
        return ""

    def raw_value(self) -> Any:
        """Return the raw underlying value (for JSON export)."""
        if self.cell_type == CellType.NUMBER:
            v = self.value
            if v is None:
                return None
            try:
                f = float(v)
            except (TypeError, ValueError):
                return v
            if math.isnan(f):
                return None
            return f
        return self.value

    def __repr__(self) -> str:
        return f"CellText({self.cell_type.name}, {self.value!r})"


# ----------------------------------------------------------------------
# FormatSpec: how to render a numeric cell
# ----------------------------------------------------------------------

class FormatType(Enum):
    GENERAL_STAT = "GeneralStat"
    COUNT = "Count"
    PERCENT = "Percent"
    MEAN = "Mean"
    STD_DEV = "StdDev"
    VARIANCE = "Variance"
    MIN = "Minimum"
    MAX = "Maximum"
    RANGE = "Range"
    SUM = "Sum"
    SEMEAN = "SEMean"
    SKEWNESS = "Skewness"
    SESKEW = "SESkew"
    KURTOSIS = "Kurtosis"
    SEKURT = "SEKurt"
    MEDIAN = "Median"
    MODE = "Mode"
    T_STAT = "tstat"
    DF = "df"
    SIG = "Sig"
    CI_LOWER = "CILower"
    CI_UPPER = "CIUpper"
    F_STAT = "Fstat"
    CHISQ = "ChiSquare"
    PHI = "Phi"
    CC = "CC"
    N = "N"
    TEXT = "text"


@dataclass
class FormatSpec:
    fmt: FormatType = FormatType.GENERAL_STAT
    decimals: Optional[int] = None


# ----------------------------------------------------------------------
# Dimension and Category
# ----------------------------------------------------------------------

class DimensionPlace(Enum):
    ROW = "row"
    COLUMN = "column"
    LAYER = "layer"


@dataclass
class Category:
    """A category label within a dimension."""
    label: CellText
    hidden: bool = False

    @classmethod
    def from_string(cls, label: str) -> "Category":
        return cls(label=CellText.string(label))

    @classmethod
    def from_number(cls, value: float) -> "Category":
        return cls(label=CellText.number(value))

    @classmethod
    def from_var_name(cls, name: str) -> "Category":
        return cls(label=CellText.var_name(name))

    def display_text(self) -> str:
        return self.label.display_text()


@dataclass
class Dimension:
    """A pivot table dimension (row, column, or layer)."""
    place: DimensionPlace
    name: str
    categories: List[Category] = field(default_factory=list)
    is_label_dimension: bool = False
    hide_name: bool = False
    hide_categories: bool = False

    def add_category(self, cat: Category) -> "Dimension":
        self.categories.append(cat)
        return self

    def add_categories(self, labels: Sequence[str]) -> "Dimension":
        for lbl in labels:
            self.categories.append(Category.from_string(lbl))
        return self

    def category_labels(self) -> List[str]:
        return [c.display_text() for c in self.categories if not c.hidden]


# ----------------------------------------------------------------------
# PivotTable
# ----------------------------------------------------------------------

class PivotTable:
    """A multi-dimensional pivot table.

    Cells are stored in a dict keyed by a tuple of category display-texts.
    The key ordering follows the table's dimensions: layer cats first, then
    row cats, then column cats.
    """

    def __init__(self, title: str = "",
                 caption: Optional[str] = None) -> None:
        self.title: str = title
        self.caption: Optional[str] = caption
        self.dimensions: List[Dimension] = []
        self.cells: Dict[Tuple[str, ...], CellText] = {}
        self.cell_formats: Dict[Tuple[str, ...], FormatSpec] = {}
        self.notes: List[str] = []

    # ------------------------------------------------------------------
    # Dimension management
    # ------------------------------------------------------------------

    def append_dimension(self, place: DimensionPlace,
                          name: str) -> Dimension:
        """Append a new dimension and return it."""
        dim = Dimension(place=place, name=name)
        self.dimensions.append(dim)
        return dim

    def append(self, place: DimensionPlace, name: str) -> Dimension:
        """Alias for append_dimension (BasePivotTable API)."""
        return self.append_dimension(place, name)

    def get_dimension(self, name: str) -> Optional[Dimension]:
        for d in self.dimensions:
            if d.name == name:
                return d
        return None

    def set_categories(self, dim_name: str,
                       categories: List[Category]) -> None:
        """Define the categories for a dimension."""
        dim = self.get_dimension(dim_name)
        if dim is None:
            raise ValueError(f"Dimension not found: {dim_name}")
        dim.categories = list(categories)

    def set_category(self, dim_name: str, index: int,
                       category: Category) -> None:
        """Set a single category by index."""
        dim = self.get_dimension(dim_name)
        if dim is None:
            raise ValueError(f"Dimension not found: {dim_name}")
        while len(dim.categories) <= index:
            dim.categories.append(Category.from_string(""))
        dim.categories[index] = category

    # ------------------------------------------------------------------
    # Cell access
    # ------------------------------------------------------------------

    def _cell_key(self, row_cats: Sequence[str],
                  col_cats: Sequence[str],
                  layer_cats: Sequence[str] = ()) -> Tuple[str, ...]:
        """Build the cell key tuple following dimension order."""
        # Order: layers, rows, columns
        return tuple(layer_cats) + tuple(row_cats) + tuple(col_cats)

    def set_cell(self, value: Union[CellText, float, str, None],
                  row_cats: Sequence[str] = (),
                  col_cats: Sequence[str] = (),
                  layer_cats: Sequence[str] = (),
                  fmt: Optional[FormatSpec] = None) -> None:
        """Set a cell value at the given category coordinates."""
        key = self._cell_key(row_cats, col_cats, layer_cats)
        if isinstance(value, CellText):
            self.cells[key] = value
        elif isinstance(value, (int, float)):
            self.cells[key] = CellText.number(float(value))
        elif isinstance(value, str):
            self.cells[key] = CellText.string(value)
        elif value is None:
            self.cells[key] = CellText.number(float("nan"))
        else:
            self.cells[key] = CellText.string(str(value))
        if fmt is not None:
            self.cell_formats[key] = fmt

    def get_cell(self, row_cats: Sequence[str] = (),
                  col_cats: Sequence[str] = (),
                  layer_cats: Sequence[str] = ()) -> Optional[CellText]:
        """Get a cell value at the given category coordinates."""
        key = self._cell_key(row_cats, col_cats, layer_cats)
        return self.cells.get(key)

    def get_cell_value(self, row_cats: Sequence[str] = (),
                        col_cats: Sequence[str] = (),
                        layer_cats: Sequence[str] = ()) -> Any:
        """Get the raw cell value (or NaN)."""
        c = self.get_cell(row_cats, col_cats, layer_cats)
        if c is None:
            return float("nan")
        return c.raw_value()

    # ------------------------------------------------------------------
    # simple_pivot_table: convenience builder
    # ------------------------------------------------------------------

    def simple_pivot_table(self,
                            rowdim: str = " ",
                            rowlabels: Optional[Sequence[str]] = None,
                            coldim: str = " ",
                            collabels: Optional[Sequence[str]] = None,
                            cells: Optional[Sequence[Any]] = None,
                            rowcats: Optional[Sequence[Category]] = None,
                            colcats: Optional[Sequence[Category]] = None,
                            fmt: Optional[FormatSpec] = None) -> None:
        """Build a simple 1-row-dim + 1-col-dim table.

        cells is a flat list of values in row-major order:
          cells[row0_col0, row0_col1, ..., row1_col0, ...]
        """
        row_labels: List[str] = list(rowlabels) if rowlabels else []
        col_labels: List[str] = list(collabels) if collabels else []
        row_cats: List[Category] = (
            list(rowcats) if rowcats
            else [Category.from_string(lbl) for lbl in row_labels]
        )
        col_cats: List[Category] = (
            list(colcats) if colcats
            else [Category.from_string(lbl) for lbl in col_labels]
        )

        # Set up dimensions (replace existing row/col dims of same name)
        self.dimensions = [
            d for d in self.dimensions
            if d.place == DimensionPlace.LAYER
        ]
        row_dim = Dimension(place=DimensionPlace.ROW, name=rowdim,
                             categories=row_cats)
        col_dim = Dimension(place=DimensionPlace.COLUMN, name=coldim,
                             categories=col_cats)
        self.dimensions.append(row_dim)
        self.dimensions.append(col_dim)

        # Fill cells
        if cells is not None:
            n_rows = len(row_cats)
            n_cols = len(col_cats)
            for i, val in enumerate(cells):
                r = i // n_cols
                c = i % n_cols
                if r >= n_rows:
                    break
                rlabel = row_cats[r].display_text()
                clabel = col_cats[c].display_text()
                self.set_cell(val, row_cats=[rlabel],
                              col_cats=[clabel], fmt=fmt)

    # ------------------------------------------------------------------
    # Dimension accessors
    # ------------------------------------------------------------------

    def row_dimensions(self) -> List[Dimension]:
        return [d for d in self.dimensions
                if d.place == DimensionPlace.ROW]

    def column_dimensions(self) -> List[Dimension]:
        return [d for d in self.dimensions
                if d.place == DimensionPlace.COLUMN]

    def layer_dimensions(self) -> List[Dimension]:
        return [d for d in self.dimensions
                if d.place == DimensionPlace.LAYER]

    def __repr__(self) -> str:
        return (f"PivotTable(title={self.title!r}, "
                f"dims={len(self.dimensions)}, cells={len(self.cells)})")