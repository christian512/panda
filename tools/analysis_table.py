"""An extensible per-inequality analysis table for panda output files.

Each panda output file may have a companion *analysis* file that records
computed quantities for its inequalities -- one row per inequality, keyed by the
line number of that inequality in the panda file. Different scripts contribute
different columns:

    line,inequality,rhs,seesaw_lower_bound,seesaw_lower_bound_dim,npa_upper_bound
    11,pAB01x1y0 +pAB10x0y1 ... <= 1,1.0,1.207106781,2,1.207106782

The format is deliberately open: ``update`` takes an arbitrary ``values`` dict,
and the column set is the union of whatever has been written. A script adding a
new quantity (an NPA upper bound, a local bound, a runtime) just writes its own
column name; existing columns and rows are preserved, so several scripts can
enrich the same file incrementally and in any order.

``line`` is the key. ``inequality`` and ``rhs`` are written once, by whichever
script gets there first, and are not overwritten by later writers -- they
describe the inequality itself rather than any one analysis of it.

Requires: the standard library only.
"""

from __future__ import annotations

import csv
from pathlib import Path

# Columns describing the inequality itself, written once and never overwritten.
KEY_COLUMN = "line"
DESCRIPTIVE_COLUMNS = ("inequality", "rhs")


class AnalysisTable:
    """Rows keyed by panda line number, with a union-of-columns CSV layout."""

    def __init__(self, rows=None, columns=None):
        self.rows: dict[int, dict[str, str]] = rows or {}
        # Column order is preserved so that a file's layout stays stable.
        self.columns: list[str] = list(columns or [KEY_COLUMN, *DESCRIPTIVE_COLUMNS])

    # ----------------------------------------------------------------------- #
    # Loading and saving
    # ----------------------------------------------------------------------- #
    @classmethod
    def load(cls, path) -> "AnalysisTable":
        """Read an existing analysis file, or return an empty table if absent."""
        path = Path(path)
        if not path.exists():
            return cls()

        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns = list(reader.fieldnames or [KEY_COLUMN, *DESCRIPTIVE_COLUMNS])
            rows = {}
            for row in reader:
                key = row.get(KEY_COLUMN)
                if key is None or not key.strip():
                    continue
                rows[int(key)] = {k: (v or "") for k, v in row.items() if k is not None}
        return cls(rows, columns)

    def save(self, path) -> None:
        """Write the table, creating parent directories as needed.

        Rows are ordered by line number and every row is padded to the full
        column set, so a file stays readable by plain ``csv.DictReader`` even
        after several scripts have added columns at different times.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns)
            writer.writeheader()
            for key in sorted(self.rows):
                row = self.rows[key]
                writer.writerow({column: row.get(column, "") for column in self.columns})

    # ----------------------------------------------------------------------- #
    # Reading and writing values
    # ----------------------------------------------------------------------- #
    def ensure_columns(self, names) -> None:
        """Register columns now, so their order does not depend on the rows.

        Columns are otherwise appended the first time some row carries one,
        which makes the layout depend on which facet happens to be written
        first -- and with facets analysed in parallel, that is whichever
        finished first. Declaring a stage's columns before the run fixes the
        order for every run over the same file.
        """
        for name in names:
            if name not in self.columns:
                self.columns.append(name)

    def has_value(self, line: int, column: str) -> bool:
        """Whether ``column`` already holds a non-empty value for this line."""
        return bool(self.rows.get(line, {}).get(column, "").strip())

    def get(self, line: int, column: str, default=""):
        return self.rows.get(line, {}).get(column, default)

    def update(self, line: int, values: dict, inequality=None, rhs=None) -> None:
        """Merge ``values`` into the row for ``line``, adding columns as needed.

        ``inequality`` and ``rhs`` are filled in only when the row does not
        already carry them, so re-running one analysis never rewrites the
        inequality text recorded by another.
        """
        row = self.rows.setdefault(line, {KEY_COLUMN: str(line)})
        row[KEY_COLUMN] = str(line)

        if inequality is not None and not row.get("inequality", "").strip():
            row["inequality"] = inequality
        if rhs is not None and not row.get("rhs", "").strip():
            row["rhs"] = str(rhs)

        for column, value in values.items():
            row[column] = "" if value is None else str(value)

        for column in (*DESCRIPTIVE_COLUMNS, *values):
            if column not in self.columns:
                self.columns.append(column)
