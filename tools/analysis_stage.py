"""The per-inequality stage protocol shared by the analysis scripts.

Each analysis in this directory -- see-saw lower bound, NPA upper bound, NPA
detection efficiency -- exposes its per-facet work as a *stage* object:

    stage = SomeStage(scenario, args, solver)  # scenario-level setup, once
    stage.describe()                           # one line for the run header
    stage.done(table, line)                    # already in the analysis file?
    stage.analyse(parsed) -> StageResult       # the work for one facet

Everything that depends only on the scenario -- the column names, the white
noise behaviour, a compiled NPA moment matrix -- is built in the constructor,
so ``analyse`` is exactly what one facet costs and may be called in any order
relative to the other stages. That is what lets each script sweep a whole file
with its own stage, while ``analyze_panda_output.py`` runs every stage on one
facet before moving to the next.

``analyse`` returns its report lines instead of printing them, because the two
callers want different layouts: a sweep over a large file prints one line per
interesting facet, whereas the combined driver prints every stage's report
underneath the facet it belongs to. ``notable`` marks the facets that are worth
a line even in the quiet per-file sweep.

Requires: the standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StageResult:
    """What one stage computed for one inequality."""

    # Column name -> value, ready for ``AnalysisTable.update``.
    values: dict
    # Lines describing the result, without indentation or a line prefix: the
    # caller adds whichever it uses.
    report: list = field(default_factory=list)
    # Whether this facet is worth reporting in a quiet per-file sweep.
    notable: bool = True
