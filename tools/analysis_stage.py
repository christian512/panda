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

``analyse_facets`` drives that protocol over a whole file, on one core or
several. It is the only place that knows about processes, so each script keeps
its own printing and its own table.

Requires: the standard library only (numpy only reaches here through stages).
"""

from __future__ import annotations

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace


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


# --------------------------------------------------------------------------- #
# Running the stages over a file, on one core or several
# --------------------------------------------------------------------------- #
@dataclass
class FacetOutcome:
    """What happened to one facet: the results, what was skipped, what failed."""

    parsed: object
    results: dict  # stage name -> StageResult, for the stages that ran
    skipped: tuple = ()  # stage names whose columns the row already carried
    failures: tuple = ()  # (stage name, message) for the stages that raised
    seconds: float = 0.0

    def values(self) -> dict:
        """Every column this facet produced, merged across its stages."""
        merged: dict = {}
        for result in self.results.values():
            merged.update(result.values)
        return merged


def resolve_jobs(jobs) -> int:
    """Worker count from a ``--jobs`` value; 0 or less means every usable core.

    ``sched_getaffinity`` rather than ``cpu_count``: on a machine where the
    process is pinned to a subset of the cores (a cgroup, a batch scheduler),
    that subset is what is actually available.
    """
    if jobs is None:
        return 1
    if jobs > 0:
        return jobs
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


# Set in the parent before forking, read by the workers it forks. Fork rather
# than spawn so that a worker inherits the stages ready-made: the efficiency
# stage's NPA problem is compiled once in the parent instead of once per worker,
# and no module is imported a second time.
_WORKER_STAGES: dict = {}


def _analyse_one(stages, parsed, todo):
    """Run the named stages on one facet, collecting results and failures."""
    results, failures = {}, []
    started = time.time()
    for name in todo:
        try:
            results[name] = stages[name].analyse(parsed)
        except KeyboardInterrupt:
            raise
        except (SystemExit, Exception) as error:  # noqa: BLE001
            # One bad facet must not end a sweep of thousands; the caller
            # reports the failure and the remaining stages still run.
            failures.append((name, str(error) or error.__class__.__name__))
    return FacetOutcome(parsed, results, (), tuple(failures), time.time() - started)


def _worker(payload):
    """Entry point in a forked worker: one facet, the stages it still needs."""
    parsed, todo = payload
    return _analyse_one(_WORKER_STAGES, parsed, todo)


def analyse_facets(facets, stages, table, jobs=1, overwrite=False):
    """Yield a ``FacetOutcome`` per facet, ``jobs`` facets at a time.

    Facets are independent -- a stage reads the analysis table only to decide
    whether its columns are already there, and that is settled here in the
    parent before anything runs -- so the work parallelises facet by facet,
    which keeps each facet's stages together and needs no locking. The caller
    remains the only writer of the table.

    With ``jobs == 1`` the stages run inline, in file order: the same serial
    loop this replaces, with no processes involved. Above that, outcomes are
    yielded as they finish, so they no longer arrive in file order -- each one
    carries its facet, and the table sorts itself by line number on save.

    A stage's own state does not survive the worker that produced it (each
    process mutates its own copy), so anything a caller wants to count has to
    be counted from the outcomes here.
    """
    by_name = {stage.name: stage for stage in stages}

    def plan(parsed):
        """The stages this facet still needs, in the order they were given."""
        return tuple(
            stage.name
            for stage in stages
            if overwrite or not stage.done(table, parsed.line_number)
        )

    if jobs <= 1:
        for parsed in facets:
            todo = plan(parsed)
            outcome = _analyse_one(by_name, parsed, todo)
            yield replace(
                outcome,
                skipped=tuple(name for name in by_name if name not in todo),
            )
        return

    global _WORKER_STAGES
    _WORKER_STAGES = by_name
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=context) as pool:
        pending = {}
        for parsed in facets:
            todo = plan(parsed)
            pending[pool.submit(_worker, (parsed, todo))] = tuple(
                name for name in by_name if name not in todo
            )
        try:
            for future in as_completed(pending):
                yield replace(future.result(), skipped=pending[future])
        except KeyboardInterrupt:
            # Without this the pool waits for every queued facet to finish.
            for future in pending:
                future.cancel()
            raise
