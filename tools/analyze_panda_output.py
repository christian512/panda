"""One-command analysis of a panda output file of full-probability Bell facets.

Runs the three per-facet analyses in this directory over a single panda file
and collects everything in that file's one analysis CSV:

    1. seesaw     see-saw *lower* bound on the quantum value at a fixed local
                  dimension, plus the critical white-noise visibility
                  (``lower_bound_q_seesaw_full_probability.py``)
    2. npa        NPA *upper* bound on the quantum value, over all dimensions
                  (``upper_bound_q_npa_full_probability.py``)
    3. efficiency minimum symmetric detection efficiency, minimised over the
                  outcome liftings that accommodate a no-click outcome
                  (``detection_efficiency_npa.py``)

The loop is facet-outer, stage-inner: one facet is put through all three
analyses before the next facet is read. Each stage is the ``analyse`` method of
its own script (see ``analysis_stage.py``), so the numbers, the columns and the
resume-on-rerun behaviour are exactly what running the three scripts by hand
produces. Nothing here re-implements an analysis; this is the driver they were
missing.

Why facet by facet. A facet is finished when the driver moves on, so the CSV
fills up as complete rows instead of three separate sweeps, each of which would
have to traverse the whole file before the next one starts. On a file too large
to analyse exhaustively that is the difference between a prefix of fully
analysed facets and a full file with only its cheapest column. It also puts the
see-saw lower bound and the NPA upper bound side by side while the facet is in
front of you, which is what makes a loose bracket -- too small a see-saw
dimension, too low an NPA level -- visible immediately rather than after the
second sweep finishes.

The stages still run cheapest first *within* a facet, so a facet's expensive
efficiency search happens only after its two bounds are in the table. The table
is saved after every stage of every facet, so an interrupted run resumes where
it stopped (without ``--overwrite``, stages whose columns a row already carries
are skipped individually).

A stage that fails on a facet does not abort the run: the failure is reported,
the remaining stages and facets still run, and the exit status is non-zero at
the end. On a multi-day sweep, losing everything because one facet tripped the
see-saw would be the worse outcome.

Requires: numpy, cvxpy. MOSEK (the default solver) is markedly faster and is
picked up from a license at ``~/mosek/mosek.lic``.

Usage:
    python tools/analyze_panda_output.py results/bell_full_probability/2222

    # a large file: 200 facets drawn at random, cheap NPA levels, fewer
    # restarts, coarser bisection
    python tools/analyze_panda_output.py results/bell/3333_129M.out.gz \
        --limit 200 --randomize 7 \
        --npa-level 1+AB --efficiency-level 1+AB --tries 3 \
        --efficiency-precision 0.01

    # skip the (expensive) efficiency stage, or run only it
    python tools/analyze_panda_output.py results/bell_full_probability/3322 \
        --skip efficiency
    python tools/analyze_panda_output.py results/bell_full_probability/3322 \
        --only efficiency
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_stage import analyse_facets, resolve_jobs  # noqa: E402
from analysis_table import AnalysisTable  # noqa: E402
from detection_efficiency_npa import (  # noqa: E402
    DEFAULT_TOLERANCE,
    EfficiencyStage,
)
from lower_bound_q_seesaw_full_probability import (  # noqa: E402
    SOLVERS,
    SeesawStage,
    default_out_path,
    load_inequalities,
    resolve_scenario,
)
from upper_bound_q_npa_full_probability import Level, NPAStage  # noqa: E402

# Cheapest first; the efficiency search is the expensive one and goes last, so
# that a facet's two bounds are recorded before it is attempted.
STAGES = ("seesaw", "npa", "efficiency")


# --------------------------------------------------------------------------- #
# Building the stages
# --------------------------------------------------------------------------- #
# Each stage reads its options off an argparse namespace of its own script's
# making. Rather than teach those scripts about a shared parser, this script
# owns one flat command line and hands each stage the namespace it expects. The
# shared options (--limit, --solver, --out, --overwrite) are handled here in the
# driver, since the per-facet loop is the driver's own.
def build_stage(name, scenario, args, solver):
    if name == "seesaw":
        return SeesawStage(
            scenario,
            Namespace(
                tries=args.tries,
                seed=args.seed,
                precision=args.seesaw_precision,
                max_iter=args.max_iter,
                verbose=args.verbose,
                tolerance=args.tolerance,
                column="seesaw_lower_bound",
                noise_column="white_noise_visibility",
            ),
            solver,
        )
    if name == "npa":
        return NPAStage(
            scenario,
            Namespace(
                level=args.npa_level,
                tolerance=args.tolerance,
                column="npa_upper_bound",
            ),
            solver,
        )
    if name == "efficiency":
        return EfficiencyStage(
            scenario,
            Namespace(
                level=args.efficiency_level,
                start=args.efficiency_start,
                precision=args.efficiency_precision,
                # The efficiency search needs slack that exceeds solver
                # accuracy, so it keeps its own default rather than the
                # tolerance the bounds use for their violated / not violated
                # verdict.
                tolerance=args.efficiency_tolerance,
                candidates=args.efficiency_candidates,
                column="detection_efficiency_npa_lifted",
            ),
            solver,
        )
    raise ValueError(f"unknown stage {name!r}")  # unreachable: argparse restricts


def bracket_line(table, line, seesaw, npa):
    """The quantum value's bracket, once both bounds are known for a facet.

    Returns ``None`` unless the row carries a see-saw lower bound and an NPA
    upper bound -- either from this run or from an earlier one. A wide bracket
    is the signal to raise ``--dim`` or ``--npa-level``, and it is only worth
    printing where both numbers are in hand.
    """
    if seesaw is None or npa is None:
        return None
    try:
        lower = float(table.get(line, seesaw.column))
        upper = float(table.get(line, npa.column))
    except ValueError:  # missing or non-numeric ("n/a"), nothing to bracket
        return None
    return (
        f"quantum value in [{lower:.9f}, {upper:.9f}]   "
        f"bracket width = {upper - lower:.2e}"
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_file(path, args, solver):
    """Analyse every facet in ``path`` with every selected stage, facet first."""
    selection = load_inequalities(path, args.limit, args.randomize)
    inequalities = selection.inequalities

    # One scenario for all three stages. The efficiency search ignores the
    # local dimension (it relaxes over all of them), so --dim reaching it is
    # harmless and keeps every stage reading the same scenario.
    scenario = resolve_scenario(path, inequalities, args.dim)
    out_path = Path(args.out) if args.out else default_out_path(path)
    table = AnalysisTable.load(out_path)
    jobs = resolve_jobs(args.jobs)

    # Scenario-level setup happens once per stage, not once per facet: the
    # efficiency stage compiles its NPA problem here, and the workers inherit
    # it through fork rather than each compiling its own.
    stages = [build_stage(name, scenario, args, solver) for name in args.stages]
    by_name = {stage.name: stage for stage in stages}
    for stage in stages:
        table.ensure_columns(stage.columns)

    print(
        f"{path}: {selection.label}, "
        f"scenario {scenario.label} (settings then outcomes), "
        f"local dimension {scenario.local_dim}"
        + (f", {jobs} facets at a time" if jobs > 1 else "")
    )
    for stage in stages:
        print(f"  {stage.name}: {stage.describe()}")
    print(f"writing analysis to {out_path}\n")

    started = time.time()
    failures = []  # (line number, stage name, message)
    completed = 0
    threshold_found = no_threshold = 0

    for count, outcome in enumerate(
        analyse_facets(inequalities, stages, table, jobs, args.overwrite), start=1
    ):
        parsed = outcome.parsed
        print(
            f"[{count}/{len(inequalities)}] line {parsed.line_number}: {parsed.text}"
        )
        for name in args.stages:
            if name in outcome.skipped:
                print(f"  {name}: already present, skipping")
                continue
            print(f"  {name}:")
            for line in outcome.results[name].report if name in outcome.results else ():
                print(f"    {line}")
        for name, message in outcome.failures:
            print(f"  {name}: FAILED: {message}")
            failures.append((parsed.line_number, name, message))

        if "efficiency" in outcome.results:
            if outcome.results["efficiency"].notable:
                threshold_found += 1
            else:
                no_threshold += 1

        if outcome.results:
            table.update(
                parsed.line_number,
                inequality=parsed.text,
                rhs=parsed.rhs,
                values=outcome.values(),
            )
            table.save(out_path)  # save as we go: long runs stay resumable

        bracket = bracket_line(
            table, parsed.line_number, by_name.get("seesaw"), by_name.get("npa")
        )
        if bracket:
            print(f"  {bracket}")
        print(f"  ({outcome.seconds:.1f}s)\n")
        completed += not outcome.failures

    print("=" * 78)
    print(
        f"{completed}/{len(inequalities)} facets analysed without error in "
        f"{time.time() - started:.1f}s -> {out_path}"
    )
    if "efficiency" in by_name:
        print(
            f"detection efficiency: {threshold_found} facets with a threshold, "
            f"{no_threshold} with no violation at eta = {args.efficiency_start:g}"
        )
    for line, name, message in failures:
        print(f"  failed: line {line}, {name}: {message}")
    return 1 if failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the see-saw lower bound, the NPA upper bound and the NPA "
            "detection efficiency on each facet of one panda output file in "
            "turn, writing all of them into that file's analysis CSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "file",
        help="panda output file, e.g. results/bell_full_probability/2222 (.gz is fine)",
    )

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--only",
        nargs="+",
        choices=STAGES,
        default=None,
        help="run only these stages",
    )
    selection.add_argument(
        "--skip",
        nargs="+",
        choices=STAGES,
        default=(),
        help="run everything except these stages",
    )

    shared = parser.add_argument_group("shared")
    shared.add_argument(
        "--limit", type=int, default=None, help="only process the first N inequalities"
    )
    shared.add_argument(
        "--randomize",
        type=int,
        default=None,
        metavar="SEED",
        help=(
            "draw the --limit inequalities uniformly at random from the whole "
            "file instead of taking the first ones; the same seed and --limit "
            "select the same facets in every analysis script, and this seed is "
            "independent of --seed, which drives the see-saw's restarts"
        ),
    )
    shared.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        metavar="N",
        help=(
            "analyse N facets at a time in separate processes; 0 or less uses "
            "every core available to this process. A facet still runs its "
            "stages in order, one worker per facet, so the results are "
            "identical to a serial run; they are printed as they finish "
            "rather than in file order (default: 1)"
        ),
    )
    shared.add_argument(
        "--solver", choices=sorted(SOLVERS), default="mosek", help="SDP solver"
    )
    shared.add_argument(
        "--out",
        default=None,
        help="analysis file (default: <dir>/analysis/<name>.csv next to the input)",
    )
    shared.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute stages whose columns a row already has a value in",
    )
    shared.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="slack before a bound counts as violating the classical bound",
    )

    seesaw = parser.add_argument_group("see-saw lower bound")
    seesaw.add_argument(
        "--dim",
        type=int,
        default=None,
        help="local Hilbert space dimension per party (default: max outcome count)",
    )
    seesaw.add_argument(
        "--tries", type=int, default=5, help="random restarts per inequality"
    )
    seesaw.add_argument(
        "--seesaw-precision",
        type=float,
        default=1e-9,
        help="see-saw convergence threshold",
    )
    seesaw.add_argument(
        "--max-iter", type=int, default=500, help="iteration cap per see-saw run"
    )
    seesaw.add_argument("--seed", type=int, default=None, help="random seed")
    seesaw.add_argument(
        "--verbose", action="store_true", help="print the value of every see-saw restart"
    )

    npa = parser.add_argument_group("NPA upper bound")
    npa.add_argument(
        "--npa-level",
        default="2",
        help=(
            "NPA level: an integer depth, or an intermediate level such as "
            "'1+AB' (level 1 plus the products A_{a|x} B_{b|y}), which is much "
            "cheaper than level 2 and usually nearly as tight"
        ),
    )

    efficiency = parser.add_argument_group("detection efficiency")
    efficiency.add_argument(
        "--efficiency-level",
        default="1+AB",
        help="NPA level for the efficiency search (see --npa-level)",
    )
    efficiency.add_argument(
        "--efficiency-start",
        type=float,
        default=0.99,
        help="upper end of the eta search; facets not violated here are skipped",
    )
    efficiency.add_argument(
        "--efficiency-candidates",
        type=int,
        default=None,
        metavar="N",
        help=(
            "search only the N liftings of lowest no-click coefficient sum, "
            "the heuristic of Cope & Colbeck (arXiv:1812.10017), who use "
            "N = 10; gives an upper bound on the minimum over all liftings, in "
            "its own column (default: search every lifting)"
        ),
    )
    efficiency.add_argument(
        "--efficiency-precision",
        type=float,
        default=0.001,
        help="bisection width on eta",
    )
    efficiency.add_argument(
        "--efficiency-tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="slack above rhs before a value counts as violating, in the eta search",
    )

    args = parser.parse_args(argv)

    args.stages = [
        stage for stage in (args.only or STAGES) if stage not in args.skip
    ]
    if not args.stages:
        parser.error("every stage was skipped; nothing to do")

    # Parse the levels up front so a malformed spec is a clean CLI error rather
    # than a traceback hours into a run.
    for option in ("npa_level", "efficiency_level"):
        try:
            setattr(args, option, Level.parse(getattr(args, option)))
        except ValueError as error:
            parser.error(f"--{option.replace('_', '-')}: {error}")

    # Every stage reads the same file, so an unreadable one is a CLI error here
    # rather than a failure reported once per facet below.
    if not Path(args.file).is_file():
        parser.error(f"no such panda output file: {args.file}")

    return run_file(args.file, args, SOLVERS[args.solver])


if __name__ == "__main__":
    raise SystemExit(main())
