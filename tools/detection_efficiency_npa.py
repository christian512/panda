"""Minimum detection efficiency per facet, certified with the NPA hierarchy.

Reads panda output files in the *full probability* parameterization and, for
each facet, brackets the smallest symmetric detection efficiency ``eta`` at
which the inequality can still certify non-locality, minimised over every
outcome lifting that accommodates the no-click outcome.

NPA rather than a see-saw, because of what a negative answer means. A see-saw
searches *inside* the quantum set at a fixed local dimension, so "no violation
found at eta" may only mean the search landed in a local optimum. NPA relaxes
*outward*: if the relaxation admits no violation at ``eta``, then no quantum
strategy whatsoever violates there, at any dimension. That turns every "no"
into a proof and is the method Cope & Colbeck use (arXiv:1812.10017, Sec. V B).

The pipeline per facet
----------------------
1. Load the facet ``sum gamma p <= rhs`` exactly as panda wrote it, in its own
   scenario ``(m_A, m_B, n_A, n_B)``. No transformation is applied. The quantum
   correlations live in this scenario, with all ``n`` outcomes clicking.

2. Outcome-lift it to ``(m_A, m_B, n_A + 1, n_B + 1)``: for each party and each
   of its settings, the new no-click outcome copies the coefficients of one
   existing outcome. There are ``n_A^m_A n_B^m_B`` such liftings (16 for 2222,
   81 for 2233, 729 for 3333); all are enumerated. The bound stays ``rhs``.

3. Form the inefficient correlation by attaching the failure outcome to the
   genuine correlations with efficiency ``eta``, and fold it into the lifted
   coefficients. Maximise over NPA level ``k`` moment matrices in the original
   scenario. A maximum of at most ``rhs`` proves no quantum violation at that
   ``eta``; above ``rhs`` exhibits one.

4. Report the smallest threshold over all liftings.

Why copying, not an empty outcome
---------------------------------
An outcome lifting (Pironio, J. Math. Phys. 46, 062112 (2005)) keeps the local
bound and turns a facet into a facet of the larger scenario: any local strategy
answering no-click scores exactly what it would by answering the copied outcome
instead. Appending a no-click outcome with *zero* coefficients does not: a local
strategy can answer no-click precisely where the facet has negative
coefficients and exceed the bound (CHSH in 2222 goes from local bound 0 to 1).
The resulting functional is violated at every efficiency, including below the
no-signalling floor.

Evaluated on an inefficient correlation, a copy lifting scores the same as the
original facet on the behaviour in which no-click events are binned into the
copied outcome. Enumerating the liftings is therefore enumerating the binning
strategies, per setting.

Finding the minimum over liftings
---------------------------------
Bisecting every lifting separately would cost ``n^(m_A + m_B)`` bisections. It
is not needed: once some lifting is known to violate at ``eta_best``, another
lifting can only lower the minimum if it violates at ``eta_best - precision``,
and a single SDP settles that. Only liftings passing that test are bisected, so
a facet costs roughly one SDP per lifting plus a handful of bisections. The
result is the same minimum, to the requested precision, as bisecting them all.

What the number means
---------------------
The relaxation's maximum upper-bounds the true quantum maximum, so a maximum at
or below ``rhs`` rules out every quantum strategy: the lower end of the bracket
is a certified *lower* bound on the quantum threshold of the lifting. A maximum
above ``rhs`` need not be attained by an actual quantum behaviour, so the true
threshold may lie above the reported value; only raising the NPA level can
close that gap. Values within a small tolerance of ``rhs`` count as local, so
that solver noise cannot masquerade as non-locality -- which also means
violations smaller than the tolerance, as occur just above the floor, are not
seen.

The floor
---------
The bisection is bracketed below by the no-signalling bound of Massar & Pironio
(PRA 68, 062109 (2003)), ``(m_A + m_B - 2)/(m_A m_B - 1)``, below which an
explicit local model reproduces any inefficient no-signalling distribution. It
depends only on the number of inputs: 2/3 for two settings a side, 1/2 for
three. A violation reported at or below it is proof of a bug, not a discovery,
and is checked for rather than assumed against.

Requires: numpy, cvxpy. MOSEK is used by default and is picked up from a
license at ``~/mosek/mosek.lic``.

Usage:
    python tools/detection_efficiency_npa.py results/bell_full_probability/2222
    python tools/detection_efficiency_npa.py results/bell_full_probability/3333 \
        --level 1+AB --limit 100 --precision 0.01 --start 0.95
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cvxpy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_stage import StageResult  # noqa: E402
from analysis_table import AnalysisTable  # noqa: E402
from upper_bound_q_npa_full_probability import (  # noqa: E402
    SOLVERS,
    IDENTITY,
    Level,
    _probability_coefficients,
    moment_structure,
)
from lower_bound_q_seesaw_full_probability import (  # noqa: E402
    Scenario,
    default_out_path,
    read_inequalities,
    resolve_scenario,
)

Array = np.ndarray

# Slack above ``rhs`` before a value counts as a violation, so that solver
# precision cannot turn a local point into a spurious one. It must exceed the
# solver's accuracy (MOSEK is good to roughly 1e-8 here). The cost is that a
# genuine but tiny violation may be missed, which raises the reported
# efficiency: CHSH, whose threshold approaches 2/3, comes out near 0.6686.
DEFAULT_TOLERANCE = 1e-6


# --------------------------------------------------------------------------- #
# NPA maximisation, compiled once per scenario and level
# --------------------------------------------------------------------------- #
class NPAMaximiser:
    """Maximise a Bell functional over NPA level-``level`` moment matrices.

    The moment matrix and its PSD constraint depend only on the scenario and
    the level; only the objective changes from one lifting and ``eta`` to the
    next. So the problem is built once with the objective weights as a cvxpy
    ``Parameter`` and re-solved with new values, which skips cvxpy's
    compilation on every call.
    """

    def __init__(self, scenario: Scenario, level, solver):
        structure = moment_structure(scenario, level)
        self.size = structure.size
        self.solver = solver

        names = [
            word for word in structure.classes if word is not None and word != IDENTITY
        ]
        slot = {word: index + 2 for index, word in enumerate(names)}
        slot[IDENTITY] = 0
        n_moments = len(names) + 2

        moments = cp.Variable(n_moments)

        n = structure.size
        index = np.ones((n, n), dtype=int)  # default slot 1 == 0, for zero entries
        for word, positions in structure.classes.items():
            if word is None:
                continue
            rows, cols = zip(*positions)
            index[list(rows), list(cols)] = slot[word]
        matrix = cp.reshape(moments[index.ravel()], (n, n), order="C")

        # Linear map from the flattened coefficient tensor to one weight per
        # moment, so that setting the objective is a single matrix product.
        shape = (scenario.n_out_a, scenario.n_out_b, scenario.n_set_a, scenario.n_set_b)
        self.transfer = np.zeros((n_moments, int(np.prod(shape))))
        for flat, (a, b, x, y) in enumerate(np.ndindex(*shape)):
            for word, coefficient in _probability_coefficients(
                scenario, a, b, x, y
            ).items():
                self.transfer[slot[word], flat] += coefficient

        self.weights = cp.Parameter(n_moments)
        constraints = [
            matrix >> 0,
            moments[0] == 1.0,  # <1> = 1
            moments[1] == 0.0,  # the slot shared by entries forced to vanish
        ]
        self.problem = cp.Problem(cp.Maximize(self.weights @ moments), constraints)

    def maximum(self, gamma: Array) -> float:
        self.weights.value = self.transfer @ gamma.ravel()
        self.problem.solve(solver=self.solver)
        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"NPA SDP did not solve: status {self.problem.status}")
        return float(self.problem.value)


# --------------------------------------------------------------------------- #
# Outcome liftings
# --------------------------------------------------------------------------- #
def outcome_lifting(gamma: Array, copy_a, copy_b) -> Array:
    """Lift ``gamma`` by one no-click outcome per party, copying existing ones.

    The no-click outcome sits at the new last index. When Alice measures ``x``
    its coefficients are those of outcome ``copy_a[x]``, and likewise for Bob
    with ``copy_b[y]``; the both-fail entry copies ``gamma[copy_a[x],
    copy_b[y]]``. The local bound is unchanged.
    """
    n_out_a, n_out_b, n_set_a, n_set_b = gamma.shape
    lifted = np.zeros((n_out_a + 1, n_out_b + 1, n_set_a, n_set_b))
    lifted[:n_out_a, :n_out_b] = gamma
    for x in range(n_set_a):
        lifted[n_out_a, :n_out_b, x] = gamma[copy_a[x], :, x]
    for y in range(n_set_b):
        # Reads from ``lifted`` so that Alice's no-click row is copied too.
        lifted[:, n_out_b, :, y] = lifted[:, copy_b[y], :, y]
    return lifted


def all_liftings(scenario: Scenario):
    """Every outcome lifting ``(copy_a, copy_b)``.

    ``copy_a[x]`` is the outcome whose coefficients Alice's no-click outcome
    takes for setting ``x``, chosen independently per setting, and likewise for
    Bob.
    """
    for copy_a in itertools.product(range(scenario.n_out_a), repeat=scenario.n_set_a):
        for copy_b in itertools.product(
            range(scenario.n_out_b), repeat=scenario.n_set_b
        ):
            yield np.array(copy_a), np.array(copy_b)


def no_signalling_floor(n_set_a: int, n_set_b: int) -> float:
    """Massar-Pironio bound ``(m_A + m_B - 2)/(m_A m_B - 1)``."""
    return (n_set_a + n_set_b - 2) / (n_set_a * n_set_b - 1)


# --------------------------------------------------------------------------- #
# The inefficient correlation
# --------------------------------------------------------------------------- #
def inefficient_functional(lifted: Array, eta: float) -> Array:
    """Fold the inefficient correlation into the lifted coefficients.

    Each detector clicks with probability ``eta`` and otherwise reports the
    no-click outcome ``N`` (the last index of ``lifted``), independently on the
    two sides. A behaviour ``p`` of the original scenario becomes

        p_eta(a, b|x, y) = eta^2      p(a, b|x, y)
        p_eta(a, N|x, y) = eta(1-eta) p_A(a|x)
        p_eta(N, b|x, y) = eta(1-eta) p_B(b|y)
        p_eta(N, N|x, y) = (1 - eta)^2

    Substituting into ``sum lifted p_eta`` and writing the marginals as sums of
    joint probabilities gives a functional on ``p`` alone, with coefficients

        eta^2 L[a,b] + eta(1-eta) (L[a,N] + L[N,b]) + (1-eta)^2 L[N,N]

    so the NPA problem stays in the original scenario and ``N`` needs no
    operator of its own.
    """
    n_out_a, n_out_b = lifted.shape[0] - 1, lifted.shape[1] - 1
    alice_only = lifted[:n_out_a, n_out_b][:, None]  # (a, N): Bob fails
    bob_only = lifted[n_out_a, :n_out_b][None, :]  # (N, b): Alice fails
    neither = lifted[n_out_a, n_out_b]
    return (
        eta**2 * lifted[:n_out_a, :n_out_b]
        + eta * (1.0 - eta) * (alice_only + bob_only)
        + (1.0 - eta) ** 2 * neither
    )


def violates(npa, lifted, rhs, eta, tolerance):
    """Whether *any* quantum strategy violates the lifted facet at ``eta``.

    Returns ``(violated, value)`` where ``value`` is the NPA maximum over the
    inefficient correlation: ``value <= rhs`` proves no quantum behaviour
    violates, ``value > rhs`` exhibits a relaxation point that does, which may
    or may not be quantum-realisable.
    """
    value = npa.maximum(inefficient_functional(lifted, eta))
    return value > rhs + tolerance, value


# --------------------------------------------------------------------------- #
# Per-facet search
# --------------------------------------------------------------------------- #
@dataclass
class EfficiencyResult:
    eta: float | None
    violated_at_start: bool
    bracket: tuple | None
    evaluations: int
    liftings: int
    copy_a: Array | None  # per-setting outcome Alice's no-click copies
    copy_b: Array | None  # per-setting outcome Bob's no-click copies
    lifted: Array | None  # the winning lifted facet


def find_threshold(
    scenario,
    gamma,
    rhs,
    npa,
    start=0.99,
    bisect_precision=0.001,
    tolerance=DEFAULT_TOLERANCE,
):
    """Smallest ``eta`` admitting a violation, over every outcome lifting.

    Monotonicity makes the bisection valid: Cope & Colbeck note that for
    ``eta_1 >= eta_2``, non-locality at ``eta_2`` implies non-locality at
    ``eta_1``, so each lifting's violating efficiencies form an interval
    reaching up to 1 and a bisection converges to its lower edge.

    The same monotonicity lets most liftings be dismissed with one SDP: a
    lifting that does not violate at ``best - precision`` has its threshold
    above that, so it cannot improve the minimum by more than the precision.
    The returned bracket is valid for the minimum over *all* liftings.
    """
    floor = no_signalling_floor(scenario.n_set_a, scenario.n_set_b)
    evaluations = liftings = 0
    best = None  # (low, high, copy_a, copy_b, lifted)
    lower = start  # smallest efficiency any lifting might still violate at

    for copy_a, copy_b in all_liftings(scenario):
        liftings += 1
        probe = start if best is None else best[1] - bisect_precision
        if probe <= floor:
            break  # already within precision of the floor, nothing to gain

        lifted = outcome_lifting(gamma, copy_a, copy_b)
        violated, _ = violates(npa, lifted, rhs, probe, tolerance)
        evaluations += 1
        if not violated:
            lower = min(lower, probe)
            continue

        low, high = floor, probe
        while high - low > bisect_precision:
            mid = 0.5 * (low + high)
            violated, _ = violates(npa, lifted, rhs, mid, tolerance)
            evaluations += 1
            if violated:
                high = mid
            else:
                low = mid
        best = (low, high, copy_a, copy_b, lifted)

    if best is None:
        return EfficiencyResult(
            None, False, None, evaluations, liftings, None, None, None
        )

    low, high, copy_a, copy_b, lifted = best
    bracket = (max(floor, min(low, lower)), high)
    return EfficiencyResult(
        high, True, bracket, evaluations, liftings, copy_a, copy_b, lifted
    )


# --------------------------------------------------------------------------- #
# Per-facet stage and entry point
# --------------------------------------------------------------------------- #
class EfficiencyStage:
    """The efficiency search for a single facet, and the columns it writes.

    See ``analysis_stage.py`` for the protocol. The NPA problem is compiled
    once here and re-solved with new objective weights for every lifting and
    every bisection step, which is what makes the search affordable: a facet
    costs roughly one SDP *solve* per lifting rather than one SDP *build*.
    """

    name = "efficiency"

    def __init__(self, scenario, args, solver):
        self.scenario = scenario
        self.args = args
        self.level = args.level if isinstance(args.level, Level) else Level.parse(args.level)
        self.floor = no_signalling_floor(scenario.n_set_a, scenario.n_set_b)
        self.npa = NPAMaximiser(scenario, self.level, solver)
        self.size = self.npa.size
        self.n_liftings = (
            scenario.n_out_a**scenario.n_set_a * scenario.n_out_b**scenario.n_set_b
        )
        self.lifted_label = (
            f"{scenario.n_set_a}{scenario.n_set_b}"
            f"{scenario.n_out_a + 1}{scenario.n_out_b + 1}"
        )
        self.column = f"{args.column}_L{self.level}"
        self.skip_column = f"{args.column}_skipped_L{self.level}"
        self.lifting_column = f"{args.column}_lifting_L{self.level}"
        self.columns = (self.column, self.lifting_column, self.skip_column)
        self.violated = self.not_violated = 0

    def describe(self) -> str:
        """One line naming the work and the columns, for a run header."""
        return (
            f"detection efficiency at NPA level {self.level} "
            f"(moment matrix {self.size}x{self.size}), {self.n_liftings} liftings "
            f"to {self.lifted_label}, floor {self.floor:.6f}, "
            f"start eta {self.args.start:g} -> column {self.column!r}"
        )

    def done(self, table, line) -> bool:
        """Either column counts: both record that the facet was searched.

        ``skip_column`` marks a facet with no violation at the starting
        efficiency, which is a result rather than a gap -- re-running should
        not repeat that search.
        """
        return table.has_value(line, self.column) or table.has_value(
            line, self.skip_column
        )

    def analyse(self, parsed) -> StageResult:
        result = find_threshold(
            self.scenario,
            parsed.to_tensor(self.scenario),
            parsed.rhs,
            self.npa,
            start=self.args.start,
            bisect_precision=self.args.precision,
            tolerance=self.args.tolerance,
        )

        if not result.violated_at_start:
            self.not_violated += 1
            return StageResult(
                values={self.skip_column: f"no_violation_at_{self.args.start:g}"},
                report=[
                    f"no violation at eta = {self.args.start:g} under any of the "
                    f"{result.liftings} liftings"
                ],
                # A facet that is not violated even at a near-perfect detector
                # is the common case in a large file; printing every one of
                # them would bury the facets that do have a threshold.
                notable=False,
            )

        # A violation at or below the floor is impossible for a valid facet:
        # Massar and Pironio give an explicit local model there. Finding one
        # means the functional being optimised is not the intended inequality,
        # so it is reported rather than recorded.
        floor_violated, floor_value = violates(
            self.npa, result.lifted, parsed.rhs, self.floor, self.args.tolerance
        )
        if floor_violated:
            raise SystemExit(
                f"line {parsed.line_number}: NPA finds a violation at the "
                f"no-signalling floor eta = {self.floor:.6f} (maximum "
                f"{floor_value:.9f} > {parsed.rhs:g}), where an explicit local "
                "model exists. The functional is not the intended inequality."
            )

        self.violated += 1
        low, high = result.bracket
        lifting = (
            f"a={''.join(map(str, result.copy_a))} "
            f"b={''.join(map(str, result.copy_b))}"
        )
        return StageResult(
            values={self.column: f"{result.eta:.6f}", self.lifting_column: lifting},
            report=[
                f"eta <= {result.eta:.6f} (bracket [{low:.6f}, {high:.6f}]); "
                f"no-click copies {lifting}; "
                f"{result.evaluations} SDP(s) over {result.liftings} liftings"
            ],
        )


def run_file(path, args, solver):
    inequalities = list(read_inequalities(path, limit=args.limit))
    if not inequalities:
        raise SystemExit(f"no inequalities found in {path}")

    scenario = resolve_scenario(path, inequalities, None)
    stage = EfficiencyStage(scenario, args, solver)

    out_path = Path(args.out) if args.out else default_out_path(path)
    table = AnalysisTable.load(out_path)

    print(
        f"{path}: {len(inequalities)} inequalities, correlations {scenario.label}, "
        f"{stage.n_liftings} outcome liftings per facet to {stage.lifted_label} "
        f"(last outcome is no-click), "
        f"NPA level {stage.level} (moment matrix {stage.size}x{stage.size})"
    )
    print(
        f"no-signalling floor (m_A+m_B-2)/(m_A m_B-1) = {stage.floor:.6f}; "
        f"start eta = {args.start}; precision = {args.precision}"
    )
    print(
        f"writing analysis to {out_path} "
        f"(columns {stage.column!r}, {stage.lifting_column!r}, {stage.skip_column!r})\n"
    )

    # A status line rewritten in place on stderr, shown only on a terminal so
    # that redirected logs do not fill up with carriage returns.
    show_progress = sys.stderr.isatty()

    def status(text=""):
        if show_progress:
            sys.stderr.write("\r\033[K" + text)
            sys.stderr.flush()

    started = time.time()
    present = 0
    for count, parsed in enumerate(inequalities, start=1):
        status(
            f"[{count}/{len(inequalities)}] line {parsed.line_number} · "
            f"{stage.violated} violated · {time.time() - started:.0f}s"
        )
        if not args.overwrite and stage.done(table, parsed.line_number):
            present += 1
            continue

        try:
            result = stage.analyse(parsed)
        except SystemExit:
            status()  # clear the status line so the message is not printed over
            raise

        if result.notable:
            status()  # clear the status line so the result does not print over it
            for line in result.report:
                print(f"line {parsed.line_number}: {line}")

        table.update(
            parsed.line_number,
            inequality=parsed.text,
            rhs=parsed.rhs,
            values=result.values,
        )
        table.save(out_path)  # save as we go: long runs stay resumable

    status()
    print(
        f"\n{stage.violated} violated, {stage.not_violated} not violated at eta = "
        f"{args.start:g}, {present} already present; "
        f"done in {time.time() - started:.1f}s -> {out_path}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Minimum symmetric detection efficiency per facet over all outcome "
            "liftings that accommodate a no-click outcome, certified with the "
            "NPA hierarchy."
        )
    )
    parser.add_argument("file", help="panda output file")
    parser.add_argument(
        "--level",
        default="1+AB",
        help="NPA level: an integer depth or an intermediate level such as "
        "1+AB (default: 1+AB)",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.99,
        help="upper end of the search: the bisection runs between the "
        "no-signalling floor and this efficiency, and facets not violated "
        "here under any lifting are skipped (default: 0.99)",
    )
    parser.add_argument(
        "--precision",
        type=float,
        default=0.001,
        help="bisection width on eta (default: 0.001)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only process the first N inequalities"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="slack above rhs before a value counts as violating",
    )
    parser.add_argument(
        "--solver", choices=sorted(SOLVERS), default="mosek", help="SDP solver"
    )
    parser.add_argument("--out", default=None, help="analysis file")
    parser.add_argument(
        "--column",
        default="detection_efficiency_npa_lifted",
        help="base column name; the NPA level is appended",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="recompute rows that already have values"
    )
    args = parser.parse_args(argv)

    try:
        args.level = Level.parse(args.level)
    except ValueError as error:
        parser.error(str(error))

    run_file(args.file, args, SOLVERS[args.solver])


if __name__ == "__main__":
    main()
