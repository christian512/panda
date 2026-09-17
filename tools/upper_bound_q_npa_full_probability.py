"""NPA upper bounds on the quantum value of full-probability Bell inequalities.

Reads panda output files written in the *full probability* parameterization
(see ``samples/generate_bell_files_full_probability.py``), where a behaviour is
given by every joint probability

    p(a, b | x, y),   a = 0..o_A-1,  b = 0..o_B-1,  x = 0..m_A-1,  y = 0..m_B-1

and a functional is the joint tensor ``S = sum gamma[a,b,x,y] p(a,b|x,y)``.
This is the same input format and the same analysis file that
``seesaw_full_probability.py`` uses; the two scripts bracket the quantum value
from opposite sides and write into different columns of one table.

The Navascues-Pironio-Acin hierarchy
------------------------------------
The see-saw searches *inside* the quantum set at a fixed local dimension, so it
returns a lower bound. NPA instead relaxes the problem: it optimizes over
moment matrices that every quantum behaviour must satisfy, over all dimensions
at once. Any feasible point of the relaxation dominates every quantum strategy,
so the optimum is an *upper* bound, and the bounds decrease monotonically with
the level:

    Q  <=  ...  <=  NPA level 3  <=  NPA level 2  <=  NPA level 1

Construction. Fix a set ``S`` of operator monomials (level 1 is ``{1} u {A} u
{B}``, level 2 adds all products of two, and so on). For a quantum strategy the
matrix ``Gamma[u, v] = <psi| u^dag v |psi>`` is positive semidefinite. Three
structural facts turn that into a finite SDP:

  * ``Gamma >> 0`` for any state and operators;
  * entries are *equal* whenever ``u^dag v`` reduces to the same monomial under
    the projector relations (``P^2 = P``, ``P_a|x P_a'|x = 0`` for ``a != a'``,
    and ``[A, B] = 0``), which is what makes the matrix a *moment* matrix
    rather than an arbitrary PSD matrix;
  * the entries indexed by ``A_{a|x} B_{b|y}`` are exactly ``p(a,b|x,y)``.

Maximizing the Bell functional over such matrices is then an SDP whose value
upper-bounds the quantum maximum. Level 1 already reproduces Tsirelson's bound
for CHSH; I3322 needs level 3-4 (see ``--level`` help and the module tests).

Levels. ``--level`` takes either an integer depth (all monomials up to that
length) or an *intermediate* level written as a depth plus monomial families:
``1+AB`` is level 1 together with the products ``A_{a|x} B_{b|y}``. These
intermediate sets are the workhorse of the Bell literature because they sit
much closer to level 1 in cost and to level 2 in tightness. For the 5522
scenario the moment matrix is 11x11 at level 1, 36x36 at ``1+AB`` and 76x76 at
level 2, costing roughly 0.02s / 0.15s / 1.0s per inequality -- which is the
difference between a feasible and an infeasible sweep over a large file.

Outcome handling. One outcome per setting is dropped and reconstructed from
normalization, so the operators are ``(o-1)`` projectors per setting rather than
``o``. This is the standard NPA convention and keeps the moment matrix small;
the dropped outcome enters the objective through
``p(o_A-1, b|x, y) = p_B(b|y) - sum_{a<o_A-1} p(a, b|x, y)`` and its analogues.

Requires: numpy, cvxpy. MOSEK (``--solver mosek``, the default when available)
is markedly faster and is picked up from a license at ``~/mosek/mosek.lic``.

Usage:
    python tools/npa_full_probability.py results/bell_full_probability/2222
    python tools/npa_full_probability.py results/bell_full_probability/3322 \
        --level 3 --out results/bell_full_probability/analysis/3322.csv
    python tools/npa_full_probability.py results/bell_full_probability/5522 \
        --level 1+AB
    python tools/npa_full_probability.py --self-test
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
from lower_bound_q_seesaw_full_probability import (  # noqa: E402
    Scenario,
    default_out_path,
    read_inequalities,
    resolve_scenario,
)

Array = np.ndarray

SOLVERS = {"scs": cp.SCS, "clarabel": cp.CLARABEL, "mosek": cp.MOSEK}


# --------------------------------------------------------------------------- #
# Operator monomials
# --------------------------------------------------------------------------- #
# A monomial is a tuple of letters; a letter is (party, outcome, setting) with
# party 0 = Alice, 1 = Bob. The empty tuple is the identity. Only outcomes
# 0..o-2 appear as operators: the last one is eliminated by normalization.
Letter = tuple  # (party, outcome, setting)
Monomial = tuple  # (Letter, ...)

IDENTITY: Monomial = ()


def _letters(scenario: Scenario, party: int):
    """Projector letters for one party, dropping the last outcome per setting."""
    if party == 0:
        n_out, n_set = scenario.n_out_a, scenario.n_set_a
    else:
        n_out, n_set = scenario.n_out_b, scenario.n_set_b
    return [(party, o, s) for s in range(n_set) for o in range(n_out - 1)]


def reduce_monomial(word):
    """Normal form of a product of projectors, or ``None`` if it is zero.

    Applies the three relations that define the NPA moment structure:
    Alice's and Bob's operators commute (so letters are sorted by party, with
    each party's internal order preserved), ``P P = P``, and two different
    outcomes of the same setting multiply to zero.
    """
    alice = [letter for letter in word if letter[0] == 0]
    bob = [letter for letter in word if letter[0] == 1]

    reduced = []
    for part in (alice, bob):
        out = []
        for letter in part:
            if out and out[-1] == letter:  # P^2 = P
                continue
            if out and out[-1][2] == letter[2] and out[-1][1] != letter[1]:
                return None  # P_{a|x} P_{a'|x} = 0 for a != a'
            out.append(letter)
        reduced.extend(out)
    return tuple(reduced)


def _dagger(word: Monomial) -> Monomial:
    return tuple(reversed(word))


def _signature(word: Monomial) -> str:
    """Party pattern of a reduced monomial, e.g. ``"AAB"`` -- sorted, A before B."""
    return "A" * sum(1 for letter in word if letter[0] == 0) + "B" * sum(
        1 for letter in word if letter[0] == 1
    )


@dataclass(frozen=True)
class Level:
    """An NPA level: every monomial up to ``depth``, plus named extra families.

    The hierarchy's integer levels take *all* monomials of length <= n. The
    intermediate levels standard in the Bell literature keep a lower depth and
    add back only some longer families -- most importantly ``1+AB``, which is
    level 1 plus the operators ``A_{a|x} B_{b|y}``. It is far cheaper than
    level 2 and in practice nearly as tight, so it is the usual workhorse for
    anything but the smallest scenarios.
    """

    depth: int
    extra: frozenset  # party signatures such as "AB", "AAB"

    @classmethod
    def parse(cls, text) -> "Level":
        """Parse ``"2"``, ``"1+AB"``, ``"1+AB+AAB"`` into a ``Level``."""
        parts = [part.strip() for part in str(text).split("+") if part.strip()]
        if not parts:
            raise ValueError(f"empty NPA level: {text!r}")

        head, *rest = parts
        if not head.isdigit():
            raise ValueError(
                f"NPA level must start with a number, got {text!r} "
                "(examples: 1, 2, 1+AB)"
            )

        extra = set()
        for part in rest:
            letters = part.upper()
            if not letters or set(letters) - {"A", "B"}:
                raise ValueError(
                    f"unknown monomial family {part!r} in level {text!r}: "
                    "families are written with A and B only, e.g. AB or AAB"
                )
            # Operators of one party commute in the moment matrix indexing, so
            # "BA" and "AB" denote the same family; store a canonical form.
            extra.add("A" * letters.count("A") + "B" * letters.count("B"))
        return cls(int(head), frozenset(extra))

    @property
    def max_length(self) -> int:
        return max([self.depth] + [len(family) for family in self.extra])

    def admits(self, word: Monomial) -> bool:
        return len(word) <= self.depth or _signature(word) in self.extra

    def __str__(self) -> str:
        return "+".join([str(self.depth), *sorted(self.extra)])


def monomials_for_level(scenario: Scenario, level) -> list:
    """Reduced monomials indexing the moment matrix, identity first.

    ``level`` is a :class:`Level` or anything it can parse. Words are generated
    up to the longest length the level mentions and then filtered, so that an
    intermediate level such as ``1+AB`` keeps ``A``, ``B`` and ``AB`` while
    dropping ``AA`` and ``BB``.
    """
    level = level if isinstance(level, Level) else Level.parse(level)
    letters = _letters(scenario, 0) + _letters(scenario, 1)

    seen = {IDENTITY}
    ordered = [IDENTITY]
    frontier = [IDENTITY]
    for _ in range(level.max_length):
        next_frontier = []
        for word in frontier:
            for letter in letters:
                reduced = reduce_monomial(word + (letter,))
                if reduced is None or reduced in seen:
                    continue
                seen.add(reduced)
                # Keep walking through words the level excludes: a longer word
                # it does admit may still be reachable only through them.
                next_frontier.append(reduced)
                if level.admits(reduced):
                    ordered.append(reduced)
        frontier = next_frontier
    return ordered


# --------------------------------------------------------------------------- #
# Moment matrix
# --------------------------------------------------------------------------- #
@dataclass
class MomentStructure:
    """The equality pattern of a moment matrix.

    ``classes`` maps a reduced monomial to the list of ``(row, col)`` positions
    whose entry must equal that monomial's moment. ``size`` is the matrix
    dimension.
    """

    classes: dict
    size: int


def moment_structure(scenario: Scenario, level) -> MomentStructure:
    """Group the entries of ``Gamma[u, v] = <u^dag v>`` into equality classes."""
    words = monomials_for_level(scenario, level)
    classes: dict = {}
    for i, u in enumerate(words):
        for j, v in enumerate(words):
            reduced = reduce_monomial(_dagger(u) + v)
            if reduced is None:
                classes.setdefault(None, []).append((i, j))  # entry is zero
            else:
                classes.setdefault(reduced, []).append((i, j))
    return MomentStructure(classes, len(words))


def _probability_coefficients(scenario: Scenario, a, b, x, y):
    """``p(a, b | x, y)`` as a mapping ``monomial -> coefficient``.

    Dropped outcomes are reconstructed from normalization. With
    ``A_last = 1 - sum_{a < o_A-1} A_{a|x}`` (and likewise for Bob), expanding
    the product gives the inclusion-exclusion form used here.
    """
    last_a = a == scenario.n_out_a - 1
    last_b = b == scenario.n_out_b - 1

    a_terms = (
        [(-1.0, (0, k, x)) for k in range(scenario.n_out_a - 1)] + [(1.0, None)]
        if last_a
        else [(1.0, (0, a, x))]
    )
    b_terms = (
        [(-1.0, (1, k, y)) for k in range(scenario.n_out_b - 1)] + [(1.0, None)]
        if last_b
        else [(1.0, (1, b, y))]
    )

    coefficients: dict = {}
    for sign_a, letter_a in a_terms:
        for sign_b, letter_b in b_terms:
            word = tuple(letter for letter in (letter_a, letter_b) if letter is not None)
            key = reduce_monomial(word)
            coefficients[key] = coefficients.get(key, 0.0) + sign_a * sign_b
    return coefficients


def npa_upper_bound(scenario: Scenario, gamma: Array, level, solver) -> float:
    """Maximize the Bell functional over NPA level-``level`` moment matrices."""
    structure = moment_structure(scenario, level)

    # One scalar per equality class, held in a single vector variable. Slot 0 is
    # pinned to the identity moment (= 1) and slot 1 to zero, so that entries
    # forced to vanish can be addressed like any other class.
    names = [word for word in structure.classes if word is not None and word != IDENTITY]
    slot = {word: index + 2 for index, word in enumerate(names)}
    slot[IDENTITY] = 0

    moments = cp.Variable(len(names) + 2)

    # The moment matrix is one gather from that vector: cvxpy then compiles a
    # single expression for the PSD constraint rather than one subexpression per
    # class, which is what keeps the larger scenarios fast to build.
    n = structure.size
    index = np.ones((n, n), dtype=int)  # default slot 1 == 0, for zero entries
    for word, positions in structure.classes.items():
        if word is None:
            continue
        rows, cols = zip(*positions)
        index[list(rows), list(cols)] = slot[word]

    matrix = cp.reshape(moments[index.ravel()], (n, n), order="C")

    variables = {word: moments[position] for word, position in slot.items()}

    # Collect the objective as one scalar coefficient per moment before handing
    # it to cvxpy, so the expression tree stays flat however many terms the
    # inequality has.
    weights: dict = {}
    for a in range(scenario.n_out_a):
        for b in range(scenario.n_out_b):
            for x in range(scenario.n_set_a):
                for y in range(scenario.n_set_b):
                    if not gamma[a, b, x, y]:
                        continue
                    for word, coefficient in _probability_coefficients(
                        scenario, a, b, x, y
                    ).items():
                        weights[word] = weights.get(word, 0.0) + gamma[a, b, x, y] * coefficient

    objective = cp.sum(
        [
            cp.multiply(coefficient, variables[word])
            for word, coefficient in weights.items()
            if coefficient
        ]
    )

    constraints = [
        matrix >> 0,
        moments[0] == 1.0,  # <1> = 1, the normalization of the moment matrix
        moments[1] == 0.0,  # the slot shared by all entries forced to vanish
    ]
    problem = cp.Problem(cp.Maximize(objective), constraints)
    problem.solve(solver=solver)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"NPA SDP did not solve: status {problem.status}")
    return float(problem.value)


# --------------------------------------------------------------------------- #
# Per-inequality stage and entry point
# --------------------------------------------------------------------------- #
class NPAStage:
    """The NPA analysis of a single inequality, and the column it writes.

    See ``analysis_stage.py`` for the protocol. The moment matrix depends only
    on the scenario and the level, so its size is settled here; the SDP itself
    is rebuilt per facet by ``npa_upper_bound``, since the objective is the
    only thing that changes and cvxpy's compilation dominates only for the
    small scenarios.
    """

    name = "npa"

    def __init__(self, scenario, args, solver):
        self.scenario = scenario
        self.args = args
        self.solver = solver
        self.level = args.level if isinstance(args.level, Level) else Level.parse(args.level)
        self.size = moment_structure(scenario, self.level).size
        # The level is part of the column name, so bounds computed at different
        # levels accumulate side by side instead of overwriting one another.
        self.column = f"{args.column}_L{self.level}"
        self.columns = (self.column,)

    def describe(self) -> str:
        """One line naming the work and the column, for a run header."""
        return (
            f"NPA upper bound at level {self.level} "
            f"(moment matrix {self.size}x{self.size}) -> column {self.column!r}"
        )

    def done(self, table, line) -> bool:
        return table.has_value(line, self.column)

    def analyse(self, parsed) -> StageResult:
        gamma = parsed.to_tensor(self.scenario)
        value = npa_upper_bound(self.scenario, gamma, self.level, self.solver)

        violation = value - parsed.rhs
        verdict = "violated" if violation > self.args.tolerance else "not violated"
        return StageResult(
            values={self.column: f"{value:.9f}"},
            report=[
                f"classical bound = {parsed.rhs:.6f}   "
                f"quantum upper bound = {value:.9f}   "
                f"violation = {violation:+.9f}  ({verdict})"
            ],
        )


def run_file(path, args, solver):
    inequalities = list(read_inequalities(path, limit=args.limit))
    if not inequalities:
        raise SystemExit(f"no inequalities found in {path}")

    scenario = resolve_scenario(path, inequalities, args.dim)
    out_path = Path(args.out) if args.out else default_out_path(path)
    table = AnalysisTable.load(out_path)
    stage = NPAStage(scenario, args, solver)

    print(
        f"{path}: {len(inequalities)} inequalities, "
        f"scenario {scenario.label} (settings then outcomes), "
        f"NPA level {stage.level} (moment matrix {stage.size}x{stage.size})"
    )
    print(f"writing analysis to {out_path} (column {stage.column!r})\n")

    started = time.time()
    for index, parsed in enumerate(inequalities):
        if not args.overwrite and stage.done(table, parsed.line_number):
            print(f"[{index}] line {parsed.line_number}: already present, skipping")
            continue

        print(f"[{index}] line {parsed.line_number}: {parsed.text}")
        result = stage.analyse(parsed)
        for line in result.report:
            print(f"    {line}")

        table.update(
            parsed.line_number,
            inequality=parsed.text,
            rhs=parsed.rhs,
            values=result.values,
        )
        table.save(out_path)  # save as we go: long runs stay resumable

    print(f"\ndone in {time.time() - started:.1f}s -> {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "NPA upper bounds on the quantum maximum of full-probability Bell "
            "inequalities read from a panda output file."
        )
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="panda output file, e.g. results/bell_full_probability/2222",
    )
    parser.add_argument(
        "--level",
        default="2",
        help=(
            "NPA hierarchy level; higher is tighter and slower. Either an "
            "integer depth (1, 2, 3) or an intermediate level written as a "
            "depth plus monomial families, e.g. '1+AB' -- level 1 together "
            "with the products A_{a|x} B_{b|y}. 1+AB is much cheaper than "
            "level 2 and usually nearly as tight, so it is the practical "
            "choice for large scenarios. Level 1 already gives Tsirelson's "
            "bound for CHSH, I3322 needs 3-4 (default: 2)"
        ),
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="override the scenario's local dimension field (does not affect the bound)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only process the first N inequalities"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="slack before a value counts as violating the classical bound",
    )
    parser.add_argument(
        "--solver",
        choices=sorted(SOLVERS),
        default="mosek",
        help="SDP solver (default: mosek)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="analysis file (default: <dir>/analysis/<name>.csv next to the input)",
    )
    parser.add_argument(
        "--column",
        default="npa_upper_bound",
        help=(
            "base column name; the level is appended, so a run at 1+AB writes "
            "'npa_upper_bound_L1+AB' (default: npa_upper_bound)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute inequalities that already have a value in this column",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="check the implementation against known analytic bounds and exit",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        from npa_self_test import run_self_test

        raise SystemExit(run_self_test(SOLVERS[args.solver]))

    if not args.file:
        parser.error("a panda output file is required (or use --self-test)")

    # Parse the level here so a malformed spec is a clean CLI error rather than
    # a traceback out of the middle of a run.
    try:
        args.level = Level.parse(args.level)
    except ValueError as error:
        parser.error(str(error))

    run_file(args.file, args, SOLVERS[args.solver])


if __name__ == "__main__":
    main()
