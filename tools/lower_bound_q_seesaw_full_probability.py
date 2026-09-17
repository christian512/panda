"""See-saw lower bounds on the quantum value of full-probability Bell inequalities.

Reads panda output files written in the *full probability* parameterization
(see ``samples/generate_bell_files_full_probability.py``), where a behaviour is
given by every joint probability

    p(a, b | x, y),   a = 0..o_A-1,  b = 0..o_B-1,  x = 0..m_A-1,  y = 0..m_B-1

with no outcome dropped. A functional is therefore purely a joint tensor

    S = sum_{a,b,x,y} gamma[a, b, x, y] * p(a, b | x, y)

which is simpler than the Collins-Gisin case handled by ``cg_functional.py`` /
``seesaw_cg.py``: there are no marginal terms, and every outcome carries a
weight (CG's dropped outcome has no analogue here).

The see-saw alternates three steps, each exact or optimal in its own variable:

    1. optimize Alice's POVMs   (SDP, state and Bob fixed)
    2. optimize Bob's POVMs     (SDP, state and Alice fixed)
    3. optimize the state       (eigendecomposition, both parties fixed)

Step 3 needs no SDP. With both parties fixed the value is ``tr[rho W]`` for the
Hermitian Bell operator ``W = sum gamma[a,b,x,y] A_{a|x} (x) B_{b|y}``. A linear
function on the convex set of density matrices is maximized at an extreme point,
and the extreme points are exactly the pure states; writing
``rho = sum_i p_i |psi_i><psi_i|`` gives
``tr[rho W] = sum_i p_i <psi_i|W|psi_i> <= lambda_max(W)``, attained by the pure
state on a top eigenvector. So ``lambda_max(W)`` is the exact optimum and the
corresponding eigenvector is an optimal state.

Because each step is a true maximization in its variable, the Bell value is
non-decreasing along the iteration and converges. The result is a *lower* bound
on the quantum maximum at the chosen local dimension: the see-saw may land in a
local optimum, hence the random restarts.

White-noise robustness
----------------------
The see-saw ends holding the optimal state and measurements, so the behaviour
``p_Q(a,b|x,y) = tr[rho A_{a|x} (x) B_{b|y}]`` that maximally violates the
inequality comes for free. Mixing it with white noise (uniformly random
outcomes) gives ``p(v) = v p_Q + (1 - v) p_noise``, and because the functional
is linear the value is a straight line in ``v``. The visibility at which the
violation vanishes therefore has a closed form,

    v* = (rhs - S_noise) / (S_Q - S_noise),

with the inequality violated for every ``v > v*``; a *smaller* ``v*`` means a
more noise-resistant inequality. This costs no extra optimization, so it is
computed alongside the bound and written to its own column. For CHSH it
reproduces the familiar ``1/sqrt(2) ~ 0.7071``.

Requires: numpy, cvxpy. MOSEK (``--solver mosek``, the default when available)
is markedly faster and is picked up from a license at ``~/mosek/mosek.lic``.

Each run writes two columns: the see-saw lower bound and the critical white-noise
visibility, both tagged with the local dimension.

Usage:
    python tools/seesaw_full_probability.py results/bell_full_probability/2222
    python tools/seesaw_full_probability.py results/bell_full_probability/3322 \
        --tries 20 --dim 3 --out results/analysis/3322.csv
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cvxpy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis_stage import StageResult  # noqa: E402
from analysis_table import AnalysisTable  # noqa: E402

Array = np.ndarray

SOLVERS = {"scs": cp.SCS, "clarabel": cp.CLARABEL, "mosek": cp.MOSEK}


# --------------------------------------------------------------------------- #
# Scenario and functional
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scenario:
    """A bipartite Bell scenario in the full-probability parameterization."""

    n_set_a: int  # m_A
    n_set_b: int  # m_B
    n_out_a: int  # o_A
    n_out_b: int  # o_B
    local_dim: int

    @property
    def joint_dim(self) -> int:
        return self.local_dim**2

    @property
    def label(self) -> str:
        """Scenario as ``"3322"``: settings for A and B, then outcomes for A and B."""
        return f"{self.n_set_a}{self.n_set_b}{self.n_out_a}{self.n_out_b}"


# Full-probability terms as emitted by panda, e.g. "-2pAB01x2y1".
# Outcome indices are single digits; setting indices may have several.
_TERM_RE = re.compile(
    r"""(?P<sign>[+-])?\s*
        (?P<coeff>\d+(?:/\d+)?)?\s*
        pAB(?P<a>\d)(?P<b>\d)x(?P<x>\d+)y(?P<y>\d+)""",
    re.VERBOSE,
)
_RELATION_RE = re.compile(r"(<=|>=|=<|=>)")
_SECTION_RE = re.compile(r"^[A-Za-z ]+:\s*$")


@dataclass
class ParsedInequality:
    """A full-probability inequality ``sum c p(a,b|x,y) <= rhs`` from panda output."""

    coeffs: dict  # (a, b, x, y) -> coefficient
    rhs: float
    text: str
    line_number: int

    def extent(self):
        """Largest ``(a, b, x, y)`` index appearing, each as a count (max + 1)."""
        na = nb = sa = sb = 0
        for a, b, x, y in self.coeffs:
            na, nb, sa, sb = max(na, a + 1), max(nb, b + 1), max(sa, x + 1), max(sb, y + 1)
        return na, nb, sa, sb

    def to_tensor(self, scenario: Scenario) -> Array:
        """Dense ``gamma[a, b, x, y]`` coefficient tensor for the given scenario."""
        gamma = np.zeros(
            (scenario.n_out_a, scenario.n_out_b, scenario.n_set_a, scenario.n_set_b)
        )
        for (a, b, x, y), c in self.coeffs.items():
            gamma[a, b, x, y] = c
        return gamma


def parse_inequality(line: str, line_number: int = 0) -> ParsedInequality:
    """Parse one ``-pAB01x1y1 +pAB11x0y1 <= 0`` style full-probability line."""
    parts = _RELATION_RE.split(line.strip(), maxsplit=1)
    if len(parts) != 3:
        raise ValueError(f"no relation symbol in inequality: {line.strip()!r}")
    lhs, relation, rhs_text = parts
    flip = relation in (">=", "=>")  # normalize everything to "<="

    rhs = float(Fraction(rhs_text.strip()))
    coeffs: dict = {}
    for m in _TERM_RE.finditer(lhs):
        value = float(Fraction(m.group("coeff") or 1))
        if (m.group("sign") == "-") != flip:
            value = -value
        key = tuple(int(m.group(g)) for g in ("a", "b", "x", "y"))
        coeffs[key] = coeffs.get(key, 0.0) + value

    leftover = _TERM_RE.sub("", lhs).strip()
    if leftover:
        raise ValueError(
            f"unrecognized term(s) {leftover!r} in: {line.strip()!r}. Expected "
            "full-probability names such as pAB01x0y1."
        )
    return ParsedInequality(
        coeffs=coeffs,
        rhs=-rhs if flip else rhs,
        text=line.strip(),
        line_number=line_number,
    )


def read_inequalities(path, limit=None):
    """Yield ``ParsedInequality`` objects from a panda full-probability output file.

    The file is streamed line by line, so large outputs are fine; ``.gz`` files
    are decompressed on the fly. Only the ``Inequalities:`` section is read --
    the ``Equations:`` section of a full-probability file holds the
    normalization / no-signaling equalities, which are constraints on the
    parameterization rather than Bell functionals. ``line_number`` is the
    1-based physical line in the file, which is what the analysis table keys on.
    """
    count = 0
    in_inequalities = True  # files without any header are taken to be inequalities
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if _SECTION_RE.match(stripped):  # e.g. "Inequalities:", "Equations:"
                in_inequalities = "inequalit" in stripped.lower()
                continue
            if not in_inequalities:
                continue
            yield parse_inequality(stripped, number)
            count += 1
            if limit is not None and count >= limit:
                return


def infer_scenario(inequalities, local_dim=None) -> Scenario:
    """Smallest scenario containing every index used by ``inequalities``.

    Unlike Collins-Gisin, full probability keeps every outcome, so the outcome
    counts are exactly the largest index seen plus one.
    """
    na = nb = sa = sb = 0
    for ineq in inequalities:
        e = ineq.extent()
        na, nb, sa, sb = max(na, e[0]), max(nb, e[1]), max(sa, e[2]), max(sb, e[3])
    return Scenario(
        n_set_a=sa,
        n_set_b=sb,
        n_out_a=na,
        n_out_b=nb,
        local_dim=local_dim or max(na, nb),
    )


def scenario_from_filename(path, local_dim=None):
    """Scenario encoded in a name like ``3322`` (m_A m_B o_A o_B), else ``None``.

    Matches the argument order of
    ``samples/generate_bell_files_full_probability.py``.
    """
    match = re.match(r"^(\d)(\d)(\d)(\d)(?:\D|$)", Path(path).name)
    if match is None:
        return None
    m_a, m_b, o_a, o_b = (int(g) for g in match.groups())
    return Scenario(
        n_set_a=m_a,
        n_set_b=m_b,
        n_out_a=o_a,
        n_out_b=o_b,
        local_dim=local_dim or max(o_a, o_b),
    )


# --------------------------------------------------------------------------- #
# Linear-algebra helpers
# --------------------------------------------------------------------------- #
def _random_psd(dim: int, rng: np.random.Generator) -> Array:
    g = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return g @ g.conj().T


def _inverse_sqrt(mat: Array) -> Array:
    """Inverse matrix square root of a Hermitian positive-definite matrix."""
    w, v = np.linalg.eigh(mat)
    return (v / np.sqrt(w)) @ v.conj().T


def random_pure_state(dim: int, rng: np.random.Generator) -> Array:
    """Random pure state ``|psi><psi|`` drawn from the Haar measure."""
    psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def random_measurements(n_out: int, n_set: int, dim: int, rng: np.random.Generator) -> Array:
    """Random ``n_out``-outcome POVMs (one per setting), each summing to the identity.

    Each POVM is built from random PSD blocks G_k, symmetrically normalized as
    ``M_k = S^{-1/2} G_k S^{-1/2}`` with ``S = sum_k G_k``, so ``sum_k M_k = I``.
    """
    effects = np.empty((n_out, n_set, dim, dim), dtype=complex)
    for s in range(n_set):
        blocks = [_random_psd(dim, rng) for _ in range(n_out)]
        root_inv = _inverse_sqrt(sum(blocks))
        for k, block in enumerate(blocks):
            effects[k, s] = root_inv @ block @ root_inv
    return effects


def random_projective_measurements(
    n_out: int, n_set: int, dim: int, rng: np.random.Generator
) -> Array:
    """Random projective measurements (one per setting), used as see-saw seeds.

    A Haar-random orthonormal basis per setting, its vectors split as evenly as
    possible over the outcomes. Sharp seeds matter: starting from flat, nearly
    proportional-to-identity effects the see-saw tends to stall on a trivial
    fixed point. Falls back to general POVMs when outcomes exceed the dimension.
    """
    if n_out > dim:
        return random_measurements(n_out, n_set, dim, rng)

    effects = np.zeros((n_out, n_set, dim, dim), dtype=complex)
    for s in range(n_set):
        basis, _ = np.linalg.qr(
            rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        )
        for k, columns in enumerate(np.array_split(np.arange(dim), n_out)):
            block = basis[:, columns]
            effects[k, s] = block @ block.conj().T
    return effects


# --------------------------------------------------------------------------- #
# See-saw sub-problems
# --------------------------------------------------------------------------- #
def _optimize_party(dim, n_out, n_set, coeff, solver):
    """Maximize ``sum_{o,s} Re tr(coeff(o, s) @ M[o, s])`` over valid POVMs.

    ``coeff(o, s)`` returns the constant ``d x d`` matrix that makes the Bell
    objective linear in the effect ``M[o, s]``. Effects are PSD and sum to the
    identity per setting. In full probability every outcome carries a weight.
    """
    effects = [
        [cp.Variable((dim, dim), hermitian=True) for _ in range(n_set)]
        for _ in range(n_out)
    ]

    objective = cp.Maximize(
        sum(
            cp.real(cp.trace(coeff(o, s) @ effects[o][s]))
            for o in range(n_out)
            for s in range(n_set)
        )
    )
    constraints = [effects[o][s] >> 0 for o in range(n_out) for s in range(n_set)]
    constraints += [
        sum(effects[o][s] for o in range(n_out)) == np.eye(dim) for s in range(n_set)
    ]

    value = cp.Problem(objective, constraints).solve(solver=solver)
    result = np.array([[effects[o][s].value for s in range(n_set)] for o in range(n_out)])
    return value, result


def optimize_alice(scenario, gamma, rho, bob, solver):
    """Optimize Alice's measurements with ``rho`` and Bob's effects fixed."""
    rho4 = rho.reshape((scenario.local_dim,) * 4)

    def coeff(a, x):
        w = np.zeros((scenario.local_dim, scenario.local_dim), dtype=complex)
        for b in range(scenario.n_out_b):
            for y in range(scenario.n_set_b):
                if gamma[a, b, x, y]:
                    w = w + gamma[a, b, x, y] * np.einsum("ijkl,lj->ik", rho4, bob[b, y])
        return w

    return _optimize_party(
        scenario.local_dim, scenario.n_out_a, scenario.n_set_a, coeff, solver
    )


def optimize_bob(scenario, gamma, rho, alice, solver):
    """Optimize Bob's measurements with ``rho`` and Alice's effects fixed."""
    rho4 = rho.reshape((scenario.local_dim,) * 4)

    def coeff(b, y):
        w = np.zeros((scenario.local_dim, scenario.local_dim), dtype=complex)
        for a in range(scenario.n_out_a):
            for x in range(scenario.n_set_a):
                if gamma[a, b, x, y]:
                    w = w + gamma[a, b, x, y] * np.einsum("ijkl,ki->jl", rho4, alice[a, x])
        return w

    return _optimize_party(
        scenario.local_dim, scenario.n_out_b, scenario.n_set_b, coeff, solver
    )


def bell_operator(scenario, gamma, alice, bob) -> Array:
    """The Bell operator ``W = sum gamma[a,b,x,y] A_{a|x} (x) B_{b|y}``."""
    dim = scenario.joint_dim
    w = np.zeros((dim, dim), dtype=complex)
    for a in range(scenario.n_out_a):
        for b in range(scenario.n_out_b):
            for x in range(scenario.n_set_a):
                for y in range(scenario.n_set_b):
                    if gamma[a, b, x, y]:
                        w += gamma[a, b, x, y] * np.kron(alice[a, x], bob[b, y])
    return w


def optimize_state(scenario, gamma, alice, bob):
    """Optimize the shared state with both parties' measurements fixed.

    Exact, and no SDP required: maximizing the linear functional ``tr[rho W]``
    over density matrices attains its maximum at an extreme point of that convex
    set, and the extreme points are exactly the pure states. Hence the optimum is
    ``lambda_max(W)``, attained by the pure state on a top eigenvector.
    """
    w = bell_operator(scenario, gamma, alice, bob)
    eigenvalues, eigenvectors = np.linalg.eigh(w)
    psi = eigenvectors[:, -1]  # eigh returns eigenvalues in ascending order
    return float(eigenvalues[-1]), np.outer(psi, psi.conj())


# --------------------------------------------------------------------------- #
# Behaviour and white-noise robustness
# --------------------------------------------------------------------------- #
def behaviour(scenario, alice, bob, rho) -> Array:
    """The behaviour ``p[a, b, x, y] = tr[rho (A_{a|x} (x) B_{b|y})]``.

    The Born rule applied to every outcome pair, giving back the same
    parameterization the inequalities are written in. Contracting this with
    ``gamma`` reproduces the see-saw value exactly, which is what makes it
    usable as the quantum end of the noise mixture.
    """
    p = np.zeros(
        (scenario.n_out_a, scenario.n_out_b, scenario.n_set_a, scenario.n_set_b)
    )
    for a in range(scenario.n_out_a):
        for x in range(scenario.n_set_a):
            for b in range(scenario.n_out_b):
                for y in range(scenario.n_set_b):
                    p[a, b, x, y] = np.real(
                        np.trace(rho @ np.kron(alice[a, x], bob[b, y]))
                    )
    return p


def white_noise_behaviour(scenario) -> Array:
    """Uniformly random outcomes, ``p(a, b|x, y) = 1 / (o_A o_B)``.

    White noise is the maximally mixed behaviour: every outcome pair equally
    likely, independently of the settings. It is normalized and no-signaling,
    so every mixture of it with a quantum behaviour is a valid behaviour.
    """
    return np.full(
        (scenario.n_out_a, scenario.n_out_b, scenario.n_set_a, scenario.n_set_b),
        1.0 / (scenario.n_out_a * scenario.n_out_b),
    )


def critical_visibility(rhs, quantum_value, noise_value):
    """Smallest visibility ``v`` for which the noisy behaviour still violates.

    Mixing the optimal quantum behaviour with white noise,

        p(v) = v p_Q + (1 - v) p_noise,

    and using that the functional is *linear* in the behaviour, the value is

        S(v) = v S_Q + (1 - v) S_noise,

    a straight line in ``v``. So the threshold where the violation disappears
    has a closed form and needs no search over ``v``:

        S(v*) = rhs   =>   v* = (rhs - S_noise) / (S_Q - S_noise).

    The inequality is violated for every ``v > v*``, so a *smaller* ``v*`` means
    a more noise-resistant inequality. Returns ``None`` when the quantum value
    does not exceed the classical bound, in which case no amount of visibility
    violates and the threshold is undefined.
    """
    if quantum_value <= rhs:
        return None
    # S_Q > rhs >= S_noise is the violating case; the denominator is then
    # positive and the quotient lands in [0, 1).
    if quantum_value <= noise_value:
        return None
    return (rhs - noise_value) / (quantum_value - noise_value)


# --------------------------------------------------------------------------- #
# See-saw driver
# --------------------------------------------------------------------------- #
@dataclass
class SeesawResult:
    alice: Array
    bob: Array
    rho: Array
    value: float
    iterations: int


def run_seesaw(scenario, gamma, rng, precision=1e-9, max_iter=500, solver=cp.SCS):
    """One see-saw run from a random starting point."""
    rho = random_pure_state(scenario.joint_dim, rng)
    alice = random_projective_measurements(
        scenario.n_out_a, scenario.n_set_a, scenario.local_dim, rng
    )
    bob = random_projective_measurements(
        scenario.n_out_b, scenario.n_set_b, scenario.local_dim, rng
    )

    value = previous = -np.inf
    iterations = 0
    for iterations in range(1, max_iter + 1):
        _, bob = optimize_bob(scenario, gamma, rho, alice, solver)
        _, alice = optimize_alice(scenario, gamma, rho, bob, solver)
        value, rho = optimize_state(scenario, gamma, alice, bob)

        if np.isnan(value) or abs(value - previous) <= precision:
            break
        previous = value

    return SeesawResult(alice, bob, rho, float(value), iterations)


def best_of(scenario, gamma, rng, n_tries=5, verbose=False, **kwargs) -> SeesawResult:
    """Run the see-saw ``n_tries`` times and return the highest-value result."""
    best = None
    for k in range(n_tries):
        result = run_seesaw(scenario, gamma, rng, **kwargs)
        if verbose:
            print(
                f"    try {k + 1}/{n_tries}: value = {result.value:.9f} "
                f"({result.iterations} iterations)"
            )
        if best is None or result.value > best.value:
            best = result
    return best


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def resolve_scenario(path, inequalities, dim):
    """Scenario for a file: filename convention if present, else inferred indices."""
    scenario = scenario_from_filename(path, dim) or infer_scenario(inequalities, dim)
    if dim:  # honour an explicit --dim even when the filename fixed the scenario
        scenario = Scenario(
            scenario.n_set_a, scenario.n_set_b, scenario.n_out_a, scenario.n_out_b, dim
        )

    needed = infer_scenario(inequalities)
    if (
        needed.n_out_a > scenario.n_out_a
        or needed.n_out_b > scenario.n_out_b
        or needed.n_set_a > scenario.n_set_a
        or needed.n_set_b > scenario.n_set_b
    ):
        raise SystemExit(f"{path}: indices exceed scenario {scenario}; needs {needed}")
    return scenario


def default_out_path(path) -> Path:
    """Analysis file that corresponds to a given panda output file."""
    source = Path(path)
    name = source.name[:-3] if source.name.endswith(".gz") else source.name
    return source.parent / "analysis" / f"{name}.csv"


# --------------------------------------------------------------------------- #
# Per-inequality stage and entry point
# --------------------------------------------------------------------------- #
class SeesawStage:
    """The see-saw analysis of a single inequality, and the columns it writes.

    See ``analysis_stage.py`` for the protocol. The scenario-level work -- the
    column names and the white-noise behaviour -- happens here in the
    constructor, so ``analyse`` is exactly the optimization one facet costs.
    """

    name = "seesaw"

    def __init__(self, scenario, args, rng, solver):
        self.scenario = scenario
        self.args = args
        self.rng = rng
        self.solver = solver
        # The local dimension is part of the column name, so bounds computed at
        # different dimensions accumulate side by side instead of overwriting
        # one another -- a see-saw at dimension 3 can only improve on one at
        # dimension 2, and it is the comparison between them that is
        # interesting.
        self.column = f"{args.column}_d{scenario.local_dim}"
        self.noise_column = f"{args.noise_column}_d{scenario.local_dim}"
        self.columns = (self.column, self.noise_column)
        # White noise does not depend on the inequality, only on the scenario.
        self.noise = white_noise_behaviour(scenario)

    def describe(self) -> str:
        """One line naming the work and the columns, for a run header."""
        return (
            f"see-saw lower bound at local dimension {self.scenario.local_dim}, "
            f"{self.args.tries} restart(s) "
            f"-> columns {self.column!r}, {self.noise_column!r}"
        )

    def done(self, table, line) -> bool:
        """Whether this facet already carries both columns.

        Both are required: a file analysed before the noise threshold existed
        still has the see-saw value, and re-running is what fills the new
        column in.
        """
        return all(table.has_value(line, column) for column in self.columns)

    def analyse(self, parsed) -> StageResult:
        gamma = parsed.to_tensor(self.scenario)
        result = best_of(
            self.scenario,
            gamma,
            self.rng,
            n_tries=self.args.tries,
            verbose=self.args.verbose,
            precision=self.args.precision,
            max_iter=self.args.max_iter,
            solver=self.solver,
        )

        violation = result.value - parsed.rhs
        verdict = "violated" if violation > self.args.tolerance else "not violated"
        report = [
            f"classical bound = {parsed.rhs:.6f}   "
            f"quantum lower bound = {result.value:.9f}   "
            f"violation = {violation:+.9f}  ({verdict})"
        ]

        # The see-saw already found the optimal state and measurements, so the
        # noise threshold costs one behaviour reconstruction and two dot
        # products -- no further optimization.
        p_quantum = behaviour(self.scenario, result.alice, result.bob, result.rho)
        quantum_value = float(np.sum(gamma * p_quantum))
        noise_value = float(np.sum(gamma * self.noise))

        # The reconstructed behaviour must reproduce the see-saw value; if it
        # does not, the state and measurements do not describe the optimum that
        # was reported and the threshold below would be meaningless.
        if abs(quantum_value - result.value) > 1e-6:
            raise SystemExit(
                f"line {parsed.line_number}: reconstructed behaviour gives "
                f"{quantum_value:.9f} but the see-saw reported {result.value:.9f}"
            )

        visibility = critical_visibility(parsed.rhs, quantum_value, noise_value)

        if visibility is None:
            report.append(
                f"white-noise value = {noise_value:.9f}   "
                "critical visibility = n/a (no violation to degrade)"
            )
        else:
            report.append(
                f"white-noise value = {noise_value:.9f}   "
                f"critical visibility = {visibility:.9f}   "
                f"(violated for v > {visibility:.6f})"
            )

        return StageResult(
            values={
                self.column: f"{result.value:.9f}",
                # "n/a" rather than blank: the threshold is genuinely undefined
                # when nothing is violated, and recording that distinguishes it
                # from "not computed yet" -- which is also what lets a re-run
                # skip the row instead of redoing the see-saw for it.
                self.noise_column: "n/a" if visibility is None else f"{visibility:.9f}",
            },
            report=report,
        )


def run_file(path, args, rng, solver):
    inequalities = list(read_inequalities(path, limit=args.limit))
    if not inequalities:
        raise SystemExit(f"no inequalities found in {path}")

    scenario = resolve_scenario(path, inequalities, args.dim)
    out_path = Path(args.out) if args.out else default_out_path(path)
    table = AnalysisTable.load(out_path)
    stage = SeesawStage(scenario, args, rng, solver)

    print(
        f"{path}: {len(inequalities)} inequalities, "
        f"scenario {scenario.label} (settings then outcomes), "
        f"local dimension {scenario.local_dim}"
    )
    print(
        f"writing analysis to {out_path} "
        f"(columns {stage.column!r}, {stage.noise_column!r})\n"
    )

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
            "See-saw lower bounds on the quantum maximum of full-probability Bell "
            "inequalities read from a panda output file."
        )
    )
    parser.add_argument(
        "file", help="panda output file, e.g. results/bell_full_probability/2222"
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="local Hilbert space dimension per party (default: max outcome count)",
    )
    parser.add_argument(
        "--tries", type=int, default=5, help="random restarts per inequality"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only process the first N inequalities"
    )
    parser.add_argument(
        "--precision", type=float, default=1e-9, help="see-saw convergence threshold"
    )
    parser.add_argument(
        "--max-iter", type=int, default=500, help="iteration cap per see-saw run"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="slack before a value counts as violating the classical bound",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--solver",
        choices=sorted(SOLVERS),
        default="mosek",
        help="SDP solver for the Alice/Bob sub-problems (default: mosek)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="analysis file (default: <dir>/analysis/<name>.csv next to the input)",
    )
    parser.add_argument(
        "--column",
        default="seesaw_lower_bound",
        help=(
            "base column name; the local dimension is appended, so a run at "
            "dimension 3 writes 'seesaw_lower_bound_d3' "
            "(default: seesaw_lower_bound)"
        ),
    )
    parser.add_argument(
        "--noise-column",
        default="white_noise_visibility",
        help=(
            "base column name for the critical white-noise visibility; the "
            "local dimension is appended, so a run at dimension 2 writes "
            "'white_noise_visibility_d2' (default: white_noise_visibility)"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute inequalities that already have a value in this column",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print the value of every restart"
    )
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    run_file(args.file, args, rng, SOLVERS[args.solver])


if __name__ == "__main__":
    main()
