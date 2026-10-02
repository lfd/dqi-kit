"""Cross-code equivalence suite for the imperfect-decoding estimators of dqi.py.

``compute_expectation`` is the single estimator of the imperfect-decoding regime
for every prime p; ``tests.reference_impl.compute_A_bar`` is the GF(2)-only
A_bar formulation used as an oracle.  Both were
rewritten as syndrome-bucketing numpy code, and during that rewrite a subtle
population-rescaling bug (applying ``scale_k1 * scale_k2`` to the ``y1 == y2``
diagonal, which is a single draw and carries only ``scale_k1``) slipped through
one reference implementation.  So the estimator is pinned here against three
*independent* oracles over a deliberately diverse set of codes:

1. ``legacy_estimators.legacy_expectation`` -- the pre-port implementation
   (commit ``393d2ce``), valid only for exhaustive benchmarks.
2. ``legacy_estimators.naive_rescaled_expectation`` -- a slow, explicit
   transcription that does rescale sampled benchmarks.  This is the oracle for
   the sampled path, i.e. exactly where the bug lived.
3. The ``a = 0`` diagonal identity at p = 2: the diagonal of the numerator
   contributes exactly ``m * r / p = m / 2`` after normalisation, independent of
   any scaling, so ``compute_expectation`` must equal
   ``m / 2 + (w^T A_bar w / norm) / 2`` with ``A_bar`` holding only the
   ``a = 1`` part.

A fourth section pins the partner lookup itself: the partner of a sampled error
is decoded from the shifted syndrome, and the two ways of getting the decoder's
answer (the coset-leader table of a ``SyndromeDecoder``, ``error_estimate``
calls for any other decoder) must agree; on sampled benchmarks the estimate
must land close to the exhaustively computed value.

The code set spans p in {2, 3, 5, 7}, m from 7 to 30, minimum distances 3 to 8
(even and odd) and r in {1, 2}; see ``CODES``.
"""

import math
import random

import numpy as np
import pytest
from sage.all import GF, Matrix, codes, vector

from decoders import (
    BenchmarkResult,
    GeneralizedReedSolomonDecoder,
    NearestNeighborDecoder,
    SyndromeDecoder,
)
from dqi import Dqi, compute_expectation, get_eigenvector, make_A
from max_lin_sat import MaxLinSat, MaxXorSat, OptimalPolynomialIntersection
from tests.legacy_estimators import legacy_expectation, naive_rescaled_expectation
from tests.oracles import golay_instance, max_lin_sat_from_B, odd_distance_instance
from tests.reference_impl import compute_A_bar

# --------------------------------------------------------------------------- #
# The code set.
#
# Diversity is the point: the rescaling and the diagonal enter the estimate
# differently for different p, r, minimum distance and decoding radius, so a
# single code (or a single l, or a single w) is not enough to pin them.
#
#   id             p   m   n   d   r   note
#   hamming_7_4    2   7   3   3   1   perfect code: weight >= 2 never decodes
#   dual_golay_2   2  24  12   8   1   perfect decoding up to l = 3
#   bch_15_7_5     2  15   8   5   1   odd minimum distance
#   xorsat_24_12   2  24  12   4   1   random 3-XORSAT, imperfect from l = 2
#   xorsat_30_15   2  30  15   4   1   larger random 3-XORSAT
#   dual_golay_3   3  11   6   6   1   non-binary, even d
#   golay_3_r2     3  11   6   6   2   r = 2 != 1, and p = 3 != 2r, so make_A
#                                      has a non-zero diagonal
#   gf5_10_4       5  10   4   3   1   p > 3
#   opi_gf7        7   7   2   3   1   OPI / dual GRS over GF(7)
# --------------------------------------------------------------------------- #


def _hamming_7_4():
    """Binary Hamming [7, 4, 3]: B^T is its parity check matrix, so ker(B^T) is the code."""
    B = codes.HammingCode(GF(2), 3).parity_check_matrix().T
    return max_lin_sat_from_B(GF(2), B, [1] * B.nrows())


def _dual_golay(p: int):
    B, m, n, d = golay_instance(p)
    return max_lin_sat_from_B(GF(p), B, [i % p for i in range(m)])


def _bch_15_7_5():
    B, m, n, d = odd_distance_instance()
    return max_lin_sat_from_B(GF(2), B, [i % 2 for i in range(m)])


def _xorsat(m: int, n: int, seed: int, k: int = 3):
    """Random k-XORSAT with m distinct constraints over n variables."""
    random.seed(seed)
    ls = MaxXorSat()
    xs = [ls.new_var(f"x{i}") for i in range(n)]
    seen = set()
    while len(seen) < m:
        term = tuple(sorted(random.sample(range(n), k)))
        if term in seen:
            continue
        seen.add(term)
        ls.add_constraint(sum(xs[i] for i in term) == random.randint(0, 1))
    return ls


def _golay_3_r2():
    """Ternary Golay B with |F_i| = 2 instead of 1, i.e. r = 2.

    ``make_A(p, r, m, l)`` has a non-zero diagonal exactly when p != 2r, so this
    is the instance that exercises the r != 1 path (p = 3, r = 2).  Built from
    ``LinSatConstraint`` directly because ``max_lin_sat_from_B`` only makes
    singleton F_i.
    """
    from constraints import LinSatConstraint

    field = GF(3)
    B, m, n, d = golay_instance(3)
    ls = MaxLinSat(field, equal_size_F_i=False)
    xs = [ls.new_var(f"x{i}") for i in range(B.ncols())]
    for i, row in enumerate(B.rows()):
        F_i = {field(0), field(1)} if i % 2 == 0 else {field(1), field(2)}
        ls.add_constraint(LinSatConstraint(field, xs, list(row), F_i), disable_warnings=True)
    return ls


def _gf5_10_4():
    """Random 10 x 4 matrix over GF(5); ker(B^T) is a [10, 6, 3] code."""
    random.seed(100)
    B = Matrix(GF(5), [[random.randrange(5) for _ in range(4)] for _ in range(10)])
    rhs = [random.randrange(5) for _ in range(10)]
    return max_lin_sat_from_B(GF(5), B, rhs)


def _opi_gf7():
    """Optimal Polynomial Intersection over GF(7), degree 1: DQI code = dual GRS [7, 5, 3]."""
    field = GF(7)
    return OptimalPolynomialIntersection(
        field,
        1,
        {field(i): {field(i)} for i in range(7)},
        GeneralizedReedSolomonDecoder.constructor(),
    )


# (builder, p, m, n, d, r).  d is recorded rather than computed: for non-binary
# instances MaxLinSat.get_minimum_distance() runs a 5 s randomised search.
CODES = {
    "hamming_7_4": (_hamming_7_4, 2, 7, 3, 3, 1),
    "dual_golay_2": (lambda: _dual_golay(2), 2, 24, 12, 8, 1),
    "bch_15_7_5": (_bch_15_7_5, 2, 15, 8, 5, 1),
    "xorsat_24_12": (lambda: _xorsat(24, 12, 0), 2, 24, 12, 4, 1),
    "xorsat_30_15": (lambda: _xorsat(30, 15, 7), 2, 30, 15, 4, 1),
    "dual_golay_3": (lambda: _dual_golay(3), 3, 11, 6, 6, 1),
    "golay_3_r2": (_golay_3_r2, 3, 11, 6, 6, 2),
    "gf5_10_4": (_gf5_10_4, 5, 10, 4, 3, 1),
    "opi_gf7": (_opi_gf7, 7, 7, 2, 3, 1),
}

BINARY_CODES = [name for name, spec in CODES.items() if spec[1] == 2]

_INSTANCES: dict[str, object] = {}


@pytest.fixture(scope="module")
def instances():
    """Lazily built, module-scoped registry of the code set.

    Module scope matters: the decoder caches its benchmarks per
    ``(l, weight, n_tries)``, so re-requesting the same benchmark from another
    test is free *and* returns the identical error list.
    """

    def get(name: str):
        if name not in _INSTANCES:
            builder, p, m, n, d, r = CODES[name]
            ls = builder()
            assert (ls.field.order(), ls.get_m(), ls.get_n(), ls.get_r()) == (p, m, n, r)
            _INSTANCES[name] = ls
        return _INSTANCES[name]

    return get


def test_code_set_is_diverse():
    """A guard on the fixture table itself: the point of the set is its spread."""
    ps = {spec[1] for spec in CODES.values()}
    ds = {spec[4] for spec in CODES.values()}
    rs = {spec[5] for spec in CODES.values()}
    assert len(CODES) >= 7
    assert ps == {2, 3, 5, 7}
    assert {d % 2 for d in ds} == {0, 1}, "need both even and odd minimum distance"
    assert rs == {1, 2}, "need an r != 1 instance for the p != 2r path of make_A"
    assert max(spec[2] for spec in CODES.values()) >= 30


# --------------------------------------------------------------------------- #
# Benchmarks and weight vectors.
# --------------------------------------------------------------------------- #


def _benchmarks(instance, l: int, n_tries: int | None, seed: int):
    """Benchmark dict for weights 0..l and the decoder that produced it.

    ``decoders.generate_errors`` draws from the legacy global NumPy RNG *and*
    from ``random``, so both are seeded here; the returned dict is then handed to
    every estimator under test, so no comparison depends on two separate
    benchmark runs agreeing.  The decoder comes along because the estimator asks
    *that* decoder for the interference partners: which errors count as
    "correctly decoded" is a property of the decoder, not of the code.
    """
    random.seed(seed)
    np.random.seed(seed)
    m = instance.get_m()
    decoder = Dqi(instance, SyndromeDecoder.constructor()).get_decoder()
    bm = {0: BenchmarkResult([vector(instance.field, [0] * m)], [], 0)}
    for k in range(l, 0, -1):
        bm[k] = decoder.get_benchmarks(l, k, n_tries)
    return bm, decoder


def _weight_vectors(p: int, r: int, m: int, l: int) -> dict[str, np.ndarray]:
    """The default w and non-default variants; a mis-scaled diagonal shows up differently
    for different w, so every comparison is run for all of them."""
    _, eigen = get_eigenvector(make_A(p, r, m, l), -1)
    eigen = eigen / np.linalg.norm(eigen)
    variants = {
        "eigenvector": eigen,
        "ones": np.ones(l + 1) / math.sqrt(l + 1),
    }
    if l >= 2:
        # a zero component drops one weight block from the state entirely
        zeroed = eigen.copy()
        zeroed[1] = 0.0
        variants["zero_at_1"] = zeroed / np.linalg.norm(zeroed)
    return variants


def _n_sampled(bm) -> int:
    return sum(len(bm[k].correct) for k in bm)


# --------------------------------------------------------------------------- #
# Oracle 1: the superseded implementation, on exhaustive benchmarks.
# --------------------------------------------------------------------------- #

# (code, l).  legacy_expectation is O(N^2 m p) *Sage matrix-vector products*, so
# only combinations whose exhaustive benchmark stays small are affordable here;
# the sampled tests below reach the larger l.
LEGACY_CASES = [
    ("hamming_7_4", 1),
    ("hamming_7_4", 2),
    ("hamming_7_4", 3),
    ("dual_golay_2", 1),
    ("bch_15_7_5", 1),
    ("bch_15_7_5", 2),
    ("xorsat_24_12", 1),
    ("xorsat_30_15", 1),
    ("dual_golay_3", 1),
    ("golay_3_r2", 1),
    ("gf5_10_4", 1),
    ("opi_gf7", 1),
    ("opi_gf7", 2),
]


@pytest.mark.parametrize("code,l", LEGACY_CASES, ids=[f"{name}-l{l}" for name, l in LEGACY_CASES])
def test_expectation_matches_superseded_implementation(instances, code, l):
    """The fast estimator must reproduce the implementation it replaced, exactly.

    Exhaustive benchmarks only: ``legacy_expectation`` has no population
    rescaling, so it is only valid where every ``scale_k`` is 1.
    """
    ls = instances(code)
    p, r, m = ls.field.order(), ls.get_r(), ls.get_m()
    bm, decoder = _benchmarks(ls, l, None, seed=0)
    assert all(len(bm[k].correct) == 0 or bm[k].epsilon < 1 for k in bm)

    for name, w in _weight_vectors(p, r, m, l).items():
        expected = legacy_expectation(ls, l, w, bm)
        got = compute_expectation(ls, l, w, bm, decoder)
        assert got == pytest.approx(expected, rel=1e-9), f"w = {name}"


# --------------------------------------------------------------------------- #
# Oracle 2: the naive rescaled transcription.
# --------------------------------------------------------------------------- #

# The transcription must first agree with the superseded code where the latter
# is valid (all scales 1); only then is it trustworthy as the oracle for the
# sampled path.
TRANSCRIPTION_CASES = [
    ("hamming_7_4", 3),
    ("bch_15_7_5", 2),
    ("dual_golay_2", 1),
    ("dual_golay_3", 1),
    ("golay_3_r2", 1),
    ("gf5_10_4", 1),
    ("opi_gf7", 2),
]


@pytest.mark.parametrize(
    "code,l",
    TRANSCRIPTION_CASES,
    ids=[f"{name}-l{l}" for name, l in TRANSCRIPTION_CASES],
)
def test_naive_transcription_agrees_with_superseded_implementation(instances, code, l):
    """Validates the sampled-path oracle before it is relied upon.

    On exhaustive benchmarks every ``scale_k`` is 1, so the rescaled
    transcription must collapse onto the unrescaled legacy formula.
    """
    ls = instances(code)
    p, r, m = ls.field.order(), ls.get_r(), ls.get_m()
    bm, decoder = _benchmarks(ls, l, None, seed=0)

    for name, w in _weight_vectors(p, r, m, l).items():
        assert naive_rescaled_expectation(ls, l, w, bm, decoder) == pytest.approx(
            legacy_expectation(ls, l, w, bm), rel=1e-9
        ), f"w = {name}"


# (code, l, n_tries).  n_tries is small enough that *every* weight class >= 1 is
# genuinely subsampled, so every scale_k differs from 1 and most partners of a
# sampled error are not in the sample themselves.
SAMPLED_CASES = [
    ("hamming_7_4", 1, 4),
    ("hamming_7_4", 2, 4),
    ("dual_golay_2", 2, 15),
    ("dual_golay_2", 3, 20),
    ("bch_15_7_5", 2, 10),
    ("bch_15_7_5", 3, 25),
    ("xorsat_24_12", 2, 15),
    ("xorsat_24_12", 3, 30),
    ("xorsat_30_15", 3, 30),
    ("dual_golay_3", 2, 15),
    ("dual_golay_3", 3, 40),
    ("golay_3_r2", 2, 15),
    ("golay_3_r2", 3, 40),
    ("gf5_10_4", 2, 30),
    ("gf5_10_4", 3, 60),
    ("opi_gf7", 2, 30),
]


@pytest.mark.parametrize(
    "code,l,n_tries",
    SAMPLED_CASES,
    ids=[f"{name}-l{l}-n{n}" for name, l, n in SAMPLED_CASES],
)
def test_expectation_matches_naive_oracle_on_sampled_benchmarks(instances, code, l, n_tries):
    """The sampled path against a plain-loop transcription of the same estimator.

    With subsampled benchmarks each weight block is rescaled by
    ``scale_k = C(m,k) (p-1)^k (1 - eps_k) / N_k`` and the partner of a sampled
    error is decoded rather than looked up in the sample.  A wrong scale (for
    instance ``scale_k1 * scale_k2`` on a term, which inflates the estimate by a
    factor of order ``scale_k`` -- a bug that once lived here), a dropped
    partner or a wrong partner amplitude is well outside the tolerance.
    """
    ls = instances(code)
    p, r, m = ls.field.order(), ls.get_r(), ls.get_m()
    bm, decoder = _benchmarks(ls, l, n_tries, seed=11)

    # the sample must actually be a sample, otherwise every scale_k is 1 and
    # this test degenerates into the exhaustive one
    subsampled = [
        k
        for k in range(1, l + 1)
        if len(bm[k].correct) + len(bm[k].incorrect) < math.comb(m, k) * (p - 1) ** k
    ]
    assert subsampled, f"n_tries={n_tries} did not subsample any weight class"

    for name, w in _weight_vectors(p, r, m, l).items():
        expected = naive_rescaled_expectation(ls, l, w, bm, decoder)
        got = compute_expectation(ls, l, w, bm, decoder)
        assert got == pytest.approx(expected, rel=1e-9), f"w = {name}"


# --------------------------------------------------------------------------- #
# Oracle 3: the a = 0 diagonal identity at p = 2.
# --------------------------------------------------------------------------- #

DIAGONAL_CASES = [(code, l, None) for code in BINARY_CODES for l in (1, 2, 3)]


@pytest.mark.parametrize(
    "code,l,n_tries",
    DIAGONAL_CASES,
    ids=[f"{c}-l{l}-{'exhaustive' if n is None else f'n{n}'}" for c, l, n in DIAGONAL_CASES],
)
def test_diagonal_contributes_exactly_m_over_p_binary(instances, code, l, n_tries):
    """``compute_expectation == m/2 + (w^T A_bar w / norm)/2`` at p = 2.

    ``A_bar`` holds only the ``a = 1`` part of the numerator, so the ``+ m/2`` is
    precisely the ``a = 0`` diagonal -- which contributes ``m * r / p`` after
    division by ``norm`` no matter how the blocks are rescaled.  The right-hand
    side is assembled here rather than taken from
    ``predict_from_A_bar`` so that its ``max(0, ...)``
    clamp cannot hide a disagreement.

    This is an independent check of the diagonal: it goes through
    ``compute_A_bar``, not through either oracle in ``legacy_estimators``.

    Exhaustive benchmarks only: ``compute_A_bar`` counts pairs of *benchmarked*
    errors, which is the population sum exactly when the benchmark is the
    population (the sampled path is covered by the naive oracle above).
    """
    ls = instances(code)
    assert ls.field.order() == 2
    m = ls.get_m()
    bm, decoder = _benchmarks(ls, l, n_tries, seed=23)
    epsilon = np.array([bm[k].epsilon for k in range(l + 1)])

    A_bar = compute_A_bar(ls.get_B(), ls.get_v(), l, bm, ls.get_minimum_distance())

    for name, w in _weight_vectors(2, 1, m, l).items():
        norm = float(np.sum(w * w * (1 - epsilon)))
        if norm == 0:
            continue  # the DQI state is empty for this w, nothing to compare
        expected = m / 2 + float(w @ A_bar @ w) / norm / 2
        got = compute_expectation(ls, l, w, bm, decoder)
        assert got == pytest.approx(expected, abs=1e-9), f"w = {name}"


# --------------------------------------------------------------------------- #
# The partner lookup.
#
# The partner of a sampled y1 is decoded from the shifted syndrome, so every y1
# contributes its complete row of (i, a) terms with weight scale_k1 alone.  The
# two ways of getting the decoder's answer (its coset-leader table for a
# SyndromeDecoder, error_estimate calls otherwise) must agree, and on sampled
# benchmarks the estimate must land close to the exhaustively computed value.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_tries", [None, 10], ids=["exhaustive", "n10"])
def test_partner_lookups_agree(n_tries):
    """The generic decoder path must reproduce the ``SyndromeDecoder`` table path.

    ``_Partners`` reads the whole coset-leader table of a ``SyndromeDecoder`` at
    once, but calls ``decoder.error_estimate`` for every syndrome it cannot
    shortcut on any other decoder.  Both are asked for the partner of the same
    syndromes, so for two decoders that agree as functions of the syndrome they
    must return the same amplitudes.

    The BCH [15, 7, 5] code at l = 2 is such a pair: a coset with two distinct
    leaders of weight <= 2 would need a codeword of weight <= 4 < d, so the
    minimum-weight leader is unique there and both decoders return it; above
    weight 2 the decoders may disagree, but those estimates are not part of the
    state and contribute 0 either way.  The decoding radius is 2 = l, so the
    nearest-neighbour path exercises both the radius shortcut (neighbours of
    weight <= 2) and real decoder calls (neighbours of weight 3).

    The sampled case is the one with teeth: on exhaustive benchmarks every
    partner is found among the sampled errors and neither lookup is consulted.
    """
    B, m, n, d = odd_distance_instance()
    ls = max_lin_sat_from_B(GF(2), B, [i % 2 for i in range(m)])
    l = 2
    bm, syndrome_decoder = _benchmarks(ls, l, n_tries, seed=17)
    nearest = Dqi(ls, NearestNeighborDecoder.constructor()).get_decoder()

    for name, w in _weight_vectors(2, 1, m, l).items():
        via_table = compute_expectation(ls, l, w, bm, syndrome_decoder)
        via_calls = compute_expectation(ls, l, w, bm, nearest)
        assert via_calls == pytest.approx(via_table, rel=1e-10), f"w = {name}"


def test_sampled_estimate_is_close_to_the_exact_value():
    """Fresh 500-sample benchmarks must land near the exhaustively computed value.

    Decoding the partner rather than looking it up among the sampled errors is
    what keeps the variance down: a pair term that only counts when *both*
    errors were drawn scatters by about 0.076 around the exact 27.2417 on this
    random 3-XORSAT with m = 40, l = 3, while decoding the partner scatters by
    about 0.014 (50 measured repetitions each, worst deviation 0.04 against
    0.22).  The threshold below is loose enough (about 7 sigma) that it is a
    regression guard, not a flake.
    """
    ls = _xorsat(40, 20, 0)
    m, l = ls.get_m(), 3
    decoder = Dqi(ls, SyndromeDecoder.constructor()).get_decoder()
    _, w = get_eigenvector(make_A(2, 1, m, l), -1)
    w = w / np.linalg.norm(w)

    def benchmark(n_tries, seed):
        random.seed(seed)
        np.random.seed(seed)
        decoder.benchmarks.clear()  # otherwise every repetition reuses one sample
        bm = {0: BenchmarkResult([vector(ls.field, [0] * m)], [], 0)}
        for k in range(l, 0, -1):
            bm[k] = decoder.get_benchmarks(l, k, n_tries)
        return bm

    exact = compute_expectation(ls, l, w, benchmark(None, 0), decoder)
    assert exact == pytest.approx(27.2417, abs=1e-3)  # pins the instance itself

    for rep in range(5):
        bm = benchmark(500, 100 + rep)
        assert sum(len(bm[k].correct) + len(bm[k].incorrect) for k in bm) < math.comb(m, l), (
            "the benchmark has to be a sample for this test to say anything"
        )
        got = compute_expectation(ls, l, w, bm, decoder)
        assert abs(got - exact) < 0.1, f"repetition {rep}: {got} vs {exact}"
