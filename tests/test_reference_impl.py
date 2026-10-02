"""Correctness and speed of the syndrome-bucketing estimator in dqi.py.

``compute_expectation`` (every prime p) finds the interference partner of every
decodable error by its syndrome.  The tests here pin it against independent oracles
(the semicircle matrix, the brute-force DQI state, and the A_bar formulation
``tests.reference_impl.compute_A_bar`` on GF(2)), pin the rescaling of sampled
benchmarks to the full error population, and pin the invariant that a benchmark
holds at most one correct error per syndrome.
"""

import random
import time

import numpy as np
import pytest
from sage.all import GF, vector

from decoders import BenchmarkResult, NearestNeighborDecoder, SyndromeDecoder
from dqi import (
    Dqi,
    _predict_dqi_performance_perfect_optimal_w,
    compute_expectation,
    get_eigenvector,
    make_A,
)
from max_lin_sat import MaxXorSat
from tests.oracles import brute_force_dqi_expectation, golay_instance, max_lin_sat_from_B
from tests.reference_impl import compute_A_bar, estimate_via_A_bar, predict_from_A_bar


def _benchmarks(dq, l, n_tries):
    m = dq.instance.get_m()
    dec = dq.get_decoder()
    bm = {0: BenchmarkResult([vector(dq.instance.field, [0] * m)], [], 0)}
    for k in range(l, 0, -1):
        bm[k] = dec.get_benchmarks(l, k, n_tries)
    return bm


def _clashing_errors(ls, weight):
    """Two distinct weight-``weight`` errors with the same syndrome.

    They differ by a codeword of ``ker(B^T)`` of weight ``2 * weight``: splitting
    its support in half gives y1 = c|first half, y2 = -c|second half with
    y1 - y2 = c, hence H y1 = H y2.  No decoder can get both right.
    """
    field = ls.field
    m = ls.get_m()
    for c in ls.get_code():
        support = [i for i in range(m) if c[i] != 0]
        if len(support) != 2 * weight:
            continue
        y1 = vector(field, [0] * m)
        y2 = vector(field, [0] * m)
        for i in support[:weight]:
            y1[i] = c[i]
        for i in support[weight:]:
            y2[i] = -c[i]
        assert y1 != y2
        assert ls.get_B().T * y1 == ls.get_B().T * y2
        return y1, y2
    raise AssertionError(f"no codeword of weight {2 * weight} in this code")


@pytest.fixture(scope="module")
def random_xorsat():
    """m=24, n=12 random 3-XORSAT with minimum distance 4 (imperfect regime for l >= 2)."""
    random.seed(0)
    ls = MaxXorSat()
    xs = [ls.new_var(f"x{i}") for i in range(12)]
    seen = set()
    while len(seen) < 24:
        tri = tuple(sorted(random.sample(range(12), 3)))
        if tri in seen:
            continue
        seen.add(tri)
        ls.add_constraint(sum(xs[i] for i in tri) == random.randint(0, 1))
    return ls


@pytest.fixture(scope="module")
def ternary_golay():
    """m=11, n=6, d=6 over GF(3): imperfect regime for l = 3 (the decoder corrects 2)."""
    B, m, n, d = golay_instance(3)
    assert (m, n, d) == (11, 6, 6)
    return max_lin_sat_from_B(GF(3), B, [i % 3 for i in range(m)])


def test_compute_A_bar_reproduces_A_in_perfect_regime():
    """With d = 8 and l = 3 every pair count must reproduce the semicircle matrix exactly."""
    B, m, n, d = golay_instance(2)
    ls = max_lin_sat_from_B(GF(2), B, [1] * m)
    dq = Dqi(ls, SyndromeDecoder.constructor())
    bm = _benchmarks(dq, 3, None)  # exhaustive: C(24,3) + C(24,2) + 24 decodes
    assert all(bm[k].epsilon == 0 for k in bm)
    # pass minimum_distance=0 to disable the "copy A" shortcut and force counting
    A_bar = compute_A_bar(B, ls.get_v(), 3, bm, minimum_distance=0)
    np.testing.assert_allclose(A_bar, make_A(2, 1, m, 3), atol=1e-12)


def test_compute_A_bar_rescales_sampled_benchmarks(random_xorsat):
    """Sampled benchmarks must estimate the same A_bar as exhaustive ones.

    The population rescaling total_size / (N_k N_k2) is what makes a sample of
    the weight-k errors stand for all of them; getting it wrong is an O(N) error,
    far outside the sampling noise this tolerance allows.
    """
    ls = random_xorsat
    B, d = ls.get_B(), ls.get_minimum_distance()
    dq = Dqi(ls, SyndromeDecoder.constructor())
    l = 3

    exhaustive = compute_A_bar(B, ls.get_v(), l, _benchmarks(dq, l, None), d)
    random.seed(1)
    np.random.seed(1)
    sampled = compute_A_bar(B, ls.get_v(), l, _benchmarks(dq, l, 500), d)

    np.testing.assert_allclose(sampled, exhaustive, atol=0.6)


def test_generic_expectation_matches_A_bar_formulation_binary(random_xorsat):
    """Two independent formulations of the imperfect-decoding model must agree (p = 2)."""
    ls = random_xorsat
    B, m = ls.get_B(), ls.get_m()
    dq = Dqi(ls, SyndromeDecoder.constructor())
    l = 3
    bm = _benchmarks(dq, l, None)
    eps = [bm[k].epsilon for k in range(l + 1)]
    _, w = get_eigenvector(make_A(2, 1, m, l), -1)

    A_bar = compute_A_bar(B, ls.get_v(), l, bm, ls.get_minimum_distance())
    via_A_bar = predict_from_A_bar(A_bar, m, l, w, eps)
    via_syndromes = compute_expectation(dq.instance, l, w, bm, dq.get_decoder())
    assert via_syndromes == pytest.approx(via_A_bar, rel=1e-9)


@pytest.mark.parametrize("p,l", [(2, 1), (2, 2), (2, 3), (3, 1), (3, 2)])
def test_generic_expectation_matches_brute_force_perfect_regime(p, l):
    B, m, n, d = golay_instance(p)
    rhs = [i % p for i in range(m)]
    ls = max_lin_sat_from_B(GF(p), B, rhs)
    dq = Dqi(ls, NearestNeighborDecoder.constructor())
    bm = _benchmarks(dq, l, None)
    _, w = get_eigenvector(make_A(p, 1, m, l), -1)
    got = compute_expectation(dq.instance, l, w, bm, dq.get_decoder())
    expected = brute_force_dqi_expectation(p, ls.get_B(), ls.get_F(), w)
    assert got == pytest.approx(expected, rel=1e-9)
    if p == 2:
        assert got == pytest.approx(
            _predict_dqi_performance_perfect_optimal_w(2, 1, m, l), rel=1e-9
        )


def test_expectation_rescales_sampled_benchmarks_binary(random_xorsat):
    """A sampled benchmark must give (approximately) the exhaustive expectation."""
    ls = random_xorsat
    dq = Dqi(ls, SyndromeDecoder.constructor())
    l = 3
    _, w = get_eigenvector(make_A(2, 1, ls.get_m(), l), -1)

    exhaustive = compute_expectation(dq.instance, l, w, _benchmarks(dq, l, None), dq.get_decoder())
    random.seed(2)
    np.random.seed(2)
    sampled = compute_expectation(dq.instance, l, w, _benchmarks(dq, l, 500), dq.get_decoder())

    assert sampled == pytest.approx(exhaustive, rel=0.05)


def test_expectation_rescales_sampled_benchmarks_non_binary(ternary_golay):
    """Same on GF(3), where the population is C(m,k) (p-1)^k, not C(m,k).

    Dropping the (p - 1)^k factor would scale the weight-3 block by 1/8 here.
    """
    ls = ternary_golay
    dq = Dqi(ls, SyndromeDecoder.constructor())
    l = 3
    _, w = get_eigenvector(make_A(3, 1, ls.get_m(), l), -1)

    exhaustive = compute_expectation(dq.instance, l, w, _benchmarks(dq, l, None), dq.get_decoder())
    random.seed(3)
    np.random.seed(3)
    sampled = compute_expectation(dq.instance, l, w, _benchmarks(dq, l, 500), dq.get_decoder())

    assert sampled == pytest.approx(exhaustive, rel=0.05)


def test_expectation_is_fast_on_ternary_golay(ternary_golay):
    """The Sage-loop version needed 54 s for this; bucketing must be well under a second."""
    ls = ternary_golay
    dq = Dqi(ls, SyndromeDecoder.constructor())
    l = 3
    bm = _benchmarks(dq, l, None)
    _, w = get_eigenvector(make_A(3, 1, ls.get_m(), l), -1)

    start = time.time()
    value = compute_expectation(dq.instance, l, w, bm, dq.get_decoder())
    elapsed = time.time() - start

    assert 0 <= value <= ls.get_m()
    assert elapsed < 2.0, f"compute_expectation took {elapsed:.2f}s"


def test_two_correct_errors_per_syndrome_are_rejected_binary(random_xorsat):
    """A syndrome decoder returns one error per syndrome, so this benchmark cannot occur.

    Summing over it anyway would model a coherent superposition of several errors
    of one coset, which is not the state DQI prepares.
    """
    ls = random_xorsat
    field, m = ls.field, ls.get_m()
    y1, y2 = _clashing_errors(ls, 2)
    bm = {
        0: BenchmarkResult([vector(field, [0] * m)], [], 0),
        1: BenchmarkResult([vector(field, [1] + [0] * (m - 1))], [], 0.5),
        2: BenchmarkResult([y1, y2], [], 0.5),
    }
    dq = Dqi(ls, SyndromeDecoder.constructor())
    with pytest.raises(ValueError, match="one error per syndrome"):
        compute_A_bar(ls.get_B(), ls.get_v(), 2, bm, ls.get_minimum_distance())
    with pytest.raises(ValueError, match="one error per syndrome"):
        compute_expectation(dq.instance, 2, np.array([1.0, 1.0, 1.0]), bm, dq.get_decoder())


def test_two_correct_errors_per_syndrome_are_rejected_non_binary(ternary_golay):
    """Same on GF(3): two weight-3 errors differing by a weight-6 codeword."""
    ls = ternary_golay
    field, m = ls.field, ls.get_m()
    y1, y2 = _clashing_errors(ls, 3)
    bm = {
        0: BenchmarkResult([vector(field, [0] * m)], [], 0),
        1: BenchmarkResult([vector(field, [1] + [0] * (m - 1))], [], 0.5),
        2: BenchmarkResult([vector(field, [1, 1] + [0] * (m - 2))], [], 0.5),
        3: BenchmarkResult([y1, y2], [], 0.5),
    }
    dq = Dqi(ls, SyndromeDecoder.constructor())
    with pytest.raises(ValueError, match="one error per syndrome"):
        compute_expectation(dq.instance, 3, np.ones(4), bm, dq.get_decoder())


def _xorsat(m, n, seed):
    random.seed(seed)
    ls = MaxXorSat()
    xs = [ls.new_var(f"x{i}") for i in range(n)]
    seen = set()
    while len(seen) < m:
        tri = tuple(sorted(random.sample(range(n), 3)))
        if tri in seen:
            continue
        seen.add(tri)
        ls.add_constraint(sum(xs[i] for i in tri) == random.randint(0, 1))
    return ls


def test_syndrome_key_has_no_length_limit():
    """n = 70 > 61 variables on GF(2): the former radix-2 key overflowed an int64 here.

    The two independent formulations (decoded partners, signed A_bar counts) must
    still agree.
    """
    ls = _xorsat(100, 70, seed=3)
    assert 2 ** ls.get_n() >= 2**62
    dq = Dqi(ls, SyndromeDecoder.constructor())
    l = 2
    bm = _benchmarks(dq, l, None)
    _, w = get_eigenvector(make_A(2, 1, ls.get_m(), l), -1)

    via_A_bar = estimate_via_A_bar(dq.instance, l, w, bm)
    via_syndromes = compute_expectation(dq.instance, l, w, bm, dq.get_decoder())
    assert ls.get_m() / 2 <= via_syndromes <= ls.get_m()
    assert abs(via_A_bar - via_syndromes) < 1e-9
