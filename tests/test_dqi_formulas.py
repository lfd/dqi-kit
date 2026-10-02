"""Check the closed-form DQI performance predictions in dqi.py against a brute-force oracle."""

import pytest
from sage.all import GF

from decoders import NearestNeighborDecoder
from dqi import (
    Dqi,
    _predict_dqi_performance_perfect,
    _predict_dqi_performance_perfect_optimal_w,
    compute_expectation,
    get_eigenvector,
    make_A,
)
from tests.oracles import brute_force_dqi_expectation, golay_instance, max_lin_sat_from_B
from tests.reference_impl import make_A_corrected, semicircle_corrected


@pytest.mark.parametrize("l", [1, 2, 3])
def test_binary_semicircle_law_matches_brute_force(l):
    B, m, n, d = golay_instance(2)
    assert d == 8 >= 2 * l + 2
    F = [{1}] * m
    _, w = get_eigenvector(make_A(2, 1, m, l), -1)
    brute = brute_force_dqi_expectation(2, B, F, w)
    assert _predict_dqi_performance_perfect(2, 1, m, l, w) == pytest.approx(brute, rel=1e-9)
    assert _predict_dqi_performance_perfect_optimal_w(2, 1, m, l) == pytest.approx(brute, rel=1e-9)


@pytest.mark.parametrize("r,l", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_ternary_semicircle_law_corrected_diagonal_matches_brute_force(r, l):
    B, m, n, d = golay_instance(3)
    assert d == 6 >= 2 * l + 2
    F = [set(range(r))] * m
    A = make_A_corrected(3, r, m, l)
    _, w = get_eigenvector(A, -1)
    brute = brute_force_dqi_expectation(3, B, F, w)
    assert semicircle_corrected(3, r, m, l) == pytest.approx(brute, rel=1e-9)


@pytest.mark.parametrize("r,l", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_ternary_semicircle_law_as_implemented(r, l):
    """D1 regression: make_A's diagonal now matches brute force for p > 2."""
    B, m, n, d = golay_instance(3)
    assert d == 6 >= 2 * l + 2
    F = [set(range(r))] * m
    _, w = get_eigenvector(make_A(3, r, m, l), -1)
    brute = brute_force_dqi_expectation(3, B, F, w)
    assert _predict_dqi_performance_perfect(3, r, m, l, w) == pytest.approx(brute, rel=1e-6)


@pytest.mark.parametrize("l", [1, 2])
def test_non_binary_exact_path_matches_brute_force(l):
    """D2 regression: the non-binary interference path matches brute force."""
    B, m, n, d = golay_instance(3)
    rhs = [i % 3 for i in range(m)]
    ls = max_lin_sat_from_B(GF(3), B, rhs)
    dq = Dqi(ls, NearestNeighborDecoder.constructor())
    _, w = get_eigenvector(make_A(3, 1, m, l), -1)
    got = compute_expectation(dq.instance, l, w, dq._benchmarks(l, None), dq.get_decoder())
    expected = brute_force_dqi_expectation(3, ls.get_B(), ls.get_F(), w)
    assert got == pytest.approx(expected, rel=1e-6)
