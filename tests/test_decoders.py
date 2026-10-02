"""Tests for the two decoding interfaces of decoders.AbstractDecoder.

A decoder implements exactly one of ``decode_syndrome`` (what the DQI circuit
does) and ``decode_codeword``; the base class turns the latter into the former
with a fixed coset representative of the syndrome.  Both paths must therefore
report exactly one error per syndrome, and must agree with each other.
"""

import random

import pytest
from sage.all import GF, matrix, vector

from decoders import (
    AbstractDecoder,
    BeliefPropagationDecoder,
    BeliefPropagationOsdDecoder,
    GeneralizedReedSolomonDecoder,
    InformationSetDecoder,
    NearestNeighborDecoder,
    SyndromeDecoder,
    generate_all_errors,
)
from max_lin_sat import MaxLinSat, MaxXorSat
from tests.oracles import golay_instance, max_lin_sat_from_B


@pytest.fixture(scope="module")
def ternary_golay():
    """Dual ternary Golay code: m = 11, n = 6, d = 6 (decoding radius 2)."""
    B, m, n, d = golay_instance(3)
    assert (m, n, d) == (11, 6, 6)
    return max_lin_sat_from_B(GF(3), B, [i % 3 for i in range(m)])


@pytest.fixture(scope="module")
def random_xorsat():
    """m=24, n=12 random 3-XORSAT with minimum distance 4 (radius 1)."""
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
    assert ls.get_minimum_distance() == 4
    return ls


def test_syndrome_and_codeword_paths_agree(ternary_golay):
    """NearestNeighbor (word based) and Syndrome (syndrome based) must agree.

    Weight 3 exceeds the decoding radius 2 of the dual ternary Golay code: only
    440 of the 1320 weight-3 errors are the decoder's estimate for their own
    syndrome, so epsilon_3 = 2/3.  Feeding the received word to the word based
    decoder instead used to break the ties per word and report all 1320.
    """
    ls = ternary_golay
    H = ls.get_B().T
    results = {}
    for cls in (NearestNeighborDecoder, SyndromeDecoder):
        dec = cls(ls)
        counts = {}
        total_correct = 1  # the weight-0 error is always decoded correctly
        for k in (1, 2, 3):
            bm = dec.get_benchmarks(3, k, None)
            counts[k] = (len(bm.correct), len(bm.incorrect))
            total_correct += len(bm.correct)
            # one error per syndrome, for both interfaces
            assert len({tuple(H * e) for e in bm.correct}) == len(bm.correct)
        results[cls.__name__] = (
            counts,
            total_correct,
            float(dec.get_benchmarks(3, 3, None).epsilon),
        )

    for counts, total_correct, epsilon in results.values():
        assert counts == {1: (22, 0), 2: (220, 0), 3: (440, 880)}
        assert total_correct == 683
        assert epsilon == pytest.approx(2 / 3, abs=1e-3)

    assert results["NearestNeighborDecoder"] == results["SyndromeDecoder"]


def test_representative_solves_the_syndrome_equation(ternary_golay):
    """rho(s) is a word with H rho(s) == s, for every syndrome in the image of H."""
    ls = ternary_golay
    H = ls.get_B().T
    dec = NearestNeighborDecoder(ls)
    for e in generate_all_errors(GF(3), ls.get_m(), 2):
        s = H * e
        rho = dec.representative(s)
        assert len(rho) == ls.get_m()
        assert H * rho == s


def test_representative_handles_rank_deficient_H():
    """H = B^T may be rank deficient (dependent columns of B)."""
    field = GF(3)
    # x2 does not occur, so B has a zero column and B^T a zero row
    B = matrix(field, [[1, 1, 0], [1, 2, 0], [2, 1, 0], [0, 1, 0]])
    ls = max_lin_sat_from_B(field, B, [0, 1, 2, 1])
    H = ls.get_B().T
    assert H.rank() < H.nrows()  # rank deficient

    class Dummy(AbstractDecoder):
        def decoding_radius(self):
            return None

        def decode_codeword(self, y, l):
            return vector(self.instance.field, [0] * len(y))

    dec = Dummy(ls)
    m = ls.get_m()
    for k in (0, 1, 2):
        for e in generate_all_errors(field, m, k):
            s = H * e
            assert H * dec.representative(s) == s
    # and the resulting estimate is a valid error for that syndrome
    for e in generate_all_errors(field, m, 1):
        s = H * e
        assert H * dec.error_estimate(s, 1) == s


def test_decoder_must_implement_exactly_one_interface(ternary_golay):
    ls = ternary_golay

    class Neither(AbstractDecoder):
        def decoding_radius(self):
            return None

    class Both(AbstractDecoder):
        def decoding_radius(self):
            return None

        def decode_syndrome(self, s, l):
            return None

        def decode_codeword(self, y, l):
            return y

    with pytest.raises(TypeError):
        Neither(ls)
    with pytest.raises(TypeError):
        Both(ls)


@pytest.mark.parametrize("cls", [BeliefPropagationDecoder, BeliefPropagationOsdDecoder])
def test_belief_propagation_decoders_correct_single_errors(cls, random_xorsat):
    """d = 4, so every weight-1 error is uniquely decodable from its syndrome."""
    ls = random_xorsat
    H = ls.get_B().T
    dec = cls(ls)
    bm = dec.get_benchmarks(2, 1, None)
    assert len(bm.correct) == ls.get_m()
    assert bm.epsilon == 0
    bm2 = dec.get_benchmarks(2, 2, None)
    assert 0 < bm2.epsilon < 1
    assert len({tuple(H * e) for e in bm2.correct}) == len(bm2.correct)


# ------------------------------------------------------ GeneralizedReedSolomonDecoder


def _opi(p: int, degree: int):
    """Optimal Polynomial Intersection over GF(p) with one point per x; DQI code = dual GRS."""
    from max_lin_sat import OptimalPolynomialIntersection

    F = GF(p)
    return OptimalPolynomialIntersection(
        F, degree, {F(i): {F(i)} for i in range(p)}, GeneralizedReedSolomonDecoder.constructor()
    )


def test_grs_decoder_is_built_on_the_dual_code():
    opi = _opi(7, 1)  # polynomial code [7, 2]; DQI code = its dual [7, 5, 3]
    dec = GeneralizedReedSolomonDecoder(opi)
    assert dec.code == opi.get_code() == opi.dual_code
    assert dec.decoding_radius() == 1


def test_grs_falls_back_to_unique_decoding_when_list_decoding_is_infeasible():
    """[7, 5]: Johnson radius 1.71, so Guruswami-Sudan cannot be built for tau = 2."""
    dec = GeneralizedReedSolomonDecoder(_opi(7, 1))
    assert dec.get_decoder_by_l(2) is dec.decoder
    assert dec.get_benchmarks(2, 2, None).epsilon > 0


def test_grs_empty_list_counts_as_decoding_failure():
    """[13, 6, 8]: unique radius 3, Guruswami-Sudan constructible for tau = 4.

    Weight-3 errors are inside the unique radius, so the list is exactly [0].
    Weight-5 errors are beyond the list radius: the list is empty (failure) or
    holds a wrong codeword; either way none may be reported correct, and the
    empty list must not abort the benchmark with an IndexError.
    """
    dec = GeneralizedReedSolomonDecoder(_opi(13, 6))
    gs = dec.get_decoder_by_l(4)
    assert gs is not dec.decoder and gs.decoding_radius() == 4
    assert dec.get_benchmarks(4, 3, 50).epsilon == 0
    assert dec.get_benchmarks(4, 5, 50).epsilon == 1.0


def test_benchmark_with_impossible_error_weight_raises(ternary_golay):
    dec = SyndromeDecoder(ternary_golay)
    with pytest.raises(ValueError, match="No errors of weight"):
        dec.get_benchmarks(1, 12, None)  # m = 11


def test_syndrome_error_array_matches_the_table(random_xorsat, ternary_golay):
    """``get_syndrome_error_array`` is the table's errors, in order, as an array.

    The estimator probes the coset-leader table with numpy, so the Sage vectors
    are converted once -- over GF(2) through ``support()``, which touches only
    the non-zero positions, and elementwise on every other field.  Both have to
    reproduce the table exactly, including the row order, because the consumer
    recomputes the syndromes from these rows (``s = B^T e``) and pairs them up
    positionally.
    """
    for instance, l in ((random_xorsat, 2), (ternary_golay, 2)):
        decoder = SyndromeDecoder(instance)
        table = decoder.get_syndrome_table(l)
        array = decoder.get_syndrome_error_array(l)

        assert array.shape == (len(table), instance.get_m())
        expected = [[int(c) for c in e] for e in table.values()]
        assert array.tolist() == expected
        # the syndrome of row j must be the key it was stored under
        B = matrix(instance.field, instance.get_B())
        for j, s in enumerate(table):
            assert B.T * vector(instance.field, array[j].tolist()) == s
        # cached, not rebuilt
        assert decoder.get_syndrome_error_array(l) is array


def test_information_set_decoder_is_reproducible():
    """Sage's ISD picks its information sets at random, and when a coset has
    several weight-1 leaders that choice decides which one is "decoded
    correctly".  With its own seed and a fixed search size the decoder makes
    the same decisions in every run; a different seed makes different ones.
    """
    ls = MaxLinSat(GF(5), equal_size_F_i=False)
    x, y, z = (ls.new_var(name) for name in "xyz")
    # weighted constraints are emitted as identical rows, so every weight-1
    # coset has two or three weight-1 leaders to choose from
    for constraint, weight in [
        (x == 1, 3),
        (y == 2, 3),
        (x + y == 0, 2),
        (x + 2 * z == 1, 2),
        (y + z == 3, 2),
    ]:
        ls.add_constraint(constraint, weight=weight)

    def correct(seed):
        decoder = InformationSetDecoder(ls, seed=seed, search_size=1)
        return [tuple(e) for e in decoder.get_benchmarks(1, 1, None).correct]

    first = correct(0)
    assert len(first) > 0
    assert correct(0) == first
    assert correct(1) != first
