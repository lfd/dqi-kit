"""Regression tests for the defects listed in report.md.

Every test asserts the *correct* behaviour and is marked ``xfail(strict=True)``.
As long as the bug is present pytest reports the test as XFAIL.  Once the bug
is fixed the test XPASSes, pytest fails the run because of ``strict=True`` and
the marker should be removed.  So: the list of xfails below IS the open bug list.
"""

import copy

import numpy as np
import pytest
from sage.all import GF

from classical_solvers import BruteForceSolver, MaxLinSatAnneal, OrToolsSolver
from constraints import LinSatConstraint, LinSatVar
from decoders import GeneralizedReedSolomonDecoder, SyndromeDecoder
from dqi import Dqi, _predict_dqi_performance_perfect_optimal_w, make_A
from max_constraint_sat import IntConstraint, IntTerms, MaxConstraintSat, Relation
from max_lin_sat import (
    MaxLinSat,
    MaxXorSat,
    MergeStrategy,
    OptimalPolynomialIntersection,
    UnweightedIsing,
)

bug = lambda reason: pytest.mark.xfail(strict=True, reason=reason)  # noqa: E731


# ----------------------------------------------------------------------------- dqi.py


def test_estimate_with_interference_method_runs():
    """Was ``rhs_approximation=False`` and dead code behind the D4 ``NameError``.

    The flag is now ``method="interference"``, which is also what ``auto`` picks in
    the imperfect regime.
    """
    ls = MaxXorSat()
    xs = [ls.new_var(f"x{i}") for i in range(4)]
    for i in range(4):
        ls.add_constraint(xs[i] + xs[(i + 1) % 4] == 1)
        ls.add_constraint(xs[i] + xs[(i + 2) % 4] == 1)
    dq = Dqi(ls, SyndromeDecoder.constructor())
    value = dq.estimate_solution_quality(
        l=2,
        method="interference",
        n_decoding_samples=None,
    )
    assert 0 <= value <= ls.get_m()
    assert dq.last_estimate.method == "interference"
    assert dq.last_estimate.exact is True  # exhaustive benchmarks


def test_g_tilde_depends_on_constraint():
    ls = MaxLinSat(GF(3), equal_size_F_i=False)
    x, y, z = (ls.new_var(c) for c in "xyz")
    ls.add_constraint(x == 0)
    ls.add_constraint(y == 1)
    ls.add_constraint(z == 2)
    ls.add_constraint(x + y == 0)
    dq = Dqi(ls)
    g_tilde = dq._get_g_tilde()
    values = [g_tilde[i](1) for i in range(4)]
    # F = [{0}, {0}, {1}, {2}] after sorting: rows 0 and 1 share F_i, rows 2, 3 differ.
    assert values[0] == pytest.approx(values[1])
    assert values[0] != pytest.approx(values[2])
    assert values[2] != pytest.approx(values[3])


def test_estimate_respects_l_in_perfect_regime():
    """D5: the perfect branch used to discard l and return the (d - 1) // 2 value."""
    from tests.oracles import golay_instance, max_lin_sat_from_B

    B, m, n, d = golay_instance(2)  # d = 8 -> semicircle law valid for l <= 3
    ls = max_lin_sat_from_B(GF(2), B, [1] * m)

    class RadiusDecoder(SyndromeDecoder):
        def decoding_radius(self):
            return 3

    dq = Dqi(ls, lambda inst: RadiusDecoder(inst))
    from dqi import _predict_dqi_performance_perfect_optimal_w

    assert dq.estimate_solution_quality(l=1) == pytest.approx(
        _predict_dqi_performance_perfect_optimal_w(2, 1, m, 1)
    )
    assert dq.estimate_solution_quality(l=2) == pytest.approx(
        _predict_dqi_performance_perfect_optimal_w(2, 1, m, 2)
    )


def test_perfect_regime_needs_2l_plus_2_at_most_d():
    """The gate used to be l <= (d - 1) // 2, one too large for odd d.

    A_bar == A requires k + k' + 1 < d for all k, k' <= l, the same condition
    compute_A_bar applies per entry, i.e. 2l + 2 <= d.  On the BCH [15, 7, 5]
    code the old gate accepted l = 2, where the exact A_bar[2, 2] is -5.14 instead
    of 0 and the closed form overshoots by 0.84.
    """
    from tests.oracles import max_lin_sat_from_B, odd_distance_instance

    B, m, n, d = odd_distance_instance()
    assert d == 5 and d % 2 == 1
    ls = max_lin_sat_from_B(GF(2), B, [1] * m)
    dq = Dqi(ls, SyndromeDecoder.constructor())

    assert dq.get_decoder().decoding_radius() == 2  # unique decoding reaches l = 2
    assert dq._has_exact_A(1, None) is True  # ... but exactness stops at l = 1
    assert dq._has_exact_A(2, None) is False

    assert dq.estimate_solution_quality(l=1) == pytest.approx(
        _predict_dqi_performance_perfect_optimal_w(2, 1, m, 1)
    )
    with pytest.raises(ValueError, match="not in the perfect-decoding regime"):
        dq.estimate_solution_quality(l=2, method="analytical")


def test_semicircle_law_default_l_and_warning():
    """semicircle_law_solution_quality took no l and hard-coded (d - 1) // 2 as a float."""
    from tests.oracles import max_lin_sat_from_B, odd_distance_instance

    B, m, n, d = odd_distance_instance()
    ls = max_lin_sat_from_B(GF(2), B, [1] * m)
    dq = Dqi(ls, SyndromeDecoder.constructor())

    assert dq.semicircle_law_solution_quality() == pytest.approx(
        _predict_dqi_performance_perfect_optimal_w(2, 1, m, d // 2 - 1)
    )
    # an explicit l outside the exact regime still computes, but says so
    with pytest.warns(UserWarning, match="A_bar is not exactly A"):
        dq.semicircle_law_solution_quality(2)


def test_default_l_is_the_largest_exact_perfect_l():
    """l=None used to mean "the decoder's radius", or a ValueError without one."""
    from tests.oracles import golay_instance, max_lin_sat_from_B

    B, m, n, d = golay_instance(2)  # d = 8
    ls = max_lin_sat_from_B(GF(2), B, [1] * m)
    dq = Dqi(ls, SyndromeDecoder.constructor())

    value = dq.estimate_solution_quality(details=True)
    assert value.l == d // 2 - 1 == 3
    assert value.method == "analytical"
    assert value.guaranteed is True
    assert value.epsilon is None  # the radius alone settles it: no decoding at all
    assert float(value) == pytest.approx(_predict_dqi_performance_perfect_optimal_w(2, 1, m, 3))


def test_perfect_regime_honours_w():
    """The perfect branch ignored w; _predict_dqi_performance_perfect was unreachable."""
    from tests.oracles import golay_instance, max_lin_sat_from_B

    B, m, n, d = golay_instance(2)
    ls = max_lin_sat_from_B(GF(2), B, [1] * m)
    dq = Dqi(ls, SyndromeDecoder.constructor())

    w = np.array([1.0, 0.0, 0.0, 0.0])  # all weight on the zero-error term
    assert dq.estimate_solution_quality(l=3, w=w) == pytest.approx(m / 2)
    assert dq.estimate_solution_quality(l=3) == pytest.approx(
        _predict_dqi_performance_perfect_optimal_w(2, 1, m, 3)
    )
    with pytest.raises(ValueError, match=r"w must have shape"):
        dq.estimate_solution_quality(l=3, w=np.ones(3))


def test_perfect_regime_detected_without_a_decoding_radius():
    """A decoder with no radius used to be barred from the perfect regime entirely.

    Now epsilon is measured, and all-zero epsilon plus 2l + 2 <= d takes the same
    closed form -- flagged as not guaranteed, because sampled benchmarks cannot
    prove epsilon = 0.
    """
    from tests.oracles import golay_instance, max_lin_sat_from_B

    B, m, n, d = golay_instance(2)
    ls = max_lin_sat_from_B(GF(2), B, [1] * m)

    class NoRadius(SyndromeDecoder):
        def decoding_radius(self):
            return None

    dq = Dqi(ls, lambda inst: NoRadius(inst))
    value = dq.estimate_solution_quality(details=True)
    assert value.l == 3 and value.method == "analytical"
    assert value.guaranteed is False
    assert value.epsilon == (0.0, 0.0, 0.0, 0.0)
    assert float(value) == pytest.approx(_predict_dqi_performance_perfect_optimal_w(2, 1, m, 3))


def test_estimate_rejects_non_prime_fields():
    """Every estimator is derived over F_p; GF(4) used to reach the closed form."""
    ls = MaxLinSat(GF(4), equal_size_F_i=False)
    xs = [ls.new_var(f"x{i}") for i in range(3)]
    for i in range(3):
        ls.add_constraint(xs[i] + xs[(i + 1) % 3] == 1)
    with pytest.raises(ValueError, match="prime field"):
        Dqi(ls, SyndromeDecoder.constructor()).estimate_solution_quality(l=1)


def test_lower_bound_average_v_indexes_epsilon_by_weight(monkeypatch):
    """_dqi_lower_bound_average_v built epsilon with ``reversed(range(1, l + 1))``.

    The benchmarks may be *computed* in that order (the comment says large errors
    first help some decoders), but the consumer reads the list positionally --
    ``norm = sum w_k^2 (1 - epsilon_k)`` -- so the reversed list attached
    epsilon_l to weight 1.  The length stayed l + 1, so the assert inside
    ``_upper_bound_dqi_performance_imperfect_average_v`` did not catch it, and on
    small instances the D6 clamp hides the wrong number; hence the direct check on
    the vector that is handed over.
    """
    ls = MaxXorSat()
    xs = [ls.new_var(f"x{i}") for i in range(5)]
    for i in range(5):
        ls.add_constraint(xs[i] + xs[(i + 1) % 5] == 1)
        ls.add_constraint(xs[i] + xs[(i + 2) % 5] == 1)
    dq = Dqi(ls, SyndromeDecoder.constructor())
    dec = dq.get_decoder()

    l = 2
    by_weight = [0.0] + [dec.get_benchmarks(l, k, None).epsilon for k in range(1, l + 1)]
    assert by_weight[1:] != by_weight[1:][::-1]  # otherwise the test is vacuous

    import dqi as dqi_module

    original = dqi_module._upper_bound_dqi_performance_imperfect_average_v
    seen = []

    def spy(m, l, epsilon, w=None):
        seen.append(list(epsilon))
        return original(m, l, epsilon, w)

    monkeypatch.setattr(dqi_module, "_upper_bound_dqi_performance_imperfect_average_v", spy)
    # the bound degenerates to m/2 on this instance (D6), which is now reported
    with pytest.warns(UserWarning, match="clamped"):
        dq.estimate_solution_quality(l=l, method="average_v_bound", n_decoding_samples=None)
    assert seen == [by_weight]


def test_g_tilde_rejects_degenerate_r():
    """_compute_g_g_tilde divides by phi = sqrt(4r(1 - r/p)), which is 0 at r = p.

    It fires before make_A on the non-binary exact path, so it needs the same
    guard.  Since L2 is fixed, compute_B_F_weights never emits F_i = F itself
    (an always-true level is dropped), so the degenerate F is set directly, as
    a subclass or a hand-built instance could.
    """
    field = GF(3)
    ls = MaxLinSat(field)
    xs = [ls.new_var(f"x{i}") for i in range(4)]
    for i in range(4):
        ls.add_constraint(LinSatConstraint(field, xs, [1, 1 + i % 2, 2, i % 3], {field(0): 1}))
    ls.get_B()
    ls.F = [set(field) for _ in range(ls.get_m())]
    assert ls.get_r() == 3  # == p, every F_i is the whole field

    with pytest.raises(ValueError, match="requires 0 < r < p"):
        Dqi(ls)._compute_g_g_tilde()


def test_make_A_rejects_degenerate_r():
    """r=0 and r=p make the D1 diagonal's sqrt(r(p-r)) vanish.

    r=p is reachable through MergeStrategy.USE_LOOSEST, so make_A must reject it
    rather than divide by zero.
    """
    assert make_A(3, 1, 11, 2).shape == (3, 3)
    for r in (0, 3):
        with pytest.raises(ValueError, match="requires 0 < r < p"):
            make_A(3, r, 11, 2)


# ------------------------------------------------------------------ max_constraint_sat.py


def test_minimize_negates_objective():
    p = MaxConstraintSat()
    a = p.new_binary_var("a")
    p.add_objective(a, minimize=True)
    assert p.objectives[0].terms == {((a.id, 1),): -1}


def test_boolean_constraint_weight_is_applied():
    p = MaxConstraintSat()
    a, b = p.new_binary_var("a"), p.new_binary_var("b")
    p.add_boolean_constraint(a | b, weight=2)
    assert p.objectives[0].terms == {
        ((a.id, 1),): 2,
        ((b.id, 1),): 2,
        ((a.id, 1), (b.id, 1)): -2,
    }


def test_greater_equal_keeps_upper_bound():
    p = MaxConstraintSat()
    a = p.new_var("a", 0, 3)
    ls = MaxLinSat(GF(5))
    v = ls.new_var("a")
    c = (a >= 1).integerize().to_lin_sat_constraint(GF(5), {a.id: v})
    assert {int(x) for x in c.rhs} == {1, 2, 3}
    c = (a > 1).integerize().to_lin_sat_constraint(GF(5), {a.id: v})
    assert {int(x) for x in c.rhs} == {2, 3}


# ------------------------------------------------------------------------ constraints.py


def test_get_first_coef_returns_leading_coefficient():
    F = GF(5)
    vs = [LinSatVar(F, 1, "y"), LinSatVar(F, 0, "x")]
    assert LinSatConstraint(F, vs, [F(3), F(2)], {0}).get_first_coef() == F(2)


def test_scaled_and_reordered_constraints_share_an_id():
    ls = MaxLinSat(GF(5))
    x, y = ls.new_var("x"), ls.new_var("y")
    assert (x + 2 * y == 1).get_id() == (2 * x + 4 * y == 2).get_id()
    assert (x + 2 * y == 1).get_id() == (2 * y + x == 1).get_id()


def test_unknown_variable_is_rejected():
    ls = MaxLinSat(GF(2))
    ls.new_var("x")
    other = MaxLinSat(GF(2))
    other.new_var("a")
    foreign = other.new_var("x")  # id 1, does not exist in `ls`
    with pytest.raises(ValueError):
        ls.add_constraint(foreign == 1)


# ------------------------------------------------------------------------ max_lin_sat.py


def test_unweighted_ising_constructs():
    UnweightedIsing(np.array([[0, 1], [1, 0]]), np.array([1, -1]))


def test_constant_offset_removed_for_all_field_elements():
    ls = MaxLinSat(GF(3), equal_size_F_i=False, merge_strategy=MergeStrategy.DUPLICATES)
    a, b = ls.new_var("a"), ls.new_var("b")
    ls.add_constraint(a + b == 0)
    ls.add_constraint(a + b == 1)
    ls.add_constraint(a + b == 2, weight=2)
    ls.add_constraint(a == 0)
    # a+b has weights {0:1, 1:1, 2:2}: the always-true level should be dropped.
    assert not any(len(F_i) == 3 for F_i in ls.get_F())


def test_minimum_distance_gf3_cycle():
    from tests.oracles import true_minimum_distance

    ls = MaxLinSat(GF(3), equal_size_F_i=False)
    a, b, c = ls.new_var("a"), ls.new_var("b"), ls.new_var("c")
    ls.add_constraint(a + b == 0)
    ls.add_constraint(b + c == 0)
    ls.add_constraint(a + c == 0)
    ls.add_constraint(a + b + c == 1)
    assert ls.get_minimum_distance() == true_minimum_distance(ls.get_B()) == 4


def test_minimum_distance_gf3_cycle_bound_lifted_to_difference_rows():
    """A cycle of rows proportional to x_u - x_v is a dependency over any field,
    so the girth bound (and its exactness) applies beyond GF(2)."""
    from tests.oracles import true_minimum_distance

    ls = MaxLinSat(GF(3), equal_size_F_i=False)
    a, b, c, d = (ls.new_var(name) for name in "abcd")
    ls.add_constraint(a - b == 0)
    ls.add_constraint(2 * b - 2 * c == 1)
    ls.add_constraint(c - a == 0)
    ls.add_constraint(c - d == 1)
    ls.add_constraint(d == 2)
    assert ls._compute_graph_minimum_distance() == (3, True)
    assert ls.get_minimum_distance() == true_minimum_distance(ls.get_B()) == 3

    # a + b rows: a 3-cycle is no dependency over GF(3), a 4-cycle is
    ls = MaxLinSat(GF(3), equal_size_F_i=False)
    a, b, c, d = (ls.new_var(name) for name in "abcd")
    ls.add_constraint(a + b == 0)
    ls.add_constraint(b + c == 0)
    ls.add_constraint(c + d == 0)
    ls.add_constraint(d + a == 0)
    ls.add_constraint(a + b + c == 1)
    assert ls._compute_graph_minimum_distance() == (4, False)
    assert ls.get_minimum_distance() == true_minimum_distance(ls.get_B()) == 4


def test_opi_default_decoder():
    F7 = GF(7)
    opi = OptimalPolynomialIntersection(F7, 1, {F7(i): {F7(i)} for i in range(7)})
    Dqi(opi).get_decoder()


def test_use_loosest_takes_the_union_of_the_rhs_groups():
    """USE_LOOSEST built ``total_rhs`` as a dict and called .update() with a set.

    That raised TypeError on every instance, so the strategy was dead code.
    """
    field = GF(3)
    ls = MaxLinSat(field, merge_strategy=MergeStrategy.USE_LOOSEST)
    xs = [ls.new_var(f"x{i}") for i in range(3)]
    for x in xs:
        ls.add_constraint(
            LinSatConstraint(field, [x], [1], {field(0): 1, field(2): 2}, n_equations=2),
            disable_warnings=True,
        )

    F = ls.get_F()
    assert len(F) == 3
    # the loosest reading keeps both weight groups, not just the strictest one
    assert all(F_i == {field(0), field(2)} for F_i in F)
    assert ls.get_r() == 2


# --------------------------------------------------------------------------- decoders.py


def test_benchmark_cache_key_includes_l():
    """get_benchmarks used to cache by (n_errors, n_tries) and ignore l."""
    ls = MaxXorSat()
    xs = [ls.new_var(f"x{i}") for i in range(5)]
    for i in range(5):
        ls.add_constraint(xs[i] + xs[(i + 1) % 5] == 1)
        ls.add_constraint(xs[i] + xs[(i + 2) % 5] == 1)
    sd = SyndromeDecoder(ls)
    sd.get_benchmarks(l=1, n_errors=1, n_tries=None)
    sd.get_benchmarks(l=3, n_errors=1, n_tries=None)
    assert len(sd.benchmarks) == 2


def test_complete_decoders_report_their_decoding_radius():
    """E5: both complete decoders returned None, which barred the perfect regime.

    Nearest-neighbour search and a minimal-weight coset-leader table are both
    provably correct inside half the minimum distance.
    """
    from decoders import NearestNeighborDecoder
    from tests.oracles import golay_instance, max_lin_sat_from_B

    B, m, n, d = golay_instance(2)
    ls = max_lin_sat_from_B(GF(2), B, [1] * m)
    assert d == 8
    for constructor in (SyndromeDecoder.constructor(), NearestNeighborDecoder.constructor()):
        dec = constructor(ls)
        assert dec.decoding_radius() == (d - 1) // 2 == 3
        # and the claim holds: no failure at any weight up to the radius
        assert dec.get_benchmarks(3, 3, 50).epsilon == 0


def test_grs_decoder_uses_dual_code():
    F7 = GF(7)
    opi = OptimalPolynomialIntersection(
        F7,
        1,
        {F7(i): {F7(i)} for i in range(7)},
        GeneralizedReedSolomonDecoder.constructor(),
    )
    assert opi.get_minimum_distance() == 3  # dual RS code [7,5,3]
    dec = GeneralizedReedSolomonDecoder(opi)
    # A distance-3 code corrects every single error, so the benchmark for k = 1
    # must be perfect; it is not, because the decoder decodes instance.code.
    assert dec.get_benchmarks(1, 1, None).epsilon == 0
    # ... while a weight-2 error is not uniquely decodable and must fail.
    assert dec.get_benchmarks(2, 2, None).epsilon > 0


def test_random_solution_value_on_a_fresh_instance():
    """L5: read ``self.F`` before anything had computed it -> TypeError."""
    ls = MaxXorSat()
    xs = [ls.new_var(f"x{i}") for i in range(3)]
    ls.add_constraint(xs[0] + xs[1] == 1)
    ls.add_constraint(xs[1] + xs[2] == 0)
    assert ls.get_random_solution_value() == 1.0  # m / 2


def test_compute_B_F_weights_leaves_constraints_alone():
    """L6: the gcd of the weights was divided out *in* the caller's constraints."""
    F = GF(2)
    ls = MaxLinSat(F, equal_size_F_i=False)
    a, b = ls.new_var("a"), ls.new_var("b")
    c = LinSatConstraint(F, [a, b], [1, 1], {F(1): 2}, n_equations=2)  # stored as is
    ls.add_constraint(c)
    ls.add_constraint(a == 1, weight=2)  # gcd of all weights is 2
    assert ls.get_B().nrows() == 2  # the gcd is still divided out: one row each
    assert c.rhs == {F(1): 2}  # ... but not inside the caller's constraint


# ------------------------------------------------------------------ max_constraint_sat.py


def test_int_constraint_keeps_required_field_order():
    """M6: the argument was overwritten with None and always recomputed."""
    c = IntConstraint(IntTerms({}, {}), Relation.EQUALS, required_field_order=(None, 7))
    assert c.required_field_order == (None, 7)


# ------------------------------------------------------------------- classical_solvers.py


def test_brute_force_solver_returns_a_vector():
    """S2: returned itertools' tuple while every other solver returns a Sage vector."""
    from sage.modules.free_module_element import FreeModuleElement

    ls = MaxXorSat()
    xs = [ls.new_var(f"x{i}") for i in range(3)]
    ls.add_constraint(xs[0] + xs[1] == 1)
    ls.add_constraint(xs[1] + xs[2] == 1)
    ls.add_constraint(xs[0] + xs[2] == 0)
    solution = BruteForceSolver(ls).get_solution()
    assert isinstance(solution, FreeModuleElement)
    assert ls.evaluate_solution(solution) == 3


def test_annealer_moves_always_change_the_state():
    """S3: ``move`` added ``random.choice(field)`` including 0, a no-op 1/p of the time."""
    ls = MaxLinSat(GF(3), equal_size_F_i=False)
    xs = [ls.new_var(f"x{i}") for i in range(3)]
    ls.add_constraint(xs[0] + xs[1] == 1)
    ls.add_constraint(xs[1] + 2 * xs[2] == 2)
    annealer = MaxLinSatAnneal(ls)
    assert annealer.updates == 0  # progress output off via the library's knob
    for _ in range(60):
        before = copy.copy(annealer.state)
        annealer.move()
        assert annealer.state != before


def test_is_optimal_before_solve():
    ls = MaxXorSat()
    xs = [ls.new_var(f"x{i}") for i in range(3)]
    ls.add_constraint(xs[0] + xs[1] == 1)
    ls.add_constraint(xs[1] + xs[2] == 1)
    ls.add_constraint(xs[0] + xs[2] == 1)
    ls.add_constraint(xs[0] == 1)
    assert OrToolsSolver(ls).is_optimal() in (True, False)


def test_benchmark_correct_set_has_one_error_per_syndrome():
    """E7 / root cause of the D9 symptom.

    Ternary Golay dual: d = 6, radius 2.  At weight 3 every error y has a partner
    y' = y - c (c a weight-6 codeword) at the same distance from 0 and from c, so
    NearestNeighborDecoder resolves a tie and reports *every* weight-3 error as
    decoded correctly (epsilon_3 = 0), 1563 correct errors on 683 syndromes.  A
    decoder that only sees the syndrome, as in the DQI circuit, can return one
    error per syndrome; SyndromeDecoder on the same instance gives 683/683 and
    epsilon_3 = 0.667.
    """
    from collections import Counter

    from decoders import NearestNeighborDecoder
    from tests.oracles import golay_instance, max_lin_sat_from_B

    B, m, n, d = golay_instance(3)
    ls = max_lin_sat_from_B(GF(3), B, [i % 3 for i in range(m)])
    assert d == 6
    dec = Dqi(ls, NearestNeighborDecoder.constructor()).get_decoder()
    bm = dec.get_benchmarks(3, 3, None)  # weight 3 > radius 2: cannot all succeed
    assert bm.epsilon > 0
    B_T = ls.get_B().T
    syndromes = Counter(tuple(B_T * y) for y in bm.correct)
    assert max(syndromes.values()) == 1
