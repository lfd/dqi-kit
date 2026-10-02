"""Oracles used by the tests: corrected closed forms and the A_bar formulation.

``compute_A_bar`` / ``predict_from_A_bar`` / ``estimate_via_A_bar`` are the
GF(2) interference-matrix route to the DQI expectation.  Mathematically they
give the same number as ``dqi.compute_expectation`` at p = 2, but by a different
formulation (signed pair counts instead of complex amplitudes), which is what
makes them a useful cross-check.  They are deliberately simple and slow-ish
(a Python loop over dict lookups) and independent of the lookup machinery in
dqi.py; ``estimate_solution_quality`` never uses them.
"""

import math

import numpy as np

from dqi import get_eigenvector, make_A


def compute_A_bar(B, v, l: int, benchmarks, minimum_distance: int) -> np.ndarray:
    """Interference matrix A_bar of the imperfect-decoding regime (GF(2), r = 1).

    ``A_bar[k, k2] = sum_i sum_{y, y2} (-1)^{v.(y + y2) + v_i} [B^T(y + y2 + e_i) = 0]``
    over the decodable errors of weight k and k2, rescaled from the benchmarked
    sample to the full population and normalised by sqrt(C(m,k) C(m,k2)); in the
    perfect-decoding regime it equals ``make_A(2, 1, m, l)`` (pass
    ``minimum_distance=0`` to disable that shortcut and force the count).

    The condition is the syndrome identity s(y) = s(y2) XOR b_i, so the partner
    of (i, y2) is looked up in a dict from syndrome to error.  A benchmark with
    two correct errors of one syndrome is rejected: a syndrome decoder returns
    exactly one error per syndrome.
    """
    if B.base_ring().order() != 2:
        raise ValueError(f"A_bar is the GF(2) formulation, got {B.base_ring()}")
    Bnp = np.array(B.list(), dtype=np.int64).reshape(B.nrows(), B.ncols())
    m = Bnp.shape[0]
    vnp = np.array([int(c) for c in v], dtype=np.int64)
    A = make_A(2, 1, m, l)

    Y, S, sign, lookup = {}, {}, {}, {}
    for k in range(l + 1):
        Y[k] = np.array(
            [[int(c) for c in y] for y in benchmarks[k].correct], dtype=np.int64
        ).reshape(-1, m)
        S[k] = (Y[k] @ Bnp) % 2
        sign[k] = 1 - 2 * ((Y[k] @ vnp) % 2)  # (-1)^{v.y}
        for j, row in enumerate(S[k]):
            key = row.tobytes()
            if key in lookup:
                k_other, j_other = lookup[key]
                raise ValueError(
                    f"invalid benchmark: the errors "
                    f"{[(k_other, Y[k_other][j_other].tolist()), (k, Y[k][j].tolist())]} "
                    f"(as (weight, error)) share the syndrome {row.tolist()}, but a "
                    f"syndrome decoder returns exactly one error per syndrome, so at "
                    f"most one of them can be decoded correctly"
                )
            lookup[key] = (k, j)

    A_bar = np.zeros((l + 1, l + 1))
    for k in range(l + 1):
        for k2 in range(k, l + 1):
            eps_k, eps_k2 = benchmarks[k].epsilon, benchmarks[k2].epsilon
            # perfect decoding and no codeword of weight <= k + k2 + 1: A_bar = A
            if k + k2 + 1 < minimum_distance and eps_k == 0 and eps_k2 == 0:
                A_bar[k, k2] = A[k, k2]
                continue
            # below the minimum distance only y + y2 + e_i = 0 can contribute
            if k + k2 + 1 < minimum_distance and k2 != k + 1:
                continue
            N_k, N_k2 = len(Y[k]), len(Y[k2])
            if N_k == 0 or N_k2 == 0:
                continue
            total_size = math.comb(m, k) * (1 - eps_k) * math.comb(m, k2) * (1 - eps_k2)
            count = 0.0
            for i in range(m):
                targets = (S[k2] + Bnp[i]) % 2  # s(y2) XOR b_i for every y2
                for j in range(N_k2):
                    hit = lookup.get(targets[j].tobytes())
                    if hit is not None and hit[0] == k:
                        count += sign[k][hit[1]] * sign[k2][j] * (1 - 2 * vnp[i])
            A_bar[k, k2] = (
                total_size * count / (N_k * N_k2) / math.sqrt(math.comb(m, k) * math.comb(m, k2))
            )
    return A_bar + (A_bar.T - np.diag(np.diag(A_bar)))


def predict_from_A_bar(A_bar: np.ndarray, m: int, l: int, w, epsilon) -> float:
    """DQI expectation from the interference matrix: (m + max(0, w^T A_bar w / norm)) / 2.

    ``norm = sum_k w_k^2 (1 - epsilon_k)`` is the squared norm of the state after
    the undecodable errors are dropped; the clamp keeps the value at or above
    random guessing, m / 2.
    """
    if isinstance(epsilon, (float, int)):
        epsilon = [float(epsilon)] * (l + 1)
    assert len(epsilon) == l + 1
    assert min(epsilon) >= 0
    assert w is None or len(w) == l + 1

    epsilon = np.array(epsilon)
    if w is None:
        _, w = get_eigenvector(make_A(2, 1, m, l), -1)
    else:
        w = np.array(w)

    norm = np.sum(w * w * (1 - epsilon))
    matrix_product = w.reshape((1, l + 1)) @ A_bar @ w.reshape((l + 1, 1))
    observable = matrix_product / norm
    return (max(0.0, observable.item()) + m) / 2


def estimate_via_A_bar(instance, l: int, w, benchmarks) -> float:
    """The GF(2) DQI expectation through ``compute_A_bar``, for cross-checking
    ``dqi.compute_expectation`` (see ``test_generic_expectation_matches_A_bar_formulation_binary``)."""
    A_bar = compute_A_bar(
        instance.get_B(),
        instance.get_v(),
        l,
        benchmarks,
        instance.get_minimum_distance(),
    )
    epsilons = [benchmarks[k].epsilon for k in range(l + 1)]
    return predict_from_A_bar(A_bar, instance.get_m(), l, w, epsilons)


def make_A_corrected(p: int, r: int, m: int, l: int) -> np.ndarray:
    """Tridiagonal matrix of the semicircle law with the diagonal that matches brute force.

    dqi.make_A uses (p - 2r)/p * k on the diagonal; the brute-force oracle shows the
    correct entry is (p - 2r)/sqrt(r (p - r)) * k.  Both agree for p = 2, r = 1.
    """
    k = np.arange(l + 1)
    a_k = np.sqrt(k * (m - k + 1))
    off = np.diag(a_k)
    return (
        np.diag((p - 2 * r) / math.sqrt(r * (p - r)) * k)
        + np.roll(off, -1, 0)
        + np.roll(off, -1, 1)
    )


def semicircle_corrected(p: int, r: int, m: int, l: int) -> float:
    A = make_A_corrected(p, r, m, l)
    lam = np.linalg.eigh(A)[0][-1]
    return m * r / p + math.sqrt(r * (p - r)) / p * lam
