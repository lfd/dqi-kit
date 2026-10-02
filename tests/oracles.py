"""Independent, slow-but-obviously-correct oracles used to check dqi.py.

Nothing in here depends on dqi.py, so a disagreement between an oracle and the
framework points at the framework (or at a wrong assumption, which is then
documented in the test).
"""

import itertools
import math

import numpy as np
from sage.all import GF, codes


def dqi_g_values(p: int, r: int) -> tuple[float, float]:
    """Normalised objective values g_i(y) for y in F_i ("yes") and y not in F_i ("no").

    Same normalisation as Dqi._compute_g_g_tilde: mean 0 and sum_y g(y)^2 = 1.
    """
    f_dash = 2 * r / p - 1
    phi = math.sqrt(4 * r * (1 - r / p))
    return (1 - f_dash) / phi, (-1 - f_dash) / phi


def brute_force_dqi_expectation(p: int, B, F, w) -> float:
    """Exact expected number of satisfied constraints of the DQI state.

    The state is written directly in the computational basis,

        psi(x)  ~  sum_k  w_k * p^{k/2} / sqrt(C(m, k)) * e_k(g_1(b_1.x), ..., g_m(b_m.x)),

    where e_k is the k-th elementary symmetric polynomial.  This is the QFT of
    sum_k w_k / sqrt(C(m,k)) sum_{|y|=k} prod g~_i(y_i) |B^T y>, valid whenever the
    syndromes of all weight <= l vectors are distinct (minimum distance > 2l).
    The expectation is computed by enumerating all p^n assignments, so this is only
    usable for tiny n, but it has no dependence on any DQI formula.
    """
    B = np.array(B.list(), dtype=int).reshape(B.nrows(), B.ncols())
    m, n = B.shape
    r = len(F[0])
    yes, no = dqi_g_values(p, r)
    l = len(w) - 1
    F_int = [{int(f) for f in F_i} for F_i in F]

    amps = []
    sats = []
    for x in itertools.product(range(p), repeat=n):
        y = (B @ np.array(x)) % p
        g = np.array([yes if int(y[i]) in F_int[i] else no for i in range(m)])
        e = np.zeros(l + 1)
        e[0] = 1.0
        for gi in g:
            for k in range(l, 0, -1):
                e[k] += e[k - 1] * gi
        amps.append(
            sum(w[k] * p ** (k / 2) / math.sqrt(math.comb(m, k)) * e[k] for k in range(l + 1))
        )
        sats.append(sum(1 for i in range(m) if int(y[i]) in F_int[i]))
    amps = np.array(amps)
    probs = amps**2 / np.sum(amps**2)
    return float(np.sum(probs * np.array(sats)))


def true_minimum_distance(B) -> int:
    """Minimum distance of ker(B^T) computed by Sage (exact, exponential)."""
    return int(codes.from_parity_check_matrix(B.T).minimum_distance())


def golay_instance(field_order: int):
    """Return (B, m, n, d) with B = G^T for the (ternary or binary) Golay code.

    ker(B^T) is the dual Golay code: [24,12,8] over GF(2), [11,5,6] over GF(3).
    Both have minimum distance >= 6, so the semicircle law is exact for l <= 2
    (l <= 3 in the binary case).
    """
    if field_order == 2:
        G = codes.GolayCode(GF(2)).generator_matrix()
    elif field_order == 3:
        G = codes.GolayCode(GF(3), extended=False).generator_matrix()
    else:
        raise ValueError(field_order)
    B = G.T
    return B, B.nrows(), B.ncols(), true_minimum_distance(B)


def odd_distance_instance():
    """Return (B, m, n, d) for the binary BCH [15, 7, 5] code, i.e. odd d.

    The case that separates the two candidate formulas for the perfect-decoding
    regime: 2l + 2 <= d holds only up to l = 1, while the unique-decoding radius
    (d - 1) // 2 is 2, where a weight-5 codeword still reaches A_bar[2, 2].
    """
    B = codes.BCHCode(GF(2), 15, 5).parity_check_matrix().T
    return B, B.nrows(), B.ncols(), true_minimum_distance(B)


def max_lin_sat_from_B(field, B, rhs):
    """Build a MaxLinSat whose B matrix is exactly the given matrix (rows in order)."""
    from max_lin_sat import MaxLinSat

    ls = MaxLinSat(field, equal_size_F_i=False)
    xs = [ls.new_var(f"x{i}") for i in range(B.ncols())]
    for row, r in zip(B.rows(), rhs, strict=False):
        ls.add_constraint(sum(int(c) * x for c, x in zip(row, xs, strict=False)) == int(r))
    return ls
