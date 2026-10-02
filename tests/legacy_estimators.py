"""Superseded / deliberately naive estimators, kept only as test oracles.

Two independent references for ``dqi.compute_expectation``:

``legacy_expectation``
    The implementation that ``compute_expectation`` replaced, taken verbatim in
    substance from ``Dqi._dqi_compute_exact_performance_non_binary`` as of commit
    ``393d2ce`` (the parent of the syndrome-bucketing port).  It is copied with
    its original loop structure and Sage operations -- being slow is the point,
    an oracle that shares the fast path's tricks would not be an oracle.  It has
    no population rescaling, so every weight class must be benchmarked
    *exhaustively* (``n_tries=None``) for it to be valid.

``naive_rescaled_expectation``
    A fresh, deliberately slow transcription of the same formula that *does*
    rescale sampled benchmarks to the full error population.  This is the oracle
    for the sampled path, which the legacy code cannot check.  For every sampled
    decodable error y1 and every shift (i, a) it asks the decoder for the error
    with syndrome ``s(y1) + a H e_i`` -- exactly what the DQI state pairs y1
    with -- and weights the term by ``scale_k1`` alone, since y1 is the only
    sampled object in it.

Neither function imports anything from the fast estimators; the only thing they
borrow from ``dqi`` is ``Dqi._get_g_tilde`` / ``make_A`` / ``get_eigenvector``
(and ``naive_rescaled_expectation`` does not even do that -- it runs its own
Fourier sum).
"""

from __future__ import annotations

import cmath
import math

import numpy as np
from sage.all import vector

from decoders import BenchmarkResult
from dqi import Dqi, get_eigenvector, make_A


def legacy_expectation(
    instance,
    l: int,
    w: np.ndarray | None,
    benchmarks: dict[int, BenchmarkResult],
) -> float:
    """``Dqi._dqi_compute_exact_performance_non_binary`` of commit ``393d2ce``.

    Valid for any prime field (the original guard was ``is_prime_field``), so it
    is an oracle at p = 2 as well.  Only correct for exhaustive benchmarks: the
    sums below count the benchmarked errors as if they were the whole weight
    class.
    """
    if not instance.field.is_prime_field():
        raise ValueError("This function only supports prime fields")

    B = instance.get_B()
    B_T = B.T
    F = instance.get_F()
    m = instance.get_m()
    r = instance.get_r()
    field = instance.field
    p = field.order()

    # obtained from the framework so the oracle does not re-derive the Fourier
    # transform of g_i; _get_g_tilde is pinned by its own tests
    g_tilde = Dqi(instance)._get_g_tilde()

    def total_g_tilde(y):
        product = 1
        for i, y_i in enumerate(y):
            if y_i != 0:
                product *= g_tilde[i](int(y_i))
        return product

    A = make_A(p, r, m, l)
    if w is None:
        _, w = get_eigenvector(A, -1)
        w = w / np.linalg.norm(w)

    # compute norm
    norm = 0
    for k in range(l + 1):
        term = 0
        for y in benchmarks[k].correct:
            product = 1
            for i, y_i in enumerate(y):
                if y_i != 0:
                    product *= abs(g_tilde[i](int(y_i))) ** 2
            term += product
        norm += w[k] ** 2 / math.comb(m, k) * term

    e = [vector(field, [0] * m) for _ in range(m)]
    for i in range(m):
        e[i][i] = 1

    omega_p = cmath.exp(2j * math.pi / p)

    expectation = 0
    for k1 in range(l + 1):
        for k2 in range(l + 1):
            factor = w[k1] * w[k2] / math.sqrt(math.comb(m, k1) * math.comb(m, k2))
            term = 0
            for y1 in benchmarks[k1].correct:
                for y2 in benchmarks[k2].correct:
                    subfactor = total_g_tilde(y1).conjugate() * total_g_tilde(y2)
                    subterm = 0
                    for i in range(m):
                        for a in field:
                            if B_T * y1 != B_T * (y2 - a * e[i]):
                                continue
                            for v in F[i]:
                                subterm += omega_p ** (-int(a) * int(v))
                    term += subfactor * subterm
            expectation += factor * term

    result = expectation / p / norm
    return float(result.real)


def naive_rescaled_expectation(
    instance,
    l: int,
    w: np.ndarray | None,
    benchmarks: dict[int, BenchmarkResult],
    decoder,
) -> float:
    """Direct transcription of the DQI objective value, with population rescaling.

    Written out with plain Python loops over the Sage error vectors, sharing no
    code and no structure with ``dqi.compute_expectation``:

        scale[k]  = C(m,k) (p-1)^k (1 - eps_k) / N_k        (1.0 when exhaustive)
        c(y)      = w[k] G(y) / sqrt(C(m,k)),  G(y) = prod_{y_i != 0} g~_i(y_i)
        norm      = sum_k scale[k] sum_{y in C_k} |c(y)|^2
        numerator = sum_{k1} sum_{y1 in C_k1} sum_i sum_a
                        scale[k1] conj(c(y1)) c(y2) phase(i, a),
                        y2 = decoder(H y1 + a H e_i), counted iff wt(y2) <= l
        <s>       = Re(numerator / p / norm)

    with ``H = B^T`` and ``phase(i, a) = sum_{u in F_i} omega^{-a u}``.  The
    partner y2 is whatever ``decoder.error_estimate`` returns for the shifted
    syndrome, so a term contains one sampled error and is rescaled once.  (For
    exhaustive benchmarks all scales are 1 and every partner of weight <= l is
    itself in the benchmark, so the transcription must reproduce
    ``legacy_expectation``, which is how it gets validated.)

    The only concessions to runtime are caching, per error, its syndrome as a
    tuple of ints, precomputing the m * p shift tuples ``a * H e_i`` and asking
    the decoder once per syndrome; the error / constraint / field-element loops
    are all still there and are all explicit.
    """
    field = instance.field
    if not field.is_prime_field():
        raise ValueError("This function only supports prime fields")
    p = int(field.order())
    m = instance.get_m()
    r = instance.get_r()
    B = instance.get_B()
    H = B.T
    F = instance.get_F()

    if w is None:
        _, w = get_eigenvector(make_A(p, r, m, l), -1)
        w = w / np.linalg.norm(w)

    # ---- g_i and its Fourier transform g~_i, computed here from scratch ----
    f_dash = 2 * r / p - 1
    phi = math.sqrt(4 * r * (1 - r / p))
    yes_value, no_value = (1 - f_dash) / phi, (-1 - f_dash) / phi
    omega = cmath.exp(2j * math.pi / p)

    def g(i: int, x: int) -> float:
        return yes_value if field(x) in F[i] else no_value

    def g_tilde(i: int, y: int) -> complex:
        return sum(omega ** ((y * x) % p) * g(i, x) for x in range(p)) / math.sqrt(p)

    # ---- population rescaling of every weight block ----
    scale = {}
    for k in range(l + 1):
        N_k = len(benchmarks[k].correct)
        if N_k == 0:
            scale[k] = 0.0
            continue
        population = math.comb(m, k) * (p - 1) ** k
        scale[k] = population * (1 - benchmarks[k].epsilon) / N_k

    # ---- amplitudes c(y) of the DQI state ----
    def amplitude(k: int, y) -> complex:
        G = 1.0 + 0j
        for i in range(m):
            y_i = int(y[i])
            if y_i != 0:
                G *= g_tilde(i, y_i)
        return w[k] * G / math.sqrt(math.comb(m, k))

    c = {k: [amplitude(k, y) for y in benchmarks[k].correct] for k in range(l + 1)}

    norm = 0.0
    for k in range(l + 1):
        for c_y in c[k]:
            norm += scale[k] * abs(c_y) ** 2

    # ---- syndromes H y, and the m * p shifts a * H e_i ----
    syndrome = {
        k: [tuple(int(s) % p for s in H * y) for y in benchmarks[k].correct] for k in range(l + 1)
    }
    e = [vector(field, [0] * m) for _ in range(m)]
    for i in range(m):
        e[i][i] = 1
    shift = [[tuple(int(s) % p for s in a * (H * e[i])) for a in range(p)] for i in range(m)]

    # ---- phase(i, a) = sum_{u in F_i} omega^{-a u} ----
    phase = [[sum(omega ** (-a * int(u)) for u in F[i]) for a in range(p)] for i in range(m)]

    # ---- the partner of a syndrome: the decoder's estimate, if it has weight <= l ----
    partner_amplitude = {}

    def partner(s: tuple) -> complex:
        if s not in partner_amplitude:
            y2 = decoder.error_estimate(vector(field, list(s)), l)
            k2 = None if y2 is None else sum(1 for x in y2 if x != 0)
            partner_amplitude[s] = 0j if k2 is None or k2 > l else amplitude(k2, y2)
        return partner_amplitude[s]

    numerator = 0j
    for k1 in range(l + 1):
        for j1 in range(len(c[k1])):
            s1 = syndrome[k1][j1]
            for i in range(m):
                for a in range(p):
                    # H y2 == H y1 + a H e_i
                    s2 = tuple((s + t) % p for s, t in zip(s1, shift[i][a], strict=True))
                    numerator += scale[k1] * c[k1][j1].conjugate() * partner(s2) * phase[i][a]

    return float((numerator / p / norm).real)
