from __future__ import annotations

import cmath
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sage.all import GF, vector
from sage.rings.finite_rings.element_base import FiniteRingElement

from decoders import AbstractDecoder, BenchmarkResult, SyndromeDecoder

METHODS = ("auto", "analytical", "interference", "average_v_bound")


@dataclass(frozen=True)
class QualityEstimate:
    """What ``estimate_solution_quality`` computed, and how much it is worth.

    ``float(estimate)`` is the objective value; the rest says which estimator ran,
    whether the number is a proven expectation or a sample estimate, and whether
    the observable hit the ``max(0.0, ...)`` floor (a clamped value is m/2, i.e.
    random guessing, and carries no information).

    ``guaranteed`` means the value holds without any measured input (the
    closed form in the perfect-decoding regime).  ``exact`` is weaker and only
    set by the interference estimator: the pair sums are exact, but for the
    benchmark that was measured -- exhaustively (``exact=True``) or on a sample
    of the error patterns of each weight (``exact=False``).
    """

    value: float
    l: int
    method: str
    guaranteed: bool
    epsilon: tuple[float, ...] | None = None
    clamped: bool = False
    exact: bool = False

    def __float__(self) -> float:
        return self.value


def _check_nondegenerate_r(p: int, r: int) -> None:
    if not 0 < r < p:
        raise ValueError(
            f"requires 0 < r < p, got r={r}, p={p}: every F_i is empty (r=0) or "
            f"the whole field (r=p), so the DQI construction does not apply"
        )


class Dqi:
    def __init__(self, instance, decoder_constructor=None):
        self.instance = instance.to_max_linsat()
        if decoder_constructor is None:
            self.decoder_constructor = self.instance.default_decoder_constructor
        else:
            self.decoder_constructor = decoder_constructor
        self.decoder = None
        self.g = None
        self.g_tilde = None
        self.last_estimate: QualityEstimate | None = None

    def get_decoder(self) -> AbstractDecoder:
        if self.decoder is None:
            self.decoder = self.decoder_constructor(self.instance)
        return self.decoder

    def estimate_solution_quality(
        self,
        l: int | None = None,
        *,
        w: np.ndarray | None = None,
        method: str = "auto",
        n_decoding_samples: int | str | None = "auto",
        details: bool = False,
    ) -> float | QualityEstimate:
        """Estimate the objective value DQI reaches on this instance.

        ``l``     how many errors the DQI state mixes in.  Defaults to the largest
                  value that still lies in the perfect-decoding regime, see
                  ``_default_l``.
        ``w``     weight vector of the DQI polynomial, length ``l + 1``.  Defaults
                  to the optimal one, the top eigenvector of A.
        ``method``
                  ``"auto"``            cheapest estimator that is valid here
                  ``"analytical"``      closed form; raises outside that regime
                  ``"interference"``    ``compute_expectation``, for every prime p
                  ``"average_v_bound"`` cheap GF(2) lower bound, often just m/2
                  Both ``"auto"`` and ``"interference"`` take the same
                  ``compute_expectation`` route on GF(2) as on GF(p); the
                  equivalent A_bar formulation lives in ``tests/reference_impl``
                  as a cross-check only.
        ``n_decoding_samples``
                  decoding attempts per error weight, for any field.  ``"auto"``
                  means 500, ``None`` means exhaustive (exponential, but then the
                  estimate needs no population rescaling).  The decoder is handed
                  to ``compute_expectation``, which decodes the shifted syndromes
                  to find the interference partner of each sampled error instead
                  of looking it up in the sample; that is what keeps a 500-error
                  sample close to the exhaustive value.
        ``details``
                  return the ``QualityEstimate`` record instead of a bare float.
                  It is also kept in ``self.last_estimate`` eitherway.
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")

        field = self.instance.field
        if not field.is_prime_field():
            raise ValueError(f"every estimator here is derived over a prime field F_p, got {field}")
        p = field.order()
        _check_nondegenerate_r(p, self.instance.get_r())
        m = self.instance.get_B().nrows()
        is_binary = p == 2

        radius = self.get_decoder().decoding_radius()

        if l is None:
            l = self._default_l(radius)
        l = int(l)
        if not 1 <= l <= m:
            raise ValueError(f"l must satisfy 1 <= l <= m = {m}, got {l}")
        if w is not None:
            w = np.asarray(w, dtype=float)
            if w.shape != (l + 1,):
                raise ValueError(f"w must have shape (l + 1,) = ({l + 1},), got {w.shape}")

        # If possible, compute solution quality analytically (without computing decoding benchmarks)
        if (
            method in ("auto", "analytical")
            and radius is not None
            # is decoding perfect
            and l <= radius
            # are semi-circle law requirements fulfilled
            and self._has_exact_A(l, radius)
        ):
            return self._finish(
                self._analytical_estimate(l, w),
                l,
                "analytical",
                guaranteed=True,
                details=details,
            )

        # requirements for analytical estimate not met
        if method == "analytical":
            raise ValueError(
                f"l = {l} is not in the perfect-decoding regime: decoding radius "
                f"{radius}, minimum distance {self._minimum_distance()} (needs "
                f'l <= radius and 2l + 2 <= d). method="auto" measures the decoding '
                f"failure rates instead of assuming they are zero"
            )

        # sampled benchmarks are rescaled to the full population by
        # compute_expectation, so they are allowed for every field
        n_tries = self._resolve_n_decoding_samples(n_decoding_samples)
        benchmarks = self._benchmarks(l, n_tries)
        epsilon = [benchmarks[k].epsilon for k in range(l + 1)]

        # When no error was incorrectly decoded, assume perfect decoding regime
        if method == "auto" and max(epsilon) == 0 and self._has_exact_A(l, None):
            return self._finish(
                self._analytical_estimate(l, w),
                l,
                "analytical",
                guaranteed=n_tries is None,
                epsilon=epsilon,
                details=details,
            )

        if method == "average_v_bound":
            if not is_binary:
                raise ValueError(
                    'the average-v bound is only derived for GF(2); use method="interference"'
                )
            value = self._dqi_lower_bound_average_v(l, w=w, benchmarks=benchmarks)
            clamped = value <= m / 2
            if clamped:
                warnings.warn(
                    f"the observable was clamped to 0, so the bound degenerated to "
                    f"m/2 = {m / 2} (random guessing); try a smaller l or "
                    'method="interference"',
                    stacklevel=2,
                )
            return self._finish(
                value,
                l,
                "average_v_bound",
                guaranteed=False,
                epsilon=epsilon,
                clamped=clamped,
                details=details,
            )

        return self._finish(
            compute_expectation(self.instance, l, w, benchmarks, decoder=self.get_decoder()),
            l,
            "interference",
            guaranteed=False,
            exact=n_tries is None,
            epsilon=epsilon,
            details=details,
        )

    def _default_l(self, radius: int | None) -> int:
        """Largest l where semi-circle law applies.

        Semi-circle law requires:
            2l + 1 < d <=>
            2l + 2 <= d <=>
            l <= d / 2 - 1 <=>
            l <= floor(d / 2 - 1) = d // 2 - 1  (since l is an integer)
        """
        l = self._minimum_distance() // 2 - 1
        if radius is not None:
            # cap l at the maximum value where the decoder guarantees perfect decoding
            l = min(l, radius)
        # if l = 0, fall back to l = 1 (then imperfect regime)
        return max(1, min(l, self.instance.get_B().nrows()))

    def _has_exact_A(self, l: int, radius: int | None) -> bool:
        """
        Check requirements for semi-circle law
        """

        # radius <= (d - 1) // 2 by the unique decoding threshold
        # radius <= (d - 1) // 2 <= (d - 1) / 2 <=> 2 radius <= d - 1
        #
        # Therefore, if l < radius <=> l + 1 <= radius, then
        # 2l + 2 <= 2 radius <= d - 1 < d
        # Thus, 2l + 1 < 2l + l < d: The requirment of the semi-circle law is met.
        # We can therefore skip computing the minimum distance
        if radius is not None and l < radius:
            return True
        return 2 * l + 2 <= self._minimum_distance()

    def _minimum_distance(self) -> int:
        return self.instance.get_minimum_distance()

    def _resolve_n_decoding_samples(self, n_decoding_samples: int | str | None) -> int | None:
        """``"auto"`` is 500 decoding attempts per error weight, on any field."""
        if n_decoding_samples == "auto":
            return 500
        return n_decoding_samples

    def _benchmarks(self, l: int, n_tries: int | None) -> dict[int, BenchmarkResult]:
        """Benchmark every error weight 1..l, keyed by weight (weight 0 never fails)."""
        decoder = self.get_decoder()
        m = self.instance.get_B().nrows()
        benchmarks = {0: BenchmarkResult([vector(self.instance.field, [0] * m)], [], 0)}
        # Starting at large errors is advantageous for some decoders
        for k in reversed(range(1, l + 1)):
            benchmarks[k] = decoder.get_benchmarks(l, k, n_tries=n_tries)
        return benchmarks

    def _analytical_estimate(self, l: int, w: np.ndarray | None) -> float:
        p = self.instance.field.order()
        r = self.instance.get_r()
        m = self.instance.get_B().nrows()
        if w is None:
            return _predict_dqi_performance_perfect_optimal_w(p, r, m, l)
        return _predict_dqi_performance_perfect(p, r, m, l, w)

    def _finish(
        self,
        value: float,
        l: int,
        method: str,
        *,
        guaranteed: bool,
        details: bool,
        epsilon: list[float] | None = None,
        clamped: bool = False,
        exact: bool = False,
    ) -> float | QualityEstimate:
        estimate = QualityEstimate(
            value=float(value),
            l=l,
            method=method,
            guaranteed=guaranteed,
            epsilon=None if epsilon is None else tuple(float(e) for e in epsilon),
            clamped=clamped,
            exact=exact,
        )
        self.last_estimate = estimate
        return estimate if details else estimate.value

    def semicircle_law_solution_quality(
        self, l: int | None = None, w: np.ndarray | None = None
    ) -> float:
        """DQI performance in the perfect-decoding regime (the semicircle law).

        Valid as long as the decoder corrects every error of weight <= l and
        2l + 2 <= d, which is what ``estimate_solution_quality`` checks before it
        takes this route.  The default l is the largest one satisfying the second
        condition; note that the unique-decoding radius (d - 1) // 2 is one too
        large for odd d, where a weight-d codeword still reaches A_bar[l, l].
        """
        d = self._minimum_distance()
        if l is None:
            l = d // 2 - 1
        l = int(l)
        if l < 1:
            raise ValueError(
                f"minimum distance d = {d} admits no l >= 1 with 2l + 2 <= d, so the "
                f"perfect-decoding regime is empty for this instance"
            )
        if w is not None:
            w = np.asarray(w, dtype=float)
            if w.shape != (l + 1,):
                raise ValueError(f"w must have shape (l + 1,) = ({l + 1},), got {w.shape}")
        if 2 * l + 2 > d:
            warnings.warn(
                f"2l + 2 = {2 * l + 2} > d = {d}: A_bar is not exactly A for l = {l}, "
                f"so the returned value is not guaranteed",
                stacklevel=2,
            )
        return self._analytical_estimate(l, w)

    def _dqi_lower_bound_average_v(
        self,
        l: int,
        w: np.ndarray | None = None,
        n_tries=500,
        benchmarks: dict[int, BenchmarkResult] | None = None,
    ):
        assert self.instance.field == GF(2) and self.instance.get_r() == 1

        if benchmarks is None:
            benchmarks = self._benchmarks(l, n_tries)
        # epsilon indexed by error weight.
        # decoding zero errors always succeeds
        epsilon = [benchmarks[k].epsilon for k in range(l + 1)]

        m = self.instance.get_B().nrows()

        return _upper_bound_dqi_performance_imperfect_average_v(m, l, epsilon, w)

    def _compute_g_g_tilde(self):
        """The objective indicators g_i and their Fourier transforms g~_i, as callables.

        The estimators use the (m, p) table ``_g_tilde_table`` directly; these
        closures are the same values with the per-element interface.
        """
        if not self.instance.field.is_prime_field():
            raise ValueError("This function only supports prime fields")
        p = self.instance.field.order()
        r = self.instance.get_r()
        _check_nondegenerate_r(p, r)
        f_dash = 2 * r / p - 1
        phi = math.sqrt(4 * r * (1 - r / p))

        yes_value = (+1 - f_dash) / phi
        no_value = (-1 - f_dash) / phi

        def make_g_lambda(F_i):
            return lambda y: yes_value if y in F_i else no_value

        self.g = [make_g_lambda(F_i) for F_i in self.instance.get_F()]

        table = _g_tilde_table(p, r, self.instance.get_F())

        def make_g_tilde_lambda(g_tilde_i):
            return lambda y: complex(g_tilde_i[int(y) % p])

        self.g_tilde = [make_g_tilde_lambda(row) for row in table]

    def _get_g_tilde(self) -> list[Callable[[FiniteRingElement], complex]]:
        if self.g is None:
            self._compute_g_g_tilde()
        return list(self.g_tilde)


def get_eigenvector(mat: np.ndarray, index: int) -> np.ndarray:
    values, vectors = np.linalg.eigh(mat)
    value = values[index]
    vector = vectors[:, index]
    return value, vector


def make_A(p: int, r: int, m: int, l: int) -> np.array:
    _check_nondegenerate_r(p, r)

    diag = (p - 2 * r) / math.sqrt(r * (p - r)) * np.diag(np.arange(0, l + 1))

    k = np.arange(0, l + 1)
    a_k = np.sqrt(k * (m - k + 1))

    off_diag = np.diag(a_k)
    return diag + np.roll(off_diag, -1, 0) + np.roll(off_diag, -1, 1)


def _predict_dqi_performance_perfect(p: int, r: int, m: int, l: int, w: int) -> float:
    A = make_A(p, r, m, l)

    assert len(w) == l + 1
    norm = np.linalg.norm(w)

    polynomial_effect = w.reshape((1, l + 1)) @ A @ w.reshape((l + 1, 1)) / (norm * norm)

    return ((m * r) / p + np.sqrt(r * (p - r)) / p * polynomial_effect).item()


def _predict_dqi_performance_perfect_optimal_w(p: int, r: int, m: int, l: int) -> float:
    A = make_A(p, r, m, l)

    lambda_1 = np.linalg.eigh(A)[0][-1]
    return (m * r) / p + np.sqrt(r * (p - r)) / p * lambda_1


def _upper_bound_dqi_performance_imperfect_average_v(
    m: int, l: int, epsilon: float, w: np.ndarray = None
) -> float:
    if isinstance(epsilon, (float, int)):
        epsilon = [float(epsilon)] * (l + 1)

    assert len(epsilon) == l + 1
    assert min(epsilon) >= 0
    assert w is None or len(w) == l + 1

    A = make_A(2, 1, m, l)

    if w is None:
        lambda_1, w = get_eigenvector(A, -1)
        matrix_product = lambda_1
    else:
        w = np.array(w)
        matrix_product = w.reshape((1, l + 1)) @ A @ w.reshape((l + 1, 1))

    largest_epsilon = max(epsilon)
    epsilon = np.array(epsilon)

    norm = np.sum(w * w * (1 - epsilon))

    observable = (matrix_product - np.sum(w * w) * 2 * largest_epsilon * (m + 1)) / norm

    return (max(0.0, observable.item()) + m) / 2


# --------------------------------------------------------------------------- #
# Imperfect-decoding estimators.
#
# The estimator sums over *pairs* of decodable errors (y1, y2).  The pair
# condition is a syndrome shift,
#
#     B^T (y1 - y2 + a e_i) = 0   <=>   s(y1) = s(y2) - a * b_i,
#
# with s(y) = B^T y the syndrome and b_i = B^T e_i the i-th column of B^T (the
# i-th row of B).  So instead of enumerating pairs, we compute every syndrome
# once, put the decodable errors into a dict keyed by their syndrome and find
# the partner of each error with one dict lookup.  That is O(m p N) lookups
# instead of O(l^2 N^2 m p) Sage operations.
# --------------------------------------------------------------------------- #


def _g_tilde_table(p: int, r: int, F) -> np.ndarray:
    """(m, p) complex table g_tilde[i, a] = 1/sqrt(p) sum_x omega^{a x} g_i(x).

    The Fourier transform of the normalised objective indicator g_i (mean 0,
    sum_x g_i(x)^2 = 1), evaluated eagerly for every field element instead of
    re-running the O(p) Fourier sum on every lookup.
    """
    _check_nondegenerate_r(p, r)
    f_dash = 2 * r / p - 1
    phi = math.sqrt(4 * r * (1 - r / p))
    yes = (1 - f_dash) / phi
    no = (-1 - f_dash) / phi
    omega = cmath.exp(2j * math.pi / p)
    m = len(F)
    g = np.full((m, p), no, dtype=float)
    for i, F_i in enumerate(F):
        for f in F_i:
            g[i, int(f)] = yes
    a = np.arange(p)
    fourier = omega ** np.outer(a, a) / math.sqrt(p)  # fourier[a, x]
    return g @ fourier.T


def _stack_rows(vectors, width: int) -> np.ndarray:
    """Stack Sage vectors into an (N, width) int array (an empty one for no vectors)."""
    if len(vectors) == 0:
        return np.zeros((0, width), dtype=np.int64)
    return np.array([[int(c) for c in v] for v in vectors], dtype=np.int64)


def _amplitudes(Y: np.ndarray, coef: np.ndarray, gfactors: np.ndarray) -> np.ndarray:
    """DQI amplitudes c(y) = w_k G(y) / sqrt(C(m, k)) of the rows of an (N, m) error array.

    G(y) = prod_{i: y_i != 0} g~_i(y_i).  With gfactors[a, i] = g~_i(a) for a != 0
    and 1 for a = 0 the product over the row of an error is G(y), the coordinates
    with y_i = 0 contributing 1.  coef[k] = w_k / sqrt(C(m, k)) for k <= l and
    coef[l + 1] = 0, so an error of weight > l, which is not part of the state,
    gets amplitude 0.
    """
    k = np.minimum(np.count_nonzero(Y, axis=1), len(coef) - 1)
    # take_along_axis produces temp[j, i] = gfactors[Y[j, i], i], so the product
    # over a row is G(y) for that error
    return coef[k] * np.take_along_axis(gfactors, Y, axis=0).prod(axis=1)


class _DuplicateSyndrome(Exception):
    """Two errors share a syndrome; ``args`` are their positions in the table."""


class _SyndromeTable:
    """Maps syndrome rows in F_p^n to the position of the error that has them.

    The key of a syndrome is the bytes of its row in the narrowest unsigned
    dtype that holds [0, p), so equal keys are equal syndromes, for any p and n.
    Probing converts a whole block of rows to one bytes buffer and slices it, so
    the per-row Python work is a bytes slice and a ``dict.get``.

    This is the only place that knows how syndromes are looked up; swapping the
    mechanism (sorted keys, hashing, ...) touches nothing else.
    """

    def __init__(self, syndromes: np.ndarray, p: int):
        # smallest unsigned integer dtype that holds p - 1 (uint8 for p <= 256), so
        # every syndrome entry is one byte and equal syndromes give equal bytes
        self._dtype = np.min_scalar_type(p - 1)
        self._n = syndromes.shape[1]
        self._table: dict[bytes, int] = {}
        # cast to that dtype and make the rows contiguous in memory, so that
        # row.tobytes() is exactly the n bytes of one syndrome
        for j, row in enumerate(np.ascontiguousarray(syndromes, dtype=self._dtype)):
            key = row.tobytes()
            if key in self._table:
                raise _DuplicateSyndrome(self._table[key], j)
            self._table[key] = j

    def find(self, rows: np.ndarray) -> np.ndarray:
        """Position of the error with each syndrome in ``rows`` (..., n); -1 where none."""
        # one bytes string for the whole block: the rows are laid out one after the
        # other (C order, n is the last axis), so row number j occupies the bytes
        # [j * step, (j + 1) * step) and can be looked up without building an array
        # object per row
        buf = np.ascontiguousarray(rows, dtype=self._dtype).tobytes()
        step = self._n * self._dtype.itemsize
        # dict.get per row slice; np.fromiter with count fills the int64 result
        # directly instead of going through a Python list
        positions = np.fromiter(
            (self._table.get(buf[j : j + step], -1) for j in range(0, len(buf), step)),
            dtype=np.int64,
            count=rows.size // self._n,
        )
        # back to the leading dimensions of rows: one position per syndrome row
        return positions.reshape(rows.shape[:-1])


class _DqiState:
    """The DQI state restricted to the errors the decoder gets right.

    |psi> ~ sum_y c(y) |B^T y> over the decodable errors y of weight <= l, with
    amplitude c(y) = w_k G(y) / sqrt(C(m, k)) and G(y) = prod_{i: y_i != 0} g~_i(y_i).

    Sampled benchmarks cover only part of each weight class, so every decodable
    error of weight k stands for ``scale_k = C(m,k) (p-1)^k (1 - epsilon_k) / N_k``
    errors of the full population (1 for exhaustive benchmarks).

    Internally every error has one flat position (the weight blocks concatenated
    in order) and ``syndromes`` and ``amplitudes`` are indexed by it.  Callers
    take a weight block with ``block`` / ``block_errors`` and look syndromes up
    with ``find``; the flat position only serves to index ``amplitudes``.

    A syndrome decoder maps every syndrome to exactly one error, so a valid
    benchmark never reports two correct errors with the same syndrome; such a
    benchmark is rejected here, because summing over it would model a state in
    which several errors of one coset interfere coherently, which DQI does not
    prepare.
    """

    def __init__(
        self,
        benchmarks: dict[int, BenchmarkResult],
        l: int,
        Bnp: np.ndarray,
        p: int,
        amplitude: Callable[[np.ndarray], np.ndarray],
    ):
        m = Bnp.shape[0]
        errors, syndromes, amplitudes = [], [], []
        self._scale_by_weight = []
        # exhaustive: every weight class was enumerated, so the sampled decodable
        # errors are *all* decodable errors of weight <= l (and every scale is 1)
        self.exhaustive = True
        for k in range(l + 1):
            Y = _stack_rows(benchmarks[k].correct, m)
            total_k = math.comb(m, k) * (p - 1) ** k  # errors of weight k that exist
            n_k = len(benchmarks[k].correct) + len(benchmarks[k].incorrect)
            # stores if all error strings were included or only a sample
            # an exhaustive state permits a shortcut in _Partners.amplitude
            self.exhaustive = self.exhaustive and n_k >= total_k
            # We estimate the total number of correctly decoded errors
            # using the ratio of correctly decoded errors (1 - epsilon) in the sample.
            population = total_k * (1 - benchmarks[k].epsilon)
            # scale_k is used to scale up all terms from the sample to the full population
            scale_k = population / max(1, len(Y))  # an empty block has nothing to scale

            errors.append(Y)
            # syndromes s(y) = B^T y for all y at once: (N_k, m) @ (m, n) = (N_k, n)
            syndromes.append((Y @ Bnp) % p)
            amplitudes.append(amplitude(Y))
            self._scale_by_weight.append(scale_k)

        # syndromes and amplitudes are stored in single arrays for quick lookup
        self.syndromes = np.concatenate(syndromes)  # (N, n)
        self.amplitudes = np.concatenate(amplitudes)  # (N,) complex
        self._errors = errors
        # block for error weight k is [start_k, start_{k+1})
        self._start = np.cumsum([0] + [len(Y) for Y in errors])

        # squared norm of the (rescaled) state: sum_k scale_k sum_{y in C_k} |c(y)|^2
        self.norm = sum(
            scale * float(np.sum(np.abs(c) ** 2))
            for scale, c in zip(self._scale_by_weight, amplitudes, strict=True)
        )
        if self.norm == 0:
            raise ValueError("no decodable errors in the benchmarks: the DQI state is empty")

        try:
            self._table = _SyndromeTable(self.syndromes, p)
        except _DuplicateSyndrome as clash:
            # clash.args are the flat positions of the two errors
            all_errors = np.concatenate(errors)
            offenders = [
                (int(np.count_nonzero(all_errors[j])), all_errors[j].tolist()) for j in clash.args
            ]
            raise ValueError(
                f"invalid benchmark: the errors {offenders} (as (weight, error)) "
                f"share the syndrome {self.syndromes[clash.args[0]].tolist()}, but a "
                f"syndrome decoder returns exactly one error per syndrome, so at "
                f"most one of them can be decoded correctly"
            ) from None

    def block(self, k: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Syndromes (N_k, n), amplitudes (N_k,) and population scale of weight k."""
        span = slice(self._start[k], self._start[k + 1])
        return self.syndromes[span], self.amplitudes[span], self._scale_by_weight[k]

    def block_errors(self, k: int) -> np.ndarray:
        """The sampled decodable errors of weight k as an (N_k, m) int array."""
        return self._errors[k]

    def find(self, rows: np.ndarray) -> np.ndarray:
        """Flat position of the sampled decodable error with each syndrome (-1 if none)."""
        return self._table.find(rows)


class _Partners:
    """The amplitude of the partner of a sampled error, for a shifted syndrome.

    The partner of y1 for the term (i, a) is the error the decoder returns for
    the syndrome s(y1) + a b_i.  It contributes iff decoding succeeds and the
    estimate has weight <= l; such an estimate is decodable by definition, since
    decoding its own syndrome returns it again.  A zero constraint row b_i = 0
    leaves the syndrome unchanged and returns y1 itself, which needs no special
    treatment.

    Three sources answer a syndrome, cheapest first:

    1. the table of *sampled* decodable errors in ``_DqiState``: if one of them
       has the syndrome, it is what the decoder returns (that is what "decoded
       correctly" means).  On exhaustive benchmarks a miss means there is no
       partner of weight <= l at all, and the lookup stops here;
    2. for a ``SyndromeDecoder`` its own coset-leader table, re-keyed like the
       state's table and built on the first miss; for any other decoder the
       neighbour y1 + a e_i whenever its weight is within the decoding radius,
       since the decoder is then guaranteed to return it;
    3. ``decoder.error_estimate``, once per syndrome (cached).
    """

    def __init__(
        self,
        decoder: AbstractDecoder,
        state: _DqiState,
        l: int,
        p: int,
        Bnp: np.ndarray,
        field,
        amplitude: Callable[[np.ndarray], np.ndarray],
    ):
        self._decoder = decoder
        self._state = state
        self._l = l
        self._p = p
        self._Bnp = Bnp
        self._field = field
        self._amplitude = amplitude
        # The syndrome decoder allows for quick decoding checks since
        # during pre-processing, it decodes all possible errors anyway.
        self._tabulated = isinstance(decoder, SyndromeDecoder)
        self._syndrome_coset_leaders: tuple[_SyndromeTable, np.ndarray] | None = None
        # decoding_radius() can be expensive (a minimum distance); only the
        # generic path on sampled benchmarks uses it
        self._radius = (
            decoder.decoding_radius() if not (self._tabulated or state.exhaustive) else None
        )
        self._cache: dict[bytes, complex] = {}

    def amplitudes(self, k: int, shifted: np.ndarray, lo: int, a: int) -> np.ndarray:
        """c(y2) for every ``shifted[i, j, :] = s(y_j) + a b_{lo + i}``, y_j of weight k."""
        position = self._state.find(shifted)  # (chunk, N_k), -1 where no error has it
        found = position >= 0
        # a -1 gathers the last amplitude, which np.where then replaces by 0
        out = np.where(found, self._state.amplitudes[position], 0)
        todo = ~found

        # If all partners are among the decoded samples or the state is exhaustive,
        # we know for sure that additional decoding will not recover any more errors.
        if self._state.exhaustive or not todo.any():
            return out

        # Shortcut: ad hoc decoding checks are free for the SyndromeDecoder.
        # Since the table stores all weight <= ell syndromes, a miss means that
        # the decoder believes the error to have weight > ell: an error that is
        # not included in the DQI state. We can therefore return early.
        if self._tabulated:
            table, amplitudes = self._get_syndrome_coset_leaders()
            position = table.find(shifted[todo])
            out[todo] = np.where(position >= 0, amplitudes[position], 0)
            return out

        # If the decoder has a guaranteed radius, all errors of weight <= radius
        # are always decoded correctly, so we can skip decoding.
        if self._radius is not None:
            # the neighbour y_j + a e_i differs from y_j in coordinate i = lo + row
            Y = self._state.block_errors(k)  # (N_k, m)
            # extract the original value at coordinate i (for each in coordinate in the chunk)
            old = Y[:, lo : lo + shifted.shape[0]].T  # (chunk, N_k)
            # compute the new value at coordinate i
            new = (old + a) % self._p  # (chunk, N_k)
            # Find all shifted syndromes, which fall inside the decoding radius
            # if non-zero -> zero: new_k = k - 1
            # if zero -> non-zero: new_k = k + 1
            # if non-zero -> non-zero: new_k = k
            # Since a != 0, the zero -> zero case never happens
            # (new != 0) - (old != 0) promoted to ints computes this weight change
            near = todo & (k + (new != 0) - (old != 0) <= self._radius)  # (chunk, N_k)

            # find the positions of all (syndrome, coordinate) pairs, which fall inside the radius
            rows, cols = np.nonzero(near)
            if len(rows):
                # Pick the relevant syndromes
                # Note that Y.shape = (N_k, m) but near.shape = (chunk, N_k).
                # Therefore, we must use Y[cols], not Y[rows]
                neighbours = Y[cols].copy()

                # Update the changed digits in these syndromes
                # (NumPy index magic new[rows, cols] = [new[rows[0], cols[0]], new[rows[1], cols[1]], ...])
                neighbours[np.arange(len(rows)), lo + rows] = new[rows, cols]
                # compute amplitude terms for these decoded syndromes
                out[near] = self._amplitude(neighbours)
            # mark these neighboring syndromes as done
            todo &= ~near

        # for all remaining syndromes in question, explicitly decode them and comput amplitudes
        for i, j in zip(*np.nonzero(todo), strict=True):
            out[i, j] = self._decode_and_compute_amplitudes(shifted[i, j])
        return out

    def _get_syndrome_coset_leaders(self) -> tuple[_SyndromeTable, np.ndarray]:
        """The coset leaders of weight <= l of a ``SyndromeDecoder``, keyed by syndrome."""
        # This function caches amplitude for all syndromes from the SyndromeDecoder table.
        # This table can be used to accelerate DQI performance estimation for
        if self._syndrome_coset_leaders is None:
            errors = self._decoder.get_syndrome_error_array(self._l)
            # the table may have been built for a larger l; leaders of weight > l
            # are not part of the state
            errors = errors[np.count_nonzero(errors, axis=1) <= self._l]
            table = _SyndromeTable((errors @ self._Bnp) % self._p, self._p)
            self._syndrome_coset_leaders = (table, self._amplitude(errors))
        return self._syndrome_coset_leaders

    def _decode_and_compute_amplitudes(self, syndrome: np.ndarray) -> complex:
        """Amplitude of the decoder's estimate for one syndrome row (0 if there is none)."""
        key = syndrome.tobytes()
        if key not in self._cache:
            estimate = self._decoder.error_estimate(
                vector(self._field, [int(c) for c in syndrome]), self._l
            )
            self._cache[key] = (
                0j
                if estimate is None
                else complex(self._amplitude(_stack_rows([estimate], len(estimate)))[0])
            )
        return self._cache[key]


def compute_expectation(
    instance,
    l: int,
    w: np.ndarray | None,
    benchmarks: dict[int, BenchmarkResult],
    decoder: AbstractDecoder,
) -> float:
    """Expected number of satisfied constraints of the DQI state, any prime p.

    The state is the one DQI actually prepares, restricted to the errors the
    decoder gets right:  |psi> ~ sum_k w_k / sqrt(C(m,k)) sum_{y in C_k} G(y) |B^T y>
    with G(y) = prod_{i: y_i != 0} g~_i(y_i).  Its objective value is

        <s> = 1/p sum_i sum_a phase[i, a] sum_{y1} conj(c(y1)) c(y2(y1, i, a))
              / sum_y |c(y)|^2

    with phase[i, a] = sum_{u in F_i} omega^{-a u} and y2(y1, i, a) the error the
    decoder returns for the syndrome s(y1) + a b_i (contributing 0 if decoding
    fails or the estimate has weight > l).  Exact for exhaustive benchmarks.

    Sampled benchmarks (``n_decoding_samples`` not ``None``) cover only part of
    each weight class: there are ``C(m,k) (p-1)^k`` errors of weight k, a
    fraction ``1 - epsilon_k`` of which is decodable, so a block of ``N_k``
    sampled errors stands for ``scale_k = C(m,k) (p-1)^k (1 - epsilon_k) / N_k``
    times as much amplitude weight, in the norm and in the numerator alike.
    Both are single sums over the sampled y1 -- the partner is decoded, not
    looked up among the sampled errors -- so every term carries the ``scale_k``
    of its y1 and nothing else, and both estimates are unbiased
    (Horvitz-Thompson; see performance_optimizations.md, section 3.9).

    ``decoder`` must be the decoder that produced ``benchmarks``: which errors
    count as decodable, and what a shifted syndrome decodes to, are properties
    of the decoder.
    """
    field = instance.field
    if not field.is_prime_field():
        raise ValueError("This function only supports prime fields")
    p = field.order()
    r = instance.get_r()
    m = instance.get_m()
    B = instance.get_B()
    Bnp = np.array(B.list(), dtype=np.int64).reshape(B.nrows(), B.ncols())
    n = Bnp.shape[1]
    F = instance.get_F()

    if w is None:
        _, w = get_eigenvector(make_A(p, r, m, l), -1)
        w = w / np.linalg.norm(w)
    w = np.asarray(w, dtype=float)
    if w.shape != (l + 1,):
        raise ValueError(f"w must have shape (l + 1,) = ({l + 1},), got {w.shape}")

    # F_i Fourier phases: phase[i, a] = sum_{u in F_i} omega^{-a u}.  These come
    # from writing the indicator [z in F_i] = 1/p sum_a sum_{u in F_i} omega^{a (z - u)};
    # the omega^{a z} part becomes the syndrome shift below, the omega^{-a u} part
    # is this table.  omega ** (-a_range * u) is the whole row a = 0..p-1 at once.
    omega = cmath.exp(2j * math.pi / p)
    a_range = np.arange(p)
    phase = np.zeros((m, p), dtype=complex)
    for i, F_i in enumerate(F):
        for u in F_i:
            phase[i] += omega ** (-a_range * int(u))

    # c(y) = coef[wt(y)] G(y), see _amplitudes
    coef = np.append(w / np.sqrt([math.comb(m, k) for k in range(l + 1)]), 0.0)
    gfactors = _g_tilde_table(p, r, F).T.copy()  # (p, m): gfactors[a, i] = g~_i(a)
    gfactors[0, :] = 1.0

    def amplitude(Y: np.ndarray) -> np.ndarray:
        return _amplitudes(Y, coef, gfactors)

    state = _DqiState(benchmarks, l, Bnp, p, amplitude)
    partners = _Partners(decoder, state, l, p, Bnp, field, amplitude)

    expectation = 0j
    for k in range(l + 1):
        S_k, c_k, scale_k = state.block(k)
        if len(c_k) == 0:
            continue
        # a = 0: no shift, so the partner is y1 itself and the sum collapses to
        # sum_i phase[i, 0] * sum_y |c(y)|^2.  Since phase[i, 0] = |F_i| = r this
        # is the random-guessing baseline m r / p after the division by p and the norm.
        expectation += scale_k * phase[:, 0].sum() * np.sum(np.abs(c_k) ** 2)
        # We chunk over the m axis, since otherwise the temporary "shifted" array
        # would contain m * N_k * n entries, surpassing memory capacity.
        # We choose the number of constraint rows per chunk so that it stays at a few million (32 MB)
        chunk = max(1, min(m, 4_000_000 // max(1, len(c_k) * n)))
        for a in range(1, p):
            for lo in range(0, m, chunk):
                b = Bnp[lo : lo + chunk]  # the rows b_i of this chunk, (chunk, n)
                # broadcasting (1, N_k, n) + (chunk, 1, n) -> (chunk, N_k, n):
                # shifted[i, j, :] = s(y_j) + a b_i mod p, the syndrome the partner
                # of y_j must have for constraint row i
                shifted = (S_k[None, :, :] + a * b[:, None, :]) % p
                # sum_i sum_{y1 in block k} phase[i, a] conj(c(y1)) c(y2) scale_k,
                # with phase as a (chunk, 1) column and conj(c) as a (1, N_k) row
                # broadcast against the (chunk, N_k) partner amplitudes
                expectation += scale_k * np.sum(
                    phase[lo : lo + chunk, a][:, None]
                    * np.conj(c_k)[None, :]
                    * partners.amplitudes(k, shifted, lo, a)
                )

    # 1/p from the character-sum form of the indicator, 1/norm from normalising
    # the state; the imaginary part is zero up to rounding since <f> is real
    return float((expectation / p / state.norm).real)
