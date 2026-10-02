"""Timing script (not a pytest test).  Run with:  .venv/bin/python tests/bench_dqi.py

Produces the numbers quoted in report.md.
"""

import os
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ldpc.mod2  # noqa: E402
import numpy as np  # noqa: E402
import scipy as sp  # noqa: E402
from sage.all import GF, vector  # noqa: E402

from decoders import BenchmarkResult, NearestNeighborDecoder, SyndromeDecoder  # noqa: E402
from dqi import Dqi, compute_expectation, get_eigenvector, make_A  # noqa: E402
from max_lin_sat import MaxLinSat, MaxXorSat  # noqa: E402
from tests.oracles import golay_instance, max_lin_sat_from_B  # noqa: E402


def timed(label, fn):
    t = time.time()
    out = fn()
    print(f"{label:<60s} {time.time() - t:8.3f} s")
    return out


def random_xorsat(m, n, seed=0):
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


def benchmarks(dq, l, n_tries):
    m = dq.instance.get_m()
    dec = dq.get_decoder()
    bm = {0: BenchmarkResult([vector(dq.instance.field, [0] * m)], [], 0)}
    for k in range(l, 0, -1):
        bm[k] = dec.get_benchmarks(l, k, n_tries)
    return bm


print("=" * 90)
print("Imperfect-decoding estimators: syndrome bucketing (numpy), binary instances")
print("=" * 90)
for m, n, l in [(24, 12, 3), (40, 20, 3)]:
    ls = random_xorsat(m, n)
    d = ls.get_minimum_distance()
    dq = Dqi(ls, SyndromeDecoder.constructor())
    bm = timed(
        f"[m={m} n={n} l={l}] benchmarks (500 samples/weight, syndrome decoder)",
        lambda: benchmarks(dq, l, 500),
    )
    _, w = get_eigenvector(make_A(2, 1, m, l), -1)
    timed(
        f"[m={m} n={n} l={l}] compute_expectation",
        lambda: compute_expectation(dq.instance, l, w, bm, dq.get_decoder()),
    )
    timed(
        f"[m={m} n={n} l={l}] Dqi.estimate_solution_quality (end to end)",
        lambda: dq.estimate_solution_quality(l=l),
    )

print()
print("=" * 90)
print("Non-binary interference estimator (was an O(N^2 m p) Sage loop: 6.4 s at l=2, 54 s at l=3)")
print("=" * 90)
B, m, n, d = golay_instance(3)
ls = max_lin_sat_from_B(GF(3), B, [i % 3 for i in range(m)])
dq = Dqi(ls, NearestNeighborDecoder.constructor())
for l in [1, 2, 3]:
    bm = timed(
        f"[ternary Golay m={m} l={l}] exhaustive benchmarks (NN decoder)",
        lambda: benchmarks(dq, l, None),
    )
    _, w = get_eigenvector(make_A(3, 1, m, l), -1)
    value = timed(
        f"[ternary Golay m={m} l={l}] compute_expectation",
        lambda: compute_expectation(dq.instance, l, w, bm, dq.get_decoder()),
    )
    print(f"    <s> = {value:.6f}")

print()
print("=" * 90)
print("compute_minimum_distance: Sage right_kernel edge-case check vs ldpc rank")
print("=" * 90)
ls = random_xorsat(2000, 200)
B = ls.get_B()
timed(
    "[m=2000 n=200] B.T.right_kernel().dimension()  (as in compute_minimum_distance)",
    lambda: B.T.right_kernel().dimension(),
)
H = sp.sparse.csc_matrix(np.array(B.T).astype(np.int8))
timed("[m=2000 n=200] ldpc.mod2.rank(B.T)", lambda: ldpc.mod2.rank(H))
timed("[m=2000 n=200] full get_minimum_distance()", lambda: ls.get_minimum_distance())

print()
print("=" * 90)
print("evaluate_solution (Sage) vs numpy")
print("=" * 90)
ls = random_xorsat(300, 100)
B = ls.get_B()
x = vector(GF(2), [random.randint(0, 1) for _ in range(100)])
timed(
    "[m=300 n=100] 2000 x evaluate_solution (Sage)",
    lambda: [ls.evaluate_solution(x) for _ in range(2000)],
)
Bnp = np.array(B).astype(np.int8)
xnp = np.array(x).astype(np.int8)
v = np.array([int(next(iter(f))) for f in ls.get_F()])
timed(
    "[m=300 n=100] 2000 x numpy (B @ x) % 2 == v",
    lambda: [int(np.sum((Bnp @ xnp) % 2 == v)) for _ in range(2000)],
)

print()
print("=" * 90)
print("Non-binary approximate minimum distance: fixed 5 s wall-clock loop")
print("=" * 90)
ls = MaxLinSat(GF(3), equal_size_F_i=False)
xs = [ls.new_var(f"x{i}") for i in range(6)]
random.seed(3)
for _ in range(14):
    ls.add_constraint(sum(random.randint(0, 2) * x for x in xs) == random.randint(0, 2))
timed(
    "[GF(3) m=14 n=6] get_minimum_distance() (approximate=True)", lambda: ls.get_minimum_distance()
)
timed(
    "[GF(3) m=14 n=6] code.minimum_distance() exact via Sage",
    lambda: ls.get_code().minimum_distance(),
)
