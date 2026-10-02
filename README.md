# DQI-Kit
DQI-Kit is a software framework for encoding combinatorial optimization problems into the Max-LINSAT and Max-XORSAT formats for the [Decoded Quantum Interferometry](https://arxiv.org/pdf/2408.08292) algorithm.

- [Features](#features)
- [Requirements and Installation](#requirements-and-installation)
- [Example](#example)
- [Usage](#usage)
    - [MaxLinSat](#maxlinsat)
    - [MaxConstraintSat](#maxconstraintsat)
    - [Dqi](#dqi)
    - [Decoders](#decoders)
    - [Solvers](#solvers)
- [Citation](#citation)
- [License](#license)

## Features
- user-defined Max-XORSAT and Max-LINSAT instances for fields of arbitrary size
- DQI performance estimation over prime fields in both the perfect and imperfect decoding regimes
- automatic transformation of domain objectives/constraints into the Max-LINSAT format:
    - integer equalities: $2 a - 3 b = 4 - 5 c$
    - integer inequalities: $\neq, \leq, <, \geq, >$
    - Boolean constraints: $\neg a \wedge (b \vee c)$
    - linear objectives: $\text{maximize} \,\,a + b + c$
    - binary polynomial objectives: $a (b + c) - 2 a$
    - binary polynomial constraints: $a (b + c) - 2 a \geq 0$
    - modular constraints: $a + b = 4 \pmod{5}$
    - range constraints: $a \in [0, 4]$
    - $\ldots$
- automatic selection of optimal DQI hyperparameters
- 6 implemented classical decoders + the ability to implement custom decoders
- 4 classical solvers + the ability to add custom solvers


## Requirements and Installation
DQI-Kit requires Python 3.12 or newer and has several dependencies, including SageMath, GAP and OR-Tools.
We thus recommend using our `Dockerfile`:
```bash
docker build -t dqi-kit . && docker run -it dqi-kit
```
The container runs `examples.py`, which applies DQI and simulated annealing to three small problems (knapsack, minimum vertex cover and maximum 3-colorable subgraph).
The image is reproducible: the base image is pinned by digest, all Arch packages (SageMath, GAP, Python, NumPy, ...) are resolved against a fixed snapshot of the [Arch Linux Archive](https://archive.archlinux.org/) selected by the `ARCH_SNAPSHOT` build argument in the `Dockerfile`, and the remaining Python packages are pinned in `requirements.txt`.
To move to a newer snapshot, run `scripts/update_docker_pins.sh YYYY/MM/DD`, which sets the date, resolves the direct dependencies from `requirements.in` inside the new image, rewrites `requirements.txt`, and runs the test suite in the rebuilt image.
DQI-Kit is a collection of top-level modules rather than an installable package: run your own scripts from the repository root (or with the root on `PYTHONPATH`) so that `import dqi` and friends resolve.

The test suite lives in `tests/` and uses pytest, which the image does not install:
```bash
docker run -it dqi-kit bash -c "pip install pytest && python -m pytest tests -q"
```

## Example
DQI for _Minimum Vertex Cover_:
```python
import networkx as nx

from classical_solvers import SimAnnealSolver
from dqi import Dqi
from max_constraint_sat import MaxConstraintSat

# Create graph
n = 10
G = nx.cycle_graph(n)

# Create problem
vertex_cover = MaxConstraintSat()

# Each vertex has one of two colors (0 = white, 1 = black)
x = [vertex_cover.new_binary_var(f"x_{i}") for i in range(n)]

# Objective: minimize the number of black vertices
vertex_cover.add_objective(
    sum(x_i for x_i in x),
    minimize=True,
)

# Constraints: for each edge (u, v) either u or v must be black
for u, v in G.edges:
    vertex_cover.add_boolean_constraint(
        x[u] | x[v],
        # constraints have priority over the objective
        weight=2,
    )

# Transform the constrained optimization problem into the Max-LINSAT format
max_lin_sat = vertex_cover.to_max_linsat()

# Estimate the DQI solution quality for degree l = 3
dqi = Dqi(max_lin_sat)
print("DQI:", dqi.estimate_solution_quality(l=3))

# Compare with a simulated annealing solver
sim_anneal = SimAnnealSolver(max_lin_sat)
print("simulated annealing:", sim_anneal.get_solution_quality())
```

## Usage

### MaxLinSat
Given a set of $m$ (weighted) linear constraints over $n$ variables, Max-LINSAT is the problem of finding the variable assignment which maximizes the (weighted) number of satisfied constraints.

The `MaxLinSat` class represents a (weighted) Max-LINSAT instance.
The finite field used is passed as a constructor argument:
```python
from max_lin_sat import MaxLinSat
from sage.all import GF

instance = MaxLinSat(GF(2 ** 8))
```
The `MaxXorSat` class from `./max_lin_sat.py` is a shorthand for `MaxLinSat(GF(2))`.
Instances over non-prime fields can be modelled and solved classically, but DQI performance estimation requires a prime field.

The constructor takes three further optional arguments:
- `default_decoder_constructor`: `Dqi` uses this decoder if none is passed explicitly (cf. decoders section).
- `merge_strategy` (default: `MergeStrategy.DUPLICATES`): Constraints with the same left-hand side (up to variable order and scaling) are collected into one group when the constraint matrix is built. This argument decides how the group is converted into rows of the Max-LINSAT matrix. `DUPLICATES` converts a weight-$k$ constraint into $k$ identical rows (if a right-hand side value has higher weight, it then appears in more rows). `WEIGHTS` produces one row per right-hand side and keeps the weights in a separate weight vector (`get_weights()`), which `evaluate_solution` and the classical solvers use but DQI estimation (as of now) does not. `USE_STRICTEST` keeps only the right-hand side with the highest weight as a single unweighted row, `USE_LOOSEST` instead keeps the union of all right-hand sides, again as a single unweighted row.
- `equal_size_F_i` (default: `True`): For (Standard) DQI, every constraint must have the same number $r$ right-hand side values. For `equal_size_F_i = True`, constraints are split, so all right-hand side sets are equal in size. Note that this can significantly increase the number of rows.
Variables are created for each instance by calling the `new_var(name: str)` method and are automatically restricted to be field elements.

Variables and scalars can be combined by using the `+`, `-`, `*`, `==` and `!=` operators to create constraints (e.g. `2 * a - 4 != b`).
Scalars are field elements from the SageMath `GF` field or `int`s, which are automatically converted to field elements.

To define set constraints, you can use the `==` and `!=` operators along with a `set` (e.g. `a + b == {0, 1}`).
Constraints are added to the `MaxLinSat` instance via the `add_constraint(constraint, weight=1)` method.
Only integer weights are supported.
Adding a constraint with an already existing left-hand side merges the two, using to the `merge_strategy` described above.
`add_gadget(gadget, *variables)` adds a precomputed block of constraints from `./gadget.py` (e.g. a Boolean AND or OR of several variables) over the given variables.

### MaxConstraintSat
The `MaxConstraintSat` class (in `./max_constraint_sat.py`) represents an instance of a constrained optimization problem.
Each `MaxConstraintSat` includes objectives and weighted constraints where the weight of each constraint represents the penalty if broken.
The goal is to find a variable assignment which maximizes the sum of all objectives minus the weighted sum of all violated constraints.

A new variable is created with the `new_var(name: str, lower: int, upper: int)` method on `MaxConstraintSat`.
Only integer variables are supported and each variable has to come with a lower and upper bound restricting its range.
Variables and scalars can be combined with `+`, `-`, `*` and `/` expressions (`/` is only supported if the right-hand side is a scalar).
As scalars, `int`s and `Fraction`s (from Python's `fractions` package) are supported.
Floats are also possible but are always converted into `Fraction`s.

An expression can be added as an objective using the `MaxConstraintSat.add_objective` method.
Objectives are maximizing by default, but `add_objective` takes an optional `minimize` Boolean argument.
The method takes an optional `weight` argument, which is a `Fraction` or `int`.

Two expressions can be combined with `==, !=, <=, <, >=, >` to form a constraint.
Constraints can be added using the `MaxConstraintSat.add_constraint` method, which again takes an optional `weight` argument.
Modular constraints can be defined using the `%` operator. Be careful with the operator precedence: `instance.add_constraint((a + b == 0) % 2)`.

`MaxConstraintSat` supports polynomial expressions and constraints like `(a + b) * (c + d) + e` or `a * b == 0`.

Binary variables can be created using the `new_binary_var` method.
Binary variables can be combined with the operators `~, &, |, ^` to create Boolean constraints.
A Boolean constraint (again with an optional `weight`) can be added using the `add_boolean_constraint` method.

A `MaxConstraintSat` instance can be converted into a `MaxLinSat` instance using the `to_max_linsat` method, which takes multiple optional arguments:
- `max_degree` (default: 3): The maximum degree of polynomial objectives/constraints and Boolean constraints is capped at this value. Any terms of larger degree are decomposed into smaller-degree terms.
A larger `max_degree` generally leads to more Max-LINSAT constraints, but fewer linearly dependent constraints.
- `var_range_constraint_factor` (default: 2): Each variable is defined with a range (`[lower, upper]`). Additional constraints are added during the transformation to restrict each variable to this range, which should generally get a larger weight. This weight is determined by this factor.
- `equality_constraint_factor` (default: 2): This argument similarly determines the weight of any additional equality constraints added when decomposing higher-order constraints/objectives.
- `default_decoder_constructor`, `merge_strategy`, `equal_size_F_i`: Passed on to the `MaxLinSat` constructor (see above).

The `Dqi` class and all solvers call `to_max_linsat` themselves, so a `MaxConstraintSat` instance can also be passed to them directly.

Note that `to_max_linsat` currently does not support the conversion of polynomial constraints over non-binary variables.

### Dqi
DQI-Kit provides a DQI solver: `Dqi` in `./dqi.py`.
The `Dqi` constructor takes two arguments, the Max-LINSAT instance and an optional decoder constructor (cf. the decoders section).
DQI performance estimation is supported for prime fields.
The main method provided by `Dqi` is `estimate_solution_quality`. Every argument after `l` is keyword-only:
- `l`: Determines the degree of the DQI polynomial: larger = potentially better performance, too large = performance degrades due to decoding errors.
If left empty, `l` defaults to the perfect decoding regime (where DQI performance can be computed analytically). Instances whose minimum distance is too small for that regime (`d <= 3`) default to `l = 1`.
- `w`: Determines the coefficients of the DQI bias polynomial as a real-valued NumPy array of length `l + 1`. If left empty, the coefficients are computed automatically. These automatically determined coefficients are optimal in the perfect decoding regime and approximately optimal in the imperfect decoding regime.
- `method` (default: `"auto"`): Selects the estimator.
`"auto"` takes the perfect-decoding closed form whenever it is valid (that is, the decoder corrects every error of weight at most `l` and `2l + 2 <= d`). If that is not possible (e.g. for heuristic decoders like belief propagation) the interference estimator is used instead.
`"analytical"` forces the closed form and raises if `l` is outside that regime.
`"interference"` forces the interference estimator, which measures the decoder's failure rates on sampled errors. It works on every prime field. When all errors of weight at most `l` are included, the result will be exact. Otherwise, it is approximate. (cf. `n_decoding_samples` below)
`"average_v_bound"` (GF(2) only) is the cheap lower bound that assumes uniformly random right-hand sides (e.g. $1$ for $a + b = 1$). it needs decoding failure rates only, but the bound can degenerate to `m / 2`, the value of random guessing, in which case a warning is issued.
- `n_decoding_samples` (default: `"auto"`): Determines how many errors are sampled for each error weight (this is only relevant for the imperfect decoding regime). Higher sample numbers lead to more accurate results, but take longer to compute. If set to `None` (or if the passed number exceeds the number of possible error patterns with a given weight) all possible errors are used. This leads to an exact result at the expense of an exponential runtime.
`"auto"` means `500` samples per error weight on every field. Empirically, the results are typically very close to the actual result (often less than 1%).
- `details` (default: `False`): If true, return a `QualityEstimate` record — the value plus the `l` and estimator actually used, the measured decoding failure rates, whether the number is guaranteed without any measurement (`guaranteed`) or exact for exhaustive benchmarks (`exact`), and whether the bound was clamped. `float(estimate)` is the value, and the record is always available afterwards as `dqi.last_estimate`.

Depending on the selected parameters `estimate_solution_quality` can take exponential time: the cost is dominated by the decoding benchmarks, which are exponential in `l` for `n_decoding_samples=None`. The interference estimator itself is cheap (well under a second for `m = 40, l = 3`).

As an alternative method, `semicircle_law_solution_quality(l=None, w=None)` computes the guaranteed solution quality of DQI in the perfect decoding regime.
`l` defaults to `d // 2 - 1` and a larger `l` is accepted but warns, since `2l + 2 <= d` is what makes the value a guarantee; note that the unique decoding radius `(d - 1) // 2` is one too large for odd `d`.
This requires Max-LINSAT instances with minimum distance at least 4.

### Decoders
Decoding algorithms for DQI are provided by the `./decoders.py` file. We provide 6 decoders:
- `NearestNeighborDecoder`: Finds the nearest codeword using brute force search. Works for arbitrary codes but is only fast enough for small values of `l`.
- `SyndromeDecoder`: Achieves the same results as `NearestNeighborDecoder` by creating an error lookup table. Requires exponential time once at creation but then achieves constant time decoding.
- `InformationSetDecoder`: A wrapper around the [`LinearCodeInformationSetDecoder`](https://doc.sagemath.org/html/en/reference/coding/sage/coding/information_set_decoder.html) from SageMath.
- `BeliefPropagationDecoder`: A wrapper around the [`BpDecoder`](https://software.roffe.eu/ldpc/ldpc/bp_decoder.html) from the `ldpc` package.
- `BeliefPropagationOsdDecoder`: A wrapper around the [`BpOsdDecoder`](https://software.roffe.eu/ldpc/ldpc/bposd_decoder.html) from the `ldpc` package.
- `GeneralizedReedSolomonDecoder`: Specialized decoder for generalized Reed-Solomon codes and the Optimal Polynomial Intersection problem (cf. [Jordan et al.](https://arxiv.org/pdf/2408.08292)).

Each decoder has a static `constructor` method that returns a constructor which can be passed to the optional `decoder_constructor` argument of the `Dqi` constructor. `InformationSetDecoder.constructor`, `BeliefPropagationDecoder.constructor` and `BeliefPropagationOsdDecoder.constructor` take optional parameters that are passed on to the underlying decoder implementations (such as `ldpc`'s `BpDecoder` for `BeliefPropagationDecoder`).
`InformationSetDecoder.constructor` additionally takes `seed` (default `0`) and `search_size` (default `None`). SageMath's information-set decoder is randomized; the seed gives each decoder its own random state, so it reproduces its decisions from run to run, and `seed=None` leaves SageMath's global random state in charge. With `search_size=None` SageMath calibrates Lee-Brickell's search size by timing a few random operations, which makes the choice depend on machine load; a fixed value (it is capped at `l`) removes that and only affects speed.
```python
from decoders import BeliefPropagationDecoder, SyndromeDecoder
from dqi import Dqi
from max_lin_sat import MaxXorSat

instance = MaxXorSat()
a, b, c, d = (instance.new_var(name) for name in "abcd")
instance.add_constraint(a + b == 1)
instance.add_constraint(b + c == 1)
instance.add_constraint(c + d == 0)
instance.add_constraint(a + d == 1)
instance.add_constraint(a + c == 0)
instance.add_constraint(b + d == 1)

# DQI with a syndrome lookup table decoder
syndrome_dqi = Dqi(
    instance,
    decoder_constructor=SyndromeDecoder.constructor(),
)
print(syndrome_dqi.estimate_solution_quality(l=2))

# DQI with a product-sum belief propagation decoder
bp_dqi = Dqi(
    instance,
    decoder_constructor=BeliefPropagationDecoder.constructor(bp_method="product_sum"),
)
print(bp_dqi.estimate_solution_quality(l=2))
```
If the user does not pass a `decoder_constructor`, one is determined automatically based on the code.

You can implement your own custom decoder by extending the `AbstractDecoder` class.
Here, you need to implement `decoding_radius()` and *exactly one* of the two decoding interfaces:
- `decoding_radius() -> int | None`: Returns how many errors are guaranteed to be corrected or `None` if no guarantees can be made. `estimate_solution_quality` uses it as a sufficient condition for error-free decoding, so it must never claim more than the decoder can prove; a radius larger than `(d - 1) // 2` is unsound for a unique decoder. A larger `l` is not an error, it just means the failure rates have to be measured. Complete decoders (`NearestNeighborDecoder`, `SyndromeDecoder`) return `(d - 1) // 2` via `AbstractDecoder._complete_decoding_radius()`, which is available to custom complete decoders as well.
- `decode_syndrome(vector, int) -> vector | None`: Gets the syndrome `s = B^T e` (a Sage vector of length `n`) and the number of errors `l`, and returns an error estimate of length `m` or `None` if decoding fails. This is what the DQI circuit does, so it is the preferred interface.
- `decode_codeword(vector, int) -> vector`: Gets a corrupted codeword (a Sage vector of length `m`) and the number of errors `l`, and returns the decoded codeword (it may raise Sage's `DecodingError`). Such a word-based decoder is turned into a syndrome decoder by `AbstractDecoder.error_estimate`, which decodes a fixed coset representative `rho` of the syndrome and returns `rho - decode_codeword(rho, l)`.

Implementing (or omitting) both raises a `TypeError`.

The interference estimator calls the decoder for syndromes that were never benchmarked: the interference partner of a benchmarked error `y` is whatever the decoder returns for the shifted syndrome `s(y) + a b_i`.
A decoder must therefore be a *function* of the syndrome — the same syndrome always decoding to the same estimate — which is what `AbstractDecoder.error_estimate` already assumes.
`SyndromeDecoder` is served from its lookup table directly, so custom decoders pay no penalty for it beyond their own decoding cost.

### Solvers
In addition to DQI, we provide classical solvers located in `./classical_solvers.py`.
All solvers get passed a `MaxLinSat` instance as the first constructor argument as well as potential additional arguments depending on the solver.
Each solver supports the following methods:
- `get_solution()`: Computes the solution or returns the previously computed solution as a Sage vector over the finite field
- `get_solution_quality()`: Returns the number of Max-LINSAT constraints satisfied by the solution.

The following solvers are provided by DQI-Kit:
- `BruteForceSolver`: Finds the optimal solution using brute force.
- `OrToolsSolver`: Uses the CP-SAT solver from Google's OR-Tools to find a solution. A time limit is passed as the second constructor parameter. If no limit is provided, the solver finds the optimal solution (with exponential runtime). The method `is_optimal()` returns `True` if the found solution is known to be optimal.
- `SimAnnealSolver`: Finds a solution using simulated annealing. Along with the Max-LINSAT instance, it accepts the following optional arguments:
    - `initial_state` (default random): The initial solution
    - `Tmax`, `Tmin` (default `25000`, `2.5`): the initial and final temperature
    - `steps` (default `10000`): the number of simulated annealing iterations
- `PrangeSolver`: Uses Prange's algorithm, which finds a linearly independent subset of $n$ constraints, satisfies them and "hopes for the best" for the remaining $m - n$ constraints.

To implement your own custom solver, simply extend the `AbstractSolver` class and implement the `compute_solution(instance)` method.

## Citation
If you use DQI-Kit in your research, please cite:

> S. Thelen and W. Mauerer, "From Constraint to Code: DQI-Kit — A Software Framework for Decoded Quantum Interferometry," in *2026 IEEE International Conference on Quantum Software (QSW)*, Sydney, Australia, 2026, pp. 117–128, doi: [10.1109/QSW72780.2026.00022](https://doi.org/10.1109/QSW72780.2026.00022).

```bibtex
@inproceedings{thelen2026dqikit,
  author    = {Thelen, Simon and Mauerer, Wolfgang},
  title     = {From Constraint to Code: {DQI-Kit} -- A Software Framework for Decoded Quantum Interferometry},
  booktitle = {2026 IEEE International Conference on Quantum Software (QSW)},
  year      = {2026},
  pages     = {117--128},
  publisher = {IEEE},
  doi       = {10.1109/QSW72780.2026.00022},
}
```

## License
DQI-Kit is released under the [Apache License 2.0](LICENSE).
