# Changelog

All notable changes to DQI-Kit are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

> **AI disclosure.** The entries below were drafted by an AI coding assistant
> from the commit history and the maintainers' review notes, and were checked
> and edited by the maintainers before release.

## [0.2] - 2026-10-02

This release reworks the DQI performance estimation end to end. The
interference estimator was rewritten and is now exact and fast on every prime
field, the decoder interface was redesigned around syndromes, the minimum
distance computation was made practical for large instances, and a test suite
with 128 tests pinned and then fixed a long list of correctness defects, some
of which changed every non-binary estimate.

### Added
- `compute_expectation` in `dqi.py`: a single exact interference estimator for
  every prime field. It buckets the decodable errors by syndrome and decodes
  the interference partner of each benchmarked error instead of enumerating
  error pairs. It replaces the Monte-Carlo pair sampler on GF(2) and the
  quadratic double loop on GF(p). On the ternary Golay code at `l = 3` it
  takes 8 ms where the old loop took 54 s, and on GF(2) with `m = 24`,
  `l = 3` it is exact where the old sampler had an error of up to 0.3 in the
  interference matrix.
- `Dqi.estimate_solution_quality(method=...)` selects the estimator
  explicitly: `"auto"`, `"analytical"` (closed form, raises outside the
  perfect-decoding regime), `"interference"`, or `"average_v_bound"` (the old
  cheap GF(2) bound).
- `Dqi.estimate_solution_quality(details=True)` returns a `QualityEstimate`
  record with the value, the `l` and method used, the measured decoding
  failure rates, and whether the result is `guaranteed`, `exact` or
  `clamped`. The record is also kept in `Dqi.last_estimate`.
- `semicircle_law_solution_quality(l=None, w=None)` accepts `l` and `w`.
- Sampled decoding benchmarks (`n_decoding_samples`) are allowed on every
  field. `"auto"` means 500 samples per error weight; non-binary fields no
  longer require exhaustive benchmarks.
- Decoders can implement either `decode_syndrome` or `decode_codeword`
  (exactly one); `AbstractDecoder.error_estimate` turns a codeword decoder
  into a syndrome decoder. `AbstractDecoder._complete_decoding_radius()` is
  available to custom complete decoders.
- `InformationSetDecoder` takes `seed` (default `0`) and `search_size`.
  SageMath's information-set decoder is randomised and calibrates its search
  size by timing, so results depended on the run and on machine load; both
  can now be fixed so that the decoder is a function of the syndrome, as the
  DQI analysis assumes.
- `SyndromeDecoder.get_syndrome_error_array` exposes the coset-leader table
  as a NumPy array.
- `MaxConstraintSat.to_max_linsat` takes `equal_size_F_i`, `merge_strategy`
  and `default_decoder_constructor` and forwards them to `MaxLinSat`.
- `OptimalPolynomialIntersection.get_code` returns the DQI code `ker(Bᵀ)`,
  so every decoder, including the default one, can be used with it.
- `make_A` and the Fourier coefficients reject the degenerate cases `r = 0`
  and `r = p` with a `ValueError` instead of dividing by zero.
- Test suite in `tests/`: a brute-force oracle for the DQI state, an
  independent reference implementation of the estimator, equivalence tests
  against the pre-rewrite estimators over nine codes and four prime fields,
  and one regression test per fixed defect. `tests/bench_dqi.py` times the
  estimator. Run with `python -m pytest tests`.
- `pyproject.toml` with a `ruff` configuration; the code base is formatted
  and lint-clean.
- The Docker image is reproducible: the base image is pinned by digest, the
  Arch packages are resolved against the Arch Linux Archive snapshot of
  2026-10-02 (SageMath 10.10, GAP 4.16.1, Python 3.14.7, NumPy 2.5.3,
  SciPy 1.18.1), and the pip packages are pinned in `requirements.txt`
  (ldpc 2.4.1, ortools 9.15.6755, simanneal 0.5.0). It also installs
  `pytest`. `scripts/update_docker_pins.sh` moves the pins to a newer
  snapshot.
- `CHANGELOG.md`; the README has a citation and a license section.

### Changed
- `Dqi.estimate_solution_quality`: every argument after `l` is keyword-only.
  The `rhs_approximation` flag was replaced by `method="average_v_bound"`,
  and `n_interference_samples` was removed because nothing is sampled on the
  interference side any more. The default `l` is now the largest degree in
  the perfect-decoding regime, `d // 2 - 1`, capped by the decoder's radius.
- `method="auto"` no longer falls back to the average-right-hand-side bound
  in the imperfect regime. That bound degenerates to `m / 2`, the value of
  random guessing, on most instances (including the README example); it is
  now reachable only on request, and a clamped result warns. On the README
  vertex-cover example the default estimate went from 10.0 to about 11.9.
- Decoders are benchmarked as a function of the syndrome, not of the
  corrupted codeword, which is what the DQI circuit does. The previous
  benchmark let a decoder break ties towards the zero codeword and reported
  unrealistically high success rates.
- `AbstractDecoder.decode` was replaced by `decode_syndrome` /
  `decode_codeword`; `get_benchmarks` takes `(l, n_errors, n_tries)`.
- `NearestNeighborDecoder` and `SyndromeDecoder` report their decoding
  radius `(d - 1) // 2` instead of `None`, so the closed form is used when it
  applies. Computing it requires the minimum distance.
- `MaxLinSat.compute_minimum_distance` is much faster on large instances: a
  graph-based upper bound and a bound from single-variable constraints are
  tried first, and non-binary codes use a sampling-based approximation when
  `approximate=True`.
- `MaxLinSat.add_constraint` recognises scaled and reordered copies of a
  constraint as duplicates. Constraints with the same left-hand side up to
  variable order and scaling are merged into one canonical row with leading
  coefficient one; a constraint without a duplicate keeps its scaling.
- `MaxLinSat.add_constraint` raises `TypeError` for a weight that is not an
  `int`. Before, `Fraction` and `float` weights were silently truncated.
- `MaxLinSat.new_var` requires a name. An unnamed variable crashed on `str()`.
- `IntTerms.scale` and `IntTerms.scalar_div` raise `TypeError` instead of
  `ValueError` for a scalar of the wrong type.
- `MaxLinSatAnneal` no longer overrides `update`; progress printing is
  disabled through the annealer's `updates = 0` setting so that callers can
  re-enable it on an instance.
- The gadget library file is now called `gadget_library.json`.
- Python 3.12 or newer is required.
- The README was rewritten: every code block runs as written, the API names
  match the code (`to_max_linsat`, `add_boolean_constraint`,
  `get_solution_quality`), and `MergeStrategy`, `equal_size_F_i`,
  `add_gadget`, the prime-field restriction of the estimator and the test
  command are documented.

### Fixed
- `make_A` had the wrong diagonal for p > 2: `(p - 2r)/p · k` instead of
  `(p - 2r)/√(r(p - r)) · k`. Every non-binary semicircle-law value and every
  automatically chosen `w` for p > 2 was affected; the binary case was
  unchanged because both expressions vanish there.
- The exact non-binary estimator returned the random-guess value `m·r/p`:
  the degree-`l` term was dropped from every sum and the norm used
  `|g̃|` instead of `|g̃|²`.
- All Fourier-coefficient closures captured the last constraint's right-hand
  side, so every constraint was treated like the last one.
- A `NameError` made the GF(2) interference path unreachable.
- `semicircle_law_solution_quality` ignored `l` and `w`, and the
  perfect-decoding gate was one `l` too large for odd `d`. The condition is
  now `2l + 2 <= d`.
- The average-right-hand-side bound indexed the failure rates in reverse
  weight order.
- For non-binary fields `estimate_solution_quality` silently ignored
  `rhs_approximation`, `n_decoding_samples`, `n_interference_samples` and `w`.
- `MergeStrategy.USE_LOOSEST` raised `TypeError` on every instance.
- `add_objective(minimize=True)` was a no-op, so the README example maximised
  the vertex cover; `add_boolean_constraint` dropped its weight; `>=` and `>`
  excluded the upper bound.
- `MaxLinSat.add_constraint` never rejected variables from another instance:
  the membership test used `==`, which the variable DSL overloads to build a
  constraint, so it was always true. The check now compares by identity.
- `MaxLinSat.compute_B_F_weights` checked the wrong length when removing a
  constant offset from a constraint whose right-hand side contains every
  field element, and it divided the weight gcd out of the user's constraint
  objects in place. The function no longer mutates the instance.
- The graph-based minimum-distance bound assumed that every cycle of
  two-variable constraints is a linear dependency. That is false over GF(p)
  for p > 2, where it returned a distance of 3 for codes of distance 4 and
  could admit the closed form outside its regime. The method was rewritten
  for binary and non-binary fields.
- `MaxLinSat.get_v` and `get_random_solution_value` read the cached
  right-hand sides directly and crashed on a fresh instance.
- `UnweightedIsing` could not be constructed.
- `IntConstraint.__init__` ignored its `required_field_order` argument.
- The decoder benchmark cache ignored `l`, so a second call with a different
  `l` returned the first result.
- `GeneralizedReedSolomonDecoder` decoded the primal Reed-Solomon code
  instead of the dual code that DQI uses, reporting weight-2 errors as
  correctly decoded. Guruswami-Sudan now falls back to Berlekamp-Welch when
  it cannot be constructed for the requested radius.
- Decoding failures (`DecodingError`, empty list-decoding results) aborted
  the benchmark instead of counting as incorrect decodings, and
  `compute_benchmarks` divided by zero when no error of the requested weight
  exists.
- The belief-propagation decoders relied on `ldpc`'s automatic input
  detection; they now pass a syndrome explicitly, built from `Bᵀ`.
- `SyndromeDecoder.constructor(**kwargs)` forwarded parameters its
  constructor did not accept, and the benchmark object was not initialised,
  which crashed DQI simulation.
- `OrToolsSolver.is_optimal()` raised `TypeError` when called before
  `get_solution()`, and then solved twice.
- `BruteForceSolver` returned a tuple where every other solver returns a Sage
  vector.
- `MaxLinSatAnneal.move` could pick a change of zero, so a fraction 1/p of
  the annealing moves were no-ops.
- Debug output in `get_minimum_distance` was removed.
- The `np.matrix` deprecation warning emitted by every conversion of a Sage
  matrix to NumPy is gone.

### Removed
- `approximate_A_bar` and `Dqi._dqi_compute_exact_performance_non_binary`,
  superseded by `compute_expectation`. The pre-rewrite estimators are
  preserved in `tests/legacy_estimators.py` for the equivalence tests.
- `MaxLinSat.get_decoding_radius`; the decoder's `decoding_radius()` is the
  single source of truth.
- `AbstractDecoder.decode`, see the decoder interface change above.
- The `rhs_approximation` and `n_interference_samples` arguments of
  `estimate_solution_quality`.

## [0.1] - 2026-03-24

Initial release, accompanying *From Constraint to Code: DQI-Kit — A Software
Framework for Decoded Quantum Interferometry* (QSW 2026).

- `MaxLinSat` and `MaxXorSat` instances over arbitrary finite fields with a
  variable and constraint DSL.
- `MaxConstraintSat`: integer, Boolean, polynomial, modular and range
  constraints and linear or polynomial objectives, transformed into
  Max-LINSAT with degree reduction.
- `Dqi`: solution-quality estimation in the perfect-decoding regime
  (semicircle law) and the imperfect regime, with automatic selection of
  `l` and the DQI polynomial.
- Six decoders (nearest neighbour, syndrome table, information set, belief
  propagation with and without OSD, generalized Reed-Solomon) and four
  classical solvers (brute force, OR-Tools CP-SAT, simulated annealing,
  Prange).
- Dockerfile with SageMath, GAP and all Python dependencies.

[0.2]: https://github.com/lfd/dqi-kit/compare/v0.1...v0.2
[0.1]: https://github.com/lfd/dqi-kit/releases/tag/v0.1
