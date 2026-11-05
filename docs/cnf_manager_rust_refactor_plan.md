# CNF Manager → Rust: Refactor Plan

Status: decisions D1–D6 taken (§8). Branch: `claude/cnf-manager-python-rust-42oyv0`.
Reviewed once by the project's simplifier agent; its findings are folded in (§3.2, §3.4, §4, §8).

## 1. Recommendation

Replace `information_gain_aml/core/cnf_manager.py` (PySAT/Minisat model enumeration plus a Python-side
solution cache that is filtered per query) with a Rust extension module
`information_gain_aml.core._sat_kb` exposing `SATKnowledgeBase`, the name the migration doc already
reserves for this component. Keep only the methods the learner actually calls, and change the engine:
an exact model counter (DPLL with unit propagation and component splitting) over an explicit, fixed
variable universe, instead of "enumerate every model with blocking clauses, cache the list, filter it
in Python". That removes the solution cache and the private-state plumbing in the learner and the
parallel workers, removes the enumeration cliff, and fixes four semantic bugs found while profiling.
Ship it as the only implementation (maturin build, abi3 wheels) and delete the Python file once parity
is proven.

What it buys, measured (§2): in greedy runs on small problems the CNF manager is 20–32% of runtime,
so the ceiling for this refactor alone is ~1.3–1.45x end to end. Under MCTS it is 45% (~1.8x). On
formulas with 16+ constrained variables the current enumeration takes seconds to minutes per query;
the counter takes tens to hundreds of microseconds; rover p00 cannot get past its third iteration
today for exactly this reason (§2.4). The largest hotspot overall is `grounding.py`
(40–58%), which this refactor does not touch (§9).

Decided direction (§8): one Rust crate with two parts, `SATKnowledgeBase` (Phases 0–3) and a gain
kernel that owns grounding and the per-grounded-action gain loop with parallelism inside Rust
(Phase 4, §11). The `ProcessPoolExecutor` path is deleted in Phase 2 (§10). The CNF part alone is
worth 1.3–1.8x end to end; the kernel is where the 10x lives, because the per-action loop is
65–95% of runtime and most of it is string grounding (§11.1).

## 2. Evidence

All numbers from this session, sequential path (`num_workers=0`), cProfile with builtin time charged
to the calling module. Driver: a loop of `select_action → env.execute → observe`, identical to
`ExperimentRunner.run_experiment`.

### 2.1 Where the time goes

| Run | ms/iter | grounding.py | cnf_manager.py (Python) | learner | PySAT C core |
|---|---|---|---|---|---|
| blocksworld p01, greedy, 150 it | 55 | 40% | 32% | 16% | 0.6% |
| depots p00, greedy, 120 it | 111 | 58% | 20% | 15% | 0.4% |
| blocksworld p01, MCTS (20 it × depth 3), 12 it | 5,019 | 42% | 45% | 12% | 0.0% |
| rover p00, greedy | cannot pass iteration 2 (§2.4) | — | — | — | — |

Inside `cnf_manager.py` nearly all of the time is pure-Python set filtering, not solving
(blocksworld greedy run):

| Function | calls | cumulative |
|---|---|---|
| `count_models_with_assumptions` → `..._via_filter` | 10,636 | 1.42 s |
| `count_models_with_temporary_clause` → `_count_models_with_clause_via_filter` | 3,765 | 0.69 s |
| `get_all_solutions` (enumeration, cache fill) | 31 | 0.63 s |
| Minisat `solve` + `get_model` + `add_clause` (C) | 27k | 0.26 s |

Formula sizes at the end of the greedy runs are tiny: 1–7 variables in clauses, 1–20 clauses,
1–544 models per action. The filter costs ~0.13 ms per query at that size and the learner issues
70–90 queries per iteration. Under MCTS the same blocksworld problem ends with 11-variable formulas
and ~2,000-model caches, the filter cost rises to 0.72 ms per query (it is linear in cache size), and
the manager answers ~3,000 queries per step: 27 s of the 60 s run.

### 2.2 The enumeration cliff, and the two counter prototypes

Random 3/4-literal clauses, same formulas for every column. "Python counter" and "Rust counter" are
~100-line prototypes of the §3.3 algorithm written for this plan (untuned; the Rust one allocates per
recursion level). Both match brute force on the verifiable cases.

| vars in clauses | clauses | models | today: `get_all_solutions` | Python counter | Rust counter |
|---|---|---|---|---|---|
| 10 | 8 | 362 | 4.1 ms (43% in Minisat) | 220 µs | 23 µs |
| 16 | 10 | 34,608 | 1,678 ms (78%) | 890 µs | 96 µs |
| 19 | 12 | 245,252 | 75,601 ms (98%) | 3,250 µs | 387 µs |
| 11 / 6 (MCTS-sized, 200 instances) | | | today's filter: 720 µs (MCTS), 130 µs (greedy) | 210 µs | 18 µs |

Blocking-clause enumeration is roughly quadratic in the number of models and the cache filters are
linear in it (1.3 ms per query with 17k cached solutions, 3.1 ms for a temporary clause). Actions
with 3–4 parameters and `learn_negative_preconditions=True` have `|La|` in the 30–60 range, so the
cliff is reachable on larger problems and under MCTS/lookahead. Subsumption (0.21 ms per call at 300
clauses) and refinement (1 ms at 300 clauses) are negligible.

### 2.3 Semantic hazards in the current implementation

Today the counting universe is "whatever variable range PySAT happens to know", so results depend on
variable-id allocation order and on call history.

| Probe | Result today | Problem |
|---|---|---|
| `(a ∨ c)` with an unused fluent `b` mapped **between** them | 6 models | `b` is enumerated as free and doubles the count |
| `(a ∨ b)` with an unused fluent `c` mapped **after** them | 3 models | same situation, other answer: count depends on id order |
| `(a ∨ b)` + assumptions on two new fluents: SAT path vs cache-filter path | 3 vs 0 | the two code paths disagree; production uses the cache path, the unit tests exercise the SAT path |
| `(a ∨ b)`, two unused fluents, assume the last one true | base 3, numerator 6 | applicability "probability" of 2.0 |
| `count_models_with_temporary_clause` | filter path if every fluent of the clause was mapped earlier (e.g. by `state_constraints_to_assumptions`), SAT path otherwise | the answer for the same formula and clause depends on what was asked before |

The subsumption rule makes the third case reachable in production: a failure constraint `B ⊇ B'` is
kept in `pre_constraints[a]` but its clause is dropped, so `⋃pre?(a)` contains literals whose
variables occur in no clause. Measured footprint on real runs (100 iterations, p01, seed 0), comparing
today's counts with the fixed-universe semantics of §3.2:

| Query | blocksworld | depots |
|---|---|---|
| applicability probability | 34 of 6,636 queries differ (0.5%): today 0.0, fixed 0.05 | 0 of 71,200 |
| failure gain (zero vs positive reduction) | 0 of 2,491 flip category; reduction differs by ≤ 2x in a few | 0 of 36,435 flip; ≤ 4x in a few |

So the change is a bug fix with a small, visible footprint, not a silent rewrite of the algorithm.
It still needs sign-off (D4) and the Phase 2 evaluation run.

### 2.4 Rover

rover p00, greedy, sequential: iteration 0 (`calibrate`, failed) took 0.4 s. Iteration 1
(`communicate_image_data`, failed) took 42 s, almost all of it enumerating the 131,071 models of
`calibrate`'s single 17-literal clause into the solution cache (`information_gain.py:618`). Iteration 2
then has to enumerate the 2^33 − 1 models of `communicate_image_data`'s single 33-literal clause; two
attempts were killed after 8 and 10 minutes inside that call. The pre-population runs before the
sequential/parallel split, so the parallel path hits the same wall. An exact counter answers both in
microseconds (a single k-literal clause has 2^k − 1 models over its k variables), and the fixed
universe makes the number independent of which fluents happen to be mapped.

## 3. Target design

### 3.1 Layout

```
rust/Cargo.toml                 crate `sat_kb`, crate-type cdylib; dependency: pyo3 0.29 (abi3-py310, extension-module)
rust/src/lib.rs                 pymodule `_sat_kb`
rust/src/formula.rs             VarMap (fluent ↔ 1-based id), clause store, subsumption, refinement
rust/src/counter.rs             exact #SAT over the formula's constrained variables
rust/src/python.rs              #[pyclass] SATKnowledgeBase
information_gain_aml/core/_sat_kb.pyi   typing stub, so `uv run mypy information_gain_aml/` stays green
information_gain_aml/core/cnf_manager.py   transition oracle in Phases 1–2, deleted in Phase 3
```

No Rust-side oracle tests: the Python property test in §6 exercises the real binding and is the single
oracle. `cargo clippy -D warnings` and `cargo fmt --check` run in CI.

Net Python file count in `information_gain_aml/`: unchanged (one `.py` removed, one `.pyi` added).

### 3.2 Semantics

The universe is explicit and fixed per action: the fluents of `La` (every literal that can ever be a
precondition), registered when the knowledge base is created (`SATKnowledgeBase(fluents)`, called from
`information_gain.py:200` with `_get_parameter_bound_literals(action)` stripped of `¬`). Every count
is over that universe `U`:

- `count_solutions()` = number of assignments to `U` satisfying `F`. Variables in no clause contribute
  a factor of 2 each. Empty formula → `2^|U|`.
- `count_models_with_assumptions(A)` = assignments to `U` satisfying `F ∧ A`. An assumption on a
  variable in no clause halves the count (it is a free hypothesis being fixed), exactly as the
  algorithm doc's `pr(app) = |SAT(cnf_pre?(a,O,s))| / |SAT(cnf_pre?(a))|` implies when both counts
  range over the same assignment space. Result is always in [0, 1].
- `count_models_with_temporary_clause(C)` = assignments to `U` satisfying `F ∧ C`. Adding a clause
  over a fixed universe can only reduce the count, so `preFailPotential` (doc line 223) is never
  negative and never needs the clamp at `information_gain.py:431` / `parallel_gain.py:531`.
- A fluent outside `U` passed to any method raises `ValueError` (it cannot be a precondition).
- `refine_clauses_by_intersection` keeps dropping clauses that become empty (today's behaviour).
  `add_clause([])` makes the formula unsatisfiable (today's behaviour).
- Counts are Python `int`s: Rust computes the exact count over the constrained variables
  (`u128`, checked) and shifts by the number of free universe variables.

Consequences for the learner, all in Phase 0: the empty-formula fallback in `_get_base_model_count`
(`information_gain.py:1510–1515`, `parallel_gain.py:441–446`) returns `2^|La|` or `2^(|La|/2)` depending
on mode; make it `2^|fluents(La)|` in both modes, which is what the knowledge base returns for an
empty formula. It is unreachable today because callers return early when `pre?(a)` is empty, so this
changes no result.

Existing unit tests in `tests/core/test_cnf_manager.py` already agree with this spec on every counting
assertion (each fixture has all variables in clauses). The unused `blocksworld_precond` fixture in
`tests/conftest.py:138–147` claims 4 models for a 3-model formula; delete it.

### 3.3 Counter

Plain DPLL counting over the residual formula: unit propagation → conflict gives 0 → no clauses left
gives `2^(unassigned variables)` → otherwise split the remaining clauses into connected components
(shared variables), multiply component counts, branch on the most frequent variable inside a
component. Two traps, both hit while writing the prototype for §2.2: pure-literal elimination is
**invalid** for counting, and a variable that drops out of every clause during propagation or
conditioning is free and contributes a factor of 2. Add memoisation of residual components only if
measurements on rover/satellite-sized formulas ask for it.

Measured on the untuned prototype: 18 µs per count at MCTS size, 387 µs for the 245k-model case.
`PyO3` argument conversion adds a few microseconds per call.

### 3.4 Python-facing surface

Only what production uses, plus the accessors the tests need. Everything else is deleted, not ported.

| Keep (same name, same signature unless noted) | Notes |
|---|---|
| `SATKnowledgeBase(fluents: Iterable[str] = ())` | new: registers the universe |
| `add_fluent`, `add_clause`, `add_clause_with_subsumption`, `refine_clauses_by_intersection`, `build_from_constraint_sets`, `clear_formula`, `has_clauses` | unchanged |
| `count_solutions()` | `max_solutions` parameter dropped (no production caller passes it) |
| `state_constraints_to_assumptions`, `count_models_with_temporary_clause` | unchanged |
| `count_models_with_assumptions(assumptions)` | `use_cache` dropped |
| `clauses`, `fluent_to_var`, `var_to_fluent`, `next_var` | read-only properties returning copies |
| `__repr__` | `SATKnowledgeBase(vars=…, clauses=…)` |
| `__reduce__` | only if the parallel path stays (D5) |

| Delete | Why |
|---|---|
| `get_all_solutions`, `get_model`, `is_satisfiable`, `get_probability`, `get_entropy`, `to_string`, `copy` | no production callers (grep of `information_gain_aml/` and `scripts/`); test-only |
| `count_models_with_assumptions_via_filter`, `_count_models_with_clause_via_filter`, `_solution_cache`, `_cache_valid`, `_solver`, `_solver_valid`, `__del__`, `cnf` (the PySAT `CNF` object) | cache and solver lifecycle gone |

Errors: every binding returns `PyResult`; no `unwrap` on inputs, so bad input raises `ValueError`
rather than a `PanicException`.

## 4. Python integration changes (exact sites)

| File | Lines | Change |
|---|---|---|
| `information_gain_aml/algorithms/information_gain.py` | 200 | `CNFManager()` → `SATKnowledgeBase(fluents of La)` |
| same | 609–618 | delete the solution-cache pre-population loop (includes the only `get_all_solutions` call) |
| same | 839–874 (`_create_context`) | replace the five `cnf_*` dicts and their constructor arguments with `cnf_managers=self.cnf_managers` |
| same | 904–919 (`_create_sequential_context`) | drop the empty `cnf_*` placeholder dicts |
| same | 981, 1202, 1482 | `cnf.cnf.clauses` → `cnf.clauses` |
| same | 1510–1515 | fallback → 2^n with n = number of fluents in La (§3.2) |
| same | 22, 117, 1461 | import and type `CNFManager` → `SATKnowledgeBase` |
| `information_gain_aml/algorithms/parallel_gain.py` | 41–46 | `ActionGainContext`: five `cnf_*` fields → `cnf_managers: Dict[str, SATKnowledgeBase]` |
| same | 83–176 | delete `_reconstruct` (`mgr.cnf = CNF()` / `mgr.cnf.append` at 142–149); `WorkerCNFCache` and `SequentialGainCache` become identical → one class, drop `GainCacheBase` |
| same | 134 | drop the `pysat.formula` import |
| same | 441–446 | fallback → 2^n with n = number of fluents in La |
| `information_gain_aml/core/__init__.py` | 4, 11 | export `SATKnowledgeBase`; keep `CNFManager = SATKnowledgeBase` for one release |
| `tests/core/test_cnf_manager.py` | whole file | rewrite around the §3.4 surface: `.cnf.clauses` → `.clauses` (16 sites), drop tests of deleted methods, drop the two `_cache_valid` asserts; add the oracle property test here |
| `tests/algorithms/test_parallel_gain.py` | 53–66, 111–157 | fixture loses the `cnf_*` fields; reconstruction tests become a pickle round-trip test (or go, under D5) |
| `tests/algorithms/test_information_gain_observation.py` | 39, 61 | `.cnf.clauses` → `.clauses` |
| `tests/conftest.py` | 138–147 | delete `blocksworld_precond` |

`scripts/`, `experiments/`, `mcts_selector.py` and `model_export.py` reach the manager only through the
learner. `grounding.py:57` has its own unrelated `to_string`.

## 5. Packaging, CI, release

Keys verified against the maturin 1.16.0 source.

| Item | Change |
|---|---|
| `pyproject.toml` | `build-backend = "maturin"`; `[tool.maturin] manifest-path = "rust/Cargo.toml"`, `module-name = "information_gain_aml.core._sat_kb"`, `features = ["pyo3/extension-module"]`; `exclude = [{ path = "information_gain_aml/experiments/**", format = "wheel" }, { path = "information_gain_aml/environments/**", format = "wheel" }]`. No `python-source`: maturin finds `information_gain_aml/__init__.py` at the project root. Drop `python-sat` from dependencies |
| Wheels | abi3 (`abi3-py310`): one wheel per platform, valid for 3.10 through 3.13+. Targets: manylinux x86_64 and aarch64, macOS arm64 and x86_64, Windows x64, plus sdist |
| `.github/workflows/ci.yml` | Rust toolchain + cargo cache; `cargo fmt --check`, `cargo clippy -D warnings`; `uv sync` then builds the extension and the existing pytest/mypy steps run unchanged |
| `.github/workflows/publish.yml` | replace `python -m build` with a `maturin-action` matrix producing the wheels above and the sdist; keep the "wheel excludes experiments/environments" check; publish all artifacts |
| Local dev (macOS) | `rustup` once; `uv sync` builds the module; `maturin develop --release` for quick rebuilds |
| Version sync | `pyproject.toml`, `information_gain_aml/__init__.py`, `rust/Cargo.toml` → update `docs/releasing.md` and `.claude/rules/releasing.md` |
| Docs | README requirements (drop python-sat; "prebuilt wheels, Rust only for source installs"), CLAUDE.md key-files table, algorithm doc "CNF/SAT Integration Components" (Minisat → exact counter, universe from §3.2), CHANGELOG |
| Version | 0.4.0 (MINOR: new engine, removed methods, alias kept) |

Risk to the AMLGym registry: none as long as a wheel exists for the platform, because the module is a
hard dependency of the package. The TestPyPI dry run in `docs/releasing.md` is the gate.

## 6. Phases

| Phase | Work | Exit criteria | Est. |
|---|---|---|---|
| 0. Prepare in Python (mergeable on its own) | add `scripts/profile_learning_loop.py` (the §2 driver; CLAUDE.md asks for profiling before optimisation work and §9 needs it again); delete the pre-population loop and the `blocksworld_precond` fixture; add the `clauses` property to `CNFManager` and migrate the §4 call sites; align the fallback; add `tests/core/test_cnf_manager.py::test_counts_match_truth_table`, a random-formula property test (≤ 10 variables) against brute force with the §2.3 cases as `xfail(strict=True)` on the current class; record D1–D5 | pytest + mypy green; the xfails document today's divergences | ½ day |
| 1. Rust core | crate, counter, bindings, `.pyi`, `maturin develop`; the Phase 0 property test parametrised over both classes; micro-benchmark on the §2.2 formulas | property test passes on Rust with the §2.3 xfails removed; 245k-model case under 1 ms | 2–3 days |
| 2. Integrate and evaluate | §4 changes; delete the process-pool path (D5, §10): `parallel_gain.py`'s worker machinery, the pool methods and `_create_parallel_context` in the learner, `test_parallel_gain.py`; the gain functions move into the learner as the single path; `num_workers` kept as an accepted no-op for one release; full `uv run pytest tests/ -v` and `uv run mypy information_gain_aml/`; runs with a fixed seed on blocksworld/depots p01 (greedy) and one MCTS run, before and after, via `run-local-experiment` / `run-aml-gym-experiment` | all tests green; learned models identical except where a §2.3 case fired, each difference explained; profile shows `_sat_kb` ≤ 3% of runtime; ms/iter before/after recorded in the PR | 1 day |
| 3. Package and release | maturin backend, wheel matrix, CI steps, docs, remove `python-sat` and `cnf_manager.py`, bump to 0.4.0, TestPyPI dry run, install check on macOS arm64 | `pip install information-gain-aml==0.4.0` works without a Rust toolchain on macOS and Linux; CI green on 3.10/3.11/3.12 | 1 day |

| 4. Gain kernel in Rust (§11) | interner, templates, binding cache, bitset state, `compute_gains` with rayon, `applicability`; learner pushes model-set changes to the kernel after each observation; hot-path grounding functions in `grounding.py` lose their callers; differential test of gains against the Python functions over 50 iterations on blocksworld/depots before those functions are deleted; update `.claude/rules/migration.md` | gains identical to the Python path on the differential run; blocksworld p01 and depots p03 ms/iter recorded with 1 and 4 threads; `parallel_gain.py` gone | 3–4 days |

Total ≈ 9–10 working days. Phase 0 can merge on its own. Between Phase 2 and Phase 4, problems with
thousands of grounded actions run about 2x slower than today (§10.1); everything else runs faster.

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| The §3.2 semantics change results | Footprint measured in §2.3 (0.5% of blocksworld applicability queries, none in depots); property test pins the semantics; Phase 2 runs make every difference visible before merge |
| Counter bug | truth-table property test on the real binding; PySAT implementation kept as the Phase 1–2 differential oracle and in git history |
| Source installs need Rust | abi3 wheels for all supported platforms; sdist builds with `rustup` |
| `uv sync` + maturin editable flow on the user's macOS | verify in Phase 1 on the user's machine; fallback `maturin develop` |
| mypy breaks on the extension module | `.pyi` stub next to the module, checked in CI |
| Panics crossing into Python | `PyResult` everywhere; `cargo clippy -D warnings` |
| Speedup oversold | §2 numbers in the PR; Phase 2 exit criterion requires measured before/after |

## 8. Decisions

| # | Decision | Outcome |
|---|---|---|
| D1 | Engine and language: (a) Rust exact counter; (b) pure-Python exact counter inside `cnf_manager.py`, no new toolchain; (c) faithful Rust port of enumeration + cache behind a CDCL crate | **Decided: Rust.** The counter replaces enumeration rather than porting it: rover's 33-literal clause has 2^33 − 1 models, which is 2.4 hours at 1 µs per model even in Rust, while the counter answers it in microseconds. Measured per query at MCTS size: today 130–720 µs, pure-Python counter 210 µs, Rust counter 18 µs |
| D2 | Distribution: single maturin-built package vs separate accelerator package with a pure-Python fallback | **Decided: no Python fallback; the knowledge base is a standalone Rust module.** It lives in its own crate (`rust/`) with no dependency on the rest of the package, so it can be published as its own PyPI package later if aml-gym or KG-AML want to reuse it. It ships inside the `information-gain-aml` wheel for now (one release pipeline); splitting it out is a packaging change only |
| D3 | Naming: `SATKnowledgeBase` + one-release `CNFManager` alias vs keep `CNFManager` | **Decided:** `SATKnowledgeBase` + alias |
| D4 | Counting universe = fluents of `La` for every count (§3.2) | **Decided: accepted.** It is what the algorithm doc's formulas assume, it removes the clamp, and the measured footprint is small |
| D5 | `ProcessPoolExecutor` path: keep (then add `__reduce__`, collapse the two caches) or delete `parallel_gain.py`'s worker machinery | **Decided: delete it; parallelism lives inside Rust** (rayon over grounded actions, GIL released, no pickling). Measurements in §10 |
| D6 | Move grounding and the per-grounded-action gain loop to Rust | **Decided: yes.** It is heavy lifting: string grounding alone is 36–53% of runtime and the loop it sits in is 65–95%. Scope and design in §11 |

## 9. Out of scope

- The hypothesis-space denominators (`3^n` for failure gain, `|La|` for success gain) and the
  2-valued CNF encoding are left exactly as they are; this plan only makes the counts consistent with
  each other.
- PDDL parsing, the domain representation and the environment (`pddl_io.py`, `up_adapter.py`,
  `lifted_domain.py`, `active_environment.py`) stay on their PDDLGym/AMLGym migration path. The Rust
  crate consumes strings and never sees PDDL.
- `ObjectSubsetManager` and `ground_all_actions*` (2–4% of runtime) stay in Python for now.

## 10. D5: the process-pool path (decided: delete)

### 10.1 Measurements

30 iterations each, this container (4 cores, Linux `fork`; macOS uses `spawn`, which makes pool
start-up slower but does not change the per-iteration picture). "Grounded" is the size of the full
grounding, the number of gain computations an iteration can issue. The context pickle is what the
learner ships to every worker on every iteration; the lean figure is the same context without the
clause, variable-map and solution-cache fields, which is what it becomes after the Rust port.

| Problem | grounded | sequential | 4 workers | speedup | context pickle today → lean |
|---|---|---|---|---|---|
| blocksworld p01 | 40 | 30 ms/iter | 50 ms/iter | 0.60x | 27 KB → 2 KB |
| depots p00 | 84 | 28 ms/iter | 34 ms/iter | 0.82x | 17 KB → 3 KB |
| blocksworld p05 | 144 | 135 ms/iter | 174 ms/iter | 0.77x | 38 KB → 2 KB |
| depots p03 | 3,528 | 624 ms/iter | 293 ms/iter | 2.13x | 18 KB → 3 KB |

The fixed cost of a parallel iteration is about 20 ms here: four chunk submissions, four context
pickles, worker-side reconstruction of every `CNFManager` from clauses and caches
(`WorkerCNFCache.update_context` throws the previous ones away), result transfer and scheduling. The
pool pays for itself only when the per-iteration work is large, which today means thousands of
grounded actions. The default is `num_workers=None` (auto), so every small and medium run, including
the AMLGym runs and the `quick`/`standard` experiment modes on small problems, is slower than it
needs to be today.

### 10.2 What changes after the Rust port

- The CNF part of a gain computation drops from 20–45% of the work to ~1%. What remains per
  grounded action is Python grounding and bookkeeping, which is still parallelisable across
  processes, so the 2x on large problems survives the port.
- The worker-side reconstruction disappears if `SATKnowledgeBase` pickles itself (`__reduce__`,
  ~10 lines of Rust); the lean context is 2–3 KB. The fixed cost per iteration shrinks but does
  not vanish; the crossover stays in the hundreds of grounded actions.
- The next refactor (§9) moves grounding into the same crate. At that point the right parallelism
  is inside Rust: a thread pool over grounded actions with the GIL released, no pickling and no
  worker processes. A `ProcessPoolExecutor` adds nothing once that exists.

### 10.3 Trade-offs

Keep the pool (simplified: pickle the Rust object, one worker cache class):

- Pro: ~2x on problems with thousands of grounded actions, today and after the port; MCTS and
  lookahead call the gain computation many times per step and reuse the pool.
- Con: a net loss by default on every problem measured except the largest; needs a size threshold
  or an opt-in default to stop hurting small runs.
- Con: two execution paths to keep identical, 531 lines in `parallel_gain.py` plus ~200 in the
  learner plus 352 test lines, per-task exceptions swallowed into a 0.0 gain, a 120 s chunk
  timeout that silently drops results, and a fallback to sequential on pool failure.
- Con: a process pool inside a library is awkward for the aml-gym target (host frameworks may run
  their own multiprocessing; `spawn` on macOS re-imports the package in every worker).
- Con: it is the main source of the caching-and-referencing complexity that motivated this refactor.

Delete the pool (one sequential path; the shared gain functions move back into
`information_gain.py`, `ActionGainContext` and the worker cache go away):

- Pro: one deterministic path, no pickling contract on `SATKnowledgeBase`, ~900 fewer lines
  including tests, simpler aml-gym integration, and it matches the 3-class target.
- Pro: faster by default on every problem under a few hundred grounded actions.
- Con: large problems lose the 2x until grounding moves to Rust (depots p03: 0.62 s/iter instead
  of 0.29 s; a 500-iteration run takes 5 minutes instead of 2.5).

Middle ground: keep the pool as explicit opt-in (`num_workers` default 0, parallel only when asked
for and only above a grounded-action threshold). Smallest change, but all of the code and both
paths stay.

### 10.4 Decision

Deleted in Phase 2. The `num_workers` keyword stays accepted for one release so existing scripts and
the README example keep working. Parallelism returns inside Rust in Phase 4 (§11). The interim cost is
confined to problems with thousands of grounded actions, and even there it is a factor of two on
wall-clock, not a capability loss.

## 11. Phase 4: the gain kernel in Rust (D6)

### 11.1 What the hot path does today

Every call to `_compute_single_action_gain` (one per grounded action per `select_action`, and per
MCTS/lookahead node) grounds the lifted literals of `pre(a)` by string parsing three times: once for
the early-termination check, once in `_calculate_applicability_probability`, once in
`_calculate_potential_gain_success` (which also grounds `eff?+` and `eff?-`). Each grounding of one
literal is `ground_parameter_bound_literal`: split on `(`, `,`, `)`, decode `?x`/`?y`/`?p10` with
`_parameter_index_from_name`, join with `_`, then a set lookup.

| | blocksworld p01 greedy, 150 it | depots p00 greedy, 120 it | blocksworld p01 MCTS, 12 it |
|---|---|---|---|
| gain computations | 6,000 | 10,080 | 29,280 |
| `_get_satisfied_literals` calls | 17,876 | 29,924 | 80,276 |
| `ground_parameter_bound_literal` calls | 347k | 769k | 2.94M |
| runtime under the gain loop | 65% | 80% | 95% |
| of which string grounding | 36% | 53% | 40% |
| of which CNF queries | 27% | 16% | 45% |
| per gain computation | 0.85 ms | 1.1 ms | 1.95 ms |

Nothing in this loop needs strings. The lifted literal templates are already positional:
`LiftedDomainKnowledge.get_parameter_bound_literals` generates parameter names with
`_generate_parameter_names` (`?x ?y ?z ?u ?v ?w ?p ?q ?r ?s ?t ?p11 …`) and
`_parameter_index_from_name` decodes the same convention, so a template is a predicate plus a tuple
of parameter positions and a sign.

### 11.2 Design

- **Interning at the boundary.** Predicates, objects and grounded fluents get `u32` ids; a grounded
  fluent is `(predicate, object ids)`. Per action schema, the `La` templates become
  `(predicate, positions, sign)` ids. Strings cross the boundary once: when the domain is loaded,
  when a state arrives, and when the learner pushes a model update.
- **Model state per action**, pushed by the learner after each observation (`set_model(action,
  pre, eff_add, eff_del, eff_maybe_add, eff_maybe_del, has_success)`): bitsets over the action's
  template ids. `pre?(a)` already lives in the `SATKnowledgeBase`; the kernel keeps the union of its
  literals as a bitset and the base model count cached until the knowledge base changes.
- **Binding table** per distinct `(schema, object tuple)`: template id → grounded fluent id, built
  once and cached. Object-subset rotation adds entries, never invalidates them.
- **State** as a bitset over grounded fluent ids, built once per call from the Python `set[str]`.
- **`compute_gains(state, grounded_actions) -> list[(name, objects, gain)]`**: rayon `par_iter`
  over grounded actions with the GIL released. Per action: satisfied literals by bitset ops; the
  early-termination test; applicability = `count_with_assumptions / base`; success potential;
  failure potential = `count_with_temporary_clause`; expected gain. The knowledge base is read-only
  during the call and counting is pure, so sharing it across threads is `&self` with no locking.
  Results sorted by `(-gain, name, objects)` as today, so the output is deterministic regardless of
  thread count.
- **`applicability(action, objects, state) -> float`** for `_filter_applicable_actions` and for
  MCTS `simulate_action`.
- **Stays in Python**: `observe`/`update_model` with `lift_fluent_set` (1% of runtime), constraint
  bookkeeping and knowledge-base updates, `ground_all_actions*` and `ObjectSubsetManager` (2–4%),
  every selection strategy including lookahead and MCTS (they call `compute_gains`),
  `_apply_injective_binding_filter`, export and metrics.

Files: `rust/src/intern.rs`, `rust/src/gain.rs`; `parallel_gain.py` and the pool code are already gone
after Phase 2; the learner's `_calculate_applicability_probability`, `_calculate_potential_gain_*`
and `_get_satisfied_literals` are replaced by kernel calls; `grounding.py` keeps the cold-path
functions (`ground_all_actions*`, `lift_grounded_fluent`, `is_injective_binding`) until PDDLGym
replaces them. `.claude/rules/migration.md` changes its line for `grounding.py`: the hot path goes to
the Rust kernel, PDDLGym may still replace the cold path, and the 3-class target becomes
`SATKnowledgeBase` + gain kernel in Rust, `InformationGainLearner` as Python orchestration over them,
`Agent` unchanged.

### 11.3 Expected effect and evidence required

Estimate, to be measured in Phase 4: a gain computation drops from 0.85–1.95 ms to a few tens of
microseconds (bitset work is about a microsecond; one or two counter queries at 18 µs each on the
untuned prototype). With the loop at 65–95% of runtime that is 10x or more single-threaded, and
near-linear scaling across cores on problems like depots p03 (3,528 grounded actions). The Phase 4
exit criteria in §6 require the before/after numbers, not the estimate.

Risks: gain parity (floating-point equality against the Python functions over a 50-iteration
differential run before deletion); the early-termination and certain-precondition logic must be
replicated exactly; the binding cache for large problems is small (3,528 actions × ~60 templates of
`u32`).

