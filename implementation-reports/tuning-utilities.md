# Implementation Report: Tuning utilities

## Module
Task 10 — Tuning utilities

## Implemented
- Added `src/Tuning.jl` and included it from `src/QuantumGraph.jl`.
- Added the explicit Optuna.jl dependency in `Project.toml` (`Optuna = "a5d0552b-b2dc-4f08-ac5c-85ca7d701b92"`, compat `0.2.1`).
- Implemented Optuna-backed and backend-neutral tuning APIs:
  - `TuningError`
  - `FixedTrial`
  - `TuningStudy`
  - `TuningBackendLimitations`
  - `is_flat_list`
  - `is_categorical_suggestion`
  - `is_float_suggestion`
  - `is_int_suggestion`
  - `get_value_of_ref`
  - `convert_to_suggestion`
  - `get_suggestion`
  - `resolve_tuning_references!`
  - `build_search_space`
  - `create_tuning_study`
  - `preferred_tuning_backend`
  - `backend_neutral_tuning_concepts`
  - `unsupported_optuna_capabilities`
  - `save_best_config`
  - `export_best_config`
- Added native integration coverage for tuning in `test/integration/test_training_tuning_workflows.jl`.

## Acceptance criteria
- Search-space conversion identifies categorical, float, int, sweep, range, random-uniform, reference, and coupled-sweep nodes.
- Trial suggestions are applied by dotted parameter path.
- Coupled sweeps produce target-to-coupled mappings and reject invalid target/length combinations.
- References and coupled-sweep mappings are resolved after suggestions are selected.
- Backend-neutral tuning concepts are exposed while constructing a concrete Optuna.jl `Study` as the preferred backend.
- Supported Optuna storage forms include in-memory, journal `.log`, SQLite, and MySQL-style RDB URLs.
- Unsupported Optuna capabilities are reported as explicit backend limitations.
- Best configuration export preserves selected search-space values and resolved references/coupled sweeps.

## BDD test results
Command:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Result for this module: PASS — `Feature: Single-machine training and Optuna-backed tuning workflow boundary` reported 7 scenarios succeeded, 0 failed.

Full-suite note: `Pkg.test()` still fails because later migration modules remain incomplete (`CUDADevice`, migration compatibility docs, public library surface).

## Native integration test results
Command:

```bash
julia --project=. -e 'using Test; using QuantumGraph; const REPO_ROOT=pwd(); function incdir(rel); d=joinpath(REPO_ROOT,rel); for f in sort(filter(endswith(".jl"), readdir(d))); include(joinpath(d,f)); end; end; @testset "native" begin incdir("test/unit"); incdir("test/integration"); end'
```

Result: PASS — 222 native unit/integration assertions passed, including tuning search-space/reference/coupled-sweep/best-config checks and concrete Optuna.jl study/storage construction.

## Dependency interface compliance verification
- Config contract: tuning operates on the same sweep/range/reference/coupled-sweep node shapes used by the config migration.
- Trial interface contract: `FixedTrial` provides deterministic trial values by dotted parameter path, matching the source tests' Optuna `FixedTrial` usage.
- Backend abstraction contract: `TuningStudy` and `TuningBackendLimitations` expose backend-neutral concepts while `create_tuning_study` constructs a concrete Optuna.jl `Study`.
- Optuna dependency contract: `Tuning.jl` imports Optuna.jl directly and delegates live trial suggestions to `Optuna.suggest_categorical`, `Optuna.suggest_float`, and `Optuna.suggest_int` when the trial is not a deterministic `FixedTrial`.
- Training workflow contract: tuning is implemented after the trainer contract is stable and can later drive training objectives without changing trainer internals.

## Decisions made
- Optuna.jl is now an explicit runtime dependency, not a later placeholder.
- Best-config export currently writes a Julia textual representation and returns the resolved config structure. YAML-specific emission should be tightened when the final YAML dependency/format decision is made.

## Decisions requiring human confirmation
None.

## Deviations from plan
None.
