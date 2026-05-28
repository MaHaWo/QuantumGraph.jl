# Implementation Report: Training orchestration

## Module
Task 11 — Single-process training orchestration

## Implemented
- Added `src/Training.jl` and included it from `src/QuantumGraph.jl`.
- Added `Serialization` as a Julia standard-library dependency for Julia-native structural checkpoint artifacts.
- Integrated Flux.jl / Optimisers.jl optimizer execution for differentiable Flux models while keeping the model-agnostic callable hook path for non-Flux or structural tests.
- Implemented model-agnostic training APIs:
  - `TrainingError`
  - `Trainer`
  - `construct_trainer`
  - `validate_training_config`
  - `prepare_training_components`
  - `start_training`
  - `fit_trainer!`
  - `run_single_machine_training!`
  - `save_julia_checkpoint`
  - `load_julia_checkpoint`
  - `write_training_config_copy`
  - `write_training_report`
  - `training_artifact_paths`
  - `local_single_machine_training`
  - `accelerator_backend_error`
  - `training_failure_error`
- Added native integration coverage in `test/integration/test_training_tuning_workflows.jl`.

## Acceptance criteria
- Trainer construction validates required configuration sections and reports missing/invalid sections with `TrainingError`.
- Trainer prepares configured dataset/model/optimizer/scheduler/evaluator/early-stopping components without starting training.
- Flux models can train with Optimisers.jl rules via `Flux.setup`, `Flux.withgradient`, and `Flux.update!` when a loss callable is supplied.
- Training applies early stopping after validation reports, saves `current_best` checkpoints on improvement, and stops before the maximum epoch count when patience is exceeded.
- Training runs locally without distributed setup or multi-machine initialization.
- Unsupported accelerator settings fail with clear backend errors.
- Deterministic fixture training writes structural artifacts: config copy, Julia-native `.jls` checkpoint, and DataFrame report artifact.
- Checkpoint write failures raise `TrainingError` and do not report a successful checkpoint.

## BDD test results
Command:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Result for this module: PASS — `Feature: Single-machine training and Optuna-backed tuning workflow boundary` reported 6 scenarios succeeded, 0 failed.

Full-suite note: `Pkg.test()` still fails because later migration modules remain incomplete (`CUDADevice`, migration compatibility docs, public library surface). The shared training/tuning BDD feature now passes, but Task 10 still needs dedicated native implementation for tuning semantics.

## Native integration test results
Command:

```bash
julia --project=. -e 'using Test; using QuantumGraph; const REPO_ROOT=pwd(); function incdir(rel); d=joinpath(REPO_ROOT,rel); for f in sort(filter(endswith(".jl"), readdir(d))); include(joinpath(d,f)); end; end; @testset "native" begin incdir("test/unit"); incdir("test/integration"); end'
```

Result: PASS — 188 native unit/integration assertions passed, including `test/integration/test_training_tuning_workflows.jl`, a Flux `Dense` + Optimisers `Adam` integration check that verifies parameter updates, and early-stopping/current-best checkpoint integration.

## Dependency interface compliance verification
- Config contract: trainer construction accepts a resolved config dictionary and validates required sections before starting training.
- Dataset/DataLoader contract: training consumes generic iterables of batches; it does not require graph-specific data structures.
- Model contract: training only requires a callable model and does not depend on `GNNModel` internals.
- Flux/Optimisers optimizer contract: Optimisers rules are initialized with `Flux.setup`, differentiated with `Flux.withgradient`, and applied with `Flux.update!`.
- Evaluation contract: evaluators return `DataFrame` reports, which are stored as Julia-native serialized artifacts.
- Early-stopping contract: validation reports are passed to `evaluate_early_stopping`; best-score/grace state is updated, current-best checkpoints are written on improvement, and training stops when the decision requests stopping.
- Filesystem artifact contract: config copies, periodic checkpoints, current-best checkpoints, and reports are written under the configured output/checkpoint paths.

## Decisions made
- Device handling is intentionally shallow. CPU is supported; a symbolic one-accelerator mode is accepted without embedding device logic deeply into datasets or models. Unsupported device strings fail early.
- Checkpoints use Julia `Serialization` into `.jls` files as a Julia-native structural format. They store epoch, model type, and config; exact learned parameters are not yet a golden-output requirement.
- Optimizer and scheduler are treated as optional callables/hooks when no differentiable loss is supplied; when `loss` and an Optimisers.jl rule are supplied, the trainer performs a real Flux optimizer step.
- The integration test verifies that a Flux `Dense` layer's weights change under `Optimisers.Adam`, but exact numeric loss trajectories remain outside scope.

## Decisions requiring human confirmation
None.

## Deviations from plan
None.
