# Implementation Report: Evaluation and early stopping

## Module
Task 9 — Evaluation and early stopping

## Implemented
- Added DataFrames.jl as the tabular reporting dependency.
- Added `src/Evaluation.jl` and `src/EarlyStopping.jl`, included from `src/QuantumGraph.jl`.
- Implemented model-agnostic evaluation APIs with QuantumGravPy-compatible monitor task semantics:
  - `EvaluationError`
  - `evaluate_batch`
  - `evaluate_iterator`
  - `evaluation_report_dataframe`
  - `loss_report_columns`
  - `task_metric_columns`
  - `graph_batch_model_input`
- Implemented DataFrame-backed early stopping APIs:
  - `EarlyStoppingError`
  - `EarlyStoppingState`
  - `EarlyStoppingDecision`
  - `early_stopping_state`
  - `evaluate_early_stopping`
  - `continue_or_stop_decision`
  - `early_stopping_best_score`
  - `early_stopping_grace_state`
- Added native integration coverage in `test/integration/test_evaluation_early_stopping.jl`.

## Acceptance criteria
- Evaluation returns a `DataFrame` with `loss_avg`, `loss_min`, and `loss_max` columns.
- Configured monitor tasks add report columns.
- Monitor tasks are called once after all batches are processed with collected model outputs and targets, and their returned values are stored directly rather than averaged by evaluation.
- Evaluation passes graph-shaped batches to the model unchanged through the approved sample boundary; it does not inspect graph internals.
- Invalid task/metric configurations fail with user-facing `EvaluationError` messages.
- Early stopping consumes `DataFrame` history and updates best score, grace state, and found-better state for `:min` and `:max` modes.
- Empty history and missing metric columns fail with user-facing `EarlyStoppingError` messages.

## BDD test results
Command:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Result for this module: PASS — `Feature: Evaluation reports and early stopping over DataFrames` reported 6 scenarios succeeded, 0 failed.

Full-suite note: `Pkg.test()` still fails because later migration modules are not fully implemented yet (`CUDADevice`, migration compatibility docs, public library surface, and parts of training/tuning workflows).

## Native integration test results
Command:

```bash
julia --project=. -e 'using Test; using QuantumGraph; const REPO_ROOT=pwd(); function incdir(rel); d=joinpath(REPO_ROOT,rel); for f in sort(filter(endswith(".jl"), readdir(d))); include(joinpath(d,f)); end; end; @testset "native" begin incdir("test/unit"); incdir("test/integration"); end'
```

Result: PASS — 142 native unit/integration assertions passed, including `test/integration/test_evaluation_early_stopping.jl`.

## Dependency interface compliance verification
- DataFrames.jl contract: evaluation reports and early-stopping history use `DataFrame` values and column names as the public tabular interface.
- Model boundary contract: evaluation calls `model(batch)` with the batch unchanged, so graph-specific and non-graph models remain supported.
- Criterion callable contract: criteria are ordinary Julia callables over observable model outputs and batches.
- Monitor task contract: monitor callables receive collected outputs and targets for the full iterator, matching the QuantumGravPy evaluator task behavior.
- Early-stopping state contract: state mutation is explicit in `EarlyStoppingState`, preserving the observed Python behavior that stopping state evolves over evaluation history while making the mutation boundary public.

## Decisions made
- Evaluation is intentionally model-agnostic. It does not import or depend on GraphNeuralNetworks.jl and only forwards batches to the supplied model callable.
- Reports aggregate only loss values over processed batches into `loss_avg`, `loss_min`, and `loss_max`. Monitor task results are stored as returned.
- Early stopping supports `:min` and `:max` modes and uses the most recent history row as the current score.

## Decisions requiring human confirmation
None.

## Deviations from plan
None.
