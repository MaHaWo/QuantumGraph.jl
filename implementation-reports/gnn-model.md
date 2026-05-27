# Implementation Report: Composite GNN model

## Module
Task 8 — Composite GNN model

## Implemented
- Added `src/GNNModel.jl` and included it from `src/QuantumGraph.jl`.
- Added `GraphNeuralNetworks.jl` as the graph-model dependency and use `GraphNeuralNetworks.GlobalPool(mean)` for graph-level pooling. `Statistics.mean` is imported only as the aggregation function passed to `GlobalPool`.
- Implemented public composite model APIs:
  - `CompositeGNNModel`
  - `GNNModelError`
  - `construct_gnn_model`
  - `evaluate_gnn_model`
  - `gnn_model_outputs`
  - `gnn_model_embedding`
  - `active_task_outputs`
  - `stable_task_identifier`
  - `gnn_model_metadata`
  - `save_gnn_model_metadata`
  - `load_gnn_model_metadata`
- Implemented config-driven downstream task heads, active-task filtering, stable Symbol task identifiers, pooling/latent embedding paths, and structural metadata round-tripping.
- Added native integration coverage in `test/integration/test_gnn_model_boundary.jl`.

## Acceptance criteria
- Empty downstream task configuration fails with a user-facing `GNNModelError` identifying missing downstream task configuration.
- Active task configuration filters output dictionaries to only active task keys.
- Output keys use stable Julia `Symbol` task identifiers and remain stable across repeated evaluations.
- Embeddings are produced through the selected pooling or latent path.
- Incompatible pooling/latent configuration is rejected with a clear configuration error.
- Public metadata preserves model structure, active task configuration, and task key mapping without requiring exact stochastic parameter values.

## BDD test results
Command:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Result for this module: PASS — `Feature: Composite GNN model boundary` reported 5 scenarios succeeded, 0 failed.

Full-suite note: `Pkg.test()` still fails because later migration modules are not implemented yet (`CUDADevice`, evaluation/early-stopping, migration compatibility docs, public library surface, and training/tuning workflows).

## Native integration test results
Command:

```bash
julia --project=. -e 'using Test; using QuantumGraph; const REPO_ROOT=pwd(); function incdir(rel); d=joinpath(REPO_ROOT,rel); for f in sort(filter(endswith(".jl"), readdir(d))); include(joinpath(d,f)); end; end; @testset "native" begin incdir("test/unit"); incdir("test/integration"); end'
```

Result: PASS — 106 native unit/integration assertions passed, including `test/integration/test_gnn_model_boundary.jl`.

## Dependency interface compliance verification
- Dataset graph sample contract: `gnn_model_embedding` accepts the approved sample shape with `graph`, `features`, `targets`, and `source` fields; the `graph` value must be GraphNeuralNetworks-compatible for the pooling path.
- GraphNeuralNetworks pooling contract: graph-level embeddings are produced by `GraphNeuralNetworks.GlobalPool(mean)`, not by custom pooling logic.
- Models/Flux contract: the composite encoder and downstream heads are `Flux.Chain` values and the model is callable through `model(input)`.
- Stable output-key contract: configured task identifiers are normalized to Julia `Symbol` keys and stored in `task_key_mapping`.
- Metadata/config contract: public metadata is represented as `ConfigMetadata` and can reconstruct a structurally equivalent model via `load_gnn_model_metadata`.

## Decisions made
- The current implementation uses a Flux dense encoder and dense downstream task heads as the minimal structural equivalent for the composite boundary.
- The pooling path delegates graph-level feature aggregation to GraphNeuralNetworks.jl via `GlobalPool(mean)`. The latent path preserves per-node encoded values.
- Task metadata records structure and active-task mapping, not learned parameter values, matching the non-numeric migration policy.

## Decisions requiring human confirmation
None.

## Deviations from plan
None.
