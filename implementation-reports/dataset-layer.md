# Implementation Report: Dataset layer and graph sample conversion

## Module
Task 6 — Dataset layer and graph sample conversion

## Implemented
- Added `src/Datasets.jl` and included it from `src/QuantumGraph.jl`.
- Implemented public dataset APIs:
  - `QuantumGraphDataset`
  - `DatasetError`
  - `dataset_sample_count`
  - `construct_dataset`
  - `map_dataset_index`
  - `read_dataset_sample`
- Updated `test/steps/datasets_graph_samples_steps.jl` to build real Zarr-backed fixture stores and exercise dataset behavior directly.
- Added MLUtils integration for the Flux-compatible dataset/dataloader boundary:
  - `MLUtils.numobs(::QuantumGraphDataset)`
  - `MLUtils.getobs(::QuantumGraphDataset, index)`
  - `dataset_dataloader(dataset; kwargs...)`
- Added `@datasets` tags for module-specific BDD execution.

## Acceptance criteria
- Sample count uses approved precedence: `num_causal_sets`, then `num_samples`, then one-dimensional dataset inference, then `adjacency_matrix` shape fallback.
- Dataset construction stores lazy Zarr handles and does not eagerly materialize every sample.
- Valid sample reads return graph structure, features, targets, and source metadata through documented fields.
- Dataset supports the MLUtils/Flux dataloader interface for sample counts, indexed observations, and mini-batches.
- Missing reader or unsupported store layouts fail with user-visible errors.
- Out-of-range indexes fail with requested index / bounds context.
- Global indexes map to the correct backing store and local sample index.

## BDD test results
Command:

```bash
julia --project=. -e 'using Behavior; ok=Behavior.runspec(pwd(); featurepath=joinpath(pwd(), "specs"), stepspath=joinpath(pwd(), "test", "steps"), tags="@datasets"); exit(ok ? 0 : 1)'
```

Result: PASS — 7 scenarios succeeded, 0 failed.

## Decisions made
- The dataset layer returns Julia named tuples with `graph`, `features`, `targets`, and `source` fields rather than introducing a QuantumGraph graph wrapper. This follows the Level 2 decision not to wrap GraphNeuralNetworks.jl data structures unnecessarily.
- The current graph boundary is represented with Julia containers (`NamedTuple` fields) that can be adapted to GraphNeuralNetworks.jl once model integration chooses the concrete graph type.
- MLUtils is the dataset abstraction dependency because Flux re-exports and uses MLUtils' dataloader interface; this avoids inventing a QuantumGraph-only batching abstraction.
- `construct_dataset` accepts a custom reader function but rejects `nothing` as a missing reader.

## Decisions requiring human confirmation
None.

## Deviations from plan
None.
