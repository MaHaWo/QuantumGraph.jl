# Implementation Report: CUDA/device behavior

## Module
Task 12 — CUDADevice.jl + CUDA validation

## Implemented
- Added `src/CUDADevice.jl` and included it from `src/QuantumGraph.jl` before `Training.jl`.
- Added explicit `CUDA.jl` dependency in `Project.toml`.
- Implemented device-boundary APIs:
  - `DeviceError`
  - `ExecutionDevice`
  - `cpu_execution_device`
  - `accelerator_execution_device`
  - `cuda_available`
  - `validate_execution_device_settings`
  - `prepare_execution_device`
  - `prepare_value_for_device`
  - `prepare_model_for_device`
  - `prepare_graph_batch_for_device`
  - `prepare_model_and_graph_for_device`
  - `accelerator_backend_unavailable_error`
  - `single_accelerator_process_setup`
  - `no_distributed_device_setup`
- Integrated training config validation and trainer construction with the device-selection boundary.
- Trainer device movement now uses the shared shallow graph-batch transfer hook.
- Adjusted `dataset_dataloader` so MLUtils owns graph mini-batch collation with `collate=true` by default.
- Adjusted graph-batch device movement so `GraphNeuralNetworks.GNNGraph` values use public Flux/Adapt movement (`Flux.gpu`) before fixture-container fallback logic.
- Added native integration coverage in `test/integration/test_cuda_device.jl` and `test/integration/test_datasets_graph_samples.jl`.

## Acceptance criteria
- CPU execution is always valid and does not require CUDA availability.
- CUDA execution can be selected explicitly when availability is reported.
- CUDA requests fail clearly when unavailable.
- Multiple accelerator requests are rejected before any partial setup.
- `MLUtils.DataLoader(...; collate=true)` batches `GNNGraph` samples into a batched `GNNGraph`.
- Device movement preserves batched graph structure, including `num_graphs`.
- Model and graph movement use public Flux APIs (`Flux.gpu`) for CUDA preparation.
- No distributed/DDP/multi-machine setup is initialized or required.
- CUDA smoke assertions are gated by `cuda_available()`.

## BDD test results
Command:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Result for this module: PASS — `Feature: Optional single-accelerator device behavior` reported 5 scenarios succeeded, 0 failed.

Full-suite note: `Pkg.test()` still fails because later migration modules remain incomplete (`migration compatibility docs`, `public library surface`).

## Native integration test results
Command:

```bash
julia --project=. -e 'using Test; using QuantumGraph; const REPO_ROOT=pwd(); function incdir(rel); d=joinpath(REPO_ROOT,rel); for f in sort(filter(endswith(".jl"), readdir(d))); include(joinpath(d,f)); end; end; @testset "native" begin incdir("test/unit"); incdir("test/integration"); end'
```

Result: PASS — 258 native unit/integration assertions passed.

## Dependency interface compliance verification
- MLUtils.jl contract: `dataset_dataloader` delegates collation to `MLUtils.DataLoader`/`MLUtils.batch`, with `collate=true` as the graph-training default and caller override support.
- GraphNeuralNetworks.jl contract: batches of graph samples are represented as batched `GNNGraph` values, preserving graph metadata such as `num_graphs`.
- CUDA.jl contract: availability is checked through `CUDA.functional()` behind `cuda_available()`; simple array fallback movement uses `CUDA.cu`.
- Flux.jl contract: model and `GNNGraph` movement use `Flux.gpu`, preserving Flux/Adapt-compatible behavior.
- Training contract: trainer owns shallow batch movement at the batch boundary and does not require graph/model internals to know about CUDA.
- Distributed contract: DDP/multi-machine setup is explicitly absent and test-visible.

## Decisions made
- `ExecutionDevice` stores backend metadata rather than raw CUDA device handles so CPU-only tests can validate behavior deterministically.
- CUDA availability can be injected in validation/preparation calls for deterministic tests, while production defaults to `CUDA.functional()`.
- `dataset_dataloader` now defaults to graph-collating behavior because GraphNeuralNetworks expects mini-batches to be represented as batched `GNNGraph` values rather than vectors of graphs at model-call time.
- Exact numeric output parity on GPU is not part of the device-selection contract; later verification can add hardware-specific smoke tests when CUDA is available.

## Decisions requiring human confirmation
None.

## Deviations from plan
None.
