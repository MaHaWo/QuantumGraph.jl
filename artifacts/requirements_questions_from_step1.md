# Requirements: QuantumGraph.jl

## Accepted discovery inputs

- source_repo: `/Users/hmack/Development/QuantumGrav/QuantumGravPy`
- output_repo: `/Users/hmack/Development/QuantumGraph.jl`
- artifact_dir: `/Users/hmack/Development/QuantumGraph.jl/artifacts`
- target_language: Julia
- source_language: Python (inferred from repository)

## Status

NOT READY: requirements intake is in progress. Planning must not begin until these requirements are completed and explicitly approved by the human.

## Evidence reviewed

- Characterization report: `/Users/hmack/Development/QuantumGraph.jl/artifacts/characterization-report.md`
- Repository overview HTML: `/Users/hmack/Development/QuantumGrav/QuantumGravPy/scout/overview.html`
- Overview summary: `/Users/hmack/Development/QuantumGraph.jl/artifacts/overview-summary.md`

## Current inferred facts

- Current system is a Python package named `QuantumGrav` for configurable GNN experiments.
- Current public use is library/import oriented; no package console scripts are declared.
- Current stack includes PyTorch, PyTorch Geometric, zarr, NumPy, pandas, PyYAML, jsonschema, joblib, Optuna, pytest, and mkdocs.
- Main current runtime path is `Trainer(config)` with zarr-backed `QGDataset`, dynamic `GNNModel` composition, evaluation/early stopping, checkpoints, DDP support, and Optuna tuning helpers.
- Characterization tests were added at `/Users/hmack/Development/QuantumGrav/QuantumGravPy/test/test_characterization_current_behavior.py`.

## Requirements questions pending human answer

### Scope and user base

1. Should the migration scope be the whole Python package, or a narrower slice such as library API, numerical/model kernel, training pipeline, dataset/Zarr layer, or tuning utilities?
2. Who is the intended user of the Julia implementation: research developers, domain scientists, students, downstream library users, or another group?

### Julia target shape

3. Should `/Users/hmack/Development/QuantumGraph.jl` be a standard Julia package exposing `QuantumGraph`/`QuantumGrav`-equivalent modules, an application/scripts repo, or both?
4. Do you want a preferred Julia ML/graph stack, or should planning keep this open? The Python stack is PyTorch/PyG; likely Julia choices affect architecture and tolerances.

### Behavioral compatibility

5. Should the Julia migration preserve the Python public API semantics closely, or provide idiomatic Julia APIs with compatibility only at behavior/file/config boundaries?
6. The current package has no console scripts. Should Julia likewise remain library-only, or should CLI/script entry points be added intentionally?
7. Should current YAML config semantics be preserved exactly, including custom tags (`!sweep`, `!coupled-sweep`, `!range`, `!random_uniform`, `!reference`, `!pyobject`), Python-object import equivalents, in-place run-name mutation, and load-time random sampling?
8. Should mutation behaviors be preserved where observed, especially `ConfigHandler` mutating model names and `DefaultEarlyStopping` mutating task dictionaries?
9. For model/training outputs, should BDD/target tests lock down stochastic numeric values, or only structural outputs such as schemas, checkpoint/file inventory, task keys, shapes, and deterministic fixture outputs?
10. What numerical tolerance policy should be used for Torch/PyG-to-Julia comparisons around graph convolutions, dropout, batching, BLAS differences, and RNG?

### Data, platform, and dependencies

11. Should Zarr files/config files/checkpoints remain compatible with existing Python-produced artifacts? If yes, which formats are mandatory: YAML configs, Zarr stores, model checkpoints, Optuna outputs, training logs/reports?
12. Is CUDA/GPU/DDP support required in Julia, or can initial migration target CPU/single-process behavior with GPU/DDP deferred?
13. Are there licensing, deployment, packaging, CI, Julia version, or platform constraints for `QuantumGraph.jl`?

### Testing and non-goals

14. Should existing Python characterization tests remain the oracle for migration, should existing tests be ported to Julia, and/or should new BDD specs be written first from requirements?
15. Are any current behaviors explicit non-goals or candidates for intentional redesign, such as Python `!pyobject`, exact `__all__` ordering, warnings, MSE broadcasting behavior, no-pooling/no-latent `GNNModel.get_embeddings`, or binary Torch checkpoint compatibility?

## Defaults proposed for confirmation

- Apply SOLID-style separation of responsibilities.
- Keep performance-critical paths free of orchestration/logging/control-flow logic where feasible.
- Make policy/configuration decisions at high levels, not buried in low-level routines.
- Separate parameterization/configuration from code.
- Separate hot-path model/data computation from training/tuning orchestration.
