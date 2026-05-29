# Implementation Plan

## Goal
Migrate the full QuantumGravPy Python library at `/Users/hmack/Development/QuantumGrav/QuantumGravPy` into a standard Julia library package at `/Users/hmack/Development/QuantumGraph.jl`, preserving behavior boundaries and YAML/Zarr/artifact compatibility while prioritizing functional/structural equivalence over exact numerical equivalence.

## Context and Approved Constraints
- Source: `/Users/hmack/Development/QuantumGrav/QuantumGravPy`
- Target: `/Users/hmack/Development/QuantumGraph.jl`
- Target language: Julia
- Target package shape: standard importable Julia library; no CLI.
- Scope: whole package migration.
- Preferred stack: `Flux.jl`, `GraphNeuralNetworks.jl`, `Zarr.jl`, Julia YAML support; investigate `Enzyme.jl` viability.
- Compatibility priority: YAML config and Zarr stores highest; training/evaluation artifacts, tuning outputs, and checkpoints must be planned explicitly.
- CUDA: important and must be planned/tested.
- Testing: BDD specs first, then executable Behavior.jl step definitions plus a separate native Julia test surface. Leaf nodes require unit tests; integration nodes require integration tests. Characterization tests are the oracle; no direct blind port of Python tests.
- Numeric policy: structural and functional equivalence only for now; exact Torch/PyG-to-Julia numeric tolerances are not required.

## Target Repository Layout

Proposed standard Julia package layout under `/Users/hmack/Development/QuantumGraph.jl`:

```text
QuantumGraph.jl/
├── Project.toml
├── README.md
├── LICENSE
├── src/
│   ├── QuantumGraph.jl
│   ├── Interfaces.jl
│   ├── Utils.jl
│   ├── Config.jl
│   ├── ZarrLoading.jl
│   ├── Datasets.jl
│   ├── Models.jl
│   ├── GNNModel.jl
│   ├── Evaluation.jl
│   ├── EarlyStopping.jl
│   ├── Training.jl
│   ├── CUDADevice.jl
│   └── Tuning.jl
├── specs/
│   ├── package-test-harness.feature
│   ├── interfaces-registry-utils.feature
│   ├── config-object-resolution.feature
│   ├── zarr-loading.feature
│   ├── datasets-graph-samples.feature
│   ├── model-components.feature
│   ├── gnn-model-boundary.feature
│   ├── evaluation-early-stopping.feature
│   ├── training-tuning-workflows.feature
│   ├── cuda-device.feature
│   ├── migration-compatibility.feature
│   └── public-library-surface.feature
├── test/
│   ├── runtests.jl
│   ├── steps/
│   │   └── *_steps.jl
│   ├── fixtures/
│   │   ├── configs/
│   │   └── zarr/
│   ├── unit/
│   │   ├── test_package_test_harness.jl
│   │   ├── test_interfaces_registry_utils.jl
│   │   └── test_zarr_loading.jl
│   └── integration/
│       ├── test_config_object_resolution.jl
│       ├── test_datasets_graph_samples.jl
│       ├── test_model_components.jl
│       ├── test_gnn_model_boundary.jl
│       ├── test_evaluation_early_stopping.jl
│       ├── test_training_tuning_workflows.jl
│       ├── test_cuda_device.jl
│       ├── test_migration_compatibility.jl
│       └── test_public_library_surface.jl
├── docs/
└── artifacts/
```

`artifacts/` remains the discovery artifact directory already in use. Runtime-generated training artifacts should be placed in user-configured output directories, not in discovery artifacts.

## Top-Down Interface Contracts

### Contract A — Public Library Surface
- Provider: `src/QuantumGraph.jl`
- Legacy mapping: `src/QuantumGrav/__init__.py`
- Inputs: user imports `QuantumGraph`; user constructs configs, datasets, models, trainers, evaluators, tuning helpers.
- Outputs: exported Julia types/functions for config loading, `QGDataset` equivalent, `GNNModel`, model blocks, evaluator/validator/tester, early stopping, trainer, tuning, and Zarr loader.
- Failure modes: import-time dependency errors must be clear; no CLI entry points are required.
- Acceptance: BDD confirms library-only import and availability of functional API equivalents; characterization C001/C002 carried forward as behavior boundary, not exact Python `__all__` ordering unless later required.

### Contract B — Config and Object Resolution
- Provider: `src/Config.jl`, `src/Interfaces.jl`, `src/Utils.jl`
- Legacy mapping: `config_utils.py`, `utils.py`, `base.py`
- Inputs: YAML config files with `!sweep`, `!coupled-sweep`, `!range`, `!random_uniform`, `!reference`, `!pyobject`-equivalent tags; nested paths; Julia object registry paths.
- Outputs: parsed config dictionaries/structs, expanded run configs, inclusive ranges, resolved registered Julia objects, serializable config output.
- Failure modes: zero step range, bad object/module/name resolution, coupled-sweep length mismatch, missing required paths.
- Data format contract: existing YAML semantics must be preserved. `!pyobject` cannot literally import Python objects in Julia; it must resolve through an explicit Julia registry/mapping while preserving config behavior boundaries.
- Acceptance: BDD for all YAML tags and config expansion; Julia tests matching characterization C004-C013 and C008 semantic random sampling without requiring exact NumPy values.

### Contract C — Zarr Loading and Dataset Boundary
- Provider: `src/ZarrLoading.jl`, `src/Datasets.jl`
- Legacy mapping: `load_zarr.py`, `dataset_base.py`, `dataset_ondisk.py`
- Inputs: existing Python-produced Zarr stores with arrays such as `adjacency_matrix`, `link_matrix`, `max_pathlen_future`, `max_pathlen_past`, `dimension`, `atomcount`, `num_samples`/`num_causal_sets`; user-supplied reader functions.
- Outputs: nested dictionaries/arrays from Zarr groups; lazy dataset indexing; graph sample objects suitable for `GraphNeuralNetworks.jl`/Flux model input; optional preprocessing output if implemented.
- Failure modes: missing input files, missing reader, out-of-range index, unsupported Zarr layout.
- Data format contract: must read existing Zarr stores and count samples using current precedence: `num_causal_sets`, `num_samples`, one-dimensional datasets, fallback to `adjacency_matrix` shape.
- Acceptance: BDD for recursive Zarr-to-dict, sample counting, lazy read, map-index errors; Julia tests using Python-created fixture Zarr stores; characterization C014-C017.

### Contract D — Model Component Boundary
- Provider: `src/Models.jl`
- Legacy mapping: `models/linear_sequential.py`, `models/skipconnection.py`, `models/gnn_block.py`
- Inputs: config-defined layers, activations, graph convolution constructors, normalizers, skip/dropout settings, tensors/graph data.
- Outputs: Flux/GraphNeuralNetworks-compatible callable model components with config serialization/deserialization.
- Failure modes: unknown registered object, invalid dimensions, unsupported graph convolution configuration.
- Numerical contract: structural shape/functionality equivalence only; exact PyTorch numeric output is not required.
- Acceptance: BDD for layer construction, forward shape, residual projection behavior, config save/load; characterization C018-C020.

### Contract E — Composite GNN Model Boundary
- Provider: `src/GNNModel.jl`
- Legacy mapping: `gnn_model.py`
- Inputs: model config, encoder, pooling or latent path, optional graph feature network, downstream task heads, active task map, graph batch data.
- Outputs: embeddings and downstream output dictionary keyed by task indices or Julia-equivalent stable keys; save/load metadata.
- Failure modes: empty downstream tasks, inconsistent pooling/latent combinations, active-task key mismatches, unknown task key activation/deactivation.
- Acceptance: BDD for active/inactive task filtering, invalid model configs, embedding paths, output keys/shapes; characterization C021-C022.

### Contract F — Evaluation and Early Stopping Boundary
- Provider: `src/Evaluation.jl`, `src/EarlyStopping.jl`
- Legacy mapping: `evaluate.py`, `early_stopping.py`
- Inputs: model, data iterator, criterion functions, task metric definitions, tabular historical evaluation data.
- Outputs: tabular report with `loss_avg`, `loss_min`, `loss_max`, optional per-task metrics; early-stopping decisions and state.
- Failure modes: empty data for early stopping, missing metric columns, invalid mode/task config.
- Data contract: Julia can use `DataFrame`-like tables, preserving column/schema behavior rather than pandas internals.
- Acceptance: BDD for evaluation report schema and early-stopping state transitions/errors; characterization C023-C024.

### Contract G — Training Orchestration Boundary
- Provider: `src/Training.jl`, `src/CUDADevice.jl`
- Legacy mapping: `train.py`, deferred subset of `train_ddp.py`
- Inputs: config, dataset constructors, model constructors, optimizer/scheduler config, evaluator/tester/validator/early stopping, output paths.
- Outputs: trained model state, checkpoints/config copies, validation/test reports, structural training artifacts.
- Failure modes: invalid config schema, missing data paths, unsupported CUDA/backend, checkpoint write failures.
- Artifact contract: file inventory and schemas matter; exact stochastic metrics do not.
- Acceptance: BDD for initialization, dataloader preparation, training loop structural outputs, checkpoint inventory, test/validation report schema; characterization C025 and CUDA requirement.

### Contract H — Tuning Boundary
- Provider: `src/Tuning.jl`
- Legacy mapping: `QGTune/tune.py`
- Inputs: config sweep/range/random/reference/coupled-sweep nodes, study settings, trial object abstraction.
- Outputs: search space suggestions, resolved configs, study metadata, best-config export compatible with YAML config semantics.
- Failure modes: unresolved references, unsupported distribution, coupled sweep mismatch, storage/backend incompatibility.
- Acceptance: BDD for search-space construction, references/coupled sweeps, best config export; existing tests and overview for `test_tune.py`; characterization suggested additional tests.

## Data Flow

1. User imports `QuantumGraph` and provides a YAML config.
2. `Config.jl` parses custom tags, expands sweeps, resolves registry objects, and produces one or more run configs.
3. `Training.jl` consumes a run config and builds datasets, model, optimizer, evaluators, early stopping, and optional tuning hooks.
4. `Datasets.jl` reads Zarr stores via `ZarrLoading.jl`, maps global sample indices to per-file samples, and returns graph sample objects.
5. `GNNModel.jl` composes `Models.jl` components and emits active downstream task outputs.
6. `Evaluation.jl` computes structural metrics/reports from model outputs and data.
7. `EarlyStopping.jl` consumes report tables and emits stop/continue and best-model state.
8. `Training.jl` writes checkpoint/config/report artifacts.
9. `Tuning.jl` drives repeated config generation and best-config export.

Serialization/deserialization ownership:
- YAML parse/emit: `Config.jl`
- Zarr read and fixture compatibility: `ZarrLoading.jl`/`Datasets.jl`
- Model/config serialization metadata: `Models.jl`/`GNNModel.jl`
- Training reports/checkpoints: `Training.jl`/`Evaluation.jl`
- Tuning best configs: `Tuning.jl`

## Tasks

1. **Establish Julia package skeleton and test harness**: Create standard Julia package structure without implementing domain behavior.
   - File: `/Users/hmack/Development/QuantumGraph.jl/Project.toml`
   - File: `/Users/hmack/Development/QuantumGraph.jl/src/QuantumGraph.jl`
   - File: `/Users/hmack/Development/QuantumGraph.jl/test/runtests.jl`
   - Changes: define package metadata, GPL-3.0 license reference, dependency placeholders/compat strategy, module include structure, `specs/`, `test/steps/`, `test/unit/`, and `test/integration/` directories.
   - node_type: `leaf`
   - dependency_interfaces: Julia `Pkg`/`Test` package entry points; Behavior.jl runner paths (`specs/` and `test/steps/`); no in-migration dependency interfaces.
   - Acceptance criteria: Behavior.jl spec `specs/package-test-harness.feature` passes; native unit test `test/unit/test_package_test_harness.jl` verifies package import, test harness wiring, and absence of CLI entry points; `julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'` runs the skeleton suite.
   - Boundary: Independent.

2. **Define common interfaces and registry contracts**: Implement abstract config/serialization interfaces and object registry design before module internals.
   - File: `/Users/hmack/Development/QuantumGraph.jl/src/Interfaces.jl`
   - File: `/Users/hmack/Development/QuantumGraph.jl/src/Utils.jl`
   - Changes: define `from_config`/`to_config`-style conventions, nested path helpers, registry lookup interface for `!pyobject` equivalents, and error conventions.
   - node_type: `leaf`
   - dependency_interfaces: Julia `Dict`/module namespace lookup semantics; registry API consumed by Config, Models, and Training; no in-migration child dependency.
   - Acceptance criteria: Behavior.jl spec `specs/interfaces-registry-utils.feature` and step definitions pass; native unit test `test/unit/test_interfaces_registry_utils.jl` covers characterization C011-C013, nested paths, registry lookup failures, and object-resolution error messages.
   - Boundary: Independent after Task 1.

3. **Implement BDD specification set before Julia behavior tests**: Write feature files that express approved behavior boundaries module-by-module.
   - File: `/Users/hmack/Development/QuantumGraph.jl/specs/*.feature`
   - File: `/Users/hmack/Development/QuantumGraph.jl/test/steps/*_steps.jl`
   - Changes: convert characterization behaviors C001-C028 and requirements into BDD scenarios with explicit non-numeric structural assertions, dependency-interface coverage sections, and executable Behavior.jl step definitions.
   - node_type: `integration`
   - dependency_interfaces: Behavior.jl feature grammar and step definition API; module interface contracts from this plan; characterization IDs C001-C028 as traceability inputs.
   - Acceptance criteria: every module has an approved `.feature` file under `specs/`; every scenario has step definitions under `test/steps/`; each feature states dependency-interface coverage or `none`; BDD review confirms coverage before native unit/integration tests are implemented.
   - Boundary: Sequential after Task 1; can run in parallel with Task 2 if interface names are stable.

4. **Migrate configuration parsing and sweep expansion**: Preserve YAML tag behavior and config expansion semantics.
   - File: `/Users/hmack/Development/QuantumGraph.jl/src/Config.jl`
   - File: `/Users/hmack/Development/QuantumGraph.jl/test/unit/test_config.jl`
   - Changes: support custom tags, inclusive ranges, load-time random sampling semantics, references, sweep/coupled-sweep expansion, model-name suffix behavior, YAML serialization where needed.
   - node_type: `integration`
   - dependency_interfaces: Interfaces/Utils registry and nested-path contract from Task 2; Julia YAML parser tag-extension API; Julia `Random` semantics for non-exact random sampling.
   - Acceptance criteria: Behavior.jl spec `specs/config-object-resolution.feature` passes; native integration test `test/integration/test_config_object_resolution.jl` covers C004-C010 and config portions of C007-C009, including bad-object and length-mismatch errors; tests verify compliance with the registry dependency interface.
   - Boundary: Sequential after Tasks 2 and 3.
   - Open validation: choose Julia YAML parser/tag extension strategy; exact NumPy random values are not required.

5. **Migrate Zarr recursive loading and fixture compatibility**: Provide direct access to Python-created Zarr stores.
   - File: `/Users/hmack/Development/QuantumGraph.jl/src/ZarrLoading.jl`
   - File: `/Users/hmack/Development/QuantumGraph.jl/test/unit/test_zarr_loading.jl`
   - File: `/Users/hmack/Development/QuantumGraph.jl/test/fixtures/zarr/`
   - Changes: implement recursive Zarr group-to-dictionary loading and establish golden structural fixtures from Python characterization data.
   - node_type: `leaf`
   - dependency_interfaces: Zarr.jl store/group/array read API; Python-created Zarr v2 layout compatibility; Julia array conversion semantics.
   - Acceptance criteria: Behavior.jl spec `specs/zarr-loading.feature` passes; native unit test `test/unit/test_zarr_loading.jl` covers C014, empty groups, array leaf loading, unsupported layouts, and Python-created fixture compatibility.
   - Boundary: Independent after Task 1; depends only on package/test harness.
   - Open validation: confirm `Zarr.jl` can read all existing store layouts; if not, plan compatibility shim.

6. **Migrate dataset layer and graph sample conversion**: Implement lazy on-disk dataset behavior over Zarr inputs.
   - File: `/Users/hmack/Development/QuantumGraph.jl/src/Datasets.jl`
   - File: `/Users/hmack/Development/QuantumGraph.jl/test/unit/test_datasets.jl`
   - Changes: implement sample counting precedence, input validation, lazy indexing, map-index behavior, reader function contract, and conversion to graph data suitable for `GraphNeuralNetworks.jl`.
   - node_type: `integration`
   - dependency_interfaces: ZarrLoading recursive dictionary contract from Task 5; GraphNeuralNetworks.jl graph sample representation; MLUtils.jl dataset/indexing expectations if used by training.
   - Acceptance criteria: Behavior.jl spec `specs/datasets-graph-samples.feature` passes; native integration test `test/integration/test_datasets_graph_samples.jl` covers C015-C017 using fixture Zarr stores, graph sample shape/field contracts, map-index errors, and DataLoader-compatible indexing.
   - Boundary: Sequential after Task 5; depends on graph data representation decision.
   - Open validation: define exact Julia graph sample type and batching interface compatible with `GraphNeuralNetworks.jl`.

7. **Migrate model components**: Implement reusable Flux/GraphNeuralNetworks model blocks.
   - File: `/Users/hmack/Development/QuantumGraph.jl/src/Models.jl`
   - File: `/Users/hmack/Development/QuantumGraph.jl/test/unit/test_models.jl`
   - Changes: implement `LinearSequential`, `SkipConnection`, and `GNNBlock` equivalents with config construction/serialization and registered activation/normalizer/conv resolution.
   - node_type: `integration`
   - dependency_interfaces: Interfaces registry contract from Task 2; Flux.jl layer/callable/functor behavior; GraphNeuralNetworks.jl convolution constructor and graph data API; graph sample contract agreed with Task 6.
   - Acceptance criteria: Behavior.jl spec `specs/model-components.feature` passes; native integration test `test/integration/test_model_components.jl` covers C018-C020 for structure, shape, config save/load, skip projection behavior, registered activation/normalizer/conv resolution, and graph convolution interface compliance.
   - Boundary: Sequential after Tasks 2, 3; can proceed in parallel with Task 6 once graph sample representation is agreed.
   - Open validation: map PyG conv/normalizer classes to GraphNeuralNetworks.jl/Flux equivalents.

8. **Migrate composite GNN model**: Compose encoder, pooling/latent path, graph feature path, downstream tasks, and active task filtering.
   - File: `/Users/hmack/Development/QuantumGraph.jl/src/GNNModel.jl`
   - File: `/Users/hmack/Development/QuantumGraph.jl/test/unit/test_gnn_model.jl`
   - Changes: implement config-driven composition, task key activation/deactivation, embedding path behavior, save/load metadata.
   - node_type: `integration`
   - dependency_interfaces: Datasets graph sample/batch contract from Task 6; Models callable/config contract from Task 7; Flux.jl callable model semantics; stable output-key convention for downstream tasks.
   - Acceptance criteria: Behavior.jl spec `specs/gnn-model-boundary.feature` passes; native integration test `test/integration/test_gnn_model_boundary.jl` covers C021-C022 with output keys/shapes, invalid config failures, active/inactive task filtering, embedding paths, and compliance with dataset/model dependency interfaces.
   - Boundary: Sequential after Tasks 6 and 7.
   - Open validation: preserve no-pooling/no-latent behavior boundary as intentional unless BDD review flags ambiguity.

9. **Migrate evaluation and early stopping**: Provide report schemas and stopping logic over Julia tables.
   - File: `/Users/hmack/Development/QuantumGraph.jl/src/Evaluation.jl`
   - File: `/Users/hmack/Development/QuantumGraph.jl/src/EarlyStopping.jl`
   - File: `/Users/hmack/Development/QuantumGraph.jl/test/unit/test_evaluation.jl`
   - File: `/Users/hmack/Development/QuantumGraph.jl/test/unit/test_early_stopping.jl`
   - Changes: implement evaluator/validator/tester schema, criterion call adapters, monitor task callables, early stopping state/grace/patience logic.
   - node_type: `integration`
   - dependency_interfaces: Interfaces/config conventions from Task 2; Tables.jl/DataFrames.jl-compatible report schema; criterion callable interface; model output collection contract when integrated after Task 8.
   - Acceptance criteria: Behavior.jl spec `specs/evaluation-early-stopping.feature` passes; native integration test `test/integration/test_evaluation_early_stopping.jl` covers C023-C024, empty data errors, report table schema, loss aggregation, monitor tasks called once over collected outputs/targets with results stored as returned, and early-stopping state transitions.
   - Boundary: Sequential after Tasks 2 and 3; evaluation integration with models depends on Task 8.
   - Open validation: choose DataFrames.jl or a lighter table abstraction.

10. **Migrate tuning utilities**: Recreate sweep/reference/search-space behavior for Julia workflows.
    - File: `/Users/hmack/Development/QuantumGraph.jl/src/Tuning.jl`
    - File: `/Users/hmack/Development/QuantumGraph.jl/test/unit/test_tuning.jl`
    - Changes: implement search-space conversion, reference/coupled-sweep resolution, study/trial abstraction, best-config YAML export.
    - node_type: `integration`
    - dependency_interfaces: Config sweep/reference/coupled-sweep contract from Task 4; backend-neutral trial suggestion interface; Julia YAML emit semantics for best-config export.
    - Acceptance criteria: Behavior.jl tuning scenarios in `specs/training-tuning-workflows.feature` pass; native integration test `test/integration/test_training_tuning_workflows.jl` covers tuning-specific `test_tune.py` behaviors, characterization gaps, mocked trial suggestions, reference/coupled-sweep compliance, and best-config YAML export.
    - Boundary: Sequential after Tasks 4 and 11; training is implemented first because tuning wraps or drives training workflows.
    - Open validation: identify Julia Optuna-equivalent or create backend-neutral tuning interface; ensure existing Optuna outputs are either read or explicitly migrated.

11. **Migrate single-process training orchestration**: Integrate configs, datasets, models, evaluation, early stopping, and checkpoint/report artifacts.
    - File: `/Users/hmack/Development/QuantumGraph.jl/src/Training.jl`
    - File: `/Users/hmack/Development/QuantumGraph.jl/test/unit/test_training.jl`
    - Changes: implement trainer initialization, dataloader preparation, structural train loop, validation/test calls, optimizer/scheduler construction, checkpoint/report writing.
    - node_type: `integration`
    - dependency_interfaces: Config run-config contract from Task 4; Datasets/MLUtils DataLoader contract from Task 6; generic callable model/output contract from Task 8 or any compatible model provider; Evaluation/EarlyStopping report/state contract from Task 9; Flux/Optimisers optimizer and scheduler APIs; filesystem artifact schema.
    - Acceptance criteria: Behavior.jl training scenarios in `specs/training-tuning-workflows.feature` pass; native integration test `test/integration/test_training_tuning_workflows.jl` covers C025 structurally: file inventory, report schemas, config copies, deterministic fixture execution, optimizer/scheduler construction, and compliance with config/dataset/model/evaluation dependency interfaces. No exact stochastic metrics.
    - Boundary: Sequential after Tasks 4, 6, 8, 9.
    - Open validation: checkpoint format must be Julia-native unless Python Torch compatibility is explicitly feasible.

12. **Plan and validate CUDA path**: Add CUDA device abstraction and compatibility tests separate from CPU functional migration.
    - File: `/Users/hmack/Development/QuantumGraph.jl/src/CUDADevice.jl`
    - File: `/Users/hmack/Development/QuantumGraph.jl/test/unit/test_training.jl`
    - Changes: design device selection and GPU transfer contracts for data/model/training; gate GPU tests by availability.
    - node_type: `integration`
    - dependency_interfaces: CUDA.jl device availability and array transfer API; Flux.jl GPU adaptation; GraphNeuralNetworks.jl graph data GPU support; dataset/model/training contracts from Tasks 6, 8, and 11.
    - Acceptance criteria: Behavior.jl spec `specs/cuda-device.feature` passes; native integration test `test/integration/test_cuda_device.jl` always passes CPU/no-CUDA behavior and runs gated CUDA smoke assertions when CUDA is available.
    - Boundary: Sequential after Tasks 6, 8, 11.
    - Open validation: Flux.jl + GraphNeuralNetworks.jl CUDA support and `Enzyme.jl` compatibility.

13. **Document checkpoint compatibility strategy**: Make artifact compatibility boundaries explicit and test non-accidental behavior.
    - File: `/Users/hmack/Development/QuantumGraph.jl/docs/migration_compatibility.md`
    - File: `/Users/hmack/Development/QuantumGraph.jl/test/unit/test_training.jl`
    - Changes: document Torch checkpoint limitations/conversion strategy and artifact compatibility support matrix.
    - node_type: `integration`
    - dependency_interfaces: Training artifact inventory from Task 11; public export surface from Task 14 when finalized; documentation/test convention for intentionally absent APIs.
    - Acceptance criteria: Behavior.jl spec `specs/migration-compatibility.feature` passes; native integration test `test/integration/test_migration_compatibility.jl` verifies documentation lists supported/deferred artifact types and checkpoint compatibility limits are explicit rather than partially broken.
    - Boundary: Sequential after Task 11.

14. **Assemble public exports and documentation**: Expose stable Julia library surface and package docs.
    - File: `/Users/hmack/Development/QuantumGraph.jl/src/QuantumGraph.jl`
    - File: `/Users/hmack/Development/QuantumGraph.jl/README.md`
    - File: `/Users/hmack/Development/QuantumGraph.jl/docs/`
    - Changes: export/import module surface, document usage analogous to Python examples, update compatibility notes.
    - node_type: `integration`
    - dependency_interfaces: stable public APIs from Config, ZarrLoading, Datasets, Models, GNNModel, Evaluation, EarlyStopping, Training, CUDADevice, and Tuning; Julia module export/import semantics; README/docs examples.
    - Acceptance criteria: Behavior.jl spec `specs/public-library-surface.feature` passes; native integration test `test/integration/test_public_library_surface.jl` verifies exports, sample import/use, docs examples where practical, and absence of CLI entry points.
    - Boundary: Sequential after Tasks 4-11; docs can draft earlier but final export must wait.

15. **Full migration verification**: Run the complete Julia test suite against characterization-derived fixtures and behavior specs.
    - File: `/Users/hmack/Development/QuantumGraph.jl/test/runtests.jl`
    - Changes: ensure all Behavior.jl BDD specs, step definitions, native unit tests, and native integration tests run in CI/local commands; record commands in README or docs.
    - node_type: `integration`
    - dependency_interfaces: package-level `Pkg.test` contract; Behavior.jl runner; native Julia `Test` unit/integration suites; Python pytest oracle in the source repository.
    - Acceptance criteria: `julia --project=. -e 'using Pkg; Pkg.test()'` passes and runs BDD plus native unit/integration suites; Python oracle tests still pass in source with `.venv/bin/pytest test -q`; every major behavior C001-C028 is mapped to a passing Julia test, documented deferral, or accepted non-numeric structural equivalent.
    - Boundary: Sequential after all implementation modules.

## Implementation Dependency Tree

```text
Root: Public QuantumGraph.jl library surface
├── Training/Tuning workflows
│   ├── Training.jl [depends: Config, Datasets, callable model, Evaluation, EarlyStopping]
│   │   └── CUDADevice.jl [depends: Training + shallow device-transfer contract]
│   └── Tuning.jl [depends: Config + Training workflow contract]
├── GNNModel.jl [depends: Datasets graph sample contract + Models]
│   └── Models.jl [depends: Interfaces/registry]
├── Evaluation.jl + EarlyStopping.jl [depends: Interfaces; integrates with GNNModel later]
├── Datasets.jl [depends: ZarrLoading + graph sample contract]
│   └── ZarrLoading.jl [leaf]
├── Config.jl [depends: Interfaces/Utils]
│   └── Interfaces.jl + Utils.jl [leaf]
└── Package skeleton + BDD specs [foundation]
```

Detailed task dependencies:

```text
Task 1  Package skeleton + test harness
├── Task 2  Interfaces, registry, utilities
│   ├── Task 4  Config.jl
│   │   ├── Task 11 Training.jl
│   │   ├── Task 10 Tuning.jl [after Task 11]
│   │   └── Task 14 Public exports + docs
│   ├── Task 7  Models.jl components
│   │   └── Task 8 GNNModel.jl composite model
│   │       ├── Task 11 Training.jl
│   │       ├── Task 12 CUDADevice.jl + CUDA validation
│   │       └── Task 14 Public exports + docs
│   └── Task 9  Evaluation.jl + EarlyStopping.jl
│       ├── Task 11 Training.jl
│       └── Task 14 Public exports + docs
├── Task 3  BDD specification set
│   ├── Task 4  Config.jl
│   ├── Task 7  Models.jl components
│   └── Task 9  Evaluation.jl + EarlyStopping.jl
└── Task 5  ZarrLoading.jl
    └── Task 6 Datasets.jl + graph sample contract
        ├── Task 8  GNNModel.jl composite model
        ├── Task 11 Training.jl
        ├── Task 12 CUDADevice.jl + CUDA validation
        └── Task 14 Public exports + docs

Task 11 Training.jl
├── Task 10 Tuning.jl
├── Task 12 CUDADevice.jl + CUDA validation
├── Task 13 checkpoint compatibility docs
└── Task 14 Public exports + docs

Task 10 Tuning.jl ──> Task 14 Public exports + docs
Task 12 CUDADevice.jl ──> Task 14 Public exports + docs
Task 13 checkpoint compatibility docs ──> Task 14 Public exports + docs
Task 14 Public exports + docs ──> Task 15 Full migration verification
```

Bottom-up implementation order:
1. Package skeleton/test harness.
2. Interfaces/registry/utilities and BDD specs.
3. Config and Zarr loading in parallel.
4. Dataset layer and model components in parallel after graph representation decisions.
5. Composite GNN model.
6. Evaluation and early stopping.
7. Training orchestration.
8. Tuning after training workflow contract is stable.
9. CUDA validation.
10. Deferred compatibility documentation.
11. Public export/docs and full verification.

## Legacy-to-Julia Module Mapping

| Legacy Python module | Target Julia module/file | Notes |
| --- | --- | --- |
| `QuantumGrav.__init__` | `src/QuantumGraph.jl` | Library public surface; no CLI. |
| `base.py` | `src/Interfaces.jl` | `Configurable`-style config protocol. |
| `utils.py` | `src/Utils.jl` | Nested paths and registry object lookup. |
| `config_utils.py` | `src/Config.jl` | YAML tags, ranges, sweeps, random sampling, serialization. |
| `load_zarr.py` | `src/ZarrLoading.jl` | Recursive Zarr to nested data. |
| `dataset_base.py`, `dataset_ondisk.py` | `src/Datasets.jl` | Lazy Zarr datasets and graph sample conversion. |
| `models/linear_sequential.py` | `src/Models.jl` | Flux dense/activation stack. |
| `models/skipconnection.py` | `src/Models.jl` | Residual projection behavior. |
| `models/gnn_block.py` | `src/Models.jl` | GraphNeuralNetworks/Flux block. |
| `gnn_model.py` | `src/GNNModel.jl` | Composite model and active tasks. |
| `evaluate.py` | `src/Evaluation.jl` | Report schemas and metric loop. |
| `early_stopping.py` | `src/EarlyStopping.jl` | State and decision logic. |
| `train.py` | `src/Training.jl` | Single-process orchestration. |
| `QGTune/tune.py` | `src/Tuning.jl` | Search-space/reference/best-config behavior. |

## Julia Testing Approach and Commands

Primary Julia command:

```sh
cd /Users/hmack/Development/QuantumGraph.jl
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Recommended module-level commands during implementation:

```sh
julia --project=. test/runtests.jl
julia --project=. -e 'using Pkg; Pkg.test(test_args=["config"])'
```

Oracle command in source repository:

```sh
cd /Users/hmack/Development/QuantumGrav/QuantumGravPy
source .venv/bin/activate
pytest test/test_characterization_current_behavior.py -q
pytest test -q
```

BDD workflow:
- Write/review `.feature` files under `specs/` first.
- Convert approved scenarios to executable Behavior.jl step definitions under `test/steps/`.
- Add the separate native Julia test surface required by each node type: unit tests under `test/unit/` for leaf nodes and integration tests under `test/integration/` for integration nodes.
- Native tests must verify each module's `dependency_interfaces` explicitly; do not rely on BDD scenarios alone for interface compatibility.
- Structural tests should assert schemas, keys, shapes, artifact inventory, and deterministic fixture behavior.
- Do not require exact stochastic loss/metric values or Torch/PyG numeric equivalence.

## Files to Modify
- `/Users/hmack/Development/QuantumGraph.jl/Project.toml` - Julia package metadata, dependencies, test targets, GPL-3.0 metadata.
- `/Users/hmack/Development/QuantumGraph.jl/README.md` - library usage, migration compatibility notes, test commands.
- `/Users/hmack/Development/QuantumGraph.jl/src/QuantumGraph.jl` - root module and public exports.
- `/Users/hmack/Development/QuantumGraph.jl/test/runtests.jl` - test harness that runs Behavior.jl specs, step definitions, native unit tests, and native integration tests.

## New Files
- `/Users/hmack/Development/QuantumGraph.jl/src/Interfaces.jl` - config/serialization and registry interface contracts.
- `/Users/hmack/Development/QuantumGraph.jl/src/Utils.jl` - nested path helpers and object resolution support.
- `/Users/hmack/Development/QuantumGraph.jl/src/Config.jl` - YAML tags, ranges, sweeps, config serialization.
- `/Users/hmack/Development/QuantumGraph.jl/src/ZarrLoading.jl` - recursive Zarr loading.
- `/Users/hmack/Development/QuantumGraph.jl/src/Datasets.jl` - Zarr-backed dataset and graph sample conversion.
- `/Users/hmack/Development/QuantumGraph.jl/src/Models.jl` - `LinearSequential`, `SkipConnection`, `GNNBlock` equivalents.
- `/Users/hmack/Development/QuantumGraph.jl/src/GNNModel.jl` - composite configurable GNN model.
- `/Users/hmack/Development/QuantumGraph.jl/src/Evaluation.jl` - evaluator/validator/tester behavior.
- `/Users/hmack/Development/QuantumGraph.jl/src/EarlyStopping.jl` - early stopping behavior.
- `/Users/hmack/Development/QuantumGraph.jl/src/Training.jl` - single-process trainer.
- `/Users/hmack/Development/QuantumGraph.jl/src/CUDADevice.jl` - CUDA/device selection contract.
- `/Users/hmack/Development/QuantumGraph.jl/src/Tuning.jl` - sweep/reference/search-space and best-config export.
- `/Users/hmack/Development/QuantumGraph.jl/specs/*.feature` - BDD specs for each module.
- `/Users/hmack/Development/QuantumGraph.jl/test/steps/*_steps.jl` - Behavior.jl executable step definitions for approved BDD specs.
- `/Users/hmack/Development/QuantumGraph.jl/test/unit/*.jl` - native Julia unit tests for leaf nodes.
- `/Users/hmack/Development/QuantumGraph.jl/test/integration/*.jl` - native Julia integration tests for integration nodes.
- `/Users/hmack/Development/QuantumGraph.jl/test/fixtures/configs/` - YAML config fixtures.
- `/Users/hmack/Development/QuantumGraph.jl/test/fixtures/zarr/` - Python-created Zarr fixtures.
- `/Users/hmack/Development/QuantumGraph.jl/docs/migration_compatibility.md` - artifact compatibility and checkpoint strategy.

## Dependencies
- Task 1 blocks all code implementation.
- Task 2 blocks config, model component registry, and object resolution behavior.
- Task 3 should precede target Julia tests for each module.
- Task 4 blocks tuning and training.
- Task 5 blocks datasets.
- Task 6 and Task 7 block composite GNN model.
- Task 8 blocks integrated evaluation/training with real model outputs.
- Task 9 blocks training integration.
- Task 10 depends on config but can proceed before training integration.
- Task 11 depends on config, datasets, GNN model, evaluation, and early stopping.
- Task 12 depends on training/model/dataset device contracts.
- Task 13 depends on training artifact decisions.
- Task 14 depends on stable module APIs.
- Task 15 depends on all implementation tasks.

## Risks
- `Enzyme.jl` may not be viable with the chosen `Flux.jl`/`GraphNeuralNetworks.jl` path; do not make it a hard dependency until validated.
- `Zarr.jl` may not read every Python-created Zarr layout exactly; fixture-based compatibility tests are mandatory.
- `!pyobject` has no direct Julia equivalent; requires a registry/mapping and may need migration documentation for config authors.
- PyTorch/PyG graph batching and Julia graph batching may differ; define graph sample and batch contracts before model/training implementation.
- Torch binary checkpoints are likely not directly usable from Julia; plan native Julia checkpoints plus explicit compatibility/conversion strategy.
- CUDA support is required but environment-sensitive; keep GPU tests gated and add a clear CUDA validation step.
- Mutation semantics in config and early stopping may differ in Julia; BDD should lock user-visible behavior rather than implementation aliasing.
- Exact numeric comparisons are intentionally out of scope; reviewers must not treat non-identical metrics as failures unless structural behavior changes.
