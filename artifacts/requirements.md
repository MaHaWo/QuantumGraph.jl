# Requirements: QuantumGraph.jl

## Status

APPROVED FOR PLANNING: the human approved these requirements as complete enough for migration planning.

## Accepted discovery inputs

- source_repo: `/Users/hmack/Development/QuantumGrav/QuantumGravPy`
- output_repo: `/Users/hmack/Development/QuantumGraph.jl`
- artifact_dir: `/Users/hmack/Development/QuantumGraph.jl/artifacts`
- source_language: Python
- target_language: Julia

## Evidence reviewed

- Characterization report: `/Users/hmack/Development/QuantumGraph.jl/artifacts/characterization-report.md`
- Repository overview HTML: `/Users/hmack/Development/QuantumGrav/QuantumGravPy/scout/overview.html`
- Overview summary: `/Users/hmack/Development/QuantumGraph.jl/artifacts/overview-summary.md`
- Characterization tests: `/Users/hmack/Development/QuantumGrav/QuantumGravPy/test/test_characterization_current_behavior.py`

## Scope

The migration scope is the whole QuantumGravPy package, not a narrower slice. The Julia implementation should preserve the major separation of concerns present in QuantumGravPy, especially separation between:

- data handling and Zarr-backed datasets,
- model definitions and model composition,
- training/evaluation orchestration,
- configuration handling,
- tuning utilities,
- testing/characterization oracle behavior.

QuantumGraph.jl is intended to be a Julia migration of the full Python package behavior boundaries, not a partial numerical-kernel port.

## User base

The intended users are domain scientists, students, researchers, and downstream experiment authors who use the package as a library for their own experiments.

## Output repository

The target repository is:

`/Users/hmack/Development/QuantumGraph.jl`

It should be a standard Julia package that can be imported into other Julia packages and experiment code. It should behave as a library, analogous to the Python package usage model.

The package should remain library-only. No CLI or console-script entry points should be added unless explicitly approved later.

The package is backed by a git repository, working with the github workflow: one main branch, many feature branches that get merged when the feature is implemented and tested.

## Target language and preferred stack

Target language: Julia.

Preferred Julia stack, subject to validation during planning and implementation:

- Graph neural networks: `GraphNeuralNetworks.jl`
- General ML/model layer: `Flux.jl`
- Data storage compatibility: `Zarr.jl`
- YAML/config parsing: `Yaml.jl` or suitable Julia YAML support
- Differentiation: investigate whether `Enzyme.jl` can be used as the differential programming backend; this is desired if viable, but requires technical validation.

Planning should flag feasibility risks for the Julia graph/ML/differentiation stack instead of silently assuming exact equivalence with PyTorch/PyG.

## Architectural requirements

The Julia package should preserve the high-level structure and intentional design boundaries of QuantumGravPy while allowing idiomatic Julia APIs where behavior boundaries are maintained.

Approved architectural defaults:

- Apply SOLID-style separation of responsibilities.
- Keep performance-critical paths free of orchestration, logging, and control-flow logic where feasible.
- Make policy/configuration decisions at high levels, not buried in low-level routines.
- Separate parameterization/configuration from code.
- Separate hot-path model/data computation from training/tuning orchestration.

The current QuantumGravPy design decisions are considered intentional and should not be treated as redesign candidates by default.

## Behavioral requirements

### Compatibility policy

The Julia migration must preserve the observed behavior boundaries of QuantumGravPy. The internal implementation and exact Julia public API may change where appropriate, but externally meaningful behavior, file/config/data compatibility, and library usage boundaries must be carried forward.

Python public API semantics do not need to be mirrored exactly where an idiomatic Julia API is more appropriate, but behavior at module boundaries, configuration/data boundaries, and experiment-workflow boundaries should remain compatible.

### Entry points and delivery

Current Python package behavior:

- importable package named `QuantumGrav`, exposed via `src/QuantumGrav/__init__.py`,
- no declared console scripts,
- main runtime through Python imports and documented script-style examples,
- primary path through `Trainer(config)`, `QGDataset`, `GNNModel`, evaluators, early stopping, and tuning helpers.

Target Julia behavior:

- standard importable Julia package,
- library-only delivery,
- no CLI unless approved later,
- package should support downstream experiment packages/scripts.

### Configuration behavior

The Julia migration should preserve current YAML configuration semantics, including:

- `!sweep`,
- `!coupled-sweep`,
- `!range`,
- `!random_uniform`,
- `!reference`,
- `!pyobject`-equivalent behavior through a Julia-appropriate registry/mapping or import mechanism,
- sweep expansion behavior,
- run-name mutation behavior where it is part of current config semantics,
- load-time random sampling behavior for `!random_uniform`.

Mutation behaviors observed in QuantumGravPy, such as `ConfigHandler` mutating model names and `DefaultEarlyStopping` mutating task dictionaries, may be adjusted as necessary in Julia, but the behavior boundary experienced by users/config workflows should remain compatible.

### Dataset and data-format behavior

The Julia migration must preserve compatibility with existing artifacts, especially:

- YAML config files,
- Zarr stores,
- training/evaluation reports/logs where applicable,
- Optuna/tuning outputs where applicable,
- model checkpoint workflows to the extent technically feasible.

Config and Zarr compatibility are especially important and should receive priority in planning and test coverage.

### Model/training behavior

The migration should preserve functional behavior and structural outputs rather than exact numerical equivalence.

BDD and target-language tests should focus on:

- schemas,
- output shapes,
- task keys,
- active/inactive task behavior,
- file/checkpoint inventory,
- deterministic fixture behavior,
- config expansion behavior,
- Zarr loading and dataset indexing behavior,
- training/evaluation workflow behavior.

Exact stochastic numerical outputs and Torch/PyG-to-Julia quantitative comparisons are not a priority at this stage.

### Numerical tolerance policy

Numerical equivalence has no priority at this stage. Quantitative comparisons between PyTorch/PyG and Julia implementations can be ignored for now unless needed for deterministic structural or functionality tests.

Functionality and behavior boundaries are more important than exact numeric tolerances.

### GPU and distributed behavior

CUDA/GPU support is important for the Julia migration.

DDP/distributed training can be left out or deferred initially. The plan should make this deferral explicit and avoid blocking the initial migration on DDP equivalence.

## Dependencies and constraints

### Licensing

Retain the GPL-3.0 license used by the QuantumGravPy parent repository.

### Platform/package constraints

No additional deployment, packaging, CI, Julia version, or platform constraints were specified beyond creating a standard Julia package in `/Users/hmack/Development/QuantumGraph.jl` and preserving GPL-3.0 licensing.

### Dependency risks to carry into planning

Planning should explicitly consider these risks from characterization and overview:

- PyTorch/PyG concepts must be mapped to Julia equivalents, likely `Flux.jl` and `GraphNeuralNetworks.jl`.
- `!pyobject` dynamic imports require a Julia-compatible mechanism.
- Python Zarr and Julia Zarr layout compatibility must be verified with fixtures.
- Criterion signatures differ between training and evaluation in the Python implementation.
- Current config and early-stopping implementations mutate dictionaries; Julia design may differ while preserving behavior boundaries.
- CUDA support is required, while DDP is deferred.
- Binary Torch checkpoints may not be directly portable and need explicit plan treatment.

## Testing strategy

The existing Python characterization tests are the behavioral oracle for migration.

Testing strategy:

1. Use characterization tests and characterization report as the source oracle.
2. Write new BDD specifications from requirements and observed behavior before implementing target tests.
3. Build Julia tests from approved BDD specs.
4. Do not simply port Python tests directly to Julia without BDD review.
5. Focus tests on functional/structural behavior and artifact compatibility rather than exact stochastic numeric equivalence.

The characterization tests currently added in the source repository are:

`/Users/hmack/Development/QuantumGrav/QuantumGravPy/test/test_characterization_current_behavior.py`

## Non-goals and deferred work

### Non-goals

No current behaviors were explicitly identified as redesign candidates or non-goals. QuantumGravPy design decisions should be treated as intentional unless later explicitly revised by the human.

The following are therefore not redesign targets by default:

- Python `!pyobject` behavior,
- exact public export behavior where it reflects behavior boundaries,
- warnings and edge-case behavior if they affect users,
- MSE broadcasting behavior if behaviorally relevant,
- no-pooling/no-latent `GNNModel.get_embeddings` behavior,
- checkpoint workflows.

### Deferred or lower-priority work

- Exact numerical equivalence between Python/PyTorch/PyG and Julia is deferred/not prioritized.
- DDP/distributed training is deferred initially.
- Exact binary Torch checkpoint compatibility may require feasibility review and should be explicitly planned rather than assumed.

## Characterization findings carried forward

| Behavior | Decision | Notes |
| --- | --- | --- |
| QuantumGravPy is an importable Python library with no console scripts. | Preserve boundary | QuantumGraph.jl should remain library-only. |
| Whole package includes config, datasets, models, training, evaluation, DDP, tuning. | Preserve scope | DDP may be deferred; rest is in migration scope. |
| YAML custom tags drive config behavior. | Preserve | Config compatibility is a high priority. |
| `!random_uniform` samples at load time. | Preserve | Exact NumPy numeric equivalence is not required, but semantic behavior should remain. |
| Config sweeps mutate run/model names. | Preserve behavior boundary | Internal mutation may differ if user-visible behavior is compatible. |
| Zarr-backed datasets are central. | Preserve | Existing Zarr stores should remain compatible. |
| PyTorch/PyG model/training stack is central. | Migrate to Julia equivalents | Prefer Flux.jl + GraphNeuralNetworks.jl; investigate Enzyme.jl. |
| Training outputs are stochastic and not golden-locked. | Structural testing only | No exact stochastic numeric equivalence required. |
| CUDA is documented/relevant. | Preserve support | CUDA support is important. |
| DDP exists in Python implementation. | Defer initially | Do not block initial migration on DDP. |
| Characterization tests were added. | Use as oracle | BDD specs should be written before Julia tests. |

## Open questions requiring planning investigation, not immediate human input

- Is `Enzyme.jl` viable as the differentiation backend with the chosen Flux.jl/GraphNeuralNetworks.jl stack?
- What is the best Julia mechanism for `!pyobject`-equivalent dynamic object resolution?
- Which Python-produced artifacts can be directly read by Julia libraries, especially Zarr stores and tuning outputs?
- How should Torch binary checkpoint compatibility be represented: direct conversion, structural replacement, or documented incompatibility with migration tooling?
- What CUDA path is feasible for the chosen Julia graph/ML stack?

## Approval record

The human approved this requirements document as complete enough for planning.
