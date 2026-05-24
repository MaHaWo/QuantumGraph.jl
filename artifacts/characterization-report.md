# Characterization Report: QuantumGravPy current observable behavior

## Scope

Characterized the current observable behavior of the Python source repository at `/Users/hmack/Development/QuantumGrav/QuantumGravPy` as a migration oracle for a future Julia implementation. The focus was public Python package entry points, configuration/YAML behavior, Zarr data loading, dataset access, model composition primitives, early stopping, evaluation/training surfaces, and currently observable test behavior.

Executable characterization tests were added under the existing pytest test structure.

## Out of scope

- No migrated Julia implementation was written.
- No production code was modified.
- Full numerical equivalence of stochastic training loops was not locked down beyond existing tests, because current training tests use random data and are not golden-output tests.
- GPU/CUDA, multi-node DDP, external Optuna storage beyond existing tests, and real research datasets were not deeply characterized.
- Documentation examples in `docs/training_a_model.md` and `docs/hparam_tuning.md` were inspected only for entry-point context, not executed end-to-end.

## Sources inspected

- `pyproject.toml` — package metadata, Python requirement, pytest configuration, dependencies, and absence of declared console scripts.
- `src/QuantumGrav/__init__.py` — public package exports.
- `src/QuantumGrav/config_utils.py` — custom YAML tags, sweep expansion, random range behavior, config serialization.
- `src/QuantumGrav/utils.py` — nested path helpers and dynamic import behavior.
- `src/QuantumGrav/load_zarr.py` — Zarr-to-dict public API.
- `src/QuantumGrav/dataset_base.py` — common dataset sample counting, metadata, preprocessing, and chunk behavior.
- `src/QuantumGrav/dataset_ondisk.py` — `QGDataset` lazy/on-disk behavior and index mapping.
- `src/QuantumGrav/models/linear_sequential.py` — linear sequential module behavior and config/save/load surface.
- `src/QuantumGrav/models/skipconnection.py` — residual/projection behavior and config surface.
- `src/QuantumGrav/models/gnn_block.py` — graph block composition, config, save/load surface.
- `src/QuantumGrav/gnn_model.py` — model composition, pooling/latent paths, active task behavior.
- `src/QuantumGrav/evaluate.py` — evaluator/tester/validator output schema and report behavior.
- `src/QuantumGrav/early_stopping.py` — early stopping state mutation and error behavior.
- `src/QuantumGrav/train.py` — trainer configuration schema and training/evaluation public surface.
- `src/QuantumGrav/train_ddp.py` — DDP initialization/cleanup and trainer surface.
- `src/QuantumGrav/QGTune/tune.py` — Optuna search-space helper behavior.
- `docs/api.md`, `docs/getting_started.md` — documented public modules and usage model.
- Existing `test/*.py` — current expected behavior across config, data, models, training, DDP, tuning, and utilities.
- Added `test/test_characterization_current_behavior.py` — focused executable characterization tests.

## Current behavior summary

QuantumGravPy is a Python package named `QuantumGrav` requiring Python `>=3.12`; the observed environment used Python 3.14.4. It exposes model, dataset, training, evaluation, config, utility, and Zarr-loading APIs through `QuantumGrav.__all__`. There are no configured console-script entry points in `pyproject.toml`; user-facing operation is currently through Python imports, pytest, and documented script-style examples.

Configuration is YAML-based with custom tags including `!sweep`, `!coupled-sweep`, `!range`, `!random_uniform`, `!reference`, and `!pyobject`. Sweep expansion mutates run names by appending `_<name_addition>_<index>`. Random-uniform YAML loading samples at load time using NumPy global RNG.

Datasets use Zarr stores and PyTorch Geometric conventions. `QGDataset` can lazily read Zarr data without creating a processed directory unless `pre_transform` or `pre_filter` is provided. Sample counts are read from `num_causal_sets` or `num_samples` when present, otherwise inferred from one-dimensional datasets or, as a fallback, from `adjacency_matrix` shape.

Models are PyTorch modules composed from configurable classes/callables. `GNNModel` requires at least one downstream task and either a pooling or latent path for normal embedding output; active downstream tasks are keyed by integer indices. Early stopping mutates the supplied tasks dictionary by adding runtime state.

## Behavior inventory

| ID | Observed Behavior | Evidence | Observation Status | Confidence | Notes |
| --- | --- | --- | --- | --- | --- |
| C001 | Public package exports are exactly `models`, `QGDataset`, path/import helpers, `GNNModel`, trainers, DDP helpers, evaluators, `DefaultEarlyStopping`, config helpers, and Zarr loaders. | `src/QuantumGrav/__init__.py`; `test/test_characterization_current_behavior.py::test_characterizes_current_public_package_exports` | observed | high | Export order is also locked by the characterization test. |
| C002 | No package console scripts are declared. | `pyproject.toml`; grep command found no `[project.scripts]` or `console_scripts`. | observed | high | CLI behavior is therefore mostly absent from packaging metadata. |
| C003 | Package metadata declares name `QuantumGrav`, version `0.1.0`, Python `>=3.12`, and dependencies including pandas, matplotlib, scikit-learn, joblib, PyYAML, zarr, and optuna. | `pyproject.toml` | observed | high | Torch dependencies are not listed in `pyproject.toml` despite package imports. |
| C004 | `range_inclusive` includes the stop value when exactly aligned, preserves integer dtype for all-integer inputs, supports negative steps, and omits stop when not aligned. | `src/QuantumGrav/config_utils.py`; added range characterization tests; existing `test/test_config_utils.py` | observed | high | Numerical tolerance in tests: `rtol=1e-12`, `atol=1e-12`. |
| C005 | `range_inclusive(..., step=0)` raises `ValueError: step must not be zero`. | `src/QuantumGrav/config_utils.py`; added characterization test | observed | high | Error message locked. |
| C006 | YAML `!pyobject` imports a fully qualified object; bad modules raise `ValueError("Importing module ... unsuccessful")`; bad names raise `ValueError("Could not load name ... from ...")`. | `src/QuantumGrav/config_utils.py`; existing and added tests | observed | high | Requires full module path, not local aliases. |
| C007 | YAML `!sweep`, `!coupled-sweep`, `!range`, `!random_uniform`, and `!reference` deserialize into dictionaries with `type`, `values`, `target`, and/or `tune_values` fields. | `src/QuantumGrav/config_utils.py`; `test/test_config_utils.py`; added characterization test | observed | high | `!reference` stores paths but does not dereference during YAML load. |
| C008 | YAML `!random_uniform` samples immediately with NumPy global RNG. With seed 12345 and log sampling from 0.001 to 0.01 size 3, observed values are `[0.008503859825373387, 0.002071932270266695, 0.0015272805171135612]`. | Added characterization test | observed | medium | Environment/NumPy-version dependent risk; NumPy 2.4.6 observed. |
| C009 | `ConfigHandler` expands sweeps into cartesian products, zips coupled sweeps with their targets, and mutates each run config model name as `<name>_<name_addition>_<index>`. | `src/QuantumGrav/config_utils.py`; existing `test/test_config_utils.py`; added characterization test | observed | high | Requires `cfg["model"]["name"]`; missing model/name was not separately characterized. |
| C010 | Coupled-sweep length mismatch raises `ValueError` containing `Incompatible lengths for coupled-sweep`. | Existing `test/test_config_utils.py` | observed | high | Error source is config expansion. |
| C011 | `assign_at_path` mutates nested dict/list paths and propagates `KeyError` for missing intermediate keys. | `src/QuantumGrav/utils.py`; existing and added tests | observed | high | Supports integer list indices as path elements. |
| C012 | `get_at_path` traverses all path elements except the last with direct indexing, then calls `.get(last, default)` on the final container. Missing intermediate keys raise `KeyError`; missing final keys return default. | `src/QuantumGrav/utils.py`; existing and added tests | observed | high | Final container must provide `.get`. |
| C013 | `import_and_get` returns imported objects and wraps missing module/name failures as `KeyError` with specific messages. | `src/QuantumGrav/utils.py`; existing and added tests | observed | high | Used by model load/config flows. |
| C014 | `zarr_file_to_dict` recursively returns nested dictionaries whose leaves are full NumPy array slices. Empty Zarr groups return `{}`. | `src/QuantumGrav/load_zarr.py`; `test/test_load_zarr.py`; added test | observed | high | Array values are loaded eagerly into memory. |
| C015 | `QGDataset` without `pre_transform`/`pre_filter` lazily reads raw Zarr samples, reports length from sample metadata, exposes `.raw_file_names`, and does not create/use processed `.pt` files. | `src/QuantumGrav/dataset_ondisk.py`; `src/QuantumGrav/dataset_base.py`; existing dataset tests; added characterization test | observed | high | Characterization test uses explicit `num_samples`. |
| C016 | `QGDataset.map_index` maps global indices across input files and raises `RuntimeError` if out of range. | `src/QuantumGrav/dataset_ondisk.py`; existing and added tests | observed | high | Error message includes index, per-file sizes, and total size. |
| C017 | `QGDataset` constructor raises `ValueError` when no reader is supplied and `FileNotFoundError` for missing input files. | `src/QuantumGrav/dataset_base.py`; existing and added tests | observed | high | Constructor validates all input paths. |
| C018 | `LinearSequential` constructs alternating PyG `Linear` and activation layers, returns a tensor from `forward`, and serializes activations as module/type-name strings. | `src/QuantumGrav/models/linear_sequential.py`; existing and added tests | observed | high | Docstring says list output, but observed return is tensor from `torch.nn.Sequential`. |
| C019 | `SkipConnection` adds either identity-projected old features or a learned bias-free linear projection to new features. | `src/QuantumGrav/models/skipconnection.py`; existing and added tests | observed | high | Projection path initialized stochastically unless weights are controlled. |
| C020 | `GNNBlock` applies conv, normalizer, activation, optional skip, then dropout; config save/load stores class paths as strings and reimports them on load. | `src/QuantumGrav/models/gnn_block.py`; existing `test/test_gnn_block.py` | observed | high | Dropout means train-mode numerical outputs can be stochastic. |
| C021 | `GNNModel` rejects empty downstream tasks, inconsistent pooling/latent combinations, and active-task key mismatches. Active tasks are integer-index keyed and `set_task_active/inactive` raise `KeyError` for unknown keys. | `src/QuantumGrav/gnn_model.py`; existing `test/test_gnn_model.py`; added characterization test | observed | high | Added test locks latent forward with deterministic local encoder. |
| C022 | `GNNModel.get_embeddings` returns output for pooling or latent paths; no-pooling/no-latent path is covered by existing tests and is a decision point. | `src/QuantumGrav/gnn_model.py`; `test/test_gnn_model.py::test_get_embeddings_no_pooling_path` | observed | medium | Needs migration decision on whether this is intended. |
| C023 | `Evaluator.evaluate` sets `model.eval()`, iterates a PyG dataloader with tqdm, calls either `apply_model` or `model(data.x, data.edge_index, data.batch)`, records `loss_avg`, `loss_min`, `loss_max`, and optional task metrics into a pandas DataFrame. | `src/QuantumGrav/evaluate.py`; existing `test/test_evaluate.py`; full pytest run | observed | high | Progress bar output is current observable command-line behavior. |
| C024 | `DefaultEarlyStopping` mutates the supplied tasks dict by adding `current_grace_period`, `best_score`, and `found_better`; empty data raises `ValueError("Cannot compute early stopping on empty data")`. | `src/QuantumGrav/early_stopping.py`; existing and added tests | observed | high | Mutation of input config may surprise callers. |
| C025 | `Trainer` and `TrainerDDP` are configuration-driven and current tests exercise initialization, dataloader preparation, checkpoints, training loops, scheduler, test/validation, and DDP CPU path. | `src/QuantumGrav/train.py`; `src/QuantumGrav/train_ddp.py`; `test/test_trainer.py`; `test/test_trainer_ddp.py`; full pytest run | observed | medium | Numeric training outputs are stochastic and not golden-locked. |
| C026 | DDP initialization sets `MASTER_ADDR`/`MASTER_PORT` and calls `torch.distributed.init_process_group`; cleanup destroys the group and removes those env vars. | `src/QuantumGrav/train_ddp.py`; existing `test/test_trainer_ddp.py` | observed | medium | Backend/environment-specific; docs call DDP untested. |
| C027 | Existing full test suite passes in the observed environment: 155 passed, 871 warnings. | Command `.venv/bin/pytest test -q` | observed | high | Warnings include PyG deprecations, Python 3.14 TorchScript deprecation, dataset pre_transform/pre_filter warnings, MPS pin_memory warning, and MSE broadcasting warnings. |
| C028 | Documentation says current QG supports CUDA 12.8, but observed tests passed on macOS arm64 CPU/MPS-capable environment. | `docs/getting_started.md`; environment command; full pytest run | observed | medium | Documentation/runtime compatibility may need migration decision. |

## Current-behavior examples

```gherkin
Feature: Current observed behavior of QuantumGrav configuration loading

  Scenario: Current YAML sweep behavior
    Given a YAML config with !sweep values [1, 2] and a coupled width [16, 32]
    When the config is loaded with QuantumGrav.get_loader and expanded by ConfigHandler
    Then the current run configs are named legacy_char_0 and legacy_char_1
    And the current layer-width pairs are (1, 16) and (2, 32)
```

```gherkin
Feature: Current observed behavior of lazy Zarr dataset access

  Scenario: Current QGDataset lazy read behavior
    Given a Zarr store with num_samples equal to 2
    And no pre_transform or pre_filter is supplied
    When QGDataset is indexed at 1
    Then the current reader is called on the raw Zarr group
    And no processed data file is required
```

```gherkin
Feature: Current observed behavior of GNNModel active tasks

  Scenario: Current active-task filtering
    Given a GNNModel with two downstream tasks and active_tasks {0: true, 1: false}
    When the model is called on deterministic inputs
    Then the current output dictionary contains only key 0
```

## Suggested characterization tests

Implemented tests cover the highest-priority externally observable behaviors. Additional useful characterization tests for later work:

- Golden-output tests for `Trainer.train` using fixed seeds, deterministic fixtures, and exact checkpoint/file inventory.
- Focused tests for `Trainer.from_config` and config schema failure messages across all required fields.
- Characterization of `QGTune.tune` end-to-end study creation and saved best-config YAML format.
- File-format golden tests for model `.save()` outputs where stable across Torch versions, or structural tests if binary golden files are too brittle.
- CUDA/DDP behavior in the intended deployment environment, if the migration must match GPU-specific behavior.
- Documentation example execution tests for `docs/training_a_model.md` and `docs/hparam_tuning.md` if those examples are migration requirements.

## Tests written

- `test/test_characterization_current_behavior.py` — added 18 characterization tests covering public exports, config/YAML tags, numerical range behavior, random sampling with seed, helper errors, Zarr format loading, lazy dataset behavior and errors, `LinearSequential`, `SkipConnection`, `DefaultEarlyStopping`, and `GNNModel` active-task behavior.

No golden files were added; all added tests are assertion-based with controlled temporary fixtures.

## Commands run

| Command | Exit code | Result |
| --- | ---: | --- |
| `pwd; find . -maxdepth 3 -type f ... head -200` | 0 | Listed repository structure and identified docs, source, tests, configs. |
| `find src/QuantumGrav -maxdepth 3 -type f ...; find test -maxdepth 1 -type f ...` | 0 | Enumerated source and test files, including model submodules and QGTune. |
| `python - <<'PY' ...` | 127 | System `python` command not found. Non-blocking; switched to `.venv/bin/python`. |
| `.venv/bin/python - <<'PY' ...` | 0 | Environment snapshot: Python 3.14.4, macOS-26.5 arm64, torch 2.12.0, torch_geometric 2.7.0, zarr 3.2.1, numpy 2.4.6, pandas 3.0.3, PyYAML 6.0.3; imported `QuantumGrav`. |
| `grep -R "[project.scripts]..." pyproject.toml src test docs` | 0 | Found no package script entry points; only documentation snippets contain `if __name__ == "__main__"`. |
| `.venv/bin/pytest test/test_characterization_current_behavior.py -q` | 1 | First run: 17 passed, 1 failed due to incorrect expected NumPy random sample values in the new characterization test. Test expectation was corrected to observed behavior. |
| `.venv/bin/pytest test/test_characterization_current_behavior.py -q` | 0 | Added characterization tests passed: 18 passed, 3 warnings. |
| `.venv/bin/pytest test -q` | 0 | Full test suite passed: 155 passed, 871 warnings in 5.78s. |

## Decision candidates for later human/BDD phase

- Should the Julia migration preserve the exact `QuantumGrav.__all__` export list and ordering, or only equivalent functional APIs?
- Should Python packaging behavior with no console scripts be mirrored, or should Julia add explicit CLI entry points?
- Should `!random_uniform` retain load-time NumPy-global-RNG sampling semantics, or should configuration loading become deterministic unless explicitly sampled?
- Should `ConfigHandler` continue mutating `model.name` in-place with appended run indices?
- Should `DefaultEarlyStopping` continue mutating the caller-provided task dictionaries?
- Should `LinearSequential.forward` returning a tensor be treated as the behavior to match despite a docstring mentioning `list[torch.Tensor]`?
- Should no-pooling/no-latent `GNNModel.get_embeddings` behavior be preserved, clarified, or rejected in future specs?
- Should migration tests lock down stochastic training metrics, or only structural outputs such as created files/checkpoints and schema-compatible logs?
- What numerical tolerances should be used for Torch/PyG-to-Julia model computations, especially around dropout, graph convolutions, batching, and BLAS differences?
- Is CUDA 12.8 actually required for the migrated system, given the observed test suite passes on macOS arm64 without CUDA?
- Should PyTorch/PyG warnings and current MSE broadcasting behavior be considered part of the observable training behavior or treated as incidental?

## Risks and limitations

- The environment uses Python 3.14.4 even though many ML ecosystems primarily support earlier Python versions; future runs may differ.
- NumPy RNG outputs are version-dependent; the characterization test controls seed but still assumes NumPy 2.4.6 behavior.
- Full training loop metrics are not deterministic golden outputs; current existing tests pass but many outputs depend on random fixture data and model initialization.
- Zarr behavior may differ between Python Zarr and Julia Zarr layouts; the code contains explicit fallback logic but only small Python-created fixtures were characterized here.
- GPU/CUDA/DDP behavior is environment-sensitive and was not validated against CUDA 12.8 or a multi-process cluster.
- Binary model save files are Torch-version-sensitive; the added tests avoid golden binary files.
- The report cites source and tests as evidence but does not claim current behavior is desirable or should be preserved.
