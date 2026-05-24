# QuantumGravPy Compact Overview for Julia Migration

## Modules
- `QuantumGrav.__init__`: public API exports dataset, model, trainers, evaluators, early stopping, config helpers, zarr loaders, and `models`.
- `base`: `Configurable` abstract interface for `from_config`/`to_config`.
- `config_utils`: PyYAML custom tags, Python object import, sweep/coupled-sweep expansion, pyobject-tag serialization.
- `dataset_base` + `dataset_ondisk`: zarr-backed PyTorch Geometric datasets, optional preprocessing to `processed/data_*.pt`, lazy item loading.
- `gnn_model`: dynamic GNN composition with encoder, pooling or latent model, graph feature net, multiple downstream heads.
- `models`: `GNNBlock`, `LinearSequential`, `SkipConnection`.
- `evaluate`: `Evaluator`, `Validator`, `Tester` collecting loss/metrics in pandas.
- `early_stopping`: metric-based multi-task early stopping.
- `train`: single-process training/eval/checkpoint orchestration.
- `train_ddp`: distributed trainer using torch.distributed/DDP.
- `QGTune/tune`: Optuna search space, references/coupled sweeps, study creation, best config export.
- `load_zarr`: recursive zarr group-to-dict utility.

## Entry points
- Package import: `import QuantumGrav as QG` from `src/QuantumGrav/__init__.py`.
- Main runtime: instantiate `Trainer(config)`, call `prepare_dataloaders`, `run_training`, then `run_test`.
- Model construction: `GNNModel.from_config(config["model"])`.
- Dataset construction: `QGDataset(input=zarr_files, output=..., reader=...)`.
- Tuning: `QGTune.tune.build_search_space`, `create_study`, `save_best_config`.

## Tech stack
Python 3.12+, hatchling, PyTorch, PyTorch Geometric, torch.distributed/DDP, zarr, NumPy, pandas, PyYAML, jsonschema, joblib, Optuna, matplotlib/scikit-learn dependencies, pytest, mkdocs-material/mkdocstrings.

## Dominant coding/testing paradigms
Object-oriented configurable components; PyTorch module composition; YAML-driven dependency injection; functional utility helpers for config/tuning; pytest unit tests with fixtures and synthetic temporary zarr stores.

## Test layout
- `test_config_utils.py`, `test/config.yaml`: custom YAML tags and config sweep expansion.
- `test_datasetbase.py`, `test_ondiskdataset.py`, `test_load_zarr.py`: zarr reading, sample counting, preprocessing, indexing, dataloaders.
- `test_gnn_block.py`, `test_skipconnection.py`, `test_linearsequential.py`, `test_gnn_model.py`: blocks, composed model, forward/backward, config, save/load.
- `test_trainer.py`, `test_trainer_ddp.py`, `test_evaluate.py`, `test_early_stopping.py`: training lifecycle, checkpointing, DDP, evaluators, early stopping.
- `test_tune.py`, `test_utils.py`: Optuna conversion, references, study storage, dynamic imports, nested paths.
Run from repo root: `python -m pytest`.

## Data fixtures
`test/conftest.py` creates temporary zarr stores containing `adjacency_matrix`, `link_matrix`, `max_pathlen_future`, `max_pathlen_past`, `dimension`, `atomcount`, optionally `num_samples`. Reader fixtures convert dense adjacency to PyG `edge_index`/`edge_attr` and node features from path-length arrays.

## Documentation pointers
`mkdocs.yml`; docs in `docs/index.md`, `docs/getting_started.md`, `docs/datasets_and_preprocessing.md`, `docs/models.md`, `docs/training_a_model.md`, `docs/hparam_tuning.md`, `docs/api.md`.

## Julia migration risks
- Core dependence on PyTorch/PyG types, modules, autograd, DataLoader, `state_dict` checkpoints.
- Arbitrary Python object imports via `!pyobject` need a Julia registry/mapping.
- zarr compatibility already has Python-vs-Julia layout fallback; preserve with golden fixtures.
- Criterion signatures differ between training (`criterion(outputs, data, trainer)`) and evaluation (`criterion(outputs, data)`).
- Config and early-stopping mutate dictionaries; watch aliasing/immutability decisions in Julia.
- DDP assumes NCCL/one GPU per process.
- Synthetic tests use random data, so add deterministic fixtures for migration validation.
