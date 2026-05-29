# Migration compatibility

QuantumGraph.jl is a Julia library for migrated QuantumGrav workflows. The current execution scope is single-machine Julia usage with CPU support and, when available, one configured accelerator.

## Supported artifacts

| Artifact type | Status | Notes |
| --- | --- | --- |
| YAML-style configuration structure | Supported | Migrated configuration tags and object resolution are represented through QuantumGraph's config and registry APIs. |
| Zarr stores produced by QuantumGravPy | Supported | Zarr-backed data is read through `ZarrLoading.jl` and `Datasets.jl` compatibility boundaries. |
| DataFrame evaluation reports | Supported | Evaluation and training reports use Julia `DataFrames.DataFrame` values and Julia-native serialization where written as artifacts. |
| Julia-native checkpoint artifacts (`.jls`) | Supported | Checkpoints written by `save_julia_checkpoint` are the supported checkpoint format and can be loaded with `load_julia_checkpoint`. |
| Optuna tuning outputs expressed as selected config parameters | Supported structurally | Tuning utilities replay selected parameters into QuantumGraph configuration structures. |

## Deferred or unsupported artifacts

| Artifact type | Status | Notes |
| --- | --- | --- |
| Python Torch checkpoint files (`.pt`, `.pth`, `.ckpt`) | Unsupported without explicit conversion | QuantumGraph does not implicitly load Torch checkpoint binaries. Convert model/config state intentionally and save a Julia-native `.jls` checkpoint before loading through QuantumGraph. |
| Exact Torch/PyG numeric checkpoint parity | Deferred | Current migration tests structural behavior, schemas, and artifact inventory rather than exact learned-parameter equivalence. |
| Python object imports from `!pyobject` | Requires registry mapping | Julia configs resolve objects through QuantumGraph's explicit registry rather than importing Python classes directly. |

Unsupported checkpoint inputs fail at the compatibility boundary with an error that identifies the artifact type and points back to these notes. This is intentional: partial binary checkpoint loading would be misleading without a verified conversion path.
