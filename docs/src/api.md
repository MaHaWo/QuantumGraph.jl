# API reference

The API reference is organized by subsystem. Pages use Documenter's `@autodocs` blocks with `Private = true` so every documented symbol in the corresponding source file is included, including documented error types and metadata structs.

```@contents
Pages = [
    "api/core.md",
    "api/interfaces.md",
    "api/config.md",
    "api/zarr-loading.md",
    "api/datasets.md",
    "api/models.md",
    "api/gnn-model.md",
    "api/evaluation.md",
    "api/early-stopping.md",
    "api/devices.md",
    "api/training.md",
    "api/tuning.md",
]
Depth = 2
```
