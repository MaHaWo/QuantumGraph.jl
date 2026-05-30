# QuantumGraph.jl

`QuantumGraph.jl` is a Julia package for working with quantum-graph datasets, Flux-compatible models, local training loops, and Optuna-based hyperparameter tuning.

The package is documented as a Julia library: examples use Julia data structures, Julia callables, Flux models, and package APIs directly.

## Main workflow

1. Open graph data from local or zipped Zarr stores.
2. Build a `QuantumGraphDataset` and dataloader.
3. Load or assemble an experiment configuration.
4. Define a Flux-compatible model.
5. Construct and run a local `Trainer`.
6. Optionally run Optuna-based hyperparameter tuning.

## Contents

```@contents
Pages = [
    "getting-started.md",
    "tutorials.md",
    "api.md",
]
Depth = 2
```
