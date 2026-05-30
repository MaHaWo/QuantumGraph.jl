# Getting started

## Installation for local development

From the repository root, activate the package environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Then load the package:

```julia
using QuantumGraph
```

## Core concepts

- **Zarr stores** hold graph arrays on disk. `QuantumGraph.jl` can open local stores and read-only zipped stores through Zarr.jl.
- **Datasets** map one or more Zarr stores into a sample-indexed `QuantumGraphDataset`.
- **Dataloaders** expose datasets through MLUtils-compatible iteration.
- **Configurations** are Julia dictionaries loaded from text or constructed directly.
- **Models** are ordinary Julia callables, including `Flux.Chain` values and custom structs.
- **Trainers** run local, single-machine training and write reports/checkpoints.
- **Tuning** builds Optuna trial-specific configurations from search-space tags.

## Minimal workflow

```julia
using QuantumGraph
using Flux
using Optimisers
using DataFrames

dataset = [(features = Float32[1, 2], targets = Dict(:loss => 1.0))]
model = Flux.Chain(Flux.Dense(2 => 1))
loss(model, batch, outputs) = sum(abs2, outputs)
evaluator(model, iterator) = DataFrame(loss = [0.0])

trainer = construct_trainer(Dict{String, Any}(
    "dataset" => dataset,
    "model" => model,
    "optimizer" => Optimisers.Adam(1.0f-3),
    "loss" => loss,
    "evaluator" => evaluator,
    "early_stopping" => early_stopping_state(metric = :loss),
    "output_path" => mktempdir(),
    "device" => "cpu",
    "num_epochs" => 1,
))

fit_trainer!(trainer)
```

For complete executable examples, see the [tutorial notebooks](tutorials.md).
