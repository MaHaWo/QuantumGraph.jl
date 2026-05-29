# QuantumGraph.jl

QuantumGraph.jl is a standard importable Julia library for migrated QuantumGrav workflows.

```julia
using QuantumGraph
```

The package exposes public APIs for configuration, Zarr-backed datasets, GraphNeuralNetworks-compatible graph samples, model construction, evaluation, early stopping, local training, CUDA device selection, and Optuna-backed tuning helpers.

## Execution scope

QuantumGraph runs as a single-machine Julia library. CPU execution is always supported; one accelerator may be used when CUDA is available and explicitly configured.

## Compatibility notes

Migration compatibility information, including supported artifact types and checkpoint boundaries, is documented in [`docs/migration_compatibility.md`](docs/migration_compatibility.md).

Supported checkpoint artifacts are Julia-native `.jls` files written by `save_julia_checkpoint` and loaded with `load_julia_checkpoint`. Python Torch checkpoint files require an explicit conversion path before they can be used by QuantumGraph.

## Example workflow sketch

```julia
using QuantumGraph

config = Dict{String, Any}(
    "dataset" => [],
    "model" => identity,
    "optimizer" => nothing,
    "evaluator" => nothing,
    "early_stopping" => nothing,
    "output_path" => "runs/example",
    "device" => "cpu",
)

# Real workflows provide configured datasets, models, evaluators, and early stopping.
# trainer = construct_trainer(config)
# fit_trainer!(trainer)
```

## Tests and verification

Run the full Julia verification suite:

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
```

When validating against the source Python oracle, run from the source repository:

```sh
cd /Users/hmack/Development/QuantumGrav/QuantumGravPy
.venv/bin/pytest test -q
```
