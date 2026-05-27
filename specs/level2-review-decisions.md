# Level 2 BDD Review Decisions

Human decisions recorded before Level 2 spec drafting:

- Zarr reader strategy: `Zarr.jl` is a public dependency for reading Python-produced Zarr stores.
- Zarr compatibility: no compatibility shim is required unless future evidence shows `Zarr.jl` cannot read an approved fixture layout.
- Graph sample type: expose GraphNeuralNetworks.jl graph data structures directly; do not introduce a QuantumGraph wrapper solely for graph samples.
- Model component mapping: specify reusable layer/block behavior structurally; exact Flux.jl and GraphNeuralNetworks.jl layer choices remain implementation decisions.
- Config parser scope: Level 2 config/interface specs should describe public QuantumGraph behavior only and avoid committing to a specific YAML parser implementation.
