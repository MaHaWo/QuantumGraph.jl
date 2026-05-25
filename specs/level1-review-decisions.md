# Level 1 BDD Review Decisions

Human decisions recorded before Level 1 spec drafting:

- Dataset/model graph boundary: use a GraphNeuralNetworks.jl-compatible graph container, or a QuantumGraph wrapper around one, populated from Zarr data.
- Model layer mapping: specify behavior structurally; leave exact Flux/GraphNeuralNetworks layer mapping to implementation because QuantumGravPy templates are configurable.
- Evaluation, early stopping, and trainer report interface: use DataFrames.jl DataFrame values as the tabular data interface.
- Tuning backend: use Optuna.jl as the preferred backend candidate while keeping QuantumGraph's tuning concepts backend-neutral where practical.
- Checkpoints: Julia-native checkpoint format only for the current migration.
- Execution scope: one machine with at most one accelerator; distributed/DDP compatibility is not part of the current public surface.
