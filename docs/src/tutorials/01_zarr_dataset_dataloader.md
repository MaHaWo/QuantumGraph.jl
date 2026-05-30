# Zarr datasets and dataloaders

This executable tutorial is maintained as a Jupyter notebook:

[`docs/src/notebooks/01_zarr_dataset_dataloader.ipynb`](../notebooks/01_zarr_dataset_dataloader.ipynb)

It covers:

- creating a small local Zarr store,
- opening it with `open_zarr_for_dataset`,
- constructing a `QuantumGraphDataset`,
- wrapping it in `dataset_dataloader`, and
- loading the same store from a read-only `.zip` archive.
