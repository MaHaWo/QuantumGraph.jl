# Custom models

This executable tutorial is maintained as a Jupyter notebook:

[`docs/src/notebooks/04_custom_models.ipynb`](../notebooks/04_custom_models.ipynb)

It uses a real config file from:

[`docs/src/examples/configs/custom_model.yml`](../examples/configs/custom_model.yml)

It covers:

- defining a callable Julia model struct,
- making it visible to Flux's parameter traversal,
- registering a model constructor, and
- using the registered constructor from a file-backed config.
