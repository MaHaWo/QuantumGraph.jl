# Optuna tuning

This executable tutorial is maintained as a Jupyter notebook:

[`docs/src/notebooks/05_optuna_tuning.ipynb`](../notebooks/05_optuna_tuning.ipynb)

It uses a real config file from:

[`docs/src/examples/configs/optuna_search_space.yml`](../examples/configs/optuna_search_space.yml)

It covers:

- loading search-space tags from a YAML file,
- building nested Julia search configs,
- resolving trial-specific values with `build_search_space`,
- creating an Optuna-backed study, and
- exporting a best configuration.
