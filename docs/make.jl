using Documenter
using QuantumGraph

DocMeta.setdocmeta!(QuantumGraph, :DocTestSetup, :(using QuantumGraph); recursive = true)

makedocs(
    sitename = "QuantumGraph.jl",
    modules = [QuantumGraph],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "",
        edit_link = nothing,
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Tutorials" => [
            "Overview" => "tutorials.md",
            "Zarr datasets and dataloaders" => "tutorials/01_zarr_dataset_dataloader.md",
            "Configuration files" => "tutorials/02_config_files.md",
            "Flux training" => "tutorials/03_flux_training.md",
            "Custom models" => "tutorials/04_custom_models.md",
            "Optuna tuning" => "tutorials/05_optuna_tuning.md",
        ],
        "API" => [
            "Overview" => "api.md",
            "Core module" => "api/core.md",
            "Interfaces and registry" => "api/interfaces.md",
            "Configuration" => "api/config.md",
            "Zarr loading" => "api/zarr-loading.md",
            "Datasets" => "api/datasets.md",
            "Model components" => "api/models.md",
            "Graph neural networks" => "api/gnn-model.md",
            "Evaluation" => "api/evaluation.md",
            "Early stopping" => "api/early-stopping.md",
            "Devices" => "api/devices.md",
            "Training" => "api/training.md",
            "Tuning" => "api/tuning.md",
        ],
    ],
)
