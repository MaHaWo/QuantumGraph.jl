using Test
using QuantumGraph

@testset "Public library surface integration contract" begin
    exported = Set(String.(names(QuantumGraph; all = false, imported = false)))

    capability_patterns = Dict(
        :config => [r"config"i, r"load"i, r"expand"i, r"sweep"i],
        :registry => [r"registr"i, r"resolv"i, r"object"i],
        :zarr => [r"zarr"i],
        :dataset => [r"dataset"i],
        :graph_model => [r"gnn"i, r"graph"i, r"model"i],
        :model_blocks => [r"block"i, r"skip"i, r"sequential"i, r"layer"i],
        :evaluation => [r"evaluat"i, r"metric"i],
        :early_stopping => [r"early"i, r"stop"i],
        :training => [r"train"i, r"fit"i],
        :tuning => [r"tun"i, r"trial"i, r"study"i, r"search"i],
        :device => [r"device"i, r"cuda"i, r"accelerator"i],
    )

    for patterns in values(capability_patterns)
        @test any(name -> any(pattern -> occursin(pattern, name), patterns), exported)
    end

    required_names = [
        "load_config",
        "expand_config",
        "register_object!",
        "resolve_registered_object",
        "open_zarr_store",
        "construct_dataset",
        "dataset_dataloader",
        "construct_model_component",
        "construct_gnn_model",
        "evaluate_iterator",
        "early_stopping_state",
        "construct_trainer",
        "fit_trainer!",
        "create_tuning_study",
        "build_search_space",
        "prepare_execution_device",
        "validate_checkpoint_input",
    ]
    for name in required_names
        @test name in exported
    end

    @test isempty([name for name in exported if occursin("__", name)])
    @test isempty([name for name in exported if endswith(name, ".py")])
    @test !isdefined(QuantumGraph, :main)
    @test !isdefined(QuantumGraph, :julia_main)

    repo_root = dirname(dirname(@__DIR__))
    readme = read(joinpath(repo_root, "README.md"), String)
    docs = read(joinpath(repo_root, "docs", "migration_compatibility.md"), String)
    combined = readme * "\n" * docs

    @test occursin(r"(?m)^\s*using\s+QuantumGraph\b", combined)
    @test occursin(r"(?i)one\s+machine|single[- ]machine", combined)
    @test occursin(r"(?i)(at\s+most\s+one|single|one)\s+(accelerator|gpu|cuda\s+device)", combined)
    @test occursin(r"(?i)configuration", combined)
    @test occursin(r"(?i)datasets", combined)
    @test occursin(r"(?i)models", combined)
    @test occursin(r"(?i)evaluation", combined)
    @test occursin(r"(?i)early stopping", combined)
    @test occursin(r"(?i)training", combined)
    @test occursin(r"(?i)tuning", combined)
    @test occursin(r"(?i)Zarr", combined)
    @test !occursin(r"(?i)(quantumgraph|quantumgrav)\s+(train|tune|evaluate|validate|test|run)\b", combined)
end
