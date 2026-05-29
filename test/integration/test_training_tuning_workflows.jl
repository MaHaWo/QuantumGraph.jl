# These integration tests cover training orchestration and tuning utilities.
# They run deterministic trainers, inspect written reports/checkpoints, exercise
# Flux optimizer updates and early stopping, then validate Optuna search helpers.
using Test
using QuantumGraph
using DataFrames
using Flux
using Optimisers
using Optuna

@testset "Training orchestration integration contract" begin
    tmp = mktempdir()
    batches = [
        (features = Float32[1, 2], targets = Dict(:loss => 1.0)),
        (features = Float32[3, 4], targets = Dict(:loss => 2.0)),
    ]
    seen_batches = Any[]
    model = batch -> begin
        push!(seen_batches, batch)
        Dict(:loss => batch.targets[:loss])
    end
    optimizer_steps = Any[]
    optimizer = (model, batch, outputs) -> push!(optimizer_steps, (batch = batch, outputs = outputs))
    scheduler_epochs = Int[]
    scheduler = epoch -> push!(scheduler_epochs, epoch)
    evaluator = (model, iterator) -> DataFrame(loss_avg = [1.5], loss_min = [1.0], loss_max = [2.0])

    config = Dict{String, Any}(
        "dataset" => batches,
        "model" => model,
        "optimizer" => optimizer,
        "scheduler" => scheduler,
        "evaluator" => evaluator,
        "early_stopping" => early_stopping_state(metric = :loss_avg),
        "output_path" => tmp,
        "device" => "cpu",
        "num_epochs" => 2,
        "checkpoint_at" => 1,
    )

    trainer = construct_trainer(config)
    @test trainer isa Trainer
    @test trainer.prepared
    @test !trainer.started
    @test local_single_machine_training(trainer)
    @test trainer.output_path == tmp
    @test trainer.checkpoint_path == joinpath(tmp, "model_checkpoints")

    @test validate_training_config(config)["output_path"] == tmp
    @test prepare_training_components(config).model === model

    invalid_err = try
        construct_trainer(Dict("dataset" => batches))
        nothing
    catch caught
        caught
    end
    @test invalid_err isa TrainingError
    @test occursin("missing required", sprint(showerror, invalid_err))

    accelerator_err = try
        construct_trainer(merge(config, Dict{String, Any}("device" => "cuda:99")))
        nothing
    catch caught
        caught
    end
    @test accelerator_err isa TrainingError
    @test occursin("unsupported accelerator backend", sprint(showerror, accelerator_err))

    fit_trainer!(trainer)
    @test trainer.started
    @test trainer.epoch == 2
    @test length(seen_batches) == 4
    @test length(optimizer_steps) == 4
    @test scheduler_epochs == [1, 2]
    @test length(trainer.reports) == 2
    @test trainer.reports[end] isa DataFrame

    artifacts = training_artifact_paths(trainer)
    @test isfile(artifacts.config_copy)
    @test isdir(artifacts.checkpoint_path)
    @test artifacts.latest_checkpoint !== nothing
    @test isfile(artifacts.latest_checkpoint)
    @test endswith(artifacts.latest_checkpoint, ".jls")
    @test isfile(joinpath(tmp, "validation_report_epoch_1.jls"))
    @test isfile(joinpath(tmp, "validation_report_epoch_2.jls"))

    checkpoint = load_julia_checkpoint(artifacts.latest_checkpoint)
    @test checkpoint["epoch"] == 2
    @test haskey(checkpoint, "config")

    bad_checkpoint_file = joinpath(tmp, "not_a_directory")
    write(bad_checkpoint_file, "occupied")
    bad_trainer = construct_trainer(merge(config, Dict{String, Any}(
        "output_path" => mktempdir(),
        "checkpoint_path" => bad_checkpoint_file,
    )))
    checkpoint_err = try
        save_julia_checkpoint(bad_trainer)
        nothing
    catch caught
        caught
    end
    @test checkpoint_err isa TrainingError
    @test occursin("checkpoint", sprint(showerror, checkpoint_err))
    @test bad_trainer.latest_checkpoint === nothing

    flux_tmp = mktempdir()
    flux_model = Flux.Chain(Flux.Dense(2 => 1))
    initial_weight = copy(flux_model[1].weight)
    flux_batches = [(Float32[1 2; 3 4], Float32[1 2])]
    flux_loss = (model, batch) -> sum(abs2, vec(model(batch[1])) .- batch[2])
    flux_evaluator = (model, iterator) -> DataFrame(loss_avg = [flux_loss(model, first(iterator))], loss_min = [flux_loss(model, first(iterator))], loss_max = [flux_loss(model, first(iterator))])
    flux_trainer = construct_trainer(Dict{String, Any}(
        "dataset" => flux_batches,
        "model" => flux_model,
        "optimizer" => Optimisers.Adam(0.01),
        "loss" => flux_loss,
        "scheduler" => nothing,
        "evaluator" => flux_evaluator,
        "early_stopping" => early_stopping_state(metric = :loss_avg),
        "output_path" => flux_tmp,
        "device" => "cpu",
        "num_epochs" => 2,
        "checkpoint_at" => 1,
    ))
    fit_trainer!(flux_trainer)
    @test flux_trainer.optimizer_state !== nothing
    @test flux_model[1].weight != initial_weight
    @test length(flux_trainer.reports) == 2
    @test isfile(training_artifact_paths(flux_trainer).latest_checkpoint)

    early_tmp = mktempdir()
    scores = [1.0, 1.2, 1.3, 1.4]
    early_evaluator = let scores = scores
        (model, iterator) -> DataFrame(loss_avg = [popfirst!(scores)], loss_min = [0.0], loss_max = [0.0])
    end
    early_trainer = construct_trainer(Dict{String, Any}(
        "dataset" => batches,
        "model" => model,
        "optimizer" => optimizer,
        "scheduler" => nothing,
        "evaluator" => early_evaluator,
        "early_stopping" => early_stopping_state(metric = :loss_avg, mode = :min, patience = 1, grace_period = 0),
        "output_path" => early_tmp,
        "device" => "cpu",
        "num_epochs" => 4,
        "checkpoint_at" => nothing,
    ))
    fit_trainer!(early_trainer)
    @test early_trainer.stopped_early
    @test early_trainer.epoch < 4
    @test early_stopping_best_score(early_trainer.early_stopping) == 1.0
    @test early_stopping_grace_state(early_trainer.early_stopping) == 1
    @test length(early_trainer.early_stopping_decisions) == 2
    @test isfile(current_best_checkpoint_path(early_trainer))
end

@testset "Tuning utilities integration contract" begin
    @test preferred_tuning_backend() == :Optuna
    @test :study in backend_neutral_tuning_concepts()
    @test :trial in backend_neutral_tuning_concepts()
    limitations = unsupported_optuna_capabilities()
    @test limitations.backend == :Optuna
    @test !isempty(limitations.limitations)

    @test is_flat_list(["relu", "tanh"])
    @test !is_flat_list([[1, 2]])
    @test is_categorical_suggestion([16, 32, 64])
    @test is_float_suggestion((0.001, 0.1, 0.001))
    @test is_float_suggestion((1e-5, 1e-1, true))
    @test is_int_suggestion((1, 3, 1))

    config = Dict{Any, Any}(
        "model" => Dict{Any, Any}(
            "layers" => Dict("type" => "sweep", "values" => [1, 2]),
            "width" => Dict("type" => "coupled-sweep", "target" => ["model", "layers"], "values" => [16, 32]),
            "activation" => Dict("type" => "sweep", "values" => ["relu", "tanh"]),
            "lr" => Dict("type" => "range", "tune_values" => (0.001, 0.1, 0.001)),
        ),
        "trainer" => Dict{Any, Any}(
            "epochs" => Dict("type" => "range", "tune_values" => (1, 3, 1)),
            "copied_width" => Dict("type" => "reference", "target" => ["model", "layers"]),
        ),
    )
    trial = FixedTrial(Dict{String, Any}(
        "model.layers" => 2,
        "model.activation" => "tanh",
        "model.lr" => 0.01,
        "trainer.epochs" => 2,
    ))

    @test get_value_of_ref(config, ["model", "layers"])["type"] == "sweep"
    @test convert_to_suggestion("model.activation", config["model"]["activation"], trial, config) == "tanh"
    coupled = convert_to_suggestion("model.width", config["model"]["width"], trial, config)
    @test coupled["type"] == "coupled-sweep-mapping"
    @test coupled["mapping"][2] == 32

    suggestions, mapping = get_suggestion(config, config, trial, String[])
    @test suggestions["model"]["layers"] == 2
    @test suggestions["model"]["activation"] == "tanh"
    @test suggestions["model"]["width"]["type"] == "coupled-sweep-mapping"
    @test haskey(mapping, "model.width")

    resolved = build_search_space(config, trial)
    @test resolved["model"]["width"] == 32
    @test resolved["trainer"]["copied_width"] == 2

    bad_coupled = deepcopy(config)
    bad_coupled["model"]["width"] = Dict("type" => "coupled-sweep", "target" => ["trainer", "epochs"], "values" => [16])
    @test_throws TuningError convert_to_suggestion("model.width", bad_coupled["model"]["width"], trial, bad_coupled)

    study = create_tuning_study(Dict("study_name" => "demo", "direction" => "minimize", "storage" => nothing))
    @test study.study_name == "demo"
    @test study.direction == :minimize
    @test study.backend == :Optuna
    @test study.study isa Optuna.Study
    @test study.storage isa Optuna.InMemoryStorage
    @test_throws TuningError create_tuning_study(Dict("study_name" => "demo"))
    @test_throws TuningError create_tuning_study(Dict("study_name" => "demo", "direction" => "sideways"))
    @test_throws TuningError create_tuning_study(Dict("study_name" => "demo", "direction" => "minimize", "storage" => "unsupported"))

    out = joinpath(mktempdir(), "best_config.txt")
    best = save_best_config(config, trial, out)
    @test isfile(out)
    @test best["model"]["layers"] == 2
    @test best["model"]["width"] == 32
    @test best["trainer"]["copied_width"] == 2
end
