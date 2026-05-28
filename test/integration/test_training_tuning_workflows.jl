using Test
using QuantumGraph
using DataFrames
using Flux
using Optimisers

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
