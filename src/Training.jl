using DataFrames
using Flux
using Optimisers
using Serialization

export TrainingError,
    Trainer,
    construct_trainer,
    validate_training_config,
    prepare_training_components,
    start_training,
    fit_trainer!,
    run_single_machine_training!,
    save_julia_checkpoint,
    load_julia_checkpoint,
    write_training_config_copy,
    write_training_report,
    training_artifact_paths,
    local_single_machine_training,
    apply_training_early_stopping!,
    current_best_checkpoint_path,
    accelerator_backend_error,
    training_failure_error

"""
    TrainingError(operation::String, field::String, message::String)

User-facing exception for training orchestration failures.
"""
struct TrainingError <: Exception
    operation::String
    field::String
    message::String
end

function Base.showerror(io::IO, err::TrainingError)
    print(io, err.operation, " failed for ", err.field, ": ", err.message)
end

"""
    Trainer

Mutable orchestration state for one local training run.

`Trainer` deliberately keeps the training loop model-agnostic. The model only
needs to be callable on a batch. When `loss` and an Optimisers.jl rule are
provided, the trainer uses the Flux/Optimisers interface for gradient updates;
otherwise, it falls back to an optional user-provided optimizer hook. Device
movement is intentionally shallow and happens at the batch boundary.
"""
mutable struct Trainer
    config::Dict{String, Any}
    dataset::Any
    model::Any
    optimizer::Any
    scheduler::Any
    optimizer_state::Any
    loss::Any
    evaluator::Any
    early_stopping::Any
    output_path::String
    checkpoint_path::String
    device::ExecutionDevice
    prepared::Bool
    started::Bool
    epoch::Int
    latest_checkpoint::Union{Nothing, String}
    reports::Vector{DataFrame}
    early_stopping_decisions::Vector{Any}
    stopped_early::Bool
end

# The minimal structural sections needed before training can be prepared.
_required_sections() = ["dataset", "model", "optimizer", "evaluator", "early_stopping", "output_path"]

# Config dictionaries may come from parsed YAML with string keys or from Julia
# callers with symbol keys. Keep both forms accepted at public boundaries.
function _cfg_has(config, key)
    haskey(config, key) || haskey(config, Symbol(key))
end

function _cfg_get(config, key, default = nothing)
    haskey(config, key) && return config[key]
    haskey(config, Symbol(key)) && return config[Symbol(key)]
    default
end

"""
    validate_training_config(config::AbstractDict) -> Dict{String, Any}

Validate the resolved training configuration and normalize top-level keys to
strings.

The trainer requires dataset, model, optimizer, evaluator, early-stopping, and
output-path sections before it can prepare components. Device settings are
validated through `CUDADevice.jl`: CPU is always accepted, while CUDA requests
must resolve to one available accelerator before artifacts are written.
"""
function validate_training_config(config::AbstractDict)
    missing = [section for section in _required_sections() if !_cfg_has(config, section)]
    isempty(missing) || throw(TrainingError("validate training config", join(missing, ","), "missing required configuration section(s)"))
    try
        prepare_execution_device(config)
    catch err
        err isa DeviceError && throw(TrainingError("validate training config", "device", sprint(showerror, err)))
        rethrow()
    end
    Dict{String, Any}(String(k) => v for (k, v) in config)
end

"""
    accelerator_backend_error(device) -> TrainingError

Build the user-facing error used when a training config requests an unsupported
accelerator backend.
"""
function accelerator_backend_error(device)
    TrainingError("validate training config", "device", "unsupported accelerator backend: $device")
end

"""
    training_failure_error(field, message) -> TrainingError

Convenience constructor for generic training failures surfaced through the public
training API.
"""
function training_failure_error(field, message)
    TrainingError("training", String(field), String(message))
end

# Resolve config-specified components through the public registry. This mirrors
# the source system's type/args/kwargs construction pattern without importing
# arbitrary names directly from strings.
function _component_from_spec(spec, field::String)
    if spec isa ObjectReference
        return resolve_registered_object(spec.identifier)
    elseif spec isa AbstractString
        return resolve_registered_object(spec)
    elseif spec isa AbstractDict && (_cfg_has(spec, "type") || _cfg_has(spec, "type_identifier"))
        type_id = _cfg_get(spec, "type", _cfg_get(spec, "type_identifier"))
        target = type_id isa ObjectReference ? resolve_registered_object(type_id.identifier) : type_id isa AbstractString ? resolve_registered_object(type_id) : type_id
        args = _cfg_get(spec, "args", Any[])
        kwargs = _cfg_get(spec, "kwargs", Dict{String, Any}())
        args = args isa Tuple ? collect(args) : args isa AbstractVector ? collect(args) : Any[args]
        kwargs isa AbstractDict || throw(TrainingError("prepare training component", field, "kwargs must be a mapping"))
        kw = Dict{Symbol, Any}(Symbol(k) => v for (k, v) in kwargs)
        try
            return target(args...; kw...)
        catch err
            try
                isempty(args) && return target(Dict{String, Any}(String(k) => v for (k, v) in kwargs))
            catch
            end
            throw(TrainingError("prepare training component", field, "component construction failed: $(sprint(showerror, err))"))
        end
    else
        return spec
    end
end

"""
    prepare_training_components(config::AbstractDict)

Resolve configured training components without starting training.

Each component may be supplied directly, as an `ObjectReference`, as a registry
identifier string, or as a constructor spec with `type`, `args`, and `kwargs`.
The returned named tuple contains the dataset, model, optimizer, scheduler, loss,
evaluator, and early-stopping components used to construct a [`Trainer`](@ref).
"""
function prepare_training_components(config::AbstractDict)
    validated = validate_training_config(config)
    (
        dataset = _component_from_spec(validated["dataset"], "dataset"),
        model = _component_from_spec(validated["model"], "model"),
        optimizer = _component_from_spec(validated["optimizer"], "optimizer"),
        scheduler = _component_from_spec(_cfg_get(validated, "scheduler", nothing), "scheduler"),
        loss = _component_from_spec(_cfg_get(validated, "loss", nothing), "loss"),
        evaluator = _component_from_spec(validated["evaluator"], "evaluator"),
        early_stopping = _component_from_spec(validated["early_stopping"], "early_stopping"),
    )
end

"""
    construct_trainer(config::AbstractDict) -> Trainer

Validate configuration, prepare components, and return a non-started trainer.

Construction is intentionally side-effect light: it resolves components and paths
but does not run epochs, write checkpoints, or create reports. This preserves the
source behavior where trainer construction prepares the workflow and training is
started explicitly.
"""
function construct_trainer(config::AbstractDict)
    validated = validate_training_config(config)
    components = prepare_training_components(validated)
    output_path = String(validated["output_path"])
    checkpoint_path = String(_cfg_get(validated, "checkpoint_path", joinpath(output_path, "model_checkpoints")))
    device = prepare_execution_device(validated)
    model = prepare_model_for_device(components.model, device)
    Trainer(
        validated,
        components.dataset,
        model,
        components.optimizer,
        components.scheduler,
        nothing,
        components.loss,
        components.evaluator,
        components.early_stopping,
        output_path,
        checkpoint_path,
        device,
        true,
        false,
        0,
        nothing,
        DataFrame[],
        Any[],
        false,
    )
end

"""
    local_single_machine_training(trainer::Trainer) -> Bool

Return `true` for the current training implementation, which intentionally does
not initialize distributed or multi-machine process state.
"""
local_single_machine_training(::Trainer) = true

# Device handling belongs at the trainer/batch boundary. Deeper model or dataset
# code should not need CUDA-specific branches.
function _move_to_device(batch, device::ExecutionDevice)
    try
        return prepare_graph_batch_for_device(batch, device)
    catch err
        err isa DeviceError && throw(TrainingError("prepare batch device", "device", sprint(showerror, err)))
        rethrow()
    end
end

# Normalize a dataset-like input into an iterable batch collection for the local
# loop. Real MLUtils loaders are already iterable; simple fixtures can be vectors.
function _as_batches(dataset)
    dataset isa AbstractVector && return dataset
    dataset isa Tuple && return collect(dataset)
    try
        return collect(dataset)
    catch
        return [dataset]
    end
end

# Generic model invocation used for structural training hooks. Tuple batches are
# common in Flux examples, so fall back to the first tuple element when needed.
function _model_outputs(model, batch)
    try
        return model(batch)
    catch err
        if batch isa Tuple && !isempty(batch)
            try
                return model(batch[1])
            catch
            end
        end
        throw(TrainingError("training model step", "model", sprint(showerror, err)))
    end
end

# Flux/Optimisers rules require a state tree created from the current model.
# Callable optimizer hooks do not use this path.
function _ensure_optimizer_state!(trainer::Trainer)
    trainer.optimizer === nothing && return nothing
    trainer.optimizer isa Function && return nothing
    trainer.optimizer_state !== nothing && return trainer.optimizer_state
    try
        trainer.optimizer_state = Flux.setup(trainer.optimizer, trainer.model)
        return trainer.optimizer_state
    catch err
        throw(TrainingError("initialize optimizer", "optimizer", "Flux.setup failed: $(sprint(showerror, err))"))
    end
end

function _call_optimizer(optimizer, model, batch, outputs)
    optimizer === nothing && return nothing
    optimizer isa Function || return nothing
    try
        return optimizer(model, batch, outputs)
    catch err
        throw(TrainingError("training optimizer step", "optimizer", sprint(showerror, err)))
    end
end

# Differentiable Flux path. If no loss is configured, the caller falls back to
# the generic optimizer hook path instead of assuming Flux semantics.
function _flux_training_step!(trainer::Trainer, batch)
    trainer.loss === nothing && return nothing
    _ensure_optimizer_state!(trainer)
    trainer.optimizer_state === nothing && return nothing
    try
        loss_value, grads = Flux.withgradient(trainer.model) do model
            trainer.loss(model, batch)
        end
        Flux.update!(trainer.optimizer_state, trainer.model, grads[1])
        return loss_value
    catch err
        throw(TrainingError("training optimizer step", "Flux/Optimisers", sprint(showerror, err)))
    end
end

function _call_scheduler(scheduler, epoch)
    scheduler === nothing && return nothing
    scheduler isa Function || return nothing
    try
        return scheduler(epoch)
    catch err
        throw(TrainingError("training scheduler step", "scheduler", sprint(showerror, err)))
    end
end

# The trainer only requires evaluator output to be a DataFrame. Evaluation internals
# remain model-agnostic and live in Evaluation.jl.
function _evaluate_for_training(evaluator, model, batches)
    if evaluator isa Function
        result = evaluator(model, batches)
        result isa DataFrame && return result
        throw(TrainingError("training evaluation", "evaluator", "evaluator must return a DataFrame"))
    end
    if hasmethod(evaluate_iterator, Tuple{typeof(model), typeof(batches)})
        return evaluate_iterator(model, batches)
    end
    DataFrame(loss_avg = [0.0], loss_min = [0.0], loss_max = [0.0])
end

"""
    write_training_config_copy(trainer::Trainer) -> String

Write a human-readable copy of the resolved trainer configuration into the run
output directory and return the written path.
"""
function write_training_config_copy(trainer::Trainer)
    mkpath(trainer.output_path)
    path = joinpath(trainer.output_path, "config.txt")
    open(path, "w") do io
        show(io, MIME("text/plain"), trainer.config)
    end
    path
end

"""
    write_training_report(trainer::Trainer, report::DataFrame; name = "validation_report.jls") -> String

Serialize a validation or test report DataFrame as a Julia-native artifact under
the trainer output directory.
"""
function write_training_report(trainer::Trainer, report::DataFrame; name = "validation_report.jls")
    mkpath(trainer.output_path)
    path = joinpath(trainer.output_path, name)
    open(path, "w") do io
        serialize(io, report)
    end
    path
end

"""
    save_julia_checkpoint(trainer::Trainer; name_addition = "") -> String

Write a Julia-native structural checkpoint and update `trainer.latest_checkpoint`.

The checkpoint records the epoch, model type, and resolved config. It is not a
Torch-compatible binary checkpoint and does not yet claim exact learned-parameter
compatibility with the Python source.
"""
function save_julia_checkpoint(trainer::Trainer; name_addition = "")
    try
        if isfile(trainer.checkpoint_path)
            throw(TrainingError("save checkpoint", trainer.checkpoint_path, "checkpoint path is a file, not a directory"))
        end
        mkpath(trainer.checkpoint_path)
        filename = isempty(name_addition) ? "model_current.jls" : "model_$(name_addition).jls"
        path = joinpath(trainer.checkpoint_path, filename)
        open(path, "w") do io
            serialize(io, Dict{String, Any}(
                "epoch" => trainer.epoch,
                "model_type" => string(typeof(trainer.model)),
                "config" => trainer.config,
            ))
        end
        trainer.latest_checkpoint = path
        return path
    catch err
        err isa TrainingError && rethrow()
        throw(TrainingError("save checkpoint", trainer.checkpoint_path, "checkpoint write failure: $(sprint(showerror, err))"))
    end
end

"""
    load_julia_checkpoint(path::AbstractString)

Load a checkpoint written by [`save_julia_checkpoint`](@ref).
"""
function load_julia_checkpoint(path::AbstractString)
    open(path, "r") do io
        deserialize(io)
    end
end

"""
    current_best_checkpoint_path(trainer::Trainer) -> String

Return the conventional path for the checkpoint saved when validation improves.
"""
current_best_checkpoint_path(trainer::Trainer) = joinpath(trainer.checkpoint_path, "model_current_best.jls")

"""
    training_artifact_paths(trainer::Trainer)

Return the structural artifact locations associated with a trainer run.
"""
function training_artifact_paths(trainer::Trainer)
    (
        output_path = trainer.output_path,
        checkpoint_path = trainer.checkpoint_path,
        latest_checkpoint = trainer.latest_checkpoint,
        current_best_checkpoint = current_best_checkpoint_path(trainer),
        config_copy = joinpath(trainer.output_path, "config.txt"),
    )
end

"""
    apply_training_early_stopping!(trainer::Trainer, report::DataFrame)

Apply the configured early-stopping state to the latest validation report.

When the report improves the monitored metric, a `current_best` checkpoint is
written. When patience is exceeded, `trainer.stopped_early` is set so the outer
training loop stops before the configured maximum epoch count.
"""
function apply_training_early_stopping!(trainer::Trainer, report::DataFrame)
    trainer.early_stopping === nothing && return nothing
    trainer.early_stopping isa EarlyStoppingState || return nothing
    decision = evaluate_early_stopping(trainer.early_stopping, report)
    push!(trainer.early_stopping_decisions, decision)
    if decision.found_better
        save_julia_checkpoint(trainer; name_addition = "current_best")
    end
    if decision.should_stop
        trainer.stopped_early = true
    end
    decision
end

"""
    run_single_machine_training!(trainer::Trainer; epochs = trainer.config["num_epochs"])

Run the local training loop.

For each epoch, batches are moved through the shallow device hook, the model is
updated either through Flux/Optimisers (`loss` configured) or through a generic
optimizer callback, the scheduler hook is called, validation is evaluated, reports
are serialized, early stopping is applied, and periodic checkpoints are written.
"""
function run_single_machine_training!(trainer::Trainer; epochs = _cfg_get(trainer.config, "num_epochs", 1))
    trainer.device.backend in (:cpu, :cuda) || throw(accelerator_backend_error(trainer.device.backend))
    trainer.started = true
    write_training_config_copy(trainer)
    batches = _as_batches(trainer.dataset)
    isempty(batches) && throw(TrainingError("start training", "dataset", "no training data is available"))
    for epoch in 1:Int(epochs)
        trainer.epoch = epoch
        for batch in batches
            moved = _move_to_device(batch, trainer.device)
            flux_loss = _flux_training_step!(trainer, moved)
            if flux_loss === nothing
                outputs = _model_outputs(trainer.model, moved)
                _call_optimizer(trainer.optimizer, trainer.model, moved, outputs)
            end
        end
        _call_scheduler(trainer.scheduler, epoch)
        if trainer.evaluator !== nothing
            report = _evaluate_for_training(trainer.evaluator, trainer.model, batches)
            push!(trainer.reports, report)
            write_training_report(trainer, report; name = "validation_report_epoch_$(epoch).jls")
            decision = apply_training_early_stopping!(trainer, report)
            decision !== nothing && decision.should_stop && break
        end
        checkpoint_at = _cfg_get(trainer.config, "checkpoint_at", 1)
        if checkpoint_at !== nothing && epoch % Int(checkpoint_at) == 0
            save_julia_checkpoint(trainer; name_addition = "$(epoch)_current")
        end
        trainer.stopped_early && break
    end
    trainer
end

"""
    start_training(trainer::Trainer; kwargs...)

Alias for [`run_single_machine_training!`](@ref).
"""
start_training(trainer::Trainer; kwargs...) = run_single_machine_training!(trainer; kwargs...)

"""
    fit_trainer!(trainer::Trainer; kwargs...)

Alias for [`run_single_machine_training!`](@ref), using Julia's mutating-function
naming convention.
"""
fit_trainer!(trainer::Trainer; kwargs...) = run_single_machine_training!(trainer; kwargs...)
