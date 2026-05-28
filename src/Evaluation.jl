using DataFrames
using Statistics: mean

export EvaluationError,
    evaluate_iterator,
    evaluate_batch,
    evaluation_report_dataframe,
    loss_report_columns,
    task_metric_columns,
    monitor_task_columns,
    monitor_task_results,
    collected_model_outputs,
    graph_batch_model_input

"""
    EvaluationError(operation::String, field::String, message::String)

User-facing exception for evaluation, loss, and metric failures.
"""
struct EvaluationError <: Exception
    operation::String
    field::String
    message::String
end

function Base.showerror(io::IO, err::EvaluationError)
    print(io, err.operation, " failed for ", err.field, ": ", err.message)
end

loss_report_columns() = [:loss_avg, :loss_min, :loss_max]
task_metric_columns(metrics::AbstractDict) = Symbol.(collect(keys(metrics)))
monitor_task_columns(monitors::AbstractDict) = Symbol.(collect(keys(monitors)))
monitor_task_results(monitors, predictions, targets) = Dict{Symbol, Any}(Symbol(name) => _call_monitor(name, spec, predictions, targets) for (name, spec) in monitors)
collected_model_outputs(results) = [result.outputs for result in results]
graph_batch_model_input(batch) = batch

function _lookup_output(outputs, task)
    if outputs isa AbstractDict
        haskey(outputs, task) && return outputs[task]
        haskey(outputs, Symbol(task)) && return outputs[Symbol(task)]
        haskey(outputs, String(task)) && return outputs[String(task)]
    elseif hasproperty(outputs, task)
        return getproperty(outputs, task)
    elseif task isa AbstractString && hasproperty(outputs, Symbol(task))
        return getproperty(outputs, Symbol(task))
    end
    throw(EvaluationError("evaluate metric", string(task), "missing task output"))
end

function _call_loss(fn, outputs, batch, task)
    output = task === nothing ? outputs : _lookup_output(outputs, task)
    try
        return Float64(fn(output, batch))
    catch err
        try
            return Float64(fn(output))
        catch
            throw(EvaluationError("evaluate loss", string(task), "criterion failed: $(sprint(showerror, err))"))
        end
    end
end

function _monitor_callable(name, spec)
    if spec isa Function
        return spec
    elseif spec isa AbstractDict
        monitor = haskey(spec, "monitor") ? spec["monitor"] : haskey(spec, :monitor) ? spec[:monitor] : haskey(spec, "fn") ? spec["fn"] : haskey(spec, :fn) ? spec[:fn] : nothing
        monitor === nothing && throw(EvaluationError("evaluate monitor", string(name), "invalid monitor configuration: missing monitor"))
        return monitor
    end
    throw(EvaluationError("evaluate monitor", string(name), "invalid monitor configuration"))
end

function _batch_target(batch)
    if hasproperty(batch, :targets)
        return getproperty(batch, :targets)
    elseif hasproperty(batch, :y)
        return getproperty(batch, :y)
    elseif batch isa AbstractDict && haskey(batch, "targets")
        return batch["targets"]
    elseif batch isa AbstractDict && haskey(batch, :targets)
        return batch[:targets]
    elseif batch isa AbstractDict && haskey(batch, "y")
        return batch["y"]
    elseif batch isa AbstractDict && haskey(batch, :y)
        return batch[:y]
    end
    return nothing
end

function _call_monitor(name, spec, predictions, targets)
    monitor = _monitor_callable(name, spec)
    try
        return monitor(predictions, targets)
    catch err
        throw(EvaluationError("evaluate monitor", string(name), "invalid monitor task: $(sprint(showerror, err))"))
    end
end

"""
    evaluate_batch(model, batch; criteria, task_metrics = Dict())

Evaluate one batch by passing it unchanged to the model. This keeps evaluation
model-agnostic: graph batches remain graph-shaped, dense batches remain dense.
"""
function evaluate_batch(model, batch; criteria = Dict{Any, Any}(), task_metrics = Dict{Any, Any}())
    outputs = model(graph_batch_model_input(batch))
    isempty(criteria) && throw(EvaluationError("evaluate batch", "criteria", "at least one criterion function is required"))

    losses = Float64[]
    for (task, criterion) in criteria
        push!(losses, _call_loss(criterion, outputs, batch, task === :__all__ ? nothing : task))
    end

    return (outputs = outputs, targets = _batch_target(batch), losses = losses)
end

function evaluation_report_dataframe(losses::AbstractVector, monitor_results::AbstractDict)
    isempty(losses) && throw(EvaluationError("build evaluation report", "losses", "no loss values were recorded"))
    data = Dict{Symbol, Any}(
        :loss_avg => [mean(losses)],
        :loss_min => [minimum(losses)],
        :loss_max => [maximum(losses)],
    )
    for (name, value) in monitor_results
        data[Symbol(name)] = [value]
    end
    DataFrame(data)
end

"""
    evaluate_iterator(model, data_iterator; criteria, task_metrics = Dict())

Run model evaluation over an iterator and return a one-row `DataFrame` report
with `loss_avg`, `loss_min`, `loss_max`, and configured metric columns.
"""
function evaluate_iterator(model, data_iterator; criteria = Dict{Any, Any}(), task_metrics = Dict{Any, Any}())
    all_losses = Float64[]
    predictions = Any[]
    targets = Any[]
    processed = 0
    for batch in data_iterator
        result = evaluate_batch(model, batch; criteria = criteria)
        append!(all_losses, result.losses)
        push!(predictions, result.outputs)
        push!(targets, result.targets)
        processed += 1
    end
    processed == 0 && throw(EvaluationError("evaluate iterator", "data_iterator", "no evaluation data is available"))

    monitor_results = Dict{Symbol, Any}()
    for (name, spec) in task_metrics
        monitor_results[Symbol(name)] = _call_monitor(name, spec, predictions, targets)
    end
    evaluation_report_dataframe(all_losses, monitor_results)
end
