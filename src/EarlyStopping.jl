using DataFrames

export EarlyStoppingError,
    EarlyStoppingState,
    EarlyStoppingDecision,
    early_stopping_state,
    evaluate_early_stopping,
    continue_or_stop_decision,
    early_stopping_best_score,
    early_stopping_grace_state

"""
    EarlyStoppingError(operation::String, field::String, message::String)

User-facing exception for early stopping state evaluation failures.
"""
struct EarlyStoppingError <: Exception
    operation::String
    field::String
    message::String
end

function Base.showerror(io::IO, err::EarlyStoppingError)
    print(io, err.operation, " failed for ", err.field, ": ", err.message)
end

mutable struct EarlyStoppingState
    metric::Symbol
    mode::Symbol
    patience::Int
    grace_period::Int
    current_grace_period::Int
    best_score::Union{Nothing, Float64}
    found_better::Bool
end

struct EarlyStoppingDecision
    should_stop::Bool
    found_better::Bool
    best_score::Float64
    current_score::Float64
    current_grace_period::Int
end

function early_stopping_state(; metric, mode = :min, patience::Integer = 3, grace_period::Integer = 0)
    mode_symbol = Symbol(mode)
    mode_symbol in (:min, :max) || throw(EarlyStoppingError("configure early stopping", "mode", "mode must be :min or :max"))
    EarlyStoppingState(Symbol(metric), mode_symbol, Int(patience), Int(grace_period), 0, nothing, false)
end

function _metric_values(history::DataFrame, metric::Symbol)
    nrow(history) > 0 || throw(EarlyStoppingError("evaluate early stopping", string(metric), "no evaluation data is available"))
    metric in Symbol.(names(history)) || throw(EarlyStoppingError("evaluate early stopping", string(metric), "missing metric column"))
    Float64.(history[!, metric])
end

_better(mode::Symbol, score, best) = best === nothing || (mode == :min ? score < best : score > best)

"""
    evaluate_early_stopping(state, history::DataFrame)

Evaluate and mutate early-stopping state from a DataFrame history.
"""
function evaluate_early_stopping(state::EarlyStoppingState, history::DataFrame)
    values = _metric_values(history, state.metric)
    current_score = values[end]
    found_better = _better(state.mode, current_score, state.best_score)
    if found_better
        state.best_score = current_score
        state.current_grace_period = 0
    else
        state.current_grace_period += 1
    end
    state.found_better = found_better
    should_stop = length(values) > state.grace_period && state.current_grace_period >= state.patience
    EarlyStoppingDecision(should_stop, found_better, Float64(state.best_score), current_score, state.current_grace_period)
end

function evaluate_early_stopping(history::DataFrame; metric, mode = :min, patience::Integer = 3, grace_period::Integer = 0)
    state = early_stopping_state(; metric = metric, mode = mode, patience = patience, grace_period = grace_period)
    evaluate_early_stopping(state, history)
end

continue_or_stop_decision(decision::EarlyStoppingDecision) = decision.should_stop ? :stop : :continue
early_stopping_best_score(state::EarlyStoppingState) = state.best_score
early_stopping_grace_state(state::EarlyStoppingState) = state.current_grace_period
