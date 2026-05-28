import Optuna

export TuningError,
    FixedTrial,
    TuningStudy,
    TuningBackendLimitations,
    is_flat_list,
    is_categorical_suggestion,
    is_float_suggestion,
    is_int_suggestion,
    get_value_of_ref,
    convert_to_suggestion,
    get_suggestion,
    resolve_tuning_references!,
    build_search_space,
    create_tuning_study,
    preferred_tuning_backend,
    backend_neutral_tuning_concepts,
    unsupported_optuna_capabilities,
    save_best_config,
    export_best_config

"""
    TuningError(operation::String, field::String, message::String)

User-facing exception for tuning search-space and best-config failures.
"""
struct TuningError <: Exception
    operation::String
    field::String
    message::String
end

function Base.showerror(io::IO, err::TuningError)
    print(io, err.operation, " failed for ", err.field, ": ", err.message)
end

"""Backend-neutral trial fixture used by tuning tests and deterministic searches."""
struct FixedTrial
    params::Dict{String, Any}
end

"""Study metadata and concrete Optuna.jl study handle for tuning runs."""
struct TuningStudy
    study_name::String
    direction::Symbol
    storage::Any
    backend::Symbol
    study::Any
end

"""Explicit limitations for the currently planned Optuna.jl-backed tuning path."""
struct TuningBackendLimitations
    backend::Symbol
    limitations::Vector{String}
end

is_flat_list(value) = value isa AbstractVector && all(v -> !(v isa AbstractVector) && !(v isa AbstractDict), value)
is_categorical_suggestion(value) = value isa AbstractVector && !isempty(value) && is_flat_list(value) && all(v -> v !== nothing && (v isa Bool || v isa AbstractString || v isa Number), value)
is_float_suggestion(value) = value isa Tuple && length(value) == 3 && value[1] isa AbstractFloat && value[2] isa AbstractFloat && (value[3] isa AbstractFloat || value[3] isa Bool)
is_int_suggestion(value) = value isa Tuple && length(value) == 3 && all(v -> v isa Integer, value)

function _trial_value(trial, name)
    if trial isa FixedTrial
        haskey(trial.params, name) || throw(TuningError("suggest value", name, "missing fixed trial parameter"))
        return trial.params[name]
    elseif hasproperty(trial, :params)
        params = getproperty(trial, :params)
        haskey(params, name) || throw(TuningError("suggest value", name, "missing trial parameter"))
        return params[name]
    end
    throw(TuningError("suggest value", name, "unsupported fixed-trial object"))
end

function _suggest_categorical(trial, name::String, choices)
    if trial isa FixedTrial || hasproperty(trial, :params)
        return _trial_value(trial, name)
    end
    return Optuna.suggest_categorical(trial, name, collect(choices))
end

function _suggest_float(trial, name::String, values)
    if trial isa FixedTrial || hasproperty(trial, :params)
        return _trial_value(trial, name)
    end
    low, high, tuning = values
    if tuning isa Bool
        return Optuna.suggest_float(trial, name, Float64(low), Float64(high); log = tuning)
    end
    return Optuna.suggest_float(trial, name, Float64(low), Float64(high); step = Float64(tuning))
end

function _suggest_int(trial, name::String, values)
    if trial isa FixedTrial || hasproperty(trial, :params)
        return _trial_value(trial, name)
    end
    low, high, step = values
    return Optuna.suggest_int(trial, name, Int(low), Int(high); step = Int(step))
end

function get_value_of_ref(config, ref_path)
    current = config
    for part in ref_path
        try
            if current isa AbstractDict
                current = haskey(current, part) ? current[part] : current[String(part)]
            elseif current isa AbstractVector
                current = current[Int(part) + (part isa Integer && part == 0 ? 1 : 0)]
            else
                throw(KeyError(part))
            end
        catch
            throw(TuningError("resolve reference", join(string.(ref_path), "."), "invalid reference path"))
        end
    end
    current
end

_node_type(node) = node isa AbstractDict ? _cfg_get(node, "type", nothing) : nothing
_node_values(node, key) = node isa AbstractDict ? _cfg_get(node, key, nothing) : nothing

function convert_to_suggestion(param_name::AbstractString, node, trial, config)
    node_type = _node_type(node)
    is_sweep = node_type == "sweep" || node isa Sweep
    is_coupled_sweep = node_type == "coupled-sweep" || node isa CoupledSweep
    is_range = node_type == "range" || node isa InclusiveRange
    is_random_uniform = node_type == "random_uniform" || node isa RandomUniform

    node_values = if node isa Sweep
        node.values
    elseif node isa CoupledSweep
        node.values
    elseif node isa InclusiveRange
        (node.start, node.stop, node.step)
    elseif node isa RandomUniform
        (node.low, node.high, true)
    elseif is_range || is_random_uniform
        _node_values(node, "tune_values")
    elseif is_sweep || is_coupled_sweep
        _node_values(node, "values")
    else
        nothing
    end

    if is_sweep && is_categorical_suggestion(node_values)
        return _suggest_categorical(trial, String(param_name), node_values)
    elseif (is_range || is_random_uniform) && is_float_suggestion(node_values)
        return _suggest_float(trial, String(param_name), node_values)
    elseif (is_range || is_random_uniform) && is_int_suggestion(node_values)
        return _suggest_int(trial, String(param_name), node_values)
    elseif is_coupled_sweep
        target_path = node isa CoupledSweep ? [_node_values(node, "target")] : _node_values(node, "target")
        target_path === nothing && throw(TuningError("convert coupled sweep", String(param_name), "missing target"))
        target_node = get_value_of_ref(config, target_path)
        target_type = _node_type(target_node)
        (target_type == "sweep" || target_node isa Sweep) || throw(TuningError("convert coupled sweep", String(param_name), "target is not a sweep node"))
        target_values = target_node isa Sweep ? target_node.values : _node_values(target_node, "values")
        length(target_values) == length(node_values) || throw(TuningError("convert coupled sweep", String(param_name), "target and coupled sweep lengths do not match"))
        return Dict{String, Any}("type" => "coupled-sweep-mapping", "target" => target_path, "mapping" => Dict(zip(target_values, node_values)))
    else
        return node
    end
end

function get_suggestion(config, current_node, trial, traced_param = String[])
    if current_node isa AbstractVector
        suggestions = Vector{Any}(undef, length(current_node))
        items = collect(enumerate(current_node))
    elseif current_node isa AbstractDict
        suggestions = Dict{Any, Any}()
        items = collect(current_node)
    else
        return current_node, Dict{String, Any}()
    end

    coupled_sweep_mapping = Dict{String, Any}()
    for (param, value) in items
        push!(traced_param, string(param))
        traced_name = join(traced_param, ".")
        suggestion = convert_to_suggestion(traced_name, value, trial, config)
        if suggestion isa AbstractDict && _cfg_get(suggestion, "type", nothing) == "coupled-sweep-mapping"
            coupled_sweep_mapping[traced_name] = suggestion["mapping"]
            suggestions[param] = Dict{String, Any}("type" => "coupled-sweep-mapping", "target" => suggestion["target"])
        elseif (suggestion isa AbstractDict && _cfg_get(suggestion, "type", nothing) != "coupled-sweep-mapping") || (suggestion isa AbstractVector && !is_flat_list(suggestion))
            nested, nested_mapping = get_suggestion(config, suggestion, trial, traced_param)
            suggestions[param] = nested
            merge!(coupled_sweep_mapping, nested_mapping)
        else
            suggestions[param] = suggestion
        end
        pop!(traced_param)
    end
    suggestions, coupled_sweep_mapping
end

function _set_value_at_path!(root, path, value)
    parent = root
    for step in path[1:end-1]
        parent = parent[step]
    end
    parent[path[end]] = value
end

function resolve_tuning_references!(config, node = config, walked_path = Any[], coupled_sweep_mapping = Dict{String, Any}())
    if node isa AbstractDict && _cfg_get(node, "type", nothing) == "reference"
        _set_value_at_path!(config, walked_path, get_value_of_ref(config, _cfg_get(node, "target")))
        return config
    elseif node isa AbstractDict && _cfg_get(node, "type", nothing) == "coupled-sweep-mapping"
        target_value = get_value_of_ref(config, _cfg_get(node, "target"))
        full_path = join(string.(walked_path), ".")
        mapping = get(coupled_sweep_mapping, full_path, nothing)
        mapping === nothing && throw(TuningError("resolve coupled sweep", full_path, "no coupled sweep mapping found"))
        haskey(mapping, target_value) || throw(TuningError("resolve coupled sweep", full_path, "target value not found in coupled sweep mapping"))
        _set_value_at_path!(config, walked_path, mapping[target_value])
        return config
    elseif node isa AbstractDict
        for (key, value) in collect(node)
            resolve_tuning_references!(config, value, vcat(walked_path, [key]), coupled_sweep_mapping)
        end
    elseif node isa AbstractVector
        for (index, value) in enumerate(node)
            resolve_tuning_references!(config, value, vcat(walked_path, [index]), coupled_sweep_mapping)
        end
    end
    config
end

function build_search_space(config::AbstractDict, trial)
    search_space, coupled = get_suggestion(config, config, trial, String[])
    resolve_tuning_references!(search_space, search_space, Any[], coupled)
end

preferred_tuning_backend() = :Optuna
backend_neutral_tuning_concepts() = [:study, :trial, :suggestion, :objective_result, :best_configuration]
unsupported_optuna_capabilities() = TuningBackendLimitations(:Optuna, ["distributed or multi-machine study execution is outside the current scope"])

function _optuna_storage(storage)
    storage === nothing && return Optuna.InMemoryStorage()
    storage_text = String(storage)
    if endswith(storage_text, ".log")
        return Optuna.JournalStorage(Optuna.JournalFileBackend(storage_text))
    elseif startswith(storage_text, "sqlite://") || startswith(storage_text, "mysql://") || startswith(storage_text, "mysql+pymysql://")
        return Optuna.RDBStorage(storage_text)
    end
    throw(TuningError("create tuning study", "storage", "unsupported Optuna storage format"))
end

function create_tuning_study(config::AbstractDict)
    _cfg_has(config, "study_name") || throw(TuningError("create tuning study", "study_name", "missing study name"))
    _cfg_has(config, "direction") || throw(TuningError("create tuning study", "direction", "missing direction"))
    direction_text = String(_cfg_get(config, "direction"))
    direction = Symbol(direction_text)
    direction in (:minimize, :maximize) || throw(TuningError("create tuning study", "direction", "direction must be minimize or maximize"))
    storage_config = _cfg_get(config, "storage", nothing)
    optuna_storage = _optuna_storage(storage_config)
    artifact_path = String(_cfg_get(config, "artifact_path", mktempdir()))
    artifact_store = Optuna.FileSystemArtifactStore(artifact_path)
    study = Optuna.Study(
        String(_cfg_get(config, "study_name")),
        artifact_store,
        optuna_storage;
        direction = direction_text,
        load_if_exists = true,
        pruner = Optuna.MedianPruner(2, 5, 5),
    )
    TuningStudy(String(_cfg_get(config, "study_name")), direction, optuna_storage, preferred_tuning_backend(), study)
end

function _set_by_dotted_path!(root, dotted_path::AbstractString, value)
    parts = split(String(dotted_path), ".")
    current = root
    for part in parts[1:end-1]
        key = try
            parse(Int, part)
        catch
            part
        end
        current = current[key]
    end
    final = try
        parse(Int, parts[end])
    catch
        parts[end]
    end
    current[final] = value
    root
end

function save_best_config(config::AbstractDict, best_trial, output_file::AbstractString)
    isempty(output_file) && throw(TuningError("save best config", "output_file", "output file path must be provided"))
    search_space, coupled = get_suggestion(config, config, best_trial, String[])
    best_config = deepcopy(search_space)
    params = best_trial isa FixedTrial ? best_trial.params : getproperty(best_trial, :params)
    for (key, value) in params
        _set_by_dotted_path!(best_config, key, value)
    end
    resolve_tuning_references!(best_config, best_config, Any[], coupled)
    open(output_file, "w") do io
        show(io, MIME("text/plain"), best_config)
    end
    best_config
end

export_best_config(args...; kwargs...) = save_best_config(args...; kwargs...)
