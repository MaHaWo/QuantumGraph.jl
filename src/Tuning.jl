import Optuna

# Tuning mirrors QuantumGravPy's Optuna-driven search-space workflow while
# keeping the Julia-facing API explicit about the small boundary it owns:
# turning config tags into trial suggestions, resolving references/coupled
# sweeps, creating an Optuna.jl study, and exporting a resolved best config.
# Training itself stays outside this module so the trainer remains model- and
# search-backend agnostic.

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

"""
    FixedTrial(params)

Deterministic trial fixture keyed by dotted config paths such as
`"model.layers"`. This is intentionally small: production optimization should
receive an Optuna.jl `Trial`, while tests and best-config export can replay a
known parameter set without sampling.
"""
struct FixedTrial
    params::Dict{String, Any}
end

"""
    TuningStudy(study_name, direction, storage, backend, study)

Container returned by [`create_tuning_study`](@ref). It exposes backend-neutral
metadata (`study_name`, `direction`, `backend`) while retaining the concrete
Optuna.jl storage and study objects needed by callers that want to run Optuna
optimization directly.
"""
struct TuningStudy
    study_name::String
    direction::Symbol
    storage::Any
    backend::Symbol
    study::Any
end

"""
    TuningBackendLimitations(backend, limitations)

Structured declaration of behavior that QuantumGraph intentionally does not
support for a tuning backend. This keeps deferred or out-of-scope Optuna
features visible instead of silently ignoring them.
"""
struct TuningBackendLimitations
    backend::Symbol
    limitations::Vector{String}
end

"""
    is_flat_list(value) -> Bool

Return `true` for vectors whose elements are scalar from the tuning parser's
point of view. Nested vectors and dictionaries are configuration structure, not
categorical choices.
"""
is_flat_list(value) = value isa AbstractVector && all(v -> !(v isa AbstractVector) && !(v isa AbstractDict), value)

"""
    is_categorical_suggestion(value) -> Bool

Recognize Optuna categorical choices. This matches the source behavior: a
non-empty flat list of scalar values becomes one categorical suggestion.
"""
is_categorical_suggestion(value) = value isa AbstractVector && !isempty(value) && is_flat_list(value) && all(v -> v !== nothing && (v isa Bool || v isa AbstractString || v isa Number), value)

"""
    is_float_suggestion(value) -> Bool

Recognize tuple-style floating distributions `(low, high, step_or_log)`, where
the final value is either a floating step size or a `Bool` log flag.
"""
is_float_suggestion(value) = value isa Tuple && length(value) == 3 && value[1] isa AbstractFloat && value[2] isa AbstractFloat && (value[3] isa AbstractFloat || value[3] isa Bool)

"""
    is_int_suggestion(value) -> Bool

Recognize tuple-style integer distributions `(low, high, step)`.
"""
is_int_suggestion(value) = value isa Tuple && length(value) == 3 && all(v -> v isa Integer, value)

# Deterministic fixed-trial lookup is shared by tests and best-config export.
# It also accepts Optuna-like frozen trials exposing `params`, which mirrors the
# Python source path used by `save_best_config`.
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

# The suggestion helpers are the seam between QuantumGraph's config tags and
# Optuna.jl's live `Trial` API. Fixed/frozen trials bypass Optuna sampling so
# tests can be deterministic and best-trial replay can be exact.
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

"""
    get_value_of_ref(config, ref_path)

Resolve a reference path inside `config`. `ref_path` is a vector of dictionary
keys and/or array positions, matching the migrated `Reference` and
`coupled-sweep` target representation.

Throws [`TuningError`](@ref) when the path does not exist.
"""
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

"""
    convert_to_suggestion(param_name, node, trial, config)

Convert one config node into either a concrete trial value or a deferred marker.
Supported source-observable cases are:

- `sweep` / [`Sweep`](@ref): categorical Optuna suggestion.
- `range` / [`InclusiveRange`](@ref): integer or float Optuna suggestion.
- `random_uniform` / [`RandomUniform`](@ref): floating Optuna suggestion.
- `coupled-sweep` / [`CoupledSweep`](@ref): deferred mapping keyed by the target
  sweep's selected value.

Non-search nodes are returned unchanged so recursive traversal preserves normal
configuration structure.
"""
function convert_to_suggestion(param_name::AbstractString, node, trial, config)
    node_type = _node_type(node)
    is_sweep = node_type == "sweep" || node isa Sweep
    is_coupled_sweep = node_type == "coupled-sweep" || node isa CoupledSweep
    is_range = node_type == "range" || node isa InclusiveRange
    is_random_uniform = node_type == "random_uniform" || node isa RandomUniform

    # Normalize struct-backed and dict-backed config tags to a single value
    # shape before deciding which Optuna suggestion function to call.
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
        # Coupled sweeps are not sampled independently. They record a mapping
        # from the target sweep choice to this node's corresponding value and
        # are resolved after all primary suggestions have been selected.
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

"""
    get_suggestion(config, current_node, trial, traced_param = String[])

Walk a nested configuration tree and replace search-space nodes with trial
values. Returns `(suggestions, coupled_sweep_mapping)`, where
`coupled_sweep_mapping` stores the deferred target-value mappings needed by
[`resolve_tuning_references!`](@ref).

`traced_param` is the current dotted path and is mutated during recursion; pass
an empty vector for top-level calls.
"""
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
        # Dotted names are the Optuna parameter names and match Python's
        # `"model.layers"` style best-trial parameter keys.
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

# Minimal path setter used while resolving references into a copied config.
function _set_value_at_path!(root, path, value)
    parent = root
    for step in path[1:end-1]
        parent = parent[step]
    end
    parent[path[end]] = value
end

"""
    resolve_tuning_references!(config, node = config, walked_path = Any[], coupled_sweep_mapping = Dict{String, Any}())

Resolve `reference` and `coupled-sweep-mapping` placeholders in-place after
primary suggestions have been selected. This mirrors QuantumGravPy's two-phase
behavior: first sample independent values, then apply dependencies that point at
those sampled values.

Returns the mutated `config` for convenience.
"""
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

"""
    build_search_space(config, trial)

Build a concrete configuration for one trial by applying suggestions and then
resolving references/coupled sweeps. This is the function objective builders
should call before constructing datasets, models, and trainers.
"""
function build_search_space(config::AbstractDict, trial)
    search_space, coupled = get_suggestion(config, config, trial, String[])
    resolve_tuning_references!(search_space, search_space, Any[], coupled)
end

"""
    preferred_tuning_backend() -> Symbol

Return the selected tuning backend. The requirements explicitly call for
Optuna.jl, so this currently returns `:Optuna`.
"""
preferred_tuning_backend() = :Optuna

"""
    backend_neutral_tuning_concepts() -> Vector{Symbol}

List the concepts QuantumGraph exposes independently of backend details. These
terms are used by documentation and tests to keep tuning integration focused on
study/trial/suggestion/objective/best-config semantics rather than a specific
trainer implementation.
"""
backend_neutral_tuning_concepts() = [:study, :trial, :suggestion, :objective_result, :best_configuration]

"""
    unsupported_optuna_capabilities() -> TuningBackendLimitations

Report Optuna features intentionally outside this migration slice. The current
scope is local/single-machine tuning, so distributed or multi-machine study
execution remains explicitly unsupported.
"""
unsupported_optuna_capabilities() = TuningBackendLimitations(:Optuna, ["distributed or multi-machine study execution is outside the current scope"])

# Translate source-compatible storage config into concrete Optuna.jl storage.
# In-memory storage is the default; `.log`, SQLite, and MySQL-style URLs mirror
# the Python implementation's accepted formats.
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

"""
    create_tuning_study(config) -> TuningStudy

Validate tuning study settings and construct a concrete Optuna.jl `Study`.
Required keys:

- `study_name`: Optuna study name.
- `direction`: `"minimize"` or `"maximize"`.

Optional keys:

- `storage`: `nothing` for in-memory storage, a `.log` journal file, or an
  SQLite/MySQL RDB URL.
- `artifact_path`: directory used by Optuna.jl's `FileSystemArtifactStore`.

The returned [`TuningStudy`](@ref) keeps the concrete Optuna study handle in
`study` while exposing backend-neutral metadata to callers.
"""
function create_tuning_study(config::AbstractDict)
    _cfg_has(config, "study_name") || throw(TuningError("create tuning study", "study_name", "missing study name"))
    _cfg_has(config, "direction") || throw(TuningError("create tuning study", "direction", "missing direction"))
    direction_text = String(_cfg_get(config, "direction"))
    direction = Symbol(direction_text)
    direction in (:minimize, :maximize) || throw(TuningError("create tuning study", "direction", "direction must be minimize or maximize"))
    storage_config = _cfg_get(config, "storage", nothing)
    optuna_storage = _optuna_storage(storage_config)
    # Optuna.jl requires an artifact store even for studies where QuantumGraph
    # does not upload artifacts. A temporary directory keeps in-memory studies
    # self-contained unless the caller supplies a persistent artifact path.
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

# Apply Optuna best-trial params back onto the nested configuration copy.
# Numeric path components address arrays, matching the dotted keys emitted by
# recursive traversal through list-valued config sections.
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

"""
    save_best_config(config, best_trial, output_file) -> Dict

Replay a completed trial's parameters onto `config`, resolve references and
coupled sweeps, write the resulting best configuration to `output_file`, and
return the resolved configuration.

`best_trial` may be a [`FixedTrial`](@ref) or an Optuna-like object exposing a
`params` dictionary keyed by dotted config paths.
"""
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

"""
    export_best_config(args...; kwargs...)

Alias for [`save_best_config`](@ref), retained because the migration plan and
BDD language refer to exporting the best configuration.
"""
export_best_config(args...; kwargs...) = save_best_config(args...; kwargs...)
