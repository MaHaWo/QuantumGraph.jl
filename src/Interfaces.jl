export ConfigMetadata,
    register_object!,
    resolve_registered_object,
    configuration_metadata,
    reconstruct_from_metadata,
    get_config_path,
    set_config_path!,
    validate_public_value,
    PublicInterfaceError

"""User-facing error for public interface, registry, and utility validation failures."""
struct PublicInterfaceError <: Exception
    operation::String
    value::String
    message::String
end

function Base.showerror(io::IO, err::PublicInterfaceError)
    print(io, err.operation, " failed for ", err.value, ": ", err.message)
end

const _OBJECT_REGISTRY = Dict{String, Any}()

"""Register a Julia object under a stable module-qualified identifier."""
function register_object!(identifier::AbstractString, object)
    validate_public_value("register object", identifier; kind = "object identifier")
    _OBJECT_REGISTRY[String(identifier)] = object
    object
end

"""Resolve a registered Julia object by stable identifier."""
function resolve_registered_object(identifier::AbstractString)
    key = String(identifier)
    if !haskey(_OBJECT_REGISTRY, key)
        throw(PublicInterfaceError("resolve registered object", key, "unresolved module or object name"))
    end
    _OBJECT_REGISTRY[key]
end

"""Serializable public reconstruction metadata for config-created objects."""
struct ConfigMetadata
    type_identifier::String
    constructor_parameters::Dict{String, Any}
end

"""Return public configuration metadata for an object that exposes it."""
function configuration_metadata(object)
    if hasproperty(object, :config_metadata)
        metadata = getproperty(object, :config_metadata)
        metadata isa ConfigMetadata && return metadata
    end
    throw(PublicInterfaceError("configuration metadata", string(typeof(object)), "object does not expose public configuration metadata"))
end

"""Reconstruct a registered callable/object from public metadata."""
function reconstruct_from_metadata(metadata::ConfigMetadata)
    target = resolve_registered_object(metadata.type_identifier)
    target isa Function && return target(metadata.constructor_parameters)
    target
end

_path_segments(path) = path isa AbstractString ? split(path, ".") : collect(path)

function _lookup_key(mapping, segment)
    if haskey(mapping, segment)
        return segment
    elseif haskey(mapping, Symbol(segment))
        return Symbol(segment)
    else
        return nothing
    end
end

"""Read a value at a nested multi-segment configuration path."""
function get_config_path(config, path)
    current = config
    for segment in _path_segments(path)
        current isa AbstractDict || throw(PublicInterfaceError("read configuration path", string(segment), "missing intermediate path segment"))
        key = _lookup_key(current, segment)
        key === nothing && throw(PublicInterfaceError("read configuration path", string(segment), "missing intermediate path segment"))
        current = current[key]
    end
    current
end

"""Update a value at an existing nested multi-segment configuration path."""
function set_config_path!(config, path, value)
    segments = _path_segments(path)
    isempty(segments) && throw(PublicInterfaceError("update configuration path", "", "path must contain at least one segment"))
    current = config
    for segment in segments[1:end-1]
        current isa AbstractDict || throw(PublicInterfaceError("update configuration path", string(segment), "missing intermediate path segment"))
        key = _lookup_key(current, segment)
        key === nothing && throw(PublicInterfaceError("update configuration path", string(segment), "missing intermediate path segment"))
        current = current[key]
    end
    final = segments[end]
    key = _lookup_key(current, final)
    key === nothing && throw(PublicInterfaceError("update configuration path", string(final), "missing intermediate path segment"))
    current[key] = value
    config
end

"""Validate user-provided public path, field, or object identifier values."""
function validate_public_value(operation::AbstractString, value; kind::AbstractString = "value")
    text = String(value)
    if isempty(strip(text)) || occursin("..", text)
        throw(PublicInterfaceError(String(operation), text, "invalid $kind"))
    end
    text
end
