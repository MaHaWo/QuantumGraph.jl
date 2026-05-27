export ConfigMetadata,
    register_object!,
    resolve_registered_object,
    configuration_metadata,
    reconstruct_from_metadata,
    get_config_path,
    set_config_path!,
    validate_public_value,
    PublicInterfaceError

"""
    PublicInterfaceError(operation::String, value::String, message::String)

User-facing exception for public interface, registry, and utility validation failures.

`PublicInterfaceError` is used when QuantumGraph rejects user-provided configuration
paths, object identifiers, field names, or registry references. Its `showerror`
message is intentionally concise and suitable for downstream users; implementation
stack details are not part of the primary explanation.

# Arguments
- `operation::String`: Public operation being attempted, such as `"resolve registered object"`.
- `value::String`: User-provided value that caused the failure.
- `message::String`: Human-readable explanation of the failure.

# Returns
- A `PublicInterfaceError` exception value.
"""
struct PublicInterfaceError <: Exception
    operation::String
    value::String
    message::String
end

function Base.showerror(io::IO, err::PublicInterfaceError)
    print(io, err.operation, " failed for ", err.value, ": ", err.message)
end

const _OBJECT_REGISTRY = Dict{String, Any}()

"""
    register_object!(identifier::AbstractString, object)

Register a Julia object under a stable public identifier.

The identifier is intended for configuration-driven object resolution. It should
be stable across runs and module-qualified where practical, for example
`"QuantumGraph.MyComponent"`. The registered `object` may be any Julia value,
including a type, callable constructor, function, or singleton configuration
object.

# Arguments
- `identifier::AbstractString`: Stable public identifier used in configuration metadata.
- `object`: Julia object to bind to `identifier`.

# Returns
- The registered `object`.

# Throws
- `PublicInterfaceError`: If `identifier` is empty or otherwise invalid.
"""
function register_object!(identifier::AbstractString, object)
    validate_public_value("register object", identifier; kind = "object identifier")
    _OBJECT_REGISTRY[String(identifier)] = object
    object
end

"""
    resolve_registered_object(identifier::AbstractString)

Resolve a Julia object from the public object registry.

Looks up `identifier` in QuantumGraph's process-local registry and returns the
registered binding. This provides the public Julia replacement for config-driven
object lookup without exposing private implementation files.

# Arguments
- `identifier::AbstractString`: Stable identifier previously registered with [`register_object!`](@ref).

# Returns
- The Julia object registered for `identifier`.

# Throws
- `PublicInterfaceError`: If no object is registered for `identifier`.
"""
function resolve_registered_object(identifier::AbstractString)
    key = String(identifier)
    if !haskey(_OBJECT_REGISTRY, key)
        throw(PublicInterfaceError("resolve registered object", key, "unresolved module or object name"))
    end
    _OBJECT_REGISTRY[key]
end

"""
    ConfigMetadata(type_identifier::String, constructor_parameters::Dict{String, Any})

Serializable public reconstruction metadata for configuration-created objects.

`ConfigMetadata` stores only the public type identifier and constructor parameters
needed to reconstruct an equivalent object structure. Runtime-only state, learned
parameters, handles, caches, and private source paths should not be stored here.

# Arguments
- `type_identifier::String`: Stable public identifier for the object's type or constructor.
- `constructor_parameters::Dict{String, Any}`: Public constructor parameters needed for reconstruction.

# Returns
- A `ConfigMetadata` value.
"""
struct ConfigMetadata
    type_identifier::String
    constructor_parameters::Dict{String, Any}
end

"""
    configuration_metadata(object)

Return public configuration metadata for `object`.

Objects that participate in configuration round-tripping expose a
`config_metadata` property containing a [`ConfigMetadata`](@ref) value. This
function reads and validates that public metadata boundary.

# Arguments
- `object`: Public QuantumGraph object constructed from configuration metadata.

# Returns
- `ConfigMetadata`: Metadata needed to reconstruct the object's approved behavior.

# Throws
- `PublicInterfaceError`: If `object` does not expose valid public configuration metadata.
"""
function configuration_metadata(object)
    if hasproperty(object, :config_metadata)
        metadata = getproperty(object, :config_metadata)
        metadata isa ConfigMetadata && return metadata
    end
    throw(PublicInterfaceError("configuration metadata", string(typeof(object)), "object does not expose public configuration metadata"))
end

"""
    reconstruct_from_metadata(metadata::ConfigMetadata)

Reconstruct an object binding from public configuration metadata.

The metadata's `type_identifier` is resolved through the public registry. If the
registered binding is callable, it is called with `metadata.constructor_parameters`.
Otherwise, the registered binding itself is returned.

# Arguments
- `metadata::ConfigMetadata`: Public reconstruction metadata.

# Returns
- The reconstructed object, or the registered object binding when the binding is not callable.

# Throws
- `PublicInterfaceError`: If `metadata.type_identifier` is not registered.
"""
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

"""
    get_config_path(config, path)

Read a value from a nested configuration mapping.

`path` may be either a dot-separated string such as `"model.hidden"` or any
iterable of path segments such as `["model", "hidden"]`. Each intermediate value
must be an `AbstractDict`. String keys are preferred; symbol keys are also
accepted for Julia-native dictionaries.

# Arguments
- `config`: Nested configuration mapping to read from.
- `path`: Dot-separated path string or iterable of path segments.

# Returns
- The value stored at `path`.

# Throws
- `PublicInterfaceError`: If an intermediate segment is missing or is not a mapping.
"""
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

"""
    set_config_path!(config, path, value)

Update a value in an existing nested configuration mapping.

`set_config_path!` mutates `config` in place. The full path must already exist;
this function does not create missing intermediate dictionaries. String keys are
preferred, and symbol keys are accepted for Julia-native dictionaries.

# Arguments
- `config`: Nested configuration mapping to update.
- `path`: Dot-separated path string or iterable of path segments.
- `value`: New value to store at `path`.

# Returns
- The mutated `config` mapping.

# Throws
- `PublicInterfaceError`: If `path` is empty, missing, or crosses a non-mapping intermediate value.
"""
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

"""
    validate_public_value(operation::AbstractString, value; kind::AbstractString = "value")

Validate a user-provided value used at a public API boundary.

This helper is used for public path, field, and object-identifier validation. It
returns the normalized string value when valid and throws a user-facing
[`PublicInterfaceError`](@ref) when invalid.

# Arguments
- `operation::AbstractString`: Public operation being attempted.
- `value`: User-provided value to validate; converted to `String`.

# Keywords
- `kind::AbstractString = "value"`: Human-readable category included in error messages.

# Returns
- `String`: The validated value converted to a string.

# Throws
- `PublicInterfaceError`: If `value` is empty, whitespace-only, or contains `".."`.
"""
function validate_public_value(operation::AbstractString, value; kind::AbstractString = "value")
    text = String(value)
    if isempty(strip(text)) || occursin("..", text)
        throw(PublicInterfaceError(String(operation), text, "invalid $kind"))
    end
    text
end
