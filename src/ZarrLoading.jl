using Zarr

export ZarrLoadingError,
    LazyZarrStore,
    open_zarr_store,
    recursive_load_zarr_store,
    validate_dataset_zarr_store,
    open_zarr_for_dataset,
    read_zarr_sample

"""
    ZarrLoadingError(operation::String, path::String, message::String)

User-facing exception for Zarr loading and dataset-boundary failures.

# Arguments
- `operation::String`: Public Zarr operation being attempted.
- `path::String`: Store path or Zarr entry path associated with the failure.
- `message::String`: Human-readable explanation.

# Returns
- A `ZarrLoadingError` exception value.
"""
struct ZarrLoadingError <: Exception
    operation::String
    path::String
    message::String
end

function Base.showerror(io::IO, err::ZarrLoadingError)
    print(io, err.operation, " failed for ", err.path, ": ", err.message)
end

"""
    LazyZarrStore(path::String, root::Zarr.ZGroup, arrays::Dict{String, Any})

Lazy handles for a Zarr store opened for dataset use.

`LazyZarrStore` keeps Zarr array handles instead of materialized array values so
sample-indexed data can be read on demand.

# Arguments
- `path::String`: Filesystem path to the Zarr store.
- `root::Zarr.ZGroup`: Open root Zarr group.
- `arrays::Dict{String, Any}`: Mapping from public array names to lazy Zarr array handles.

# Returns
- A `LazyZarrStore` value.
"""
struct LazyZarrStore
    path::String
    root::Zarr.ZGroup
    arrays::Dict{String, Any}
end

function _zarr_entry_path(parent::AbstractString, name::AbstractString)
    isempty(parent) ? String(name) : string(parent, "/", name)
end

function _as_array_value(array)
    try
        return Array(array)
    catch err
        throw(ZarrLoadingError("read Zarr array", string(getproperty(array, :path)), sprint(showerror, err)))
    end
end

"""
    open_zarr_store(path::AbstractString)

Open an existing Zarr store through Zarr.jl.

# Arguments
- `path::AbstractString`: Filesystem path to an existing Zarr store.

# Returns
- `Zarr.ZGroup`: Open root group for the store.

# Throws
- `ZarrLoadingError`: If `path` does not exist or cannot be opened as a Zarr store.
"""
function open_zarr_store(path::AbstractString)
    store_path = String(path)
    ispath(store_path) || throw(ZarrLoadingError("open Zarr store", store_path, "missing store path"))
    try
        root = Zarr.zopen(store_path)
        root isa Zarr.ZGroup || throw(ZarrLoadingError("open Zarr store", store_path, "expected a Zarr group at store root"))
        return root
    catch err
        err isa ZarrLoadingError && rethrow()
        throw(ZarrLoadingError("open Zarr store", store_path, sprint(showerror, err)))
    end
end

function _recursive_load(group::Zarr.ZGroup)
    result = Dict{String, Any}()
    for (name, subgroup) in group.groups
        result[name] = _recursive_load(subgroup)
    end
    for (name, array) in group.arrays
        result[name] = _as_array_value(array)
    end
    result
end

"""
    recursive_load_zarr_store(path::AbstractString)
    recursive_load_zarr_store(group::Zarr.ZGroup)

Recursively load a Zarr group into nested Julia mappings and arrays.

Groups become `Dict{String, Any}` values, array leaves become Julia arrays, and
empty groups become empty dictionaries.

# Arguments
- `path::AbstractString`: Filesystem path to a Zarr store.
- `group::Zarr.ZGroup`: Already opened Zarr group.

# Returns
- `Dict{String, Any}`: Nested mapping mirroring the Zarr group/array structure.

# Throws
- `ZarrLoadingError`: If the store cannot be opened or an array cannot be read.
"""
recursive_load_zarr_store(path::AbstractString) = _recursive_load(open_zarr_store(path))
recursive_load_zarr_store(group::Zarr.ZGroup) = _recursive_load(group)

function _collect_arrays!(out::Dict{String, Any}, group::Zarr.ZGroup, prefix::String = "")
    for (name, array) in group.arrays
        out[_zarr_entry_path(prefix, name)] = array
        out[name] = array
    end
    for (name, subgroup) in group.groups
        _collect_arrays!(out, subgroup, _zarr_entry_path(prefix, name))
    end
    out
end

"""
    validate_dataset_zarr_store(path::AbstractString; required_arrays = String[])

Validate that a Zarr store exposes arrays required by the dataset boundary.

# Arguments
- `path::AbstractString`: Filesystem path to a Zarr store.

# Keywords
- `required_arrays = String[]`: Array names or paths that must be present.

# Returns
- `Zarr.ZGroup`: Open root group when validation succeeds.

# Throws
- `ZarrLoadingError`: If required arrays are missing or the layout is unsupported.
"""
function validate_dataset_zarr_store(path::AbstractString; required_arrays = String[])
    root = open_zarr_store(path)
    arrays = _collect_arrays!(Dict{String, Any}(), root)
    for name in required_arrays
        if !haskey(arrays, name)
            throw(ZarrLoadingError("validate Zarr dataset layout", String(path), "unsupported layout: unexpected or missing group or array entry '$name'"))
        end
    end
    root
end

"""
    open_zarr_for_dataset(path::AbstractString; required_arrays = String[])

Open a Zarr store for lazy dataset access.

The returned [`LazyZarrStore`](@ref) contains array handles, not materialized array
values. Use [`read_zarr_sample`](@ref) to read sample-indexed values.

# Arguments
- `path::AbstractString`: Filesystem path to a Zarr store.

# Keywords
- `required_arrays = String[]`: Array names or paths that must be present.

# Returns
- `LazyZarrStore`: Lazy store wrapper containing Zarr array handles.

# Throws
- `ZarrLoadingError`: If the store is missing or does not expose required arrays.
"""
function open_zarr_for_dataset(path::AbstractString; required_arrays = String[])
    root = validate_dataset_zarr_store(path; required_arrays = required_arrays)
    arrays = _collect_arrays!(Dict{String, Any}(), root)
    LazyZarrStore(String(path), root, arrays)
end

"""
    read_zarr_sample(store::LazyZarrStore, array_names, index::Integer)

Read selected arrays for one sample index from a lazy Zarr store.

For arrays with at least one dimension, `index` is applied to the first dimension
and all remaining dimensions are read for that sample. Scalar arrays are returned
as scalar values.

# Arguments
- `store::LazyZarrStore`: Lazy Zarr store returned by [`open_zarr_for_dataset`](@ref).
- `array_names`: Iterable of array names or paths to read.
- `index::Integer`: One-based sample index.

# Returns
- `Dict{String, Any}`: Mapping from requested array name to sample value.

# Throws
- `ZarrLoadingError`: If an array is missing, `index` is out of bounds, or a read fails.
"""
function read_zarr_sample(store::LazyZarrStore, array_names, index::Integer)
    index >= 1 || throw(ZarrLoadingError("read Zarr sample", store.path, "sample index out of bounds: $index"))
    result = Dict{String, Any}()
    for name in array_names
        key = String(name)
        haskey(store.arrays, key) || throw(ZarrLoadingError("read Zarr sample", store.path, "missing array '$key'"))
        array = store.arrays[key]
        try
            dims = size(array)
            if isempty(dims)
                result[key] = array[]
            elseif index > first(dims)
                throw(ZarrLoadingError("read Zarr sample", key, "sample index out of bounds: $index"))
            else
                selectors = ntuple(i -> i == 1 ? index : Colon(), length(dims))
                result[key] = array[selectors...]
            end
        catch err
            err isa ZarrLoadingError && rethrow()
            throw(ZarrLoadingError("read Zarr sample", key, sprint(showerror, err)))
        end
    end
    result
end
