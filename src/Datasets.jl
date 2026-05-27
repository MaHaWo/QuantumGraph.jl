export DatasetError,
    QuantumGraphDataset,
    dataset_sample_count,
    construct_dataset,
    map_dataset_index,
    read_dataset_sample

"""
    DatasetError(operation::String, detail::String, message::String)

User-facing exception for dataset construction, indexing, and sample conversion failures.
"""
struct DatasetError <: Exception
    operation::String
    detail::String
    message::String
end

function Base.showerror(io::IO, err::DatasetError)
    print(io, err.operation, " failed for ", err.detail, ": ", err.message)
end

"""
    QuantumGraphDataset(stores, counts, offsets, reader)

Lazy dataset spanning one or more Zarr-backed stores.

The dataset keeps lazy store handles and reads sample data only when an index is
requested. Samples are returned as Julia named tuples with graph structure,
features, targets, and source metadata fields suitable for later conversion to
GraphNeuralNetworks.jl model inputs.
"""
struct QuantumGraphDataset
    stores::Vector{LazyZarrStore}
    counts::Vector{Int}
    offsets::Vector{Int}
    reader::Function
end

Base.length(dataset::QuantumGraphDataset) = isempty(dataset.counts) ? 0 : sum(dataset.counts)
Base.getindex(dataset::QuantumGraphDataset, index::Integer) = read_dataset_sample(dataset, index)

function _array_scalar(value)
    if value isa Zarr.ZArray
        data = Array(value)
        return isempty(data) ? 0 : Int(first(data))
    elseif value isa AbstractArray
        return isempty(value) ? 0 : Int(first(value))
    else
        return Int(value)
    end
end

function _array_first_dim(value)
    if value isa Zarr.ZArray || value isa AbstractArray
        dims = size(value)
        return isempty(dims) ? nothing : first(dims)
    end
    return nothing
end

"""
    dataset_sample_count(store::LazyZarrStore)
    dataset_sample_count(arrays::AbstractDict)

Return the sample count using QuantumGraph's approved precedence.

Precedence is `num_causal_sets`, then `num_samples`, then one-dimensional dataset
inference, then `adjacency_matrix` first-dimension fallback.
"""
dataset_sample_count(store::LazyZarrStore) = dataset_sample_count(store.arrays)

function dataset_sample_count(arrays::AbstractDict)
    haskey(arrays, "num_causal_sets") && return _array_scalar(arrays["num_causal_sets"])
    haskey(arrays, "num_samples") && return _array_scalar(arrays["num_samples"])

    for (name, value) in arrays
        name in ("num_causal_sets", "num_samples") && continue
        dims = value isa Zarr.ZArray || value isa AbstractArray ? size(value) : ()
        length(dims) == 1 && return first(dims)
    end

    if haskey(arrays, "adjacency_matrix")
        dim = _array_first_dim(arrays["adjacency_matrix"])
        dim === nothing || return dim
    end

    throw(DatasetError("count dataset samples", "store layout", "unsupported layout: no sample count indicator"))
end

function _offsets(counts::Vector{Int})
    offsets = Int[]
    total = 0
    for count in counts
        push!(offsets, total)
        total += count
    end
    offsets
end

"""
    construct_dataset(paths; reader = read_dataset_sample, required_arrays = String[])

Construct a lazy dataset from one or more Zarr store paths.

# Arguments
- `paths`: A store path or collection of store paths.

# Keywords
- `reader`: Sample reader function. Must not be `nothing`.
- `required_arrays`: Array names required by the dataset layout.

# Returns
- `QuantumGraphDataset`: Lazy dataset over the provided stores.
"""
function construct_dataset(paths; reader = read_dataset_sample, required_arrays = String[])
    reader === nothing && throw(DatasetError("construct dataset", "reader", "missing reader function"))
    path_list = paths isa AbstractString ? [paths] : collect(paths)
    stores = LazyZarrStore[]
    counts = Int[]
    try
        for path in path_list
            store = open_zarr_for_dataset(path; required_arrays = required_arrays)
            push!(stores, store)
            push!(counts, dataset_sample_count(store))
        end
    catch err
        err isa DatasetError && rethrow()
        err isa ZarrLoadingError && throw(DatasetError("construct dataset", string(paths), sprint(showerror, err)))
        rethrow()
    end
    QuantumGraphDataset(stores, counts, _offsets(counts), reader)
end

"""
    map_dataset_index(dataset::QuantumGraphDataset, index::Integer)

Map a one-based global sample index to a backing store and local sample index.

# Returns
- A named tuple `(store_index, local_index, store)`.
"""
function map_dataset_index(dataset::QuantumGraphDataset, index::Integer)
    total = length(dataset)
    (1 <= index <= total) || throw(DatasetError("map dataset index", string(index), "sample index out of range 1:$total"))
    for i in eachindex(dataset.stores)
        first_index = dataset.offsets[i] + 1
        last_index = dataset.offsets[i] + dataset.counts[i]
        if first_index <= index <= last_index
            return (store_index = i, local_index = index - dataset.offsets[i], store = dataset.stores[i])
        end
    end
    throw(DatasetError("map dataset index", string(index), "sample index out of range 1:$total"))
end

const _GRAPH_ARRAYS = ["adjacency_matrix", "link_matrix"]
const _FEATURE_ARRAYS = ["dimension", "atomcount", "max_pathlen_future", "max_pathlen_past"]

function _available(store::LazyZarrStore, names)
    [name for name in names if haskey(store.arrays, name)]
end

function _default_read_sample(dataset::QuantumGraphDataset, index::Integer)
    mapped = map_dataset_index(dataset, index)
    store = mapped.store
    names = vcat(_available(store, _GRAPH_ARRAYS), _available(store, _FEATURE_ARRAYS))
    isempty(names) && throw(DatasetError("read dataset sample", string(index), "unsupported layout: no graph or feature arrays"))
    values = read_zarr_sample(store, names, mapped.local_index)
    graph = NamedTuple(Symbol(name) => values[name] for name in _available(store, _GRAPH_ARRAYS))
    features = NamedTuple(Symbol(name) => values[name] for name in _available(store, _FEATURE_ARRAYS))
    targets = Dict{Symbol, Any}()
    return (graph = graph, features = features, targets = targets, source = (store_index = mapped.store_index, local_index = mapped.local_index))
end

"""
    read_dataset_sample(dataset::QuantumGraphDataset, index::Integer)

Read and convert one dataset sample for model consumption.

The returned value exposes documented fields/accessors: `graph`, `features`,
`targets`, and `source`. Graph data is represented with Julia containers at this
layer; no eager materialization of unrelated samples occurs.
"""
function read_dataset_sample(dataset::QuantumGraphDataset, index::Integer)
    try
        dataset.reader === read_dataset_sample && return _default_read_sample(dataset, index)
        return dataset.reader(dataset, index)
    catch err
        err isa DatasetError && rethrow()
        err isa ZarrLoadingError && throw(DatasetError("read dataset sample", string(index), sprint(showerror, err)))
        throw(DatasetError("read dataset sample", string(index), sprint(showerror, err)))
    end
end
