using Behavior
using Zarr
import QuantumGraph

function qg_dataset_store(builder)
    path = joinpath(mktempdir(), "dataset.zarr")
    root = Zarr.zgroup(path)
    builder(root)
    path
end

function qg_dataset_array(group, name, values; chunks = size(values))
    array = Zarr.zcreate(eltype(values), group, name, size(values)...; chunks = chunks)
    array[:] = values
    array
end

function qg_graph_store(count::Int)
    qg_dataset_store() do root
        qg_dataset_array(root, "adjacency_matrix", reshape(collect(1:(count * 4)), count, 2, 2); chunks = (1, 2, 2))
        qg_dataset_array(root, "link_matrix", reshape(collect(1:(count * 6)), count, 2, 3); chunks = (1, 2, 3))
        qg_dataset_array(root, "dimension", collect(1:count); chunks = (1,))
    end
end

# specs/datasets-graph-samples.feature
# Background: Given existing QuantumGravPy Zarr stores may contain adjacency_matrix, link_matrix, max_pathlen_future, max_pathlen_past, dimension, atomcount, num_samples, or num_causal_sets arrays
@given("existing QuantumGravPy Zarr stores may contain adjacency_matrix, link_matrix, max_pathlen_future, max_pathlen_past, dimension, atomcount, num_samples, or num_causal_sets arrays") do context
    context[:zarr_fields] = ["adjacency_matrix", "link_matrix", "max_pathlen_future", "max_pathlen_past", "dimension", "atomcount", "num_samples", "num_causal_sets"]
end

# specs/datasets-graph-samples.feature
# Background: And the graph sample boundary is a GraphNeuralNetworks.jl-compatible graph container or QuantumGraph wrapper around one
@given("the graph sample boundary is a GraphNeuralNetworks.jl-compatible graph container or QuantumGraph wrapper around one") do context
    context[:graph_boundary] = "GraphNeuralNetworks-compatible Julia graph data"
    @expect :QuantumGraphDataset in names(QuantumGraph; all = false)
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reports sample count using approved precedence
@given("a Zarr store contains multiple possible sample count indicators") do context
    context[:zarr_path] = qg_dataset_store() do root
        qg_dataset_array(root, "num_causal_sets", [7])
        qg_dataset_array(root, "num_samples", [5])
        qg_dataset_array(root, "dimension", [1, 2, 3])
        qg_dataset_array(root, "adjacency_matrix", reshape(collect(1:8), 2, 2, 2); chunks = (1, 2, 2))
    end
    context[:num_samples_path] = qg_dataset_store() do root
        qg_dataset_array(root, "num_samples", [5])
        qg_dataset_array(root, "dimension", [1, 2, 3])
        qg_dataset_array(root, "adjacency_matrix", reshape(collect(1:8), 2, 2, 2); chunks = (1, 2, 2))
    end
    context[:one_dim_path] = qg_dataset_store() do root
        qg_dataset_array(root, "dimension", [1, 2, 3])
        qg_dataset_array(root, "adjacency_matrix", reshape(collect(1:8), 2, 2, 2); chunks = (1, 2, 2))
    end
    context[:adjacency_only_path] = qg_dataset_store() do root
        qg_dataset_array(root, "adjacency_matrix", reshape(collect(1:8), 2, 2, 2); chunks = (1, 2, 2))
    end
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reports sample count using approved precedence
@when("a QuantumGraph dataset is constructed for that store") do context
    context[:dataset] = QuantumGraph.construct_dataset(context[:zarr_path])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reports sample count using approved precedence
@then("the sample count is selected using num_causal_sets before num_samples") do context
    @expect length(context[:dataset]) == 7
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reports sample count using approved precedence
@then("num_samples is selected before one-dimensional dataset inference") do context
    @expect length(QuantumGraph.construct_dataset(context[:num_samples_path])) == 5
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reports sample count using approved precedence
@then("one-dimensional dataset inference is selected before adjacency_matrix shape fallback") do context
    @expect length(QuantumGraph.construct_dataset(context[:one_dim_path])) == 3
    @expect length(QuantumGraph.construct_dataset(context[:adjacency_only_path])) == 2
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reads samples lazily by index
@given("a valid dataset references one or more Zarr stores") do context
    context[:dataset_paths] = [qg_graph_store(2), qg_graph_store(3)]
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reads samples lazily by index
@when("the dataset is constructed") do context
    context[:dataset] = QuantumGraph.construct_dataset(context[:dataset_paths])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reads samples lazily by index
@then("sample data is not eagerly materialized for every sample") do context
    @expect context[:dataset].stores[1].arrays["adjacency_matrix"] isa Zarr.ZArray
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reads samples lazily by index
@when("a valid sample index is requested") do context
    context[:sample] = QuantumGraph.read_dataset_sample(context[:dataset], 1)
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reads samples lazily by index
@then("only the requested sample is read and converted for model consumption") do context
    @expect hasproperty(context[:sample], :graph)
    @expect size(context[:sample].graph.adjacency_matrix) == (2, 2)
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset converts a valid sample to the graph boundary type
@given("a Zarr sample contains graph structure and feature arrays required by the approved data contract") do context
    context[:dataset] = QuantumGraph.construct_dataset(qg_graph_store(2))
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset converts a valid sample to the graph boundary type
@when("the sample is requested from the dataset") do context
    context[:sample] = context[:dataset][1]
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset converts a valid sample to the graph boundary type
@then("the result is compatible with GraphNeuralNetworks.jl model input") do context
    @expect hasproperty(context[:sample], :graph)
    @expect hasproperty(context[:sample].graph, :adjacency_matrix)
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset converts a valid sample to the graph boundary type
@then("graph structure, node or graph features, and task targets are available through documented fields or accessors") do context
    @expect hasproperty(context[:sample], :graph)
    @expect hasproperty(context[:sample], :features)
    @expect hasproperty(context[:sample], :targets)
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset rejects missing reader or unsupported store layout
@given("a dataset is configured with a missing reader function or unsupported Zarr layout") do context
    context[:unsupported_path] = qg_dataset_store() do root
        qg_dataset_array(root, "unexpected", [1, 2, 3])
    end
    try
        QuantumGraph.construct_dataset(context[:unsupported_path]; required_arrays = ["adjacency_matrix"])
    catch err
        context[:dataset_error] = err
    end
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset rejects missing reader or unsupported store layout
@when("the dataset attempts to read a sample") do context
    context[:read_attempted] = true
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset rejects missing reader or unsupported store layout
@then("the read fails with an error identifying the missing reader or unsupported layout") do context
    @expect haskey(context, :dataset_error)
    @expect occursin(r"unsupported layout|missing reader"i, sprint(showerror, context[:dataset_error]))
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset rejects missing reader or unsupported store layout
@then("no silently malformed graph sample is returned") do context
    @expect !haskey(context, :sample)
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset rejects out-of-range indexes
@given("a dataset contains a known number of samples") do context
    context[:dataset] = QuantumGraph.construct_dataset(qg_graph_store(3))
    context[:sample_count] = length(context[:dataset])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset rejects out-of-range indexes
@when("a caller requests an index outside the dataset bounds") do context
    context[:requested_index] = context[:sample_count] + 1
    try
        QuantumGraph.read_dataset_sample(context[:dataset], context[:requested_index])
    catch err
        context[:dataset_error] = err
    end
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset rejects out-of-range indexes
@then("the request fails with an out-of-range error") do context
    @expect haskey(context, :dataset_error)
    @expect occursin(r"out of range|bounds"i, sprint(showerror, context[:dataset_error]))
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset rejects out-of-range indexes
@then("the error identifies the requested index or valid bounds") do context
    @expect occursin(string(context[:requested_index]), sprint(showerror, context[:dataset_error]))
end

# specs/datasets-graph-samples.feature
# Scenario: Map-index behavior selects the correct backing store and local sample
@given("a dataset spans multiple backing Zarr stores") do context
    context[:dataset] = QuantumGraph.construct_dataset([qg_graph_store(2), qg_graph_store(4)])
end

# specs/datasets-graph-samples.feature
# Scenario: Map-index behavior selects the correct backing store and local sample
@when("a global sample index is requested") do context
    context[:global_index] = 3
    context[:mapped_index] = QuantumGraph.map_dataset_index(context[:dataset], context[:global_index])
end

# specs/datasets-graph-samples.feature
# Scenario: Map-index behavior selects the correct backing store and local sample
@then("QuantumGraph maps the global index to the correct backing store") do context
    @expect context[:mapped_index].store_index == 2
end

# specs/datasets-graph-samples.feature
# Scenario: Map-index behavior selects the correct backing store and local sample
@then("maps it to the correct local sample index within that store") do context
    @expect context[:mapped_index].local_index == 1
end
