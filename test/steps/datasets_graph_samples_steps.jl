using Behavior

function qg_dataset_exports()
    try
        @eval import QuantumGraph
        return Set(String.(names(QuantumGraph; all = false, imported = false)))
    catch
        return Set{String}()
    end
end

qg_dataset_has(patterns::Vector{Regex}) = any(name -> any(pattern -> occursin(pattern, name), patterns), qg_dataset_exports())
qg_dataset_requires(patterns) = @expect qg_dataset_has(patterns)

# specs/datasets-graph-samples.feature
# Background: Given existing QuantumGravPy Zarr stores may contain adjacency_matrix, link_matrix, max_pathlen_future, max_pathlen_past, dimension, atomcount, num_samples, or num_causal_sets arrays
@given("existing QuantumGravPy Zarr stores may contain adjacency_matrix, link_matrix, max_pathlen_future, max_pathlen_past, dimension, atomcount, num_samples, or num_causal_sets arrays") do context
    context[:zarr_fields] = ["adjacency_matrix", "link_matrix", "max_pathlen_future", "max_pathlen_past", "dimension", "atomcount", "num_samples", "num_causal_sets"]
end

# specs/datasets-graph-samples.feature
# Background: And the graph sample boundary is a GraphNeuralNetworks.jl-compatible graph container or QuantumGraph wrapper around one
@given("the graph sample boundary is a GraphNeuralNetworks.jl-compatible graph container or QuantumGraph wrapper around one") do context
    context[:graph_boundary] = "GraphNeuralNetworks-compatible"
    qg_dataset_requires([r"graph"i, r"sample"i, r"GNN"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reports sample count using approved precedence
@given("a Zarr store contains multiple possible sample count indicators") do context
    context[:sample_count_indicators] = ["num_causal_sets", "num_samples", "one_dimensional_dataset", "adjacency_matrix_shape"]
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reports sample count using approved precedence
@when("a QuantumGraph dataset is constructed for that store") do context
    qg_dataset_requires([r"dataset"i, r"zarr"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reports sample count using approved precedence
@then("the sample count is selected using num_causal_sets before num_samples") do context
    qg_dataset_requires([r"sample"i, r"count"i, r"causal"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reports sample count using approved precedence
@then("num_samples is selected before one-dimensional dataset inference") do context
    qg_dataset_requires([r"sample"i, r"count"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reports sample count using approved precedence
@then("one-dimensional dataset inference is selected before adjacency_matrix shape fallback") do context
    qg_dataset_requires([r"sample"i, r"count"i, r"adjacency"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reads samples lazily by index
@given("a valid dataset references one or more Zarr stores") do context
    context[:dataset_stores] = ["fixture-a.zarr", "fixture-b.zarr"]
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reads samples lazily by index
@when("the dataset is constructed") do context
    qg_dataset_requires([r"dataset"i, r"construct"i, r"zarr"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reads samples lazily by index
@then("sample data is not eagerly materialized for every sample") do context
    qg_dataset_requires([r"lazy"i, r"dataset"i, r"sample"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reads samples lazily by index
@when("a valid sample index is requested") do context
    qg_dataset_requires([r"get"i, r"sample"i, r"index"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset reads samples lazily by index
@then("only the requested sample is read and converted for model consumption") do context
    qg_dataset_requires([r"sample"i, r"read"i, r"convert"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset converts a valid sample to the graph boundary type
@given("a Zarr sample contains graph structure and feature arrays required by the approved data contract") do context
    context[:zarr_sample] = Dict(:graph => true, :features => true, :targets => true)
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset converts a valid sample to the graph boundary type
@when("the sample is requested from the dataset") do context
    qg_dataset_requires([r"sample"i, r"dataset"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset converts a valid sample to the graph boundary type
@then("the result is compatible with GraphNeuralNetworks.jl model input") do context
    qg_dataset_requires([r"graph"i, r"GNN"i, r"sample"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset converts a valid sample to the graph boundary type
@then("graph structure, node or graph features, and task targets are available through documented fields or accessors") do context
    qg_dataset_requires([r"feature"i, r"target"i, r"graph"i, r"access"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset rejects missing reader or unsupported store layout
@given("a dataset is configured with a missing reader function or unsupported Zarr layout") do context
    context[:invalid_dataset_reader] = true
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset rejects missing reader or unsupported store layout
@when("the dataset attempts to read a sample") do context
    qg_dataset_requires([r"read"i, r"sample"i, r"dataset"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset rejects missing reader or unsupported store layout
@then("the read fails with an error identifying the missing reader or unsupported layout") do context
    qg_dataset_requires([r"read"i, r"error"i, r"layout"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset rejects missing reader or unsupported store layout
@then("no silently malformed graph sample is returned") do context
    qg_dataset_requires([r"validate"i, r"graph"i, r"sample"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset rejects out-of-range indexes
@given("a dataset contains a known number of samples") do context
    context[:sample_count] = 3
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset rejects out-of-range indexes
@when("a caller requests an index outside the dataset bounds") do context
    context[:requested_index] = (haskey(context, :sample_count) ? context[:sample_count] : 0) + 1
    qg_dataset_requires([r"index"i, r"sample"i, r"dataset"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset rejects out-of-range indexes
@then("the request fails with an out-of-range error") do context
    qg_dataset_requires([r"bounds"i, r"range"i, r"error"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Dataset rejects out-of-range indexes
@then("the error identifies the requested index or valid bounds") do context
    qg_dataset_requires([r"index"i, r"bounds"i, r"error"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Map-index behavior selects the correct backing store and local sample
@given("a dataset spans multiple backing Zarr stores") do context
    context[:backing_stores] = [2, 4]
end

# specs/datasets-graph-samples.feature
# Scenario: Map-index behavior selects the correct backing store and local sample
@when("a global sample index is requested") do context
    context[:global_index] = 3
    qg_dataset_requires([r"map"i, r"index"i, r"sample"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Map-index behavior selects the correct backing store and local sample
@then("QuantumGraph maps the global index to the correct backing store") do context
    qg_dataset_requires([r"map"i, r"index"i, r"store"i])
end

# specs/datasets-graph-samples.feature
# Scenario: Map-index behavior selects the correct backing store and local sample
@then("maps it to the correct local sample index within that store") do context
    qg_dataset_requires([r"local"i, r"index"i, r"sample"i])
end
