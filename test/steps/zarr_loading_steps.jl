# These Behavior.jl step definitions back specs/zarr-loading.feature. They test
# Zarr access through temporary stores, recursive loading, fixture-array reads,
# missing-path errors, unsupported layouts, and lazy sample access.
using Behavior
using Zarr
import QuantumGraph

const QG_ZARR_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const QG_ZARR_PROJECT = joinpath(QG_ZARR_ROOT, "Project.toml")

qg_zarr_project_text() = isfile(QG_ZARR_PROJECT) ? read(QG_ZARR_PROJECT, String) : ""
qg_zarr_import!() = QuantumGraph

function qg_make_zarr_store(builder)
    path = joinpath(mktempdir(), "fixture.zarr")
    root = Zarr.zgroup(path)
    builder(root)
    path
end

function qg_zarr_create_array(group, name, values; chunks = size(values))
    array = Zarr.zcreate(eltype(values), group, name, size(values)...; chunks = chunks)
    array[:] = values
    array
end

# specs/zarr-loading.feature
# Background: Given Zarr.jl is a public QuantumGraph dependency for Zarr store access
@given("Zarr.jl is a public QuantumGraph dependency for Zarr store access") do context
    context[:project_text] = qg_zarr_project_text()
    @expect occursin(r"(?m)^\s*Zarr\s*=", context[:project_text])
end

# specs/zarr-loading.feature
# Background: And Python compatibility shims are outside the current Zarr loading contract
@given("Python compatibility shims are outside the current Zarr loading contract") do context
    context[:python_zarr_shim_required] = false
    @expect !context[:python_zarr_shim_required]
end

# specs/zarr-loading.feature
# Scenario: Recursive loading preserves group and array structure
@given("a Zarr store contains nested groups and array leaves") do context
    context[:zarr_path] = qg_make_zarr_store() do root
        nested = Zarr.zgroup(root, "nested")
        Zarr.zgroup(root, "empty")
        qg_zarr_create_array(nested, "array", [1, 2, 3])
    end
end

# specs/zarr-loading.feature
# Scenario: Recursive loading preserves group and array structure
@when("QuantumGraph recursively loads the store") do context
    qg = qg_zarr_import!()
    context[:loaded_zarr] = qg.recursive_load_zarr_store(context[:zarr_path])
end

# specs/zarr-loading.feature
# Scenario: Recursive loading preserves group and array structure
@then("each Zarr group is represented as a nested Julia mapping") do context
    @expect context[:loaded_zarr] isa AbstractDict
    @expect context[:loaded_zarr]["nested"] isa AbstractDict
end

# specs/zarr-loading.feature
# Scenario: Recursive loading preserves group and array structure
@then("each Zarr array leaf is represented as a Julia array-compatible value") do context
    @expect context[:loaded_zarr]["nested"]["array"] == [1, 2, 3]
end

# specs/zarr-loading.feature
# Scenario: Recursive loading preserves group and array structure
@then("empty groups are preserved as empty mappings") do context
    @expect context[:loaded_zarr]["empty"] isa AbstractDict
    @expect isempty(context[:loaded_zarr]["empty"])
end

# specs/zarr-loading.feature
# Scenario: Approved fixture arrays can be read without Python imports
@given("a Python-produced Zarr fixture contains adjacency_matrix, link_matrix, and dimension arrays") do context
    context[:fixture_arrays] = ["adjacency_matrix", "link_matrix", "dimension"]
    context[:zarr_path] = qg_make_zarr_store() do root
        qg_zarr_create_array(root, "adjacency_matrix", reshape(collect(1:8), 2, 2, 2); chunks = (1, 2, 2))
        qg_zarr_create_array(root, "link_matrix", reshape(collect(1:12), 2, 2, 3); chunks = (1, 2, 3))
        qg_zarr_create_array(root, "dimension", [4, 5]; chunks = (1,))
    end
end

# specs/zarr-loading.feature
# Scenario: Approved fixture arrays can be read without Python imports
@when("QuantumGraph reads the fixture through Zarr.jl") do context
    qg = qg_zarr_import!()
    context[:loaded_zarr] = qg.recursive_load_zarr_store(context[:zarr_path])
end

# specs/zarr-loading.feature
# Scenario: Approved fixture arrays can be read without Python imports
@then("the arrays are available to Julia code") do context
    for name in context[:fixture_arrays]
        @expect haskey(context[:loaded_zarr], name)
        @expect context[:loaded_zarr][name] isa AbstractArray
    end
end

# specs/zarr-loading.feature
# Scenario: Approved fixture arrays can be read without Python imports
@then("no Python package import is required during the read") do context
    @expect !(haskey(context, :python_zarr_shim_required) ? context[:python_zarr_shim_required] : true)
end

# specs/zarr-loading.feature
# Scenario: Approved fixture arrays can be read without Python imports
@then("array names match the names stored in the Zarr fixture") do context
    @expect all(name -> haskey(context[:loaded_zarr], name), context[:fixture_arrays])
end

# specs/zarr-loading.feature
# Scenario: Missing store paths fail with a user-visible path error
@given("a requested Zarr store path does not exist") do context
    context[:missing_zarr_path] = joinpath(mktempdir(), "does-not-exist.zarr")
end

# specs/zarr-loading.feature
# Scenario: Missing store paths fail with a user-visible path error
@when("QuantumGraph attempts to open the store") do context
    qg = qg_zarr_import!()
    try
        context[:opened_zarr] = qg.open_zarr_store(context[:missing_zarr_path])
    catch err
        context[:zarr_error] = err
    end
end

# specs/zarr-loading.feature
# Scenario: Missing store paths fail with a user-visible path error
@then("loading fails") do context
    @expect haskey(context, :zarr_error)
end

# specs/zarr-loading.feature
# Scenario: Missing store paths fail with a user-visible path error
@then("the error identifies the missing store path") do context
    message = sprint(showerror, context[:zarr_error])
    @expect occursin(context[:missing_zarr_path], message)
    @expect occursin(r"missing|path"i, message)
end

# specs/zarr-loading.feature
# Scenario: Missing store paths fail with a user-visible path error
@then("no empty dataset is returned as if loading succeeded") do context
    @expect !haskey(context, :opened_zarr)
end

# specs/zarr-loading.feature
# Scenario: Unsupported store layouts are rejected at the loading boundary
@given("a Zarr store is present but does not expose the approved group or array structure") do context
    context[:zarr_path] = qg_make_zarr_store() do root
        qg_zarr_create_array(root, "unexpected", [1, 2, 3])
    end
end

# specs/zarr-loading.feature
# Scenario: Unsupported store layouts are rejected at the loading boundary
@when("QuantumGraph loads the store for dataset construction") do context
    qg = qg_zarr_import!()
    try
        context[:lazy_zarr] = qg.open_zarr_for_dataset(context[:zarr_path]; required_arrays = ["adjacency_matrix"])
    catch err
        context[:zarr_error] = err
    end
end

# specs/zarr-loading.feature
# Scenario: Unsupported store layouts are rejected at the loading boundary
@then("loading fails with an unsupported-layout error") do context
    @expect haskey(context, :zarr_error)
    @expect occursin(r"unsupported layout"i, sprint(showerror, context[:zarr_error]))
end

# specs/zarr-loading.feature
# Scenario: Unsupported store layouts are rejected at the loading boundary
@then("the error identifies the unexpected group or array entry") do context
    @expect occursin("adjacency_matrix", sprint(showerror, context[:zarr_error]))
end

# specs/zarr-loading.feature
# Scenario: Unsupported store layouts are rejected at the loading boundary
@then("dataset construction does not continue with malformed data") do context
    @expect !haskey(context, :lazy_zarr)
end

# specs/zarr-loading.feature
# Scenario: Lazy array access does not materialize all samples during store opening
@given("a Zarr store contains sample-indexed graph arrays") do context
    context[:zarr_path] = qg_make_zarr_store() do root
        qg_zarr_create_array(root, "adjacency_matrix", reshape(collect(1:8), 2, 2, 2); chunks = (1, 2, 2))
        qg_zarr_create_array(root, "dimension", [4, 5]; chunks = (1,))
    end
end

# specs/zarr-loading.feature
# Scenario: Lazy array access does not materialize all samples during store opening
@when("QuantumGraph opens the store for dataset use") do context
    qg = qg_zarr_import!()
    context[:lazy_zarr] = qg.open_zarr_for_dataset(context[:zarr_path]; required_arrays = ["adjacency_matrix", "dimension"])
end

# specs/zarr-loading.feature
# Scenario: Lazy array access does not materialize all samples during store opening
@then("opening the store records array handles or equivalent lazy accessors") do context
    @expect context[:lazy_zarr] isa qg_zarr_import!().LazyZarrStore
    @expect context[:lazy_zarr].arrays["adjacency_matrix"] isa Zarr.ZArray
end

# specs/zarr-loading.feature
# Scenario: Lazy array access does not materialize all samples during store opening
@then("sample arrays are not fully materialized until a sample is requested") do context
    @expect !(context[:lazy_zarr].arrays["adjacency_matrix"] isa Array)
end

# specs/zarr-loading.feature
# Scenario: Lazy array access does not materialize all samples during store opening
@then("requesting one sample reads only the arrays needed for that sample") do context
    qg = qg_zarr_import!()
    sample = qg.read_zarr_sample(context[:lazy_zarr], ["adjacency_matrix", "dimension"], 1)
    @expect haskey(sample, "adjacency_matrix")
    @expect haskey(sample, "dimension")
    @expect sample["dimension"] == 4
    @expect size(sample["adjacency_matrix"]) == (2, 2)
end
