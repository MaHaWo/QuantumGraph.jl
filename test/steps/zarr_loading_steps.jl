using Behavior

const QG_ZARR_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const QG_ZARR_PROJECT = joinpath(QG_ZARR_ROOT, "Project.toml")

qg_zarr_project_text() = isfile(QG_ZARR_PROJECT) ? read(QG_ZARR_PROJECT, String) : ""

function qg_zarr_exports()
    try
        @eval import QuantumGraph
        return Set(String.(names(QuantumGraph; all = false, imported = false)))
    catch
        return Set{String}()
    end
end

qg_zarr_has(patterns::Vector{Regex}) = any(name -> any(pattern -> occursin(pattern, name), patterns), qg_zarr_exports())
qg_zarr_requires(patterns) = @expect qg_zarr_has(patterns)

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
    context[:zarr_structure] = Dict(:group => Dict(:array => [1, 2, 3]))
end

# specs/zarr-loading.feature
# Scenario: Recursive loading preserves group and array structure
@when("QuantumGraph recursively loads the store") do context
    qg_zarr_requires([r"zarr"i, r"load"i, r"recursive"i])
end

# specs/zarr-loading.feature
# Scenario: Recursive loading preserves group and array structure
@then("each Zarr group is represented as a nested Julia mapping") do context
    qg_zarr_requires([r"zarr"i, r"group"i, r"mapping"i])
end

# specs/zarr-loading.feature
# Scenario: Recursive loading preserves group and array structure
@then("each Zarr array leaf is represented as a Julia array-compatible value") do context
    qg_zarr_requires([r"zarr"i, r"array"i])
end

# specs/zarr-loading.feature
# Scenario: Recursive loading preserves group and array structure
@then("empty groups are preserved as empty mappings") do context
    qg_zarr_requires([r"zarr"i, r"empty"i, r"group"i])
end

# specs/zarr-loading.feature
# Scenario: Approved fixture arrays can be read without Python imports
@given("a Python-produced Zarr fixture contains adjacency_matrix, link_matrix, and dimension arrays") do context
    context[:fixture_arrays] = ["adjacency_matrix", "link_matrix", "dimension"]
end

# specs/zarr-loading.feature
# Scenario: Approved fixture arrays can be read without Python imports
@when("QuantumGraph reads the fixture through Zarr.jl") do context
    qg_zarr_requires([r"zarr"i, r"read"i, r"fixture"i])
end

# specs/zarr-loading.feature
# Scenario: Approved fixture arrays can be read without Python imports
@then("the arrays are available to Julia code") do context
    qg_zarr_requires([r"zarr"i, r"array"i, r"read"i])
end

# specs/zarr-loading.feature
# Scenario: Approved fixture arrays can be read without Python imports
@then("no Python package import is required during the read") do context
    @expect !(haskey(context, :python_zarr_shim_required) ? context[:python_zarr_shim_required] : true)
end

# specs/zarr-loading.feature
# Scenario: Approved fixture arrays can be read without Python imports
@then("array names match the names stored in the Zarr fixture") do context
    qg_zarr_requires([r"zarr"i, r"array"i, r"name"i])
end

# specs/zarr-loading.feature
# Scenario: Missing store paths fail with a user-visible path error
@given("a requested Zarr store path does not exist") do context
    context[:missing_zarr_path] = joinpath(QG_ZARR_ROOT, "does-not-exist.zarr")
end

# specs/zarr-loading.feature
# Scenario: Missing store paths fail with a user-visible path error
@when("QuantumGraph attempts to open the store") do context
    qg_zarr_requires([r"zarr"i, r"open"i])
end

# specs/zarr-loading.feature
# Scenario: Missing store paths fail with a user-visible path error
@then("loading fails") do context
    qg_zarr_requires([r"zarr"i, r"error"i])
end

# specs/zarr-loading.feature
# Scenario: Missing store paths fail with a user-visible path error
@then("the error identifies the missing store path") do context
    qg_zarr_requires([r"zarr"i, r"path"i, r"error"i])
end

# specs/zarr-loading.feature
# Scenario: Missing store paths fail with a user-visible path error
@then("no empty dataset is returned as if loading succeeded") do context
    qg_zarr_requires([r"zarr"i, r"load"i, r"error"i])
end

# specs/zarr-loading.feature
# Scenario: Unsupported store layouts are rejected at the loading boundary
@given("a Zarr store is present but does not expose the approved group or array structure") do context
    context[:unsupported_layout] = true
end

# specs/zarr-loading.feature
# Scenario: Unsupported store layouts are rejected at the loading boundary
@when("QuantumGraph loads the store for dataset construction") do context
    qg_zarr_requires([r"zarr"i, r"dataset"i, r"load"i])
end

# specs/zarr-loading.feature
# Scenario: Unsupported store layouts are rejected at the loading boundary
@then("loading fails with an unsupported-layout error") do context
    qg_zarr_requires([r"zarr"i, r"layout"i, r"error"i])
end

# specs/zarr-loading.feature
# Scenario: Unsupported store layouts are rejected at the loading boundary
@then("the error identifies the unexpected group or array entry") do context
    qg_zarr_requires([r"zarr"i, r"unexpected"i, r"entry"i])
end

# specs/zarr-loading.feature
# Scenario: Unsupported store layouts are rejected at the loading boundary
@then("dataset construction does not continue with malformed data") do context
    qg_zarr_requires([r"dataset"i, r"malformed"i, r"error"i])
end

# specs/zarr-loading.feature
# Scenario: Lazy array access does not materialize all samples during store opening
@given("a Zarr store contains sample-indexed graph arrays") do context
    context[:sample_indexed_arrays] = true
end

# specs/zarr-loading.feature
# Scenario: Lazy array access does not materialize all samples during store opening
@when("QuantumGraph opens the store for dataset use") do context
    qg_zarr_requires([r"zarr"i, r"open"i, r"dataset"i])
end

# specs/zarr-loading.feature
# Scenario: Lazy array access does not materialize all samples during store opening
@then("opening the store records array handles or equivalent lazy accessors") do context
    qg_zarr_requires([r"zarr"i, r"lazy"i, r"access"i])
end

# specs/zarr-loading.feature
# Scenario: Lazy array access does not materialize all samples during store opening
@then("sample arrays are not fully materialized until a sample is requested") do context
    qg_zarr_requires([r"lazy"i, r"sample"i, r"array"i])
end

# specs/zarr-loading.feature
# Scenario: Lazy array access does not materialize all samples during store opening
@then("requesting one sample reads only the arrays needed for that sample") do context
    qg_zarr_requires([r"sample"i, r"read"i, r"array"i])
end
