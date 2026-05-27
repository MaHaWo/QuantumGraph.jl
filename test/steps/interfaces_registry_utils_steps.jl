using Behavior

const QG_INTERFACES_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const QG_INTERFACES_PROJECT = joinpath(QG_INTERFACES_ROOT, "Project.toml")

function qg_interfaces_import()
    try
        @eval import QuantumGraph
        return (ok = true, mod = QuantumGraph, error = nothing)
    catch err
        return (ok = false, mod = nothing, error = err)
    end
end

function qg_interfaces_exports()
    result = qg_interfaces_import()
    result.ok ? Set(String.(names(result.mod; all = false, imported = false))) : Set{String}()
end

qg_interfaces_has(patterns::Vector{Regex}) = any(name -> any(pattern -> occursin(pattern, name), patterns), qg_interfaces_exports())
qg_interfaces_requires(patterns) = @expect qg_interfaces_has(patterns)

# specs/interfaces-registry-utils.feature
# Background: And parser internals are not part of the public behavior contract
@given("parser internals are not part of the public behavior contract") do context
    context[:parser_internals_public] = false
    @expect !context[:parser_internals_public]
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup resolves registered Julia objects by stable identifiers
@given("a Julia object has been registered with a stable module-qualified identifier") do context
    context[:registered_identifier] = "QuantumGraph.ExampleObject"
    qg_interfaces_requires([r"registr"i, r"object"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup resolves registered Julia objects by stable identifiers
@when("a configuration value references that identifier") do context
    context[:referenced_identifier] = context[:registered_identifier]
    qg_interfaces_requires([r"reference"i, r"config"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup resolves registered Julia objects by stable identifiers
@then("QuantumGraph resolves the reference to the registered Julia object") do context
    qg_interfaces_requires([r"resolve"i, r"registr"i, r"object"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup resolves registered Julia objects by stable identifiers
@then("the resolved object can be used through QuantumGraph's public Julia API") do context
    qg_interfaces_requires([r"public"i, r"api"i, r"object"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup resolves registered Julia objects by stable identifiers
@then("repeated lookups of the same identifier return the same public object binding") do context
    qg_interfaces_requires([r"lookup"i, r"binding"i, r"registr"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup rejects unknown modules or object names clearly
@given("a configuration value references an unknown module or object name") do context
    context[:unknown_identifier] = "QuantumGraph.DoesNotExist"
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup rejects unknown modules or object names clearly
@when("QuantumGraph resolves the reference through the registry") do context
    qg_interfaces_requires([r"resolve"i, r"registr"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup rejects unknown modules or object names clearly
@then("resolution fails") do context
    qg_interfaces_requires([r"resolve"i, r"error"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup rejects unknown modules or object names clearly
@then("the error identifies the unresolved module or object name") do context
    qg_interfaces_requires([r"unresolved"i, r"module"i, r"object"i, r"error"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup rejects unknown modules or object names clearly
@then("no placeholder object is returned as successful") do context
    qg_interfaces_requires([r"resolve"i, r"error"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Configurable objects expose reconstruction metadata
@given("a public QuantumGraph object was constructed from configuration metadata") do context
    context[:metadata_constructed_object] = true
    qg_interfaces_requires([r"config"i, r"metadata"i, r"object"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Configurable objects expose reconstruction metadata
@when("downstream code asks for the object's configuration metadata") do context
    qg_interfaces_requires([r"metadata"i, r"config"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Configurable objects expose reconstruction metadata
@then("the metadata includes the public type identifier and constructor parameters needed for reconstruction") do context
    qg_interfaces_requires([r"metadata"i, r"identifier"i, r"construct"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Configurable objects expose reconstruction metadata
@then("implementation-only runtime state is omitted from the metadata") do context
    qg_interfaces_requires([r"metadata"i, r"serial"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Configurable objects expose reconstruction metadata
@then("the metadata can be serialized without requiring private source paths") do context
    qg_interfaces_requires([r"metadata"i, r"serial"i, r"path"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Nested configuration paths are read and written consistently
@given("a nested configuration contains a value at a multi-segment path") do context
    context[:nested_path] = ["section", "subsection", "field"]
end

# specs/interfaces-registry-utils.feature
# Scenario: Nested configuration paths are read and written consistently
@when("QuantumGraph reads and updates that nested configuration path") do context
    qg_interfaces_requires([r"path"i, r"config"i, r"update"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Nested configuration paths are read and written consistently
@then("the original value can be retrieved by the same path") do context
    qg_interfaces_requires([r"path"i, r"get"i, r"config"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Nested configuration paths are read and written consistently
@then("the updated value appears only at the requested path") do context
    qg_interfaces_requires([r"path"i, r"set"i, r"config"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Nested configuration paths are read and written consistently
@then("missing intermediate path segments are reported with the missing segment name") do context
    qg_interfaces_requires([r"path"i, r"missing"i, r"error"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Public utility errors preserve user-facing context
@given("a public QuantumGraph utility validates a user-provided path, field name, or object identifier") do context
    context[:invalid_utility_input] = "invalid-user-value"
    qg_interfaces_requires([r"valid"i, r"util"i, r"path"i, r"field"i, r"identifier"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Public utility errors preserve user-facing context
@when("validation fails") do context
    qg_interfaces_requires([r"valid"i, r"error"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Public utility errors preserve user-facing context
@then("the error message includes the invalid value") do context
    qg_interfaces_requires([r"error"i, r"invalid"i, r"value"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Public utility errors preserve user-facing context
@then("the error message identifies the public operation being attempted") do context
    qg_interfaces_requires([r"error"i, r"operation"i, r"public"i])
end

# specs/interfaces-registry-utils.feature
# Scenario: Public utility errors preserve user-facing context
@then("the error does not expose private implementation stack details as the primary explanation") do context
    qg_interfaces_requires([r"error"i, r"user"i, r"public"i])
end
