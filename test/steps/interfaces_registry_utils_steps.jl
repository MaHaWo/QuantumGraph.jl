using Behavior

function qg_interfaces_import!()
    @eval import QuantumGraph
    QuantumGraph
end

struct QGInterfaceFixture
    config_metadata
end

# specs/interfaces-registry-utils.feature
# Background: And parser internals are not part of the public behavior contract
@given("parser internals are not part of the public behavior contract") do context
    context[:parser_internals_public] = false
    @expect !context[:parser_internals_public]
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup resolves registered Julia objects by stable identifiers
@given("a Julia object has been registered with a stable module-qualified identifier") do context
    qg = qg_interfaces_import!()
    context[:registered_identifier] = "QuantumGraph.InterfaceFixture"
    context[:registered_object] = (; name = "fixture")
    qg.register_object!(context[:registered_identifier], context[:registered_object])
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup resolves registered Julia objects by stable identifiers
@when("a configuration value references that identifier") do context
    qg = qg_interfaces_import!()
    context[:resolved_object] = qg.resolve_registered_object(context[:registered_identifier])
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup resolves registered Julia objects by stable identifiers
@then("QuantumGraph resolves the reference to the registered Julia object") do context
    @expect context[:resolved_object] == context[:registered_object]
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup resolves registered Julia objects by stable identifiers
@then("the resolved object can be used through QuantumGraph's public Julia API") do context
    @expect haskey(context[:resolved_object], :name)
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup resolves registered Julia objects by stable identifiers
@then("repeated lookups of the same identifier return the same public object binding") do context
    qg = qg_interfaces_import!()
    @expect qg.resolve_registered_object(context[:registered_identifier]) == context[:resolved_object]
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup rejects unknown modules or object names clearly
@given("a configuration value references an unknown module or object name") do context
    context[:unknown_identifier] = "QuantumGraph.DoesNotExist"
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup rejects unknown modules or object names clearly
@when("QuantumGraph resolves the reference through the registry") do context
    qg = qg_interfaces_import!()
    try
        context[:unknown_resolution] = qg.resolve_registered_object(context[:unknown_identifier])
    catch err
        context[:unknown_resolution_error] = err
    end
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup rejects unknown modules or object names clearly
@then("resolution fails") do context
    @expect haskey(context, :unknown_resolution_error)
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup rejects unknown modules or object names clearly
@then("the error identifies the unresolved module or object name") do context
    message = sprint(showerror, context[:unknown_resolution_error])
    @expect occursin(context[:unknown_identifier], message)
    @expect occursin(r"unresolved|object|module"i, message)
end

# specs/interfaces-registry-utils.feature
# Scenario: Registry lookup rejects unknown modules or object names clearly
@then("no placeholder object is returned as successful") do context
    @expect !haskey(context, :unknown_resolution)
end

# specs/interfaces-registry-utils.feature
# Scenario: Configurable objects expose reconstruction metadata
@given("a public QuantumGraph object was constructed from configuration metadata") do context
    qg = qg_interfaces_import!()
    metadata = qg.ConfigMetadata("QuantumGraph.InterfaceFixture", Dict{String, Any}("width" => 3))
    context[:metadata_object] = QGInterfaceFixture(metadata)
end

# specs/interfaces-registry-utils.feature
# Scenario: Configurable objects expose reconstruction metadata
@when("downstream code asks for the object's configuration metadata") do context
    qg = qg_interfaces_import!()
    context[:metadata] = qg.configuration_metadata(context[:metadata_object])
end

# specs/interfaces-registry-utils.feature
# Scenario: Configurable objects expose reconstruction metadata
@then("the metadata includes the public type identifier and constructor parameters needed for reconstruction") do context
    @expect context[:metadata].type_identifier == "QuantumGraph.InterfaceFixture"
    @expect context[:metadata].constructor_parameters["width"] == 3
end

# specs/interfaces-registry-utils.feature
# Scenario: Configurable objects expose reconstruction metadata
@then("implementation-only runtime state is omitted from the metadata") do context
    @expect !haskey(context[:metadata].constructor_parameters, "runtime_state")
end

# specs/interfaces-registry-utils.feature
# Scenario: Configurable objects expose reconstruction metadata
@then("the metadata can be serialized without requiring private source paths") do context
    text = repr(context[:metadata])
    @expect !occursin(r"/src/|\\src\\|\.jl"i, text)
end

# specs/interfaces-registry-utils.feature
# Scenario: Nested configuration paths are read and written consistently
@given("a nested configuration contains a value at a multi-segment path") do context
    context[:nested_config] = Dict("section" => Dict("subsection" => Dict("field" => "original", "sibling" => "unchanged")))
    context[:nested_path] = ["section", "subsection", "field"]
end

# specs/interfaces-registry-utils.feature
# Scenario: Nested configuration paths are read and written consistently
@when("QuantumGraph reads and updates that nested configuration path") do context
    qg = qg_interfaces_import!()
    context[:original_value] = qg.get_config_path(context[:nested_config], context[:nested_path])
    qg.set_config_path!(context[:nested_config], context[:nested_path], "updated")
    context[:updated_value] = qg.get_config_path(context[:nested_config], context[:nested_path])
    try
        qg.get_config_path(context[:nested_config], ["section", "missing", "field"])
    catch err
        context[:missing_path_error] = err
    end
end

# specs/interfaces-registry-utils.feature
# Scenario: Nested configuration paths are read and written consistently
@then("the original value can be retrieved by the same path") do context
    @expect context[:original_value] == "original"
end

# specs/interfaces-registry-utils.feature
# Scenario: Nested configuration paths are read and written consistently
@then("the updated value appears only at the requested path") do context
    @expect context[:updated_value] == "updated"
    @expect context[:nested_config]["section"]["subsection"]["sibling"] == "unchanged"
end

# specs/interfaces-registry-utils.feature
# Scenario: Nested configuration paths are read and written consistently
@then("missing intermediate path segments are reported with the missing segment name") do context
    @expect haskey(context, :missing_path_error)
    @expect occursin("missing", sprint(showerror, context[:missing_path_error]))
end

# specs/interfaces-registry-utils.feature
# Scenario: Public utility errors preserve user-facing context
@given("a public QuantumGraph utility validates a user-provided path, field name, or object identifier") do context
    context[:invalid_utility_input] = "../private"
    context[:utility_operation] = "validate config path"
end

# specs/interfaces-registry-utils.feature
# Scenario: Public utility errors preserve user-facing context
@when("public utility validation fails") do context
    qg = qg_interfaces_import!()
    try
        qg.validate_public_value(context[:utility_operation], context[:invalid_utility_input]; kind = "path")
    catch err
        context[:validation_error] = err
    end
end

# specs/interfaces-registry-utils.feature
# Scenario: Public utility errors preserve user-facing context
@then("the error message includes the invalid value") do context
    @expect occursin(context[:invalid_utility_input], sprint(showerror, context[:validation_error]))
end

# specs/interfaces-registry-utils.feature
# Scenario: Public utility errors preserve user-facing context
@then("the error message identifies the public operation being attempted") do context
    @expect occursin(context[:utility_operation], sprint(showerror, context[:validation_error]))
end

# specs/interfaces-registry-utils.feature
# Scenario: Public utility errors preserve user-facing context
@then("the error does not expose private implementation stack details as the primary explanation") do context
    message = sprint(showerror, context[:validation_error])
    @expect !occursin(r"Stacktrace|src/Interfaces.jl|MethodError"i, message)
end
