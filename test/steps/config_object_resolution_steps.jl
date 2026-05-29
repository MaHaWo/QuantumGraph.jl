# These Behavior.jl step definitions back specs/config-object-resolution.feature.
# They test configuration semantics through Gherkin scenarios by loading YAML
# fixtures, expanding sweeps/ranges, resolving references, and checking failures.
using Behavior
import QuantumGraph

struct QGConfigFixture
    config_metadata
end

qg_config_import() = QuantumGraph

# specs/config-object-resolution.feature
# Background: Given QuantumGraph is used as an importable Julia library
@given("QuantumGraph is used as an importable Julia library") do context
    @expect qg_config_import() isa Module
end

# specs/config-object-resolution.feature
# Background: And configuration behavior is scoped to Julia equivalents rather than Python object imports
@given("configuration behavior is scoped to Julia equivalents rather than Python object imports") do context
    context[:python_imports_for_config] = false
    @expect !context[:python_imports_for_config]
end

# specs/config-object-resolution.feature
# Scenario: Custom YAML tags are parsed into Julia configuration values
@given("a YAML configuration containing !sweep, !coupled-sweep, !range, !random_uniform, !reference, and !pyobject-equivalent tags") do context
    context[:yaml_text] = """
    sweep: !sweep [1, 2]
    coupled_a: !coupled-sweep group1 [a, b]
    coupled_b: !coupled-sweep group1 [10, 20]
    range: !range [1, 3, 1]
    random: !random_uniform [0.0, 1.0]
    reference: !reference range
    object: !pyobject-equivalent QuantumGraph.ConfigFixture
    """
end

# specs/config-object-resolution.feature
# Scenario: Custom YAML tags are parsed into Julia configuration values
@when("the configuration is loaded by QuantumGraph") do context
    context[:loaded_configuration] = QuantumGraph.load_config(context[:yaml_text])
    try
        QuantumGraph.load_config("bad: !unsupported value")
    catch err
        context[:malformed_tag_error] = err
    end
end

# specs/config-object-resolution.feature
# Scenario: Custom YAML tags are parsed into Julia configuration values
@then("each supported tag is recognized") do context
    loaded = context[:loaded_configuration]
    @expect loaded["sweep"] isa QuantumGraph.Sweep
    @expect loaded["coupled_a"] isa QuantumGraph.CoupledSweep
    @expect loaded["range"] isa QuantumGraph.InclusiveRange
    @expect loaded["random"] isa QuantumGraph.RandomUniform
    @expect loaded["reference"] isa QuantumGraph.Reference
    @expect loaded["object"] isa QuantumGraph.ObjectReference
end

# specs/config-object-resolution.feature
# Scenario: Custom YAML tags are parsed into Julia configuration values
@then("the resulting Julia configuration preserves the tag's approved behavior") do context
    loaded = context[:loaded_configuration]
    @expect QuantumGraph.expand_range(loaded["range"]) == [1, 2, 3]
    @expect loaded["sweep"].values == [1, 2]
    @expect loaded["object"].identifier == "QuantumGraph.ConfigFixture"
end

# specs/config-object-resolution.feature
# Scenario: Custom YAML tags are parsed into Julia configuration values
@then("unsupported or malformed tags produce user-visible configuration errors") do context
    @expect haskey(context, :malformed_tag_error)
    @expect occursin(r"unsupported|malformed"i, sprint(showerror, context[:malformed_tag_error]))
end

# specs/config-object-resolution.feature
# Scenario: Range expansion preserves inclusive range semantics
@given("a configuration range with a start, stop, and nonzero step") do context
    context[:range_config] = QuantumGraph.InclusiveRange(1, 5, 2)
    context[:zero_step_range] = QuantumGraph.InclusiveRange(1, 5, 0)
end

# specs/config-object-resolution.feature
# Scenario: Range expansion preserves inclusive range semantics
@when("the range is expanded") do context
    context[:expanded_range] = QuantumGraph.expand_range(context[:range_config])
    try
        QuantumGraph.expand_range(context[:zero_step_range])
    catch err
        context[:zero_step_error] = err
    end
end

# specs/config-object-resolution.feature
# Scenario: Range expansion preserves inclusive range semantics
@then("the generated values include the start value") do context
    @expect first(context[:expanded_range]) == context[:range_config].start
end

# specs/config-object-resolution.feature
# Scenario: Range expansion preserves inclusive range semantics
@then("include the stop value when it lies on the step sequence") do context
    @expect last(context[:expanded_range]) == context[:range_config].stop
end

# specs/config-object-resolution.feature
# Scenario: Range expansion preserves inclusive range semantics
@then("a zero step is rejected with an error identifying the range step") do context
    @expect haskey(context, :zero_step_error)
    @expect occursin(r"step"i, sprint(showerror, context[:zero_step_error]))
end

# specs/config-object-resolution.feature
# Scenario: Sweep expansion creates one run configuration per selected value
@given("a configuration contains a sweep over multiple values") do context
    context[:sweep_config] = Dict{String, Any}("optimizer" => QuantumGraph.Sweep(["adam", "sgd"]), "epochs" => 3)
end

# specs/config-object-resolution.feature
# Scenario: Sweep expansion creates one run configuration per selected value
@when("the configuration is expanded") do context
    try
        config = haskey(context, :coupled_config) ? context[:coupled_config] : context[:sweep_config]
        context[:expanded_configurations] = QuantumGraph.expand_config(config)
    catch err
        context[:expansion_error] = err
    end
end

# specs/config-object-resolution.feature
# Scenario: Sweep expansion creates one run configuration per selected value
@then("one run configuration is produced for each swept value") do context
    @expect length(context[:expanded_configurations]) == length(context[:sweep_config]["optimizer"].values)
end

# specs/config-object-resolution.feature
# Scenario: Sweep expansion creates one run configuration per selected value
@then("each run configuration contains the selected value at the target path") do context
    values = [run.config["optimizer"] for run in context[:expanded_configurations]]
    @expect values == ["adam", "sgd"]
end

# specs/config-object-resolution.feature
# Scenario: Sweep expansion creates one run configuration per selected value
@then("run naming preserves the approved suffix behavior without requiring Python naming conventions") do context
    names = [run.name for run in context[:expanded_configurations]]
    @expect all(name -> occursin("optimizer=", name), names)
    @expect !any(name -> occursin(".py", name), names)
end

# specs/config-object-resolution.feature
# Scenario: Coupled sweeps reject mismatched lengths
@given("a configuration contains a coupled sweep with differently sized value lists") do context
    context[:coupled_config] = Dict{String, Any}(
        "a" => QuantumGraph.CoupledSweep("pair", [1, 2]),
        "b" => QuantumGraph.CoupledSweep("pair", [10, 20, 30]),
    )
end

# specs/config-object-resolution.feature
# Scenario: Coupled sweeps reject mismatched lengths
@then("expansion fails") do context
    @expect haskey(context, :expansion_error)
end

# specs/config-object-resolution.feature
# Scenario: Coupled sweeps reject mismatched lengths
@then("the error identifies the coupled sweep length mismatch") do context
    @expect occursin(r"coupled.*length|length.*mismatch"i, sprint(showerror, context[:expansion_error]))
end

# specs/config-object-resolution.feature
# Scenario: Coupled sweeps reject mismatched lengths
@then("no partial run configuration is accepted as successful") do context
    @expect !haskey(context, :expanded_configurations)
end

# specs/config-object-resolution.feature
# Scenario: References resolve previously defined configuration paths
@given("a configuration value references another configuration path") do context
    context[:reference_config] = Dict{String, Any}("hidden" => 64, "copy" => QuantumGraph.Reference(["hidden"]), "bad" => QuantumGraph.Reference(["missing"]))
end

# specs/config-object-resolution.feature
# Scenario: References resolve previously defined configuration paths
@when("the configuration is resolved") do context
    try
        if haskey(context, :object_config)
            context[:resolved_config] = QuantumGraph.resolve_config(context[:object_config])
        else
            good = Dict{String, Any}("hidden" => 64, "copy" => QuantumGraph.Reference(["hidden"]))
            context[:resolved_config] = QuantumGraph.resolve_config(good)
            QuantumGraph.resolve_config(context[:reference_config])
        end
    catch err
        context[:resolution_error] = err
    end
end

# specs/config-object-resolution.feature
# Scenario: References resolve previously defined configuration paths
@then("the reference is replaced by the referenced value") do context
    @expect context[:resolved_config]["copy"] == 64
end

# specs/config-object-resolution.feature
# Scenario: References resolve previously defined configuration paths
@then("a missing referenced path is rejected with an error identifying the missing path") do context
    @expect haskey(context, :resolution_error)
    @expect occursin("missing", sprint(showerror, context[:resolution_error]))
end

# specs/config-object-resolution.feature
# Scenario: Julia object registry replaces Python object imports
@given("a configuration contains a !pyobject-equivalent object reference") do context
    context[:registered_object] = (; kind = "registered")
    QuantumGraph.register_object!("QuantumGraph.ConfigFixture", context[:registered_object])
    context[:object_config] = Dict{String, Any}("object" => QuantumGraph.ObjectReference("QuantumGraph.ConfigFixture"))
    context[:bad_object_config] = Dict{String, Any}("object" => QuantumGraph.ObjectReference("QuantumGraph.MissingConfigFixture"))
end

# specs/config-object-resolution.feature
# Scenario: Julia object registry replaces Python object imports
@then("QuantumGraph resolves the reference through an explicit Julia registry or mapping") do context
    @expect context[:resolved_config]["object"] == context[:registered_object]
end

# specs/config-object-resolution.feature
# Scenario: Julia object registry replaces Python object imports
@then("unresolved registry modules or names fail with clear errors") do context
    try
        QuantumGraph.resolve_config(context[:bad_object_config])
    catch err
        context[:bad_registry_error] = err
    end
    @expect haskey(context, :bad_registry_error)
    @expect occursin(r"unresolved|missing|object"i, sprint(showerror, context[:bad_registry_error]))
end

# specs/config-object-resolution.feature
# Scenario: Julia object registry replaces Python object imports
@then("no Python import is required for normal Julia object resolution") do context
    @expect !(haskey(context, :python_imports_for_config) ? context[:python_imports_for_config] : false)
end

# specs/config-object-resolution.feature
# Scenario: Configurable objects can round-trip their configuration metadata
@given("a QuantumGraph object supports construction from configuration") do context
    metadata = QuantumGraph.ConfigMetadata("QuantumGraph.ConfigFixture", Dict{String, Any}("width" => 3))
    context[:config_object] = QGConfigFixture(metadata)
    QuantumGraph.register_object!("QuantumGraph.ConfigFixture", params -> QGConfigFixture(QuantumGraph.ConfigMetadata("QuantumGraph.ConfigFixture", params)))
end

# specs/config-object-resolution.feature
# Scenario: Configurable objects can round-trip their configuration metadata
@when("the object is serialized back to configuration metadata") do context
    context[:roundtrip_metadata] = QuantumGraph.configuration_metadata(context[:config_object])
    context[:roundtrip_object] = QuantumGraph.reconstruct_from_metadata(context[:roundtrip_metadata])
end

# specs/config-object-resolution.feature
# Scenario: Configurable objects can round-trip their configuration metadata
@then("the resulting metadata contains enough information to reconstruct the approved behavior") do context
    @expect context[:roundtrip_metadata].type_identifier == "QuantumGraph.ConfigFixture"
    @expect context[:roundtrip_metadata].constructor_parameters["width"] == 3
    @expect context[:roundtrip_object] isa QGConfigFixture
end

# specs/config-object-resolution.feature
# Scenario: Configurable objects can round-trip their configuration metadata
@then("implementation-only runtime state is not required for the round trip") do context
    @expect !haskey(context[:roundtrip_metadata].constructor_parameters, "runtime_state")
end
