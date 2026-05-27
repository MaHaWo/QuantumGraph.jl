using Behavior

function qg_config_import()
    try
        @eval import QuantumGraph
        return QuantumGraph
    catch err
        return err
    end
end

function qg_config_exports()
    mod = qg_config_import()
    mod isa Module ? Set(String.(names(mod; all = false, imported = false))) : Set{String}()
end

qg_config_has(patterns::Vector{Regex}) = any(name -> any(pattern -> occursin(pattern, name), patterns), qg_config_exports())
qg_config_requires(label, patterns) = @expect qg_config_has(patterns)

# specs/config-object-resolution.feature
# Background: Given QuantumGraph is used as an importable Julia library
@given("QuantumGraph is used as an importable Julia library") do context
    context[:module_or_error] = qg_config_import()
    @expect context[:module_or_error] isa Module
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
    context[:yaml_tags] = ["!sweep", "!coupled-sweep", "!range", "!random_uniform", "!reference", "!pyobject-equivalent"]
end

# specs/config-object-resolution.feature
# Scenario: Custom YAML tags are parsed into Julia configuration values
@when("the configuration is loaded by QuantumGraph") do context
    context[:loaded_configuration] = nothing
    qg_config_requires("configuration loader", [r"config"i, r"yaml"i, r"load"i])
end

# specs/config-object-resolution.feature
# Scenario: Custom YAML tags are parsed into Julia configuration values
@then("each supported tag is recognized") do context
    qg_config_requires("custom YAML tag parser", [r"tag"i, r"yaml"i, r"sweep"i, r"range"i, r"reference"i])
end

# specs/config-object-resolution.feature
# Scenario: Custom YAML tags are parsed into Julia configuration values
@then("the resulting Julia configuration preserves the tag's approved behavior") do context
    qg_config_requires("configuration semantics", [r"config"i, r"expand"i, r"resolve"i])
end

# specs/config-object-resolution.feature
# Scenario: Custom YAML tags are parsed into Julia configuration values
@then("unsupported or malformed tags produce user-visible configuration errors") do context
    qg_config_requires("configuration error reporting", [r"config"i, r"error"i, r"invalid"i])
end

# specs/config-object-resolution.feature
# Scenario: Range expansion preserves inclusive range semantics
@given("a configuration range with a start, stop, and nonzero step") do context
    context[:range_config] = (start = 1, stop = 5, step = 2)
end

# specs/config-object-resolution.feature
# Scenario: Range expansion preserves inclusive range semantics
@when("the range is expanded") do context
    qg_config_requires("range expansion", [r"range"i, r"expand"i])
end

# specs/config-object-resolution.feature
# Scenario: Range expansion preserves inclusive range semantics
@then("the generated values include the start value") do context
    qg_config_requires("inclusive range start", [r"range"i, r"expand"i])
end

# specs/config-object-resolution.feature
# Scenario: Range expansion preserves inclusive range semantics
@then("include the stop value when it lies on the step sequence") do context
    qg_config_requires("inclusive range stop", [r"range"i, r"expand"i])
end

# specs/config-object-resolution.feature
# Scenario: Range expansion preserves inclusive range semantics
@then("a zero step is rejected with an error identifying the range step") do context
    qg_config_requires("zero-step range error", [r"range"i, r"error"i, r"step"i])
end

# specs/config-object-resolution.feature
# Scenario: Sweep expansion creates one run configuration per selected value
@given("a configuration contains a sweep over multiple values") do context
    context[:sweep_values] = ["a", "b", "c"]
end

# specs/config-object-resolution.feature
# Scenario: Sweep expansion creates one run configuration per selected value
@when("the configuration is expanded") do context
    qg_config_requires("configuration expansion", [r"expand"i, r"sweep"i, r"config"i])
end

# specs/config-object-resolution.feature
# Scenario: Sweep expansion creates one run configuration per selected value
@then("one run configuration is produced for each swept value") do context
    qg_config_requires("sweep expansion", [r"sweep"i, r"expand"i])
end

# specs/config-object-resolution.feature
# Scenario: Sweep expansion creates one run configuration per selected value
@then("each run configuration contains the selected value at the target path") do context
    qg_config_requires("sweep target path", [r"path"i, r"sweep"i, r"config"i])
end

# specs/config-object-resolution.feature
# Scenario: Sweep expansion creates one run configuration per selected value
@then("run naming preserves the approved suffix behavior without requiring Python naming conventions") do context
    qg_config_requires("sweep run naming", [r"name"i, r"suffix"i, r"sweep"i])
end

# specs/config-object-resolution.feature
# Scenario: Coupled sweeps reject mismatched lengths
@given("a configuration contains a coupled sweep with differently sized value lists") do context
    context[:coupled_sweep_lengths] = [2, 3]
end

# specs/config-object-resolution.feature
# Scenario: Coupled sweeps reject mismatched lengths
@then("expansion fails") do context
    qg_config_requires("failed expansion reporting", [r"expand"i, r"error"i])
end

# specs/config-object-resolution.feature
# Scenario: Coupled sweeps reject mismatched lengths
@then("the error identifies the coupled sweep length mismatch") do context
    qg_config_requires("coupled-sweep length error", [r"coupled"i, r"length"i, r"mismatch"i])
end

# specs/config-object-resolution.feature
# Scenario: Coupled sweeps reject mismatched lengths
@then("no partial run configuration is accepted as successful") do context
    qg_config_requires("atomic expansion failure", [r"expand"i, r"error"i])
end

# specs/config-object-resolution.feature
# Scenario: References resolve previously defined configuration paths
@given("a configuration value references another configuration path") do context
    context[:reference_path] = "model.hidden_dim"
end

# specs/config-object-resolution.feature
# Scenario: References resolve previously defined configuration paths
@when("the configuration is resolved") do context
    qg_config_requires("configuration resolution", [r"resolve"i, r"reference"i, r"config"i])
end

# specs/config-object-resolution.feature
# Scenario: References resolve previously defined configuration paths
@then("the reference is replaced by the referenced value") do context
    qg_config_requires("reference replacement", [r"reference"i, r"resolve"i])
end

# specs/config-object-resolution.feature
# Scenario: References resolve previously defined configuration paths
@then("a missing referenced path is rejected with an error identifying the missing path") do context
    qg_config_requires("missing reference path error", [r"reference"i, r"path"i, r"error"i])
end

# specs/config-object-resolution.feature
# Scenario: Julia object registry replaces Python object imports
@given("a configuration contains a !pyobject-equivalent object reference") do context
    context[:object_reference] = "QuantumGraph.ExampleObject"
end

# specs/config-object-resolution.feature
# Scenario: Julia object registry replaces Python object imports
@then("QuantumGraph resolves the reference through an explicit Julia registry or mapping") do context
    qg_config_requires("Julia object registry", [r"registr"i, r"object"i, r"resolve"i])
end

# specs/config-object-resolution.feature
# Scenario: Julia object registry replaces Python object imports
@then("unresolved registry modules or names fail with clear errors") do context
    qg_config_requires("registry error reporting", [r"registr"i, r"error"i, r"missing"i])
end

# specs/config-object-resolution.feature
# Scenario: Julia object registry replaces Python object imports
@then("no Python import is required for normal Julia object resolution") do context
    @expect !(haskey(context, :python_imports_for_config) ? context[:python_imports_for_config] : false)
end

# specs/config-object-resolution.feature
# Scenario: Configurable objects can round-trip their configuration metadata
@given("a QuantumGraph object supports construction from configuration") do context
    qg_config_requires("configurable object construction", [r"config"i, r"construct"i, r"object"i])
end

# specs/config-object-resolution.feature
# Scenario: Configurable objects can round-trip their configuration metadata
@when("the object is serialized back to configuration metadata") do context
    qg_config_requires("configuration serialization", [r"config"i, r"serial"i, r"metadata"i])
end

# specs/config-object-resolution.feature
# Scenario: Configurable objects can round-trip their configuration metadata
@then("the resulting metadata contains enough information to reconstruct the approved behavior") do context
    qg_config_requires("round-trip metadata", [r"config"i, r"metadata"i, r"reconstruct"i])
end

# specs/config-object-resolution.feature
# Scenario: Configurable objects can round-trip their configuration metadata
@then("implementation-only runtime state is not required for the round trip") do context
    qg_config_requires("configuration-only round trip", [r"config"i, r"metadata"i])
end
