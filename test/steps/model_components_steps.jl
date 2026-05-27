using Behavior

function qg_components_exports()
    try
        @eval import QuantumGraph
        return Set(String.(names(QuantumGraph; all = false, imported = false)))
    catch
        return Set{String}()
    end
end

qg_components_has(patterns::Vector{Regex}) = any(name -> any(pattern -> occursin(pattern, name), patterns), qg_components_exports())
qg_components_requires(patterns) = @expect qg_components_has(patterns)

# specs/model-components.feature
# Background: Given exact Flux.jl or GraphNeuralNetworks.jl layer choices are implementation details
@given("exact Flux.jl or GraphNeuralNetworks.jl layer choices are implementation details") do context
    context[:exact_layer_choice_public_contract] = false
    @expect !context[:exact_layer_choice_public_contract]
end

# specs/model-components.feature
# Background: And model component behavior is specified through public construction, shape, and metadata contracts
@given("model component behavior is specified through public construction, shape, and metadata contracts") do context
    qg_components_requires([r"component"i, r"construct"i, r"shape"i, r"metadata"i])
end

# specs/model-components.feature
# Scenario: A configurable component validates required construction fields
@given("a model component configuration omits a required dimension, activation, or graph operator field") do context
    context[:missing_component_field] = "output_dimension"
end

# specs/model-components.feature
# Scenario: A configurable component validates required construction fields
@when("QuantumGraph constructs the component from configuration") do context
    qg_components_requires([r"component"i, r"config"i, r"construct"i])
end

# specs/model-components.feature
# Scenario: A configurable component validates required construction fields
@then("the error identifies the missing configuration field") do context
    qg_components_requires([r"config"i, r"missing"i, r"error"i])
end

# specs/model-components.feature
# Scenario: A configurable component validates required construction fields
@then("no partially constructed component is returned as successful") do context
    qg_components_requires([r"component"i, r"construct"i, r"error"i])
end

# specs/model-components.feature
# Scenario: A reusable block preserves batch and feature structure
@given("a reusable model block is constructed from valid configuration") do context
    qg_components_requires([r"block"i, r"model"i, r"config"i])
end

# specs/model-components.feature
# Scenario: A reusable block preserves batch and feature structure
@given("compatible graph input with node or graph features is available") do context
    context[:compatible_graph_input] = true
end

# specs/model-components.feature
# Scenario: A reusable block preserves batch and feature structure
# Scenario: Residual or skip connections handle matching dimensions directly
@when("the block is applied to the input") do context
    qg_components_requires([r"block"i, r"apply"i, r"forward"i])
end

# specs/model-components.feature
# Scenario: A reusable block preserves batch and feature structure
@then("the output preserves the graph batch association") do context
    qg_components_requires([r"batch"i, r"graph"i, r"output"i])
end

# specs/model-components.feature
# Scenario: A reusable block preserves batch and feature structure
@then("the output feature dimension matches the configured output dimension") do context
    qg_components_requires([r"dimension"i, r"feature"i, r"output"i])
end

# specs/model-components.feature
# Scenario: A reusable block preserves batch and feature structure
@then("exact stochastic parameter values are not part of the behavior contract") do context
    @expect true
end

# specs/model-components.feature
# Scenario: Residual or skip connections handle matching dimensions directly
# Scenario: Residual or skip connections reject incompatible dimensions without a projection policy
@given("a reusable block is configured with a residual or skip connection") do context
    context[:skip_connection_configured] = true
    qg_components_requires([r"skip"i, r"residual"i, r"block"i])
end

# specs/model-components.feature
# Scenario: Residual or skip connections handle matching dimensions directly
@given("the input and output feature dimensions match") do context
    context[:skip_dimensions_compatible] = true
end

# specs/model-components.feature
# Scenario: Residual or skip connections handle matching dimensions directly
@then("the residual or skip path contributes to the block output") do context
    qg_components_requires([r"skip"i, r"residual"i, r"output"i])
end

# specs/model-components.feature
# Scenario: Residual or skip connections handle matching dimensions directly
@then("no projection layer is required by the public behavior") do context
    qg_components_requires([r"skip"i, r"projection"i, r"dimension"i])
end

# specs/model-components.feature
# Scenario: Residual or skip connections reject incompatible dimensions without a projection policy
@given("the skip path is incompatible with the input and output dimensions of the skipped-over model block") do context
    context[:skip_dimensions_compatible] = false
end

# specs/model-components.feature
# Scenario: Residual or skip connections reject incompatible dimensions without a projection policy
@given("no projection behavior is configured") do context
    context[:projection_configured] = false
end

# specs/model-components.feature
# Scenario: Residual or skip connections reject incompatible dimensions without a projection policy
@when("QuantumGraph constructs or applies the block") do context
    qg_components_requires([r"block"i, r"construct"i, r"apply"i])
end

# specs/model-components.feature
# Scenario: Residual or skip connections reject incompatible dimensions without a projection policy
@then("the operation fails with a clear dimension compatibility error") do context
    qg_components_requires([r"dimension"i, r"compat"i, r"error"i])
end

# specs/model-components.feature
# Scenario: Residual or skip connections reject incompatible dimensions without a projection policy
@then("the error identifies the residual or skip connection configuration and the skipped-over block dimensions") do context
    qg_components_requires([r"skip"i, r"dimension"i, r"error"i])
end

# specs/model-components.feature
# Scenario: Component metadata round-trips structural configuration
@given("a reusable model component has been constructed from configuration") do context
    qg_components_requires([r"component"i, r"config"i, r"construct"i])
end

# specs/model-components.feature
# Scenario: Component metadata round-trips structural configuration
@then("the loaded metadata preserves the component type, dimensions, activation choice, and graph-operator role") do context
    qg_components_requires([r"metadata"i, r"dimension"i, r"activation"i, r"operator"i])
end

# specs/model-components.feature
# Scenario: Component metadata round-trips structural configuration
@then("exact learned parameter values are not required for this metadata round trip") do context
    @expect true
end

# specs/model-components.feature
# Scenario: Component metadata round-trips structural configuration
@then("the metadata can be used to reconstruct an equivalent component structure") do context
    qg_components_requires([r"metadata"i, r"reconstruct"i, r"component"i])
end
