# These Behavior.jl step definitions back specs/model-components.feature. They
# test reusable component behavior through exported capability checks, scenario
# fixtures for block construction/application, residuals, activations, and errors.
using Behavior
import QuantumGraph

function qg_valid_block_config(; input_dim = 3, output_dim = 2, residual = false, projection = "none")
    Dict{String, Any}(
        "input_dim" => input_dim,
        "output_dim" => output_dim,
        "activation" => "relu",
        "graph_operator_role" => "node_update",
        "residual" => residual,
        "projection" => projection,
    )
end

# specs/model-components.feature
# Background: Given exact Flux.jl or GraphNeuralNetworks.jl layer choices are implementation details
@given("exact Flux.jl or GraphNeuralNetworks.jl layer choices are implementation details") do context
    context[:graph_neural_networks_deferred] = true
    @expect context[:graph_neural_networks_deferred]
end

# specs/model-components.feature
# Background: And model component behavior is specified through public construction, shape, and metadata contracts
@given("model component behavior is specified through public construction, shape, and metadata contracts") do context
    @expect :construct_model_component in names(QuantumGraph; all = false)
    @expect :model_component_metadata in names(QuantumGraph; all = false)
end

# specs/model-components.feature
# Scenario: A configurable component validates required construction fields
@given("a model component configuration omits a required dimension, activation, or graph operator field") do context
    context[:invalid_component_config] = Dict{String, Any}(
        "input_dim" => 3,
        "activation" => "relu",
        "graph_operator_role" => "node_update",
    )
end

# specs/model-components.feature
# Scenario: A configurable component validates required construction fields
@when("QuantumGraph constructs the component from configuration") do context
    try
        config = haskey(context, :valid_component_config) ? context[:valid_component_config] : context[:invalid_component_config]
        context[:component] = QuantumGraph.construct_model_component(config)
    catch err
        context[:component_error] = err
    end
end

# specs/model-components.feature
# Scenario: A configurable component validates required construction fields
@then("component construction fails") do context
    @expect haskey(context, :component_error)
end

# specs/model-components.feature
# Scenario: A configurable component validates required construction fields
@then("the error identifies the missing configuration field") do context
    @expect occursin("output_dim", sprint(showerror, context[:component_error]))
end

# specs/model-components.feature
# Scenario: A configurable component validates required construction fields
@then("no partially constructed component is returned as successful") do context
    @expect !haskey(context, :component)
end

# specs/model-components.feature
# Scenario: A reusable block preserves batch and feature structure
@given("a reusable model block is constructed from valid configuration") do context
    context[:valid_component_config] = qg_valid_block_config(input_dim = 3, output_dim = 2)
    context[:component] = QuantumGraph.construct_model_component(context[:valid_component_config])
end

# specs/model-components.feature
# Scenario: A reusable block preserves batch and feature structure
@given("compatible graph input with node or graph features is available") do context
    context[:graph_input] = (features = ones(Float32, 3, 4), batch = [:g1, :g1, :g2, :g2])
end

# specs/model-components.feature
# Scenario: A reusable block preserves batch and feature structure
# Scenario: Residual or skip connections handle matching dimensions directly
@when("the block is applied to the input") do context
    context[:block_output] = QuantumGraph.apply_model_block(context[:component], context[:graph_input])
end

# specs/model-components.feature
# Scenario: Residual or skip connections handle matching dimensions directly
@when("the block is applied to compatible graph input") do context
    context[:block_output] = QuantumGraph.apply_model_block(context[:component], context[:graph_input])
end

# specs/model-components.feature
# Scenario: A reusable block preserves batch and feature structure
@then("the output preserves the graph batch association") do context
    @expect context[:block_output].batch == context[:graph_input].batch
end

# specs/model-components.feature
# Scenario: A reusable block preserves batch and feature structure
@then("the output feature dimension matches the configured output dimension") do context
    @expect size(context[:block_output].features, 1) == context[:valid_component_config]["output_dim"]
    @expect size(context[:block_output].features, 2) == size(context[:graph_input].features, 2)
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
end

# specs/model-components.feature
# Scenario: Residual or skip connections handle matching dimensions directly
@given("the input and output feature dimensions match") do context
    context[:valid_component_config] = qg_valid_block_config(input_dim = 3, output_dim = 3, residual = true)
    context[:component] = QuantumGraph.construct_model_component(context[:valid_component_config])
    context[:graph_input] = (features = ones(Float32, 3, 2), batch = [:g1, :g2])
end

# specs/model-components.feature
# Scenario: Residual or skip connections handle matching dimensions directly
@then("the residual or skip path contributes to the block output") do context
    no_skip_config = copy(context[:valid_component_config])
    no_skip_config["residual"] = false
    no_skip_block = QuantumGraph.construct_model_component(no_skip_config)
    no_skip_output = QuantumGraph.apply_model_block(no_skip_block, context[:graph_input])
    @expect context[:block_output].features != no_skip_output.features
end

# specs/model-components.feature
# Scenario: Residual or skip connections handle matching dimensions directly
@then("no projection layer is required by the public behavior") do context
    @expect context[:component].projection === nothing
end

# specs/model-components.feature
# Scenario: Residual or skip connections reject incompatible dimensions without a projection policy
@given("the skip path is incompatible with the input and output dimensions of the skipped-over model block") do context
    context[:incompatible_component_config] = qg_valid_block_config(input_dim = 3, output_dim = 2, residual = true, projection = "none")
end

# specs/model-components.feature
# Scenario: Residual or skip connections reject incompatible dimensions without a projection policy
@given("no projection behavior is configured") do context
    context[:incompatible_component_config]["projection"] = "none"
end

# specs/model-components.feature
# Scenario: Residual or skip connections reject incompatible dimensions without a projection policy
@when("QuantumGraph constructs or applies the block") do context
    try
        context[:component] = QuantumGraph.construct_model_component(context[:incompatible_component_config])
    catch err
        context[:component_error] = err
    end
end

# specs/model-components.feature
# Scenario: Residual or skip connections reject incompatible dimensions without a projection policy
@then("the operation fails with a clear dimension compatibility error") do context
    @expect haskey(context, :component_error)
    @expect occursin(r"dimension compatibility"i, sprint(showerror, context[:component_error]))
end

# specs/model-components.feature
# Scenario: Residual or skip connections reject incompatible dimensions without a projection policy
@then("the error identifies the residual or skip connection configuration and the skipped-over block dimensions") do context
    message = sprint(showerror, context[:component_error])
    @expect occursin("residual", message)
    @expect occursin("input_dim=3", message)
    @expect occursin("output_dim=2", message)
end

# specs/model-components.feature
# Scenario: Component metadata round-trips structural configuration
@given("a reusable model component has been constructed from configuration") do context
    context[:valid_component_config] = qg_valid_block_config(input_dim = 3, output_dim = 2, residual = false)
    context[:component] = QuantumGraph.construct_model_component(context[:valid_component_config])
end

# specs/model-components.feature
# Scenario: Component metadata round-trips structural configuration
@when("the component's public metadata is saved and loaded") do context
    context[:metadata] = QuantumGraph.model_component_metadata(context[:component])
end

# specs/model-components.feature
# Scenario: Component metadata round-trips structural configuration
@then("the loaded metadata preserves the component type, dimensions, activation choice, and graph-operator role") do context
    metadata = context[:metadata]
    @expect metadata.type_identifier == "QuantumGraph.ReusableBlock"
    @expect metadata.constructor_parameters["input_dim"] == 3
    @expect metadata.constructor_parameters["output_dim"] == 2
    @expect metadata.constructor_parameters["activation"] == "relu"
    @expect metadata.constructor_parameters["graph_operator_role"] == "node_update"
end

# specs/model-components.feature
# Scenario: Component metadata round-trips structural configuration
@then("exact learned parameter values are not required for this metadata round trip") do context
    @expect !haskey(context[:metadata].constructor_parameters, "weight")
    @expect !haskey(context[:metadata].constructor_parameters, "bias")
end

# specs/model-components.feature
# Scenario: Component metadata round-trips structural configuration
@then("the metadata can be used to reconstruct an equivalent component structure") do context
    reconstructed = QuantumGraph.construct_model_component(context[:metadata].constructor_parameters)
    @expect reconstructed.input_dim == context[:component].input_dim
    @expect reconstructed.output_dim == context[:component].output_dim
    @expect reconstructed.activation == context[:component].activation
    @expect reconstructed.graph_operator_role == context[:component].graph_operator_role
end
