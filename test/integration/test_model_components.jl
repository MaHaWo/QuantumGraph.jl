using Test
using QuantumGraph
using Flux

@testset "Model components integration contract" begin
    block = construct_model_component(Dict{String, Any}(
        "input_dim" => 2,
        "output_dim" => 3,
        "activation" => "identity",
        "graph_operator_role" => "node-feature-transform",
    ))
    @test block isa ReusableBlock
    @test block.input_dim == 2
    @test block.output_dim == 3
    @test block.graph_operator_role == "node-feature-transform"

    x = ones(Float32, 2, 4)
    y = apply_model_block(block, x)
    @test size(y) == (3, 4)

    sample = (features = ones(Float32, 2, 4), graph = (;))
    transformed = block(sample)
    @test hasproperty(transformed, :features)
    @test hasproperty(transformed, :graph)
    @test size(transformed.features) == (3, 4)

    residual = construct_model_component(Dict{String, Any}(
        "input_dim" => 2,
        "output_dim" => 3,
        "activation" => "identity",
        "graph_operator_role" => "residual-projection",
        "residual" => true,
        "projection" => "linear",
    ))
    @test residual.projection !== nothing
    @test size(residual(x)) == (3, 4)

    chain = Flux.Chain(block)
    @test size(chain(x)) == (3, 4)

    metadata = model_component_metadata(block)
    @test metadata.type_identifier == "QuantumGraph.ReusableBlock"
    @test metadata.constructor_parameters["input_dim"] == 2
    @test configuration_metadata(block).type_identifier == "QuantumGraph.ReusableBlock"

    custom = register_activation!("double", z -> 2 .* z)
    @test resolve_activation("double") === custom

    @test_throws ModelComponentError resolve_activation("missing-activation")
    @test_throws ModelComponentError construct_model_component(Dict("input_dim" => 2, "output_dim" => 3, "activation" => "identity"))
    @test_throws ModelComponentError construct_model_component(Dict(
        "input_dim" => 2,
        "output_dim" => 3,
        "activation" => "identity",
        "graph_operator_role" => "bad-residual",
        "residual" => true,
        "projection" => "none",
    ))
    @test_throws ModelComponentError apply_model_block(block, ones(Float32, 4, 1))
end
