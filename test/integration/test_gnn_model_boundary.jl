using Test
using QuantumGraph
using GraphNeuralNetworks

@testset "Composite GNN model integration contract" begin
    base_config = Dict{String, Any}(
        "input_dim" => 4,
        "embedding_dim" => 3,
        "embedding_path" => "pooling",
        "task_heads" => Dict{String, Any}(
            "mass" => Dict{String, Any}("output_dim" => 1),
            "dimension" => Dict{String, Any}("output_dim" => 2),
        ),
        "active_tasks" => ["mass"],
    )

    missing_err = try
        construct_gnn_model(Dict("input_dim" => 4, "embedding_dim" => 3, "task_heads" => String[]))
        nothing
    catch caught
        caught
    end
    @test missing_err isa GNNModelError
    @test occursin("missing downstream task configuration", sprint(showerror, missing_err))

    model = construct_gnn_model(base_config)
    @test model isa CompositeGNNModel
    @test active_task_outputs(model) == [:mass]
    @test haskey(model.task_heads, :mass)
    @test haskey(model.task_heads, :dimension)
    @test gnn_model_pooling_layer(model) isa GraphNeuralNetworks.GlobalPool

    sample = (
        graph = GNNGraph([1, 2], [2, 1]),
        features = Float32[
            4.0 5.0;
            8.0 9.0;
            2.0 3.0;
            1.0 2.0
        ],
        targets = Dict{Symbol, Any}(),
        source = (store_index = 1, local_index = 1),
    )

    outputs = evaluate_gnn_model(model, sample)
    @test collect(keys(outputs)) == [:mass]
    @test !haskey(outputs, :dimension)
    @test size(outputs[:mass]) == (1, 1)

    outputs_again = gnn_model_outputs(model, sample)
    @test collect(keys(outputs_again)) == collect(keys(outputs))

    embedding = gnn_model_embedding(model, sample)
    @test size(embedding) == (3, 1)

    latent_model = construct_gnn_model(merge(base_config, Dict{String, Any}(
        "embedding_path" => "latent",
        "use_pooling" => false,
        "use_latent" => true,
    )))
    latent_embedding = gnn_model_embedding(latent_model, ones(Float32, 4, 2))
    @test size(latent_embedding) == (3, 2)

    incompatible_err = try
        construct_gnn_model(merge(base_config, Dict{String, Any}(
            "embedding_path" => "pooling",
            "use_pooling" => true,
            "use_latent" => true,
        )))
        nothing
    catch caught
        caught
    end
    @test incompatible_err isa GNNModelError
    @test occursin("incompatible pooling and latent", sprint(showerror, incompatible_err))

    unknown_err = try
        construct_gnn_model(merge(base_config, Dict{String, Any}("active_tasks" => ["unknown"])))
        nothing
    catch caught
        caught
    end
    @test unknown_err isa GNNModelError
    @test occursin("unknown active task", sprint(showerror, unknown_err))

    dimension_err = try
        evaluate_gnn_model(model, ones(Float32, 2, 1))
        nothing
    catch caught
        caught
    end
    @test dimension_err isa GNNModelError
    @test occursin("dimension compatibility", sprint(showerror, dimension_err))

    metadata = gnn_model_metadata(model)
    @test metadata.type_identifier == "QuantumGraph.CompositeGNNModel"
    @test metadata.constructor_parameters["task_heads"] == ["mass", "dimension"]
    @test metadata.constructor_parameters["active_tasks"] == ["mass"]
    @test configuration_metadata(model).type_identifier == "QuantumGraph.CompositeGNNModel"

    saved = save_gnn_model_metadata(model)
    loaded = load_gnn_model_metadata(saved)
    @test loaded isa CompositeGNNModel
    @test loaded.active_tasks == model.active_tasks
    @test loaded.task_key_mapping == model.task_key_mapping
    @test loaded.embedding_path == model.embedding_path
    @test collect(keys(evaluate_gnn_model(loaded, sample))) == [:mass]

    loaded_from_metadata = load_gnn_model_metadata(metadata)
    @test loaded_from_metadata.active_tasks == model.active_tasks
end
