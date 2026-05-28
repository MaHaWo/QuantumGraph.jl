using Test
using QuantumGraph
using CUDA
using Flux
using DataFrames
using GraphNeuralNetworks
import MLUtils

@testset "CUDA device integration contract" begin
    cpu = prepare_execution_device(nothing)
    @test cpu isa ExecutionDevice
    @test cpu.backend == :cpu
    @test cpu.index === nothing
    @test cpu.available
    @test cpu_execution_device().backend == :cpu
    @test no_distributed_device_setup()
    @test single_accelerator_process_setup()

    explicit_cpu = validate_execution_device_settings("cpu")
    @test explicit_cpu.backend == :cpu

    requested_cuda = validate_execution_device_settings("cuda:0"; availability = true)
    @test requested_cuda.backend == :cuda
    @test requested_cuda.index == 0
    @test requested_cuda.available

    unavailable_err = try
        prepare_execution_device("cuda:0"; availability = false)
        nothing
    catch caught
        caught
    end
    @test unavailable_err isa DeviceError
    @test occursin("cuda:0", sprint(showerror, unavailable_err))
    @test occursin("unavailable", sprint(showerror, unavailable_err))

    multiple_err = try
        validate_execution_device_settings(["cuda:0", "cuda:1"])
        nothing
    catch caught
        caught
    end
    @test multiple_err isa DeviceError
    @test occursin("only one accelerator", sprint(showerror, multiple_err))

    unsupported_err = try
        validate_execution_device_settings("tpu:0")
        nothing
    catch caught
        caught
    end
    @test unsupported_err isa DeviceError
    @test occursin("unsupported accelerator", sprint(showerror, unsupported_err))

    graphs = [GNNGraph([1, 2], [2, 1], ndata = (x = fill(Float32(i), 3, 2),)) for i in 1:4]
    graph_loader = dataset_dataloader(graphs; batchsize = 2, shuffle = false)
    batched_graph = first(graph_loader)
    @test batched_graph isa GNNGraph
    @test batched_graph.num_graphs == 2
    @test batched_graph.num_nodes == 4

    batch = (
        graph = batched_graph,
        features = Dict(:x => Float32[1, 2]),
        targets = (y = Float32[3],),
    )
    moved_cpu = prepare_graph_batch_for_device(batched_graph, cpu)
    @test moved_cpu === batched_graph
    prepared = prepare_model_and_graph_for_device(Flux.Chain(Flux.Dense(2 => 1)), batched_graph, cpu)
    @test prepared.graph_batch === batched_graph
    @test prepared.model isa Flux.Chain

    train_tmp = mktempdir()
    trainer = construct_trainer(Dict{String, Any}(
        "dataset" => [batch],
        "model" => x -> x,
        "optimizer" => nothing,
        "evaluator" => (model, iterator) -> DataFrames.DataFrame(loss_avg = [0.0], loss_min = [0.0], loss_max = [0.0]),
        "early_stopping" => early_stopping_state(metric = :loss_avg),
        "output_path" => train_tmp,
        "device" => "cpu",
        "num_epochs" => 1,
        "checkpoint_at" => nothing,
    ))
    @test trainer.device.backend == :cpu
    @test local_single_machine_training(trainer)

    cuda_train_err = try
        construct_trainer(Dict{String, Any}(
            "dataset" => [batch],
            "model" => x -> x,
            "optimizer" => nothing,
            "evaluator" => (model, iterator) -> DataFrames.DataFrame(loss_avg = [0.0], loss_min = [0.0], loss_max = [0.0]),
            "early_stopping" => early_stopping_state(metric = :loss_avg),
            "output_path" => mktempdir(),
            "device" => "cuda:99",
        ))
        nothing
    catch caught
        caught
    end
    @test cuda_train_err isa TrainingError
    @test occursin("device", sprint(showerror, cuda_train_err))

    if cuda_available()
        gpu_device = prepare_execution_device("cuda:0")
        gpu_array = prepare_value_for_device(Float32[1, 2, 3], gpu_device)
        @test gpu_array isa CUDA.CuArray
        gpu_batch = prepare_graph_batch_for_device(batched_graph, gpu_device)
        @test gpu_batch isa GNNGraph
        @test gpu_batch.num_graphs == batched_graph.num_graphs
        @test gpu_batch.ndata.x isa CUDA.CuArray
    else
        @test !cuda_available() # CUDA smoke assertions are gated on hardware/runtime availability.
    end
end
