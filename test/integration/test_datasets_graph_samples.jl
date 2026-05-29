# These integration tests cover dataset sizing, indexing, and graph batching.
# They exercise public dataset helpers with dictionary and empty-dataset
# fixtures, then verify GraphNeuralNetworks batches are collated as expected.
using Test
using QuantumGraph
using GraphNeuralNetworks
import MLUtils

@testset "Datasets and graph samples integration contract" begin
    @test dataset_sample_count(Dict("num_causal_sets" => [4], "num_samples" => [2])) == 4
    @test dataset_sample_count(Dict("num_samples" => [3])) == 3
    @test dataset_sample_count(Dict("dimension" => [2, 3, 4])) == 3
    @test dataset_sample_count(Dict("adjacency_matrix" => zeros(5, 2, 2))) == 5
    @test_throws DatasetError dataset_sample_count(Dict("metadata" => 1))

    empty_dataset = QuantumGraphDataset(LazyZarrStore[], Int[], Int[], read_dataset_sample)
    @test length(empty_dataset) == 0
    @test MLUtils.numobs(empty_dataset) == 0
    @test_throws DatasetError map_dataset_index(empty_dataset, 1)
    @test_throws DatasetError read_dataset_sample(empty_dataset, 1)

    @test_throws DatasetError construct_dataset(String[]; reader = nothing)

    graphs = [GNNGraph([1, 2], [2, 1], ndata = (x = fill(Float32(i), 3, 2),)) for i in 1:4]
    loader = dataset_dataloader(graphs; batchsize = 2, shuffle = false)
    first_batch = first(loader)
    @test first_batch isa GNNGraph
    @test first_batch.num_graphs == 2
    @test first_batch.num_nodes == 4
    @test size(first_batch.ndata.x) == (3, 4)

    uncollated = first(dataset_dataloader(graphs; batchsize = 2, shuffle = false, collate = false))
    @test uncollated isa AbstractVector
    @test length(uncollated) == 2
    @test first(uncollated) isa GNNGraph
end
