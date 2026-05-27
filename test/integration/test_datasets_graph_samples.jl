using Test
using QuantumGraph
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
end
