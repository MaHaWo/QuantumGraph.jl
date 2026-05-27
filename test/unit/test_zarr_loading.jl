using Test
using QuantumGraph

@testset "Zarr loading unit contract" begin
    missing_path = joinpath(tempdir(), "quantumgraph-missing-zarr-$(rand(UInt))")

    open_err = try
        open_zarr_store(missing_path)
        nothing
    catch caught
        caught
    end
    @test open_err isa ZarrLoadingError
    @test occursin("missing store path", sprint(showerror, open_err))

    validate_err = try
        validate_dataset_zarr_store(missing_path; required_arrays = ["adjacency_matrix"])
        nothing
    catch caught
        caught
    end
    @test validate_err isa ZarrLoadingError
    @test occursin("missing store path", sprint(showerror, validate_err))

    load_err = try
        recursive_load_zarr_store(missing_path)
        nothing
    catch caught
        caught
    end
    @test load_err isa ZarrLoadingError
    @test occursin("open Zarr store", sprint(showerror, load_err))
end
