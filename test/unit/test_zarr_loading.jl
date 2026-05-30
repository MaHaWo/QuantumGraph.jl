# These unit tests cover the Zarr loading error boundary. They use missing
# temporary paths to confirm store opening, validation, and recursive loading
# all fail with QuantumGraph-specific user-visible errors.
using Test
using QuantumGraph
using Zarr
import ZipFile

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

    source_path = joinpath(mktempdir(), "zip-fixture.zarr")
    root = Zarr.zgroup(source_path)
    count = Zarr.zcreate(Int, root, "num_samples", 1; chunks = (1,))
    count[:] = [2]
    adjacency = Zarr.zcreate(Float32, root, "adjacency_matrix", 2, 2, 2; chunks = (2, 2, 2))
    adjacency[:] = reshape(Float32[0, 1, 1, 0, 0, 1, 1, 0], 2, 2, 2)

    zip_path = joinpath(mktempdir(), "zip-fixture.zarr.zip")
    zip = ZipFile.Writer(zip_path)
    try
        for (root_dir, _, files) in walkdir(source_path)
            for file in files
                full_path = joinpath(root_dir, file)
                rel_path = relpath(full_path, dirname(source_path))
                writer = ZipFile.addfile(zip, rel_path)
                write(writer, read(full_path))
            end
        end
    finally
        close(zip)
    end

    dataset = construct_dataset(zip_path; required_arrays = ["adjacency_matrix"])
    @test length(dataset) == 2
    @test dataset[1].source.local_index == 1
end
