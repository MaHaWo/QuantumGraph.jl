# This is the package test entry point. It verifies the native unit tests,
# native integration tests, and approved Behavior.jl acceptance specs by
# loading each test directory in a deterministic order before running BDD specs.
using Test
using Behavior
using QuantumGraph

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))

function include_test_directory(relative_path::AbstractString)
    directory = joinpath(REPO_ROOT, relative_path)
    isdir(directory) || return nothing
    for file in sort(filter(name -> endswith(name, ".jl"), readdir(directory)))
        include(joinpath(directory, file))
    end
    return nothing
end

@testset "QuantumGraph.jl native unit tests" begin
    include_test_directory(joinpath("test", "unit"))
end

@testset "QuantumGraph.jl native integration tests" begin
    include_test_directory(joinpath("test", "integration"))
end

@testset "QuantumGraph.jl Behavior.jl BDD acceptance tests" begin
    success = Behavior.runspec(
        REPO_ROOT;
        featurepath = joinpath(REPO_ROOT, "specs"),
        stepspath = joinpath(REPO_ROOT, "test", "steps"),
        tags = "@approved",
    )
    @test success
end
