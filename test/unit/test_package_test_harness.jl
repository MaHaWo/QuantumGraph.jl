# These unit tests cover the package skeleton and local test harness contract.
# They check that QuantumGraph imports as a library module, keeps CLI entry
# points out of the public module, and preserves the expected test directories.
using Test
using QuantumGraph

@testset "Package skeleton and test harness unit contract" begin
    @test QuantumGraph isa Module
    @test dummy() === nothing
    @test !isdefined(QuantumGraph, :main)
    @test !isdefined(QuantumGraph, :julia_main)

    repo_root = abspath(joinpath(@__DIR__, "..", ".."))
    @test isdir(joinpath(repo_root, "specs"))
    @test isdir(joinpath(repo_root, "test", "steps"))
    @test isdir(joinpath(repo_root, "test", "unit"))
    @test isdir(joinpath(repo_root, "test", "integration"))
end
