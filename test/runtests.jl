using Test
using Behavior

const REPO_ROOT = abspath(joinpath(@__DIR__, ".."))

@testset "QuantumGraph.jl Behavior.jl BDD acceptance tests" begin
    success = Behavior.runspec(
        REPO_ROOT;
        featurepath = joinpath(REPO_ROOT, "specs"),
        stepspath = joinpath(REPO_ROOT, "test", "steps"),
        tags = "@approved",
    )
    @test success
end
