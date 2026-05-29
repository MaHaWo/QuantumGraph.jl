# These integration tests cover migration-compatibility messaging and checkpoint
# validation. They inspect README/docs promises and verify that Julia-native
# checkpoints pass while legacy Torch/Python artifacts produce guided errors.
using Test
using QuantumGraph

@testset "Migration compatibility integration contract" begin
    repo_root = dirname(dirname(@__DIR__))
    docs_path = joinpath(repo_root, "docs", "migration_compatibility.md")
    readme_path = joinpath(repo_root, "README.md")
    docs = read(docs_path, String)
    readme = read(readme_path, String)

    @test occursin(r"(?i)Julia[- ]native.*checkpoint|checkpoint.*Julia[- ]native", docs)
    @test occursin(r"(?i)Torch|Python", docs)
    @test occursin(r"(?i)unsupported|conversion|deferred", docs)
    @test occursin(r"(?i)supported artifacts", docs)
    @test occursin("docs/migration_compatibility.md", readme)

    @test validate_checkpoint_input("model_current.jls") == "model_current.jls"

    err = try
        validate_checkpoint_input("model.pt")
        nothing
    catch caught
        caught
    end
    @test err isa TrainingError
    @test occursin("unsupported checkpoint artifact type", sprint(showerror, err))
    @test occursin("docs/migration_compatibility.md", sprint(showerror, err))

    compat_err = checkpoint_compatibility_error("legacy.pth")
    @test compat_err isa TrainingError
    @test occursin("legacy.pth", sprint(showerror, compat_err))
    @test unsupported_checkpoint_error("legacy.ckpt") isa TrainingError
end
