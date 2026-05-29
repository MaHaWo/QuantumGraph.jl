# These integration tests cover configuration parsing and object resolution.
# They load YAML tag fixtures, expand sweeps and ranges, resolve references and
# registered Julia objects, and assert that malformed configurations fail clearly.
using Test
using QuantumGraph

@testset "Config object resolution integration contract" begin
    text = """
    widths: !sweep [16, 32]
    paired_lr: !coupled-sweep train [0.1, 0.01]
    paired_batch: !coupled-sweep train [8, 16]
    depth: !range [1, 5, 2]
    sampled: !random_uniform [0.0, 1.0]
    copied: !reference base_value
    activation: !pyobject-equivalent QuantumGraph.Tests.Activation
    """

    config = load_config(text)
    @test config["widths"] isa Sweep
    @test config["paired_lr"] isa CoupledSweep
    @test config["depth"] isa InclusiveRange
    @test config["sampled"] isa RandomUniform
    @test config["copied"] isa Reference
    @test config["activation"] isa ObjectReference
    @test supported_config_tags() == Set(["!sweep", "!coupled-sweep", "!range", "!random_uniform", "!reference", "!pyobject-equivalent"])

    @test expand_range(config["depth"]) == [1, 3, 5]
    @test_throws ConfigError expand_range(InclusiveRange(1, 5, 0))

    runs = expand_config(config)
    @test length(runs) == 4
    @test all(run -> haskey(run.config, "widths"), runs)
    @test all(run -> haskey(run.config, "paired_lr") && haskey(run.config, "paired_batch"), runs)
    @test any(run -> occursin("widths=16", run.name), runs)
    @test any(run -> occursin("train[1]", run.name), runs)

    mismatched = Dict{String, Any}(
        "a" => CoupledSweep("bad", Any[1, 2]),
        "b" => CoupledSweep("bad", Any[3]),
    )
    @test_throws ConfigError expand_config(mismatched)

    register_object!("QuantumGraph.Tests.Activation", identity)
    resolved = resolve_config(Dict{String, Any}(
        "base_value" => 7,
        "copied" => Reference(["base_value"]),
        "activation" => ObjectReference("QuantumGraph.Tests.Activation"),
        "range" => InclusiveRange(1, 3, 1),
    ))
    @test resolved["copied"] == 7
    @test resolved["activation"] === identity
    @test resolved["range"] == [1, 2, 3]

    err = try
        resolve_config(Dict{String, Any}("missing" => ObjectReference("QuantumGraph.Tests.Unknown")))
        nothing
    catch caught
        caught
    end
    @test err isa ConfigError
    @test occursin("QuantumGraph.Tests.Unknown", sprint(showerror, err))
end
