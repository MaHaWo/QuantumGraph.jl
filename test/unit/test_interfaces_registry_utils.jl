# These unit tests cover the public registry and shared configuration utilities.
# They exercise direct registration/resolution, metadata round-tripping, nested
# config-path reads and writes, and public error reporting for invalid values.
using Test
using QuantumGraph

@testset "Interfaces, registry, and utilities unit contract" begin
    registered = register_object!("QuantumGraph.Tests.Object", :registered_object)
    @test registered === :registered_object
    @test resolve_registered_object("QuantumGraph.Tests.Object") === :registered_object
    @test resolve_registered_object("QuantumGraph.Tests.Object") === resolve_registered_object("QuantumGraph.Tests.Object")

    err = try
        resolve_registered_object("QuantumGraph.Tests.Missing")
        nothing
    catch caught
        caught
    end
    @test err isa PublicInterfaceError
    @test occursin("QuantumGraph.Tests.Missing", sprint(showerror, err))
    @test occursin("unresolved module or object name", sprint(showerror, err))

    constructor = params -> (; value = params["value"])
    register_object!("QuantumGraph.Tests.Constructor", constructor)
    metadata = ConfigMetadata("QuantumGraph.Tests.Constructor", Dict{String, Any}("value" => 42))
    reconstructed = reconstruct_from_metadata(metadata)
    @test reconstructed.value == 42

    object_with_metadata = (; config_metadata = metadata, runtime_cache = Dict(:private => true))
    extracted = configuration_metadata(object_with_metadata)
    @test extracted === metadata
    @test !haskey(extracted.constructor_parameters, "runtime_cache")

    config = Dict{Any, Any}(
        "model" => Dict{String, Any}("hidden" => 16),
        :training => Dict{Symbol, Any}(:epochs => 3),
    )
    @test get_config_path(config, "model.hidden") == 16
    @test get_config_path(config, ["training", "epochs"]) == 3
    set_config_path!(config, "model.hidden", 32)
    @test get_config_path(config, "model.hidden") == 32

    @test_throws PublicInterfaceError get_config_path(config, "model.missing")
    @test_throws PublicInterfaceError set_config_path!(config, "model.missing", 1)
    @test_throws PublicInterfaceError validate_public_value("register object", "..bad.."; kind = "object identifier")
end
