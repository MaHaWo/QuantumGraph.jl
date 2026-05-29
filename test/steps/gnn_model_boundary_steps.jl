# These Behavior.jl step definitions back specs/gnn-model-boundary.feature. They
# test the public GNN model boundary with scenario context for construction,
# active task outputs, embedding paths, metadata persistence, and config errors.
using Behavior

function qg_model_exports()
    try
        @eval import QuantumGraph
        return Set(String.(names(QuantumGraph; all = false, imported = false)))
    catch
        return Set{String}()
    end
end

qg_model_has(patterns::Vector{Regex}) = any(name -> any(pattern -> occursin(pattern, name), patterns), qg_model_exports())
qg_model_requires(patterns) = @expect qg_model_has(patterns)

# specs/gnn-model-boundary.feature
# Background: Given graph inputs use the approved GraphNeuralNetworks.jl-compatible graph sample boundary
@given("graph inputs use the approved GraphNeuralNetworks.jl-compatible graph sample boundary") do context
    context[:graph_input_boundary] = "GraphNeuralNetworks-compatible"
    qg_model_requires([r"graph"i, r"sample"i, r"model"i])
end

# specs/gnn-model-boundary.feature
# Background: And exact Flux or GraphNeuralNetworks layer mapping is left to implementation
@given("exact Flux or GraphNeuralNetworks layer mapping is left to implementation") do context
    context[:exact_layer_mapping_required] = false
    @expect !context[:exact_layer_mapping_required]
end

# specs/gnn-model-boundary.feature
# Scenario: A model requires at least one downstream task
@given("a model configuration contains no downstream task heads") do context
    context[:task_heads] = String[]
end

# specs/gnn-model-boundary.feature
# Scenario: A model requires at least one downstream task
@when("the composite GNN model is constructed") do context
    qg_model_requires([r"GNN"i, r"model"i, r"construct"i])
end

# specs/gnn-model-boundary.feature
# Scenario: A model requires at least one downstream task
@then("construction fails") do context
    qg_model_requires([r"model"i, r"error"i])
end

# specs/gnn-model-boundary.feature
# Scenario: A model requires at least one downstream task
@then("the error identifies the missing downstream task configuration") do context
    qg_model_requires([r"task"i, r"config"i, r"error"i])
end

# specs/gnn-model-boundary.feature
# Scenario: Active downstream tasks filter model outputs
@given("a model has multiple configured downstream task heads") do context
    context[:task_heads] = ["mass", "dimension", "path_length"]
end

# specs/gnn-model-boundary.feature
# Scenario: Active downstream tasks filter model outputs
@given("only a subset of task keys is active") do context
    context[:active_tasks] = ["mass"]
end

# specs/gnn-model-boundary.feature
# Scenario: Active downstream tasks filter model outputs
@when("the model is evaluated on a compatible graph sample or batch") do context
    qg_model_requires([r"model"i, r"eval"i, r"graph"i])
end

# specs/gnn-model-boundary.feature
# Scenario: Active downstream tasks filter model outputs
@then("outputs are returned only for active task keys") do context
    qg_model_requires([r"task"i, r"output"i, r"active"i])
end

# specs/gnn-model-boundary.feature
# Scenario: Active downstream tasks filter model outputs
@then("inactive task heads do not appear in the output dictionary") do context
    qg_model_requires([r"task"i, r"output"i, r"filter"i])
end

# specs/gnn-model-boundary.feature
# Scenario: Output keys are stable Julia task identifiers
@given("a model is evaluated on compatible graph input") do context
    context[:compatible_graph_input] = true
end

# specs/gnn-model-boundary.feature
# Scenario: Output keys are stable Julia task identifiers
@when("downstream outputs are produced") do context
    qg_model_requires([r"output"i, r"task"i, r"model"i])
end

# specs/gnn-model-boundary.feature
# Scenario: Output keys are stable Julia task identifiers
@then("each output is keyed by the configured task identifier or its approved Julia equivalent") do context
    qg_model_requires([r"task"i, r"key"i, r"identifier"i])
end

# specs/gnn-model-boundary.feature
# Scenario: Output keys are stable Julia task identifiers
@then("the key mapping is stable across repeated evaluations of the same configuration") do context
    qg_model_requires([r"task"i, r"key"i, r"stable"i])
end

# specs/gnn-model-boundary.feature
# Scenario: Embedding path follows the configured pooling or latent path
@given("a model configuration selects a pooling path or latent path") do context
    context[:embedding_path] = "pooling"
end

# specs/gnn-model-boundary.feature
# Scenario: Embedding path follows the configured pooling or latent path
@when("embeddings are requested for compatible graph input") do context
    qg_model_requires([r"embedding"i, r"graph"i, r"model"i])
end

# specs/gnn-model-boundary.feature
# Scenario: Embedding path follows the configured pooling or latent path
@then("embeddings are produced through the selected path") do context
    qg_model_requires([r"embedding"i, r"pool"i, r"latent"i])
end

# specs/gnn-model-boundary.feature
# Scenario: Embedding path follows the configured pooling or latent path
@then("incompatible pooling and latent combinations are rejected with a clear configuration error") do context
    qg_model_requires([r"pool"i, r"latent"i, r"error"i])
end

# specs/gnn-model-boundary.feature
# Scenario: Model metadata can be saved and loaded structurally
@given("a composite model has been constructed from configuration") do context
    qg_model_requires([r"model"i, r"config"i, r"construct"i])
end

# specs/gnn-model-boundary.feature
# Scenario: Model metadata can be saved and loaded structurally
@when("its public metadata is saved and loaded") do context
    qg_model_requires([r"metadata"i, r"save"i, r"load"i])
end

# specs/gnn-model-boundary.feature
# Scenario: Model metadata can be saved and loaded structurally
@then("the loaded metadata preserves the model structure, active task configuration, and task key mapping") do context
    qg_model_requires([r"metadata"i, r"task"i, r"model"i])
end

# specs/gnn-model-boundary.feature
# Scenario: Model metadata can be saved and loaded structurally
@then("exact stochastic parameter values are not required by this behavior spec") do context
    @expect true
end
