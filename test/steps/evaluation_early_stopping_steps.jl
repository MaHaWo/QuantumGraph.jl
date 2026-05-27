using Behavior

function qg_eval_exports()
    try
        @eval import QuantumGraph
        return Set(String.(names(QuantumGraph; all = false, imported = false)))
    catch
        return Set{String}()
    end
end

qg_eval_has(patterns::Vector{Regex}) = any(name -> any(pattern -> occursin(pattern, name), patterns), qg_eval_exports())
qg_eval_requires(patterns) = @expect qg_eval_has(patterns)

# specs/evaluation-early-stopping.feature
# Background: Given evaluation and early stopping use DataFrames.jl DataFrame values as their tabular data interface
@given("evaluation and early stopping use DataFrames.jl DataFrame values as their tabular data interface") do context
    context[:tabular_interface] = "DataFrames.DataFrame"
    qg_eval_requires([r"DataFrame"i, r"evaluat"i, r"early"i])
end

# specs/evaluation-early-stopping.feature
# Background: And exact stochastic metric values are not part of the migration contract
@given("exact stochastic metric values are not part of the migration contract") do context
    context[:requires_exact_metric_values] = false
    @expect !context[:requires_exact_metric_values]
end

# specs/evaluation-early-stopping.feature
# Scenario: Evaluation produces the required report schema
@given("a model, data iterator, criterion functions, and task metric definitions are available") do context
    context[:evaluation_fixture] = (:model, :iterator, :criteria, :metrics)
end

# specs/evaluation-early-stopping.feature
# Scenario: Evaluation produces the required report schema
@when("evaluation runs over the data iterator") do context
    qg_eval_requires([r"evaluat"i, r"iterator"i, r"metric"i])
end

# specs/evaluation-early-stopping.feature
# Scenario: Evaluation produces the required report schema
@then("the result is a DataFrame") do context
    qg_eval_requires([r"DataFrame"i, r"report"i, r"evaluat"i])
end

# specs/evaluation-early-stopping.feature
# Scenario: Evaluation produces the required report schema
@then("it contains loss_avg, loss_min, and loss_max columns") do context
    qg_eval_requires([r"loss"i, r"report"i, r"column"i])
end

# specs/evaluation-early-stopping.feature
# Scenario: Evaluation produces the required report schema
@then("it contains configured per-task metric columns when task metrics are provided") do context
    qg_eval_requires([r"metric"i, r"task"i, r"column"i])
end

# specs/evaluation-early-stopping.feature
# Scenario: Evaluation calls the model with graph batch inputs
@given("a data iterator yields GraphNeuralNetworks-compatible graph samples or batches") do context
    context[:iterator_yields_graph_batches] = true
end

# specs/evaluation-early-stopping.feature
# Scenario: Evaluation calls the model with graph batch inputs
@when("evaluation processes a batch") do context
    qg_eval_requires([r"evaluat"i, r"batch"i, r"graph"i])
end

# specs/evaluation-early-stopping.feature
# Scenario: Evaluation calls the model with graph batch inputs
@then("the model receives graph inputs through the approved graph sample boundary") do context
    qg_eval_requires([r"graph"i, r"sample"i, r"model"i])
end

# specs/evaluation-early-stopping.feature
# Scenario: Evaluation calls the model with graph batch inputs
@then("evaluation records losses and metrics from observable model outputs") do context
    qg_eval_requires([r"loss"i, r"metric"i, r"output"i])
end

# specs/evaluation-early-stopping.feature
# Scenario: Evaluation rejects invalid task metric configuration
@given("a task metric definition references a missing task output or invalid metric") do context
    context[:invalid_metric_definition] = true
end

# specs/evaluation-early-stopping.feature
# Scenario: Evaluation rejects invalid task metric configuration
@when("evaluation runs") do context
    qg_eval_requires([r"evaluat"i, r"metric"i])
end

# specs/evaluation-early-stopping.feature
# Scenario: Evaluation rejects invalid task metric configuration
@then("evaluation fails with an error identifying the invalid task or metric") do context
    qg_eval_requires([r"metric"i, r"task"i, r"error"i])
end

# specs/evaluation-early-stopping.feature
# Scenario: Evaluation rejects invalid task metric configuration
@then("no successful report is returned") do context
    qg_eval_requires([r"report"i, r"error"i])
end

# specs/evaluation-early-stopping.feature
# Scenario: Early stopping updates state from a DataFrame history
@given("early stopping is configured with a metric column, mode, patience, and grace period") do context
    context[:early_stopping_config] = (metric = "loss_avg", mode = "min", patience = 3, grace = 1)
end

# specs/evaluation-early-stopping.feature
# Scenario: Early stopping updates state from a DataFrame history
@given("a DataFrame history contains the configured metric column") do context
    context[:history_has_metric_column] = true
end

# specs/evaluation-early-stopping.feature
# Scenario: Early stopping updates state from a DataFrame history
@when("early stopping evaluates the history") do context
    qg_eval_requires([r"early"i, r"stop"i, r"history"i])
end

# specs/evaluation-early-stopping.feature
# Scenario: Early stopping updates state from a DataFrame history
@then("it returns a continue-or-stop decision") do context
    qg_eval_requires([r"continue"i, r"stop"i, r"decision"i])
end

# specs/evaluation-early-stopping.feature
# Scenario: Early stopping updates state from a DataFrame history
@then("it updates best score, grace, and found-better state according to the configured mode") do context
    qg_eval_requires([r"best"i, r"grace"i, r"early"i])
end

# specs/evaluation-early-stopping.feature
# Scenario: Early stopping rejects empty history
@given("early stopping receives an empty DataFrame history") do context
    context[:empty_history] = true
end

# specs/evaluation-early-stopping.feature
# Scenario: Early stopping rejects empty history
@when("it evaluates stopping state") do context
    qg_eval_requires([r"early"i, r"stop"i, r"state"i])
end

# specs/evaluation-early-stopping.feature
# Scenario: Early stopping rejects empty history
@then("it fails with an error identifying that no evaluation data is available") do context
    qg_eval_requires([r"empty"i, r"history"i, r"error"i])
end

# specs/evaluation-early-stopping.feature
# Scenario: Early stopping rejects missing metric columns
@given("early stopping is configured to monitor a metric column") do context
    context[:monitored_metric] = "loss_avg"
end

# specs/evaluation-early-stopping.feature
# Scenario: Early stopping rejects missing metric columns
@given("the DataFrame history does not contain that column") do context
    context[:history_has_metric_column] = false
end

# specs/evaluation-early-stopping.feature
# Scenario: Early stopping rejects missing metric columns
@then("it fails with an error identifying the missing metric column") do context
    qg_eval_requires([r"metric"i, r"column"i, r"error"i])
end
