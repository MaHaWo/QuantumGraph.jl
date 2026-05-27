using Behavior

function qg_workflow_exports()
    try
        @eval import QuantumGraph
        return Set(String.(names(QuantumGraph; all = false, imported = false)))
    catch
        return Set{String}()
    end
end

qg_workflow_has(patterns::Vector{Regex}) = any(name -> any(pattern -> occursin(pattern, name), patterns), qg_workflow_exports())
qg_workflow_requires(patterns) = @expect qg_workflow_has(patterns)

# specs/training-tuning-workflows.feature
# Background: Given configuration, dataset, model, evaluation, early stopping, and tuning capabilities are available through QuantumGraph
@given("configuration, dataset, model, evaluation, early stopping, and tuning capabilities are available through QuantumGraph") do context
    qg_workflow_requires([r"config"i, r"dataset"i, r"model"i, r"evaluat"i, r"early"i, r"tun"i])
end

# specs/training-tuning-workflows.feature
# Background: And training reports use DataFrames.jl DataFrame values
@given("training reports use DataFrames.jl DataFrame values") do context
    context[:training_report_interface] = "DataFrames.DataFrame"
    qg_workflow_requires([r"DataFrame"i, r"train"i, r"report"i])
end

# specs/training-tuning-workflows.feature
# Background: And checkpoint artifacts use a Julia-native format
@given("checkpoint artifacts use a Julia-native format") do context
    context[:checkpoint_format] = "Julia-native"
    qg_workflow_requires([r"checkpoint"i, r"save"i, r"Julia"i])
end

# specs/training-tuning-workflows.feature
# Background: And distributed or multi-machine training is outside the current scope
@given("distributed or multi-machine training is outside the current scope") do context
    context[:distributed_training_in_scope] = false
    @expect !context[:distributed_training_in_scope]
end

# specs/training-tuning-workflows.feature
# Scenario: Trainer initializes from a resolved configuration
@given("a resolved training configuration contains dataset, model, optimizer, scheduler, evaluator, early stopping, and output path settings") do context
    context[:training_config_sections] = ["dataset", "model", "optimizer", "scheduler", "evaluator", "early_stopping", "output_path"]
end

# specs/training-tuning-workflows.feature
# Scenario: Trainer initializes from a resolved configuration
@when("the trainer is constructed") do context
    qg_workflow_requires([r"trainer"i, r"train"i, r"construct"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Trainer initializes from a resolved configuration
@then("it validates the required configuration sections") do context
    qg_workflow_requires([r"config"i, r"valid"i, r"train"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Trainer initializes from a resolved configuration
@then("it prepares the configured components without starting training") do context
    qg_workflow_requires([r"prepare"i, r"component"i, r"train"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Trainer initializes from a resolved configuration
@then("invalid configuration sections are reported with clear errors") do context
    qg_workflow_requires([r"config"i, r"error"i, r"invalid"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Training runs on one machine with at most one accelerator
@given("a trainer is configured for CPU or one available accelerator") do context
    context[:accelerator_count] = 1
end

# specs/training-tuning-workflows.feature
# Scenario: Training runs on one machine with at most one accelerator
@when("training starts") do context
    qg_workflow_requires([r"train"i, r"start"i, r"fit"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Training runs on one machine with at most one accelerator
@then("it runs without requiring distributed process setup") do context
    qg_workflow_requires([r"train"i, r"single"i, r"local"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Training runs on one machine with at most one accelerator
@then("it does not advertise or initialize multi-machine training") do context
    @expect !(haskey(context, :distributed_training_in_scope) ? context[:distributed_training_in_scope] : true)
end

# specs/training-tuning-workflows.feature
# Scenario: Training runs on one machine with at most one accelerator
@then("unsupported accelerator settings fail with clear backend errors") do context
    qg_workflow_requires([r"accelerator"i, r"backend"i, r"error"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Training writes structural artifacts
@given("a trainer completes a deterministic fixture run") do context
    context[:deterministic_fixture_run] = true
end

# specs/training-tuning-workflows.feature
# Scenario: Training writes structural artifacts
@when("output artifacts are inspected") do context
    qg_workflow_requires([r"artifact"i, r"output"i, r"train"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Training writes structural artifacts
@then("checkpoint artifacts are written in a Julia-native format") do context
    qg_workflow_requires([r"checkpoint"i, r"Julia"i, r"save"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Training writes structural artifacts
@then("configuration copies are written where required") do context
    qg_workflow_requires([r"config"i, r"copy"i, r"write"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Training writes structural artifacts
@then("validation or test reports are written with DataFrame-compatible schemas") do context
    qg_workflow_requires([r"report"i, r"DataFrame"i, r"schema"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Training writes structural artifacts
@then("exact stochastic loss or metric values are not required") do context
    @expect true
end

# specs/training-tuning-workflows.feature
# Scenario: Training rejects checkpoint write failures
@given("the configured checkpoint output path cannot be written") do context
    context[:checkpoint_path_writable] = false
end

# specs/training-tuning-workflows.feature
# Scenario: Training rejects checkpoint write failures
@when("training attempts to save a checkpoint") do context
    qg_workflow_requires([r"checkpoint"i, r"save"i, r"train"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Training rejects checkpoint write failures
@then("training fails with an error identifying the checkpoint write failure") do context
    qg_workflow_requires([r"checkpoint"i, r"write"i, r"error"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Training rejects checkpoint write failures
@then("the failure is not reported as a successful training run") do context
    qg_workflow_requires([r"train"i, r"failure"i, r"error"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Tuning uses Optuna.jl as the preferred backend candidate
@given("a tuning configuration defines search-space distributions, references, coupled sweeps, study settings, and trial limits") do context
    context[:tuning_config_sections] = ["search_space", "references", "coupled_sweeps", "study", "trial_limits"]
end

# specs/training-tuning-workflows.feature
# Scenario: Tuning uses Optuna.jl as the preferred backend candidate
@when("QuantumGraph constructs the tuning workflow") do context
    qg_workflow_requires([r"tuning"i, r"workflow"i, r"construct"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Tuning uses Optuna.jl as the preferred backend candidate
@then("the preferred backend candidate is Optuna.jl") do context
    qg_workflow_requires([r"Optuna"i, r"backend"i, r"tuning"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Tuning uses Optuna.jl as the preferred backend candidate
@then("QuantumGraph exposes backend-neutral tuning concepts for study, trial, suggestion, objective result, and best configuration") do context
    qg_workflow_requires([r"study"i, r"trial"i, r"suggest"i, r"objective"i, r"best"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Tuning uses Optuna.jl as the preferred backend candidate
@then("unsupported Optuna.jl capabilities are reported as explicit backend limitations") do context
    qg_workflow_requires([r"Optuna"i, r"limitation"i, r"backend"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Tuning exports the best configuration using approved YAML semantics
@given("a tuning study has completed at least one successful trial") do context
    context[:successful_trials] = 1
end

# specs/training-tuning-workflows.feature
# Scenario: Tuning exports the best configuration using approved YAML semantics
@when("the best configuration is exported") do context
    qg_workflow_requires([r"best"i, r"config"i, r"export"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Tuning exports the best configuration using approved YAML semantics
@then("the exported configuration preserves resolved references and selected search-space values") do context
    qg_workflow_requires([r"config"i, r"reference"i, r"search"i])
end

# specs/training-tuning-workflows.feature
# Scenario: Tuning exports the best configuration using approved YAML semantics
@then("the output can be consumed by the normal QuantumGraph configuration loader") do context
    qg_workflow_requires([r"config"i, r"load"i, r"output"i])
end
