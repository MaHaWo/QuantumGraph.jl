# These integration tests cover evaluation reports and early-stopping state.
# They run deterministic model/batch fixtures through batch and iterator
# evaluators, validate monitor metric handling, and step early-stopping decisions.
using Test
using QuantumGraph
using DataFrames
using GraphNeuralNetworks

@testset "Evaluation and early stopping integration contract" begin
    seen_batches = Any[]
    model = batch -> begin
        push!(seen_batches, batch)
        Dict(:mass => [2.0, 4.0], :dimension => [3.0])
    end
    graph_batch = (
        graph = GNNGraph([1, 2], [2, 1]),
        features = Float32[1 2; 3 4],
        targets = Dict(:mass => [1.0, 5.0], :dimension => [2.0]),
    )
    criteria = Dict(:mass => (output, batch) -> sum(abs.(output .- batch.targets[:mass])))
    monitor_calls = Any[]
    metrics = Dict(:mass_error => (predictions, targets) -> begin
        push!(monitor_calls, (predictions = predictions, targets = targets))
        "monitor-result-$(length(predictions))-$(length(targets))"
    end)

    batch_result = evaluate_batch(model, graph_batch; criteria = criteria, task_metrics = metrics)
    @test seen_batches[end] === graph_batch
    @test batch_result.losses == [2.0]
    @test batch_result.outputs[:mass] == [2.0, 4.0]
    @test batch_result.targets === graph_batch.targets

    report = evaluate_iterator(model, [graph_batch, graph_batch]; criteria = criteria, task_metrics = metrics)
    @test report isa DataFrame
    @test all(col -> col in Symbol.(names(report)), loss_report_columns())
    @test :mass_error in Symbol.(names(report))
    @test report.loss_avg[1] == 2.0
    @test report.loss_min[1] == 2.0
    @test report.loss_max[1] == 2.0
    @test report.mass_error[1] == "monitor-result-2-2"
    @test length(monitor_calls) == 1
    @test length(monitor_calls[1].predictions) == 2
    @test monitor_calls[1].targets == [graph_batch.targets, graph_batch.targets]
    @test task_metric_columns(metrics) == [:mass_error]
    @test monitor_task_columns(metrics) == [:mass_error]

    invalid_task_err = try
        evaluate_iterator(model, [graph_batch]; criteria = criteria, task_metrics = Dict(:bad => predictions -> 0.0))
        nothing
    catch caught
        caught
    end
    @test invalid_task_err isa EvaluationError
    @test occursin("invalid monitor task", sprint(showerror, invalid_task_err))

    invalid_metric_err = try
        evaluate_iterator(model, [graph_batch]; criteria = criteria, task_metrics = Dict(:bad => Dict("task" => :mass)))
        nothing
    catch caught
        caught
    end
    @test invalid_metric_err isa EvaluationError
    @test occursin("monitor", sprint(showerror, invalid_metric_err))

    @test_throws EvaluationError evaluate_iterator(model, []; criteria = criteria)

    history = DataFrame(loss_avg = [3.0, 2.0, 2.5], accuracy = [0.1, 0.2, 0.15])
    state = early_stopping_state(metric = :loss_avg, mode = :min, patience = 2, grace_period = 1)
    d1 = evaluate_early_stopping(state, history[1:1, :])
    @test continue_or_stop_decision(d1) == :continue
    @test d1.found_better
    @test d1.best_score == 3.0
    @test early_stopping_best_score(state) == 3.0
    @test early_stopping_grace_state(state) == 0

    d2 = evaluate_early_stopping(state, history[1:2, :])
    @test d2.found_better
    @test d2.best_score == 2.0
    @test early_stopping_grace_state(state) == 0

    d3 = evaluate_early_stopping(state, history[1:3, :])
    @test !d3.found_better
    @test continue_or_stop_decision(d3) == :continue
    @test early_stopping_grace_state(state) == 1

    empty_err = try
        evaluate_early_stopping(DataFrame(loss_avg = Float64[]); metric = :loss_avg)
        nothing
    catch caught
        caught
    end
    @test empty_err isa EarlyStoppingError
    @test occursin("no evaluation data", sprint(showerror, empty_err))

    missing_column_err = try
        evaluate_early_stopping(DataFrame(other = [1.0]); metric = :loss_avg)
        nothing
    catch caught
        caught
    end
    @test missing_column_err isa EarlyStoppingError
    @test occursin("missing metric column", sprint(showerror, missing_column_err))

    max_state = early_stopping_state(metric = :accuracy, mode = :max, patience = 1, grace_period = 0)
    max_decision = evaluate_early_stopping(max_state, history[1:2, :])
    @test max_decision.found_better
    @test max_decision.best_score == 0.2
end
