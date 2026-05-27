Feature: Evaluation reports and early stopping over DataFrames
  As a training workflow
  I want evaluation and early stopping to exchange DataFrames
  So that reporting and stopping decisions have a stable tabular Julia interface

  Background:
    Given evaluation and early stopping use DataFrames.jl DataFrame values as their tabular data interface
    And exact stochastic metric values are not part of the migration contract

  @approved
  Scenario: Evaluation produces the required report schema
    Given a model, data iterator, criterion functions, and task metric definitions are available
    When evaluation runs over the data iterator
    Then the result is a DataFrame
    And it contains loss_avg, loss_min, and loss_max columns
    And it contains configured per-task metric columns when task metrics are provided

  @approved
  Scenario: Evaluation calls the model with graph batch inputs
    Given a data iterator yields GraphNeuralNetworks-compatible graph samples or batches
    When evaluation processes a batch
    Then the model receives graph inputs through the approved graph sample boundary
    And evaluation records losses and metrics from observable model outputs

  @approved
  Scenario: Evaluation rejects invalid task metric configuration
    Given a task metric definition references a missing task output or invalid metric
    When evaluation runs
    Then evaluation fails with an error identifying the invalid task or metric
    And no successful report is returned

  @approved
  Scenario: Early stopping updates state from a DataFrame history
    Given early stopping is configured with a metric column, mode, patience, and grace period
    And a DataFrame history contains the configured metric column
    When early stopping evaluates the history
    Then it returns a continue-or-stop decision
    And it updates best score, grace, and found-better state according to the configured mode

  @approved
  Scenario: Early stopping rejects empty history
    Given early stopping receives an empty DataFrame history
    When it evaluates stopping state
    Then it fails with an error identifying that no evaluation data is available

  @approved
  Scenario: Early stopping rejects missing metric columns
    Given early stopping is configured to monitor a metric column
    And the DataFrame history does not contain that column
    When it evaluates stopping state
    Then it fails with an error identifying the missing metric column
