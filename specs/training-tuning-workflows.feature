Feature: Single-machine training and Optuna-backed tuning workflow boundary
  As an experiment author
  I want training and tuning workflows to compose the migrated public APIs
  So that experiments can run structurally equivalent Julia workflows on one machine with at most one accelerator

  Background:
    Given configuration, dataset, model, evaluation, early stopping, and tuning capabilities are available through QuantumGraph
    And training reports use DataFrames.jl DataFrame values
    And checkpoint artifacts use a Julia-native format
    And distributed or multi-machine training is outside the current scope

  @approved
  Scenario: Trainer initializes from a resolved configuration
    Given a resolved training configuration contains dataset, model, optimizer, scheduler, evaluator, early stopping, and output path settings
    When the trainer is constructed
    Then it validates the required configuration sections
    And it prepares the configured components without starting training
    And invalid configuration sections are reported with clear errors

  @approved
  Scenario: Training runs on one machine with at most one accelerator
    Given a trainer is configured for CPU or one available accelerator
    When training starts
    Then it runs without requiring distributed process setup
    And it does not advertise or initialize multi-machine training
    And unsupported accelerator settings fail with clear backend errors

  @approved
  Scenario: Training writes structural artifacts
    Given a trainer completes a deterministic fixture run
    When output artifacts are inspected
    Then checkpoint artifacts are written in a Julia-native format
    And configuration copies are written where required
    And validation or test reports are written with DataFrame-compatible schemas
    And exact stochastic loss or metric values are not required

  @approved
  Scenario: Training applies early stopping after validation reports
    Given a trainer is configured with validation and early stopping
    When validation reports stop improving beyond patience
    Then training stops before the maximum epoch count
    And the early-stopping state records the best score and grace count
    And a current-best checkpoint is written when a better score is found

  @approved
  Scenario: Training rejects checkpoint write failures
    Given the configured checkpoint output path cannot be written
    When training attempts to save a checkpoint
    Then training fails with an error identifying the checkpoint write failure
    And the failure is not reported as a successful training run

  @approved
  Scenario: Tuning uses Optuna.jl as the preferred backend candidate
    Given a tuning configuration defines search-space distributions, references, coupled sweeps, study settings, and trial limits
    When QuantumGraph constructs the tuning workflow
    Then the preferred backend candidate is Optuna.jl
    And QuantumGraph exposes backend-neutral tuning concepts for study, trial, suggestion, objective result, and best configuration
    And unsupported Optuna.jl capabilities are reported as explicit backend limitations

  @approved
  Scenario: Tuning exports the best configuration using approved YAML semantics
    Given a tuning study has completed at least one successful trial
    When the best configuration is exported
    Then the exported configuration preserves resolved references and selected search-space values
    And the output can be consumed by the normal QuantumGraph configuration loader
