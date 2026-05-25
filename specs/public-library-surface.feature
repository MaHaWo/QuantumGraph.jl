Feature: Public QuantumGraph.jl library surface
  As a downstream Julia user
  I want QuantumGraph to be a normal importable Julia library
  So that experiment code can use migrated capabilities without a command-line application

  Background:
    Given the QuantumGraph.jl package is available in a Julia project

  @approved
  Scenario: Importing the package succeeds without running work
    When downstream code imports QuantumGraph
    Then the import succeeds
    And no command-line entry point is required
    And no dataset is opened
    And no model is constructed
    And no training, tuning, checkpoint, or report artifact is created

  @approved
  Scenario: The package remains library-only
    When the package metadata and repository entry points are inspected
    Then no command-line interface is declared
    And documented usage starts from Julia import or Julia script examples

  @approved
  Scenario: Public capabilities are available through Julia-idiomatic names
    Given QuantumGraph has been imported
    When downstream code inspects the public library surface
    Then configuration loading and expansion capabilities are available
    And object registry or object resolution capabilities are available
    And Zarr loading capabilities are available
    And dataset construction capabilities are available
    And graph model construction capabilities are available
    And reusable model block capabilities are available
    And evaluation capabilities are available
    And early stopping capabilities are available
    And single-machine training capabilities are available
    And tuning helper capabilities are available
    And the public names follow Julia naming conventions rather than preserving Python spelling

  @approved
  Scenario: Downstream scripts do not depend on private implementation files
    Given QuantumGraph has been imported in a downstream Julia script
    When the script references the public capabilities needed to configure data, build models, evaluate results, train on one machine, and tune experiments
    Then those capabilities resolve through the QuantumGraph public surface
    And the script does not include private source files directly
    And the script does not import implementation files by path

  @pending
  Scenario: Missing dependencies produce clear import-time errors
    Given a required QuantumGraph dependency is unavailable
    When downstream code imports QuantumGraph
    Then the import fails
    And the error identifies the missing dependency or dependency group
    And the error indicates that the failure occurred while loading QuantumGraph
    And the error is not reported as an unrelated undefined-name or syntax failure

  @approved
  Scenario: Documentation describes import-based usage
    Given the README or package documentation is available
    When a downstream user reads the usage examples
    Then the examples show import-based Julia usage
    And the examples show public APIs for configuration, datasets, models, evaluation, early stopping, single-machine training, tuning, or Zarr loading
    And the examples do not instruct users to run a QuantumGraph command-line application

  @approved
  Scenario: The documented execution scope is one machine with one accelerator
    Given the README or package documentation is available
    When a downstream user reads the execution-scope notes
    Then the supported scope is one machine
    And the supported accelerator scope is at most one accelerator
    And no distributed training compatibility surface is advertised
