Feature: Optional single-accelerator device behavior
  As a training workflow author
  I want QuantumGraph to support CPU execution and optionally one configured accelerator
  So that migrated workflows run on one machine without requiring distributed setup

  Background:
    Given CPU execution is always supported by QuantumGraph
    And accelerator execution is optional and limited to at most one configured accelerator

  Scenario: CPU execution is selected when no accelerator is requested
    Given a training or model workflow does not request an accelerator
    When QuantumGraph prepares the execution device
    Then the selected device is CPU
    And no CUDA or accelerator backend initialization is required
    And training can proceed without distributed process setup

  Scenario: One available accelerator can be selected explicitly
    Given a workflow requests one available accelerator
    When QuantumGraph prepares the execution device
    Then the selected device represents that accelerator
    And model and graph batch values are prepared for that device through public APIs
    And no additional accelerator or multi-process setup is initialized

  Scenario: Unavailable accelerator requests fail clearly
    Given a workflow requests an accelerator that is not available
    When QuantumGraph prepares the execution device
    Then device preparation fails
    And the error identifies the requested accelerator setting
    And the error explains that the requested backend or device is unavailable

  Scenario: Multiple accelerator requests are rejected
    Given a workflow requests more than one accelerator
    When QuantumGraph validates the execution device settings
    Then validation fails
    And the error identifies that only one accelerator is supported
    And no partial multi-accelerator setup is attempted

  Scenario: Device movement preserves graph sample and model structure
    Given a model and GraphNeuralNetworks-compatible graph batch are available
    When QuantumGraph prepares them for the selected CPU or accelerator device
    Then the model structure remains usable through the public model API
    And the graph batch remains compatible with the approved graph sample boundary
    And exact numeric output values are not part of the device-selection behavior contract
