Feature: Reusable graph model components
  As a model author
  I want reusable QuantumGraph model components to have stable structural behavior
  So that composite GNN models can be assembled without having to specify implementation details right away

  Background:
    Given exact Flux.jl or GraphNeuralNetworks.jl layer choices are implementation details
    And model component behavior is specified through public construction, shape, and metadata contracts

  @models @approved
  Scenario: A configurable component validates required construction fields
    Given a model component configuration omits a required dimension, activation, or graph operator field
    When QuantumGraph constructs the component from configuration
    Then component construction fails
    And the error identifies the missing configuration field
    And no partially constructed component is returned as successful

  @models @approved
  Scenario: A reusable block preserves batch and feature structure
    Given a reusable model block is constructed from valid configuration
    And compatible graph input with node or graph features is available
    When the block is applied to the input
    Then the output preserves the graph batch association
    And the output feature dimension matches the configured output dimension
    And exact stochastic parameter values are not part of the behavior contract

  @models @approved
  Scenario: Residual or skip connections handle matching dimensions directly
    Given a reusable block is configured with a residual or skip connection
    And the input and output feature dimensions match
    When the block is applied to compatible graph input
    Then the residual or skip path contributes to the block output
    And no projection layer is required by the public behavior

  @models @approved
  Scenario: Residual or skip connections reject incompatible dimensions without a projection policy
    Given a reusable block is configured with a residual or skip connection
    And the skip path is incompatible with the input and output dimensions of the skipped-over model block
    And no projection behavior is configured
    When QuantumGraph constructs or applies the block
    Then the operation fails with a clear dimension compatibility error
    And the error identifies the residual or skip connection configuration and the skipped-over block dimensions

  @models @approved
  Scenario: Component metadata round-trips structural configuration
    Given a reusable model component has been constructed from configuration
    When the component's public metadata is saved and loaded
    Then the loaded metadata preserves the component type, dimensions, activation choice, and graph-operator role
    And exact learned parameter values are not required for this metadata round trip
    And the metadata can be used to reconstruct an equivalent component structure
