Feature: Composite GNN model boundary
  As a training or evaluation workflow
  I want a configurable composite graph neural network model
  So that migrated experiments can produce embeddings and downstream task outputs from graph samples

  Background:
    Given graph inputs use the approved GraphNeuralNetworks.jl-compatible graph sample boundary
    And exact Flux or GraphNeuralNetworks layer mapping is left to implementation

  @approved
  Scenario: A model requires at least one downstream task
    Given a model configuration contains no downstream task heads
    When the composite GNN model is constructed
    Then construction fails
    And the error identifies the missing downstream task configuration

  @approved
  Scenario: Active downstream tasks filter model outputs
    Given a model has multiple configured downstream task heads
    And only a subset of task keys is active
    When the model is evaluated on a compatible graph sample or batch
    Then outputs are returned only for active task keys
    And inactive task heads do not appear in the output dictionary

  @approved
  Scenario: Output keys are stable Julia task identifiers
    Given a model is evaluated on compatible graph input
    When downstream outputs are produced
    Then each output is keyed by the configured task identifier or its approved Julia equivalent
    And the key mapping is stable across repeated evaluations of the same configuration

  @approved
  Scenario: Embedding path follows the configured pooling or latent path
    Given a model configuration selects a pooling path or latent path
    When embeddings are requested for compatible graph input
    Then embeddings are produced through the selected path
    And incompatible pooling and latent combinations are rejected with a clear configuration error

  @approved
  Scenario: Model metadata can be saved and loaded structurally
    Given a composite model has been constructed from configuration
    When its public metadata is saved and loaded
    Then the loaded metadata preserves the model structure, active task configuration, and task key mapping
    And exact stochastic parameter values are not required by this behavior spec
