Feature: Dataset layer and GraphNeuralNetworks-compatible graph samples
  As a model training workflow
  I want datasets to lazily read Python-produced Zarr stores into graph samples
  So that Julia models can consume migrated data without eager preprocessing

  Background:
    Given existing QuantumGravPy Zarr stores may contain adjacency_matrix, link_matrix, max_pathlen_future, max_pathlen_past, dimension, atomcount, num_samples, or num_causal_sets arrays
    And the graph sample boundary is a GraphNeuralNetworks.jl-compatible graph container or QuantumGraph wrapper around one

  @approved
  Scenario: Dataset reports sample count using approved precedence
    Given a Zarr store contains multiple possible sample count indicators
    When a QuantumGraph dataset is constructed for that store
    Then the sample count is selected using num_causal_sets before num_samples
    And num_samples is selected before one-dimensional dataset inference
    And one-dimensional dataset inference is selected before adjacency_matrix shape fallback

  @approved
  Scenario: Dataset reads samples lazily by index
    Given a valid dataset references one or more Zarr stores
    When the dataset is constructed
    Then sample data is not eagerly materialized for every sample
    When a valid sample index is requested
    Then only the requested sample is read and converted for model consumption

  @approved
  Scenario: Dataset converts a valid sample to the graph boundary type
    Given a Zarr sample contains graph structure and feature arrays required by the approved data contract
    When the sample is requested from the dataset
    Then the result is compatible with GraphNeuralNetworks.jl model input
    And graph structure, node or graph features, and task targets are available through documented fields or accessors

  @approved
  Scenario: Dataset rejects missing reader or unsupported store layout
    Given a dataset is configured with a missing reader function or unsupported Zarr layout
    When the dataset attempts to read a sample
    Then the read fails with an error identifying the missing reader or unsupported layout
    And no silently malformed graph sample is returned

  @approved
  Scenario: Dataset rejects out-of-range indexes
    Given a dataset contains a known number of samples
    When a caller requests an index outside the dataset bounds
    Then the request fails with an out-of-range error
    And the error identifies the requested index or valid bounds

  @approved
  Scenario: Map-index behavior selects the correct backing store and local sample
    Given a dataset spans multiple backing Zarr stores
    When a global sample index is requested
    Then QuantumGraph maps the global index to the correct backing store
    And maps it to the correct local sample index within that store
