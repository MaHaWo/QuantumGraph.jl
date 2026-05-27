Feature: Zarr.jl-backed recursive loading
  As a dataset implementation
  I want QuantumGraph to read Python-produced Zarr stores through Zarr.jl
  So that migrated Julia datasets can consume approved fixture stores without a Python compatibility layer

  Background:
    Given Zarr.jl is a public QuantumGraph dependency for Zarr store access
    And Python compatibility shims are outside the current Zarr loading contract

  @zarr_loading @approved
  Scenario: Recursive loading preserves group and array structure
    Given a Zarr store contains nested groups and array leaves
    When QuantumGraph recursively loads the store
    Then each Zarr group is represented as a nested Julia mapping
    And each Zarr array leaf is represented as a Julia array-compatible value
    And empty groups are preserved as empty mappings

  @zarr_loading @approved
  Scenario: Approved fixture arrays can be read without Python imports
    Given a Python-produced Zarr fixture contains adjacency_matrix, link_matrix, and dimension arrays
    When QuantumGraph reads the fixture through Zarr.jl
    Then the arrays are available to Julia code
    And no Python package import is required during the read
    And array names match the names stored in the Zarr fixture

  @zarr_loading @approved
  Scenario: Missing store paths fail with a user-visible path error
    Given a requested Zarr store path does not exist
    When QuantumGraph attempts to open the store
    Then loading fails
    And the error identifies the missing store path
    And no empty dataset is returned as if loading succeeded

  @zarr_loading @approved
  Scenario: Unsupported store layouts are rejected at the loading boundary
    Given a Zarr store is present but does not expose the approved group or array structure
    When QuantumGraph loads the store for dataset construction
    Then loading fails with an unsupported-layout error
    And the error identifies the unexpected group or array entry
    And dataset construction does not continue with malformed data

  @zarr_loading @approved
  Scenario: Lazy array access does not materialize all samples during store opening
    Given a Zarr store contains sample-indexed graph arrays
    When QuantumGraph opens the store for dataset use
    Then opening the store records array handles or equivalent lazy accessors
    And sample arrays are not fully materialized until a sample is requested
    And requesting one sample reads only the arrays needed for that sample
