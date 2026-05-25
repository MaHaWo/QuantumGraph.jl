Feature: Configuration parsing and object resolution boundary
  As an experiment author
  I want QuantumGraph to preserve approved YAML configuration semantics
  So that migrated Julia workflows can consume existing experiment configuration patterns

  Background:
    Given QuantumGraph is used as an importable Julia library
    And configuration behavior is scoped to Julia equivalents rather than Python object imports

  Scenario: Custom YAML tags are parsed into Julia configuration values
    Given a YAML configuration containing !sweep, !coupled-sweep, !range, !random_uniform, !reference, and !pyobject-equivalent tags
    When the configuration is loaded by QuantumGraph
    Then each supported tag is recognized
    And the resulting Julia configuration preserves the tag's approved behavior
    And unsupported or malformed tags produce user-visible configuration errors

  Scenario: Range expansion preserves inclusive range semantics
    Given a configuration range with a start, stop, and nonzero step
    When the range is expanded
    Then the generated values include the start value
    And include the stop value when it lies on the step sequence
    And a zero step is rejected with an error identifying the range step

  Scenario: Sweep expansion creates one run configuration per selected value
    Given a configuration contains a sweep over multiple values
    When the configuration is expanded
    Then one run configuration is produced for each swept value
    And each run configuration contains the selected value at the target path
    And run naming preserves the approved suffix behavior without requiring Python naming conventions

  Scenario: Coupled sweeps reject mismatched lengths
    Given a configuration contains a coupled sweep with differently sized value lists
    When the configuration is expanded
    Then expansion fails
    And the error identifies the coupled sweep length mismatch
    And no partial run configuration is accepted as successful

  Scenario: References resolve previously defined configuration paths
    Given a configuration value references another configuration path
    When the configuration is resolved
    Then the reference is replaced by the referenced value
    And a missing referenced path is rejected with an error identifying the missing path

  Scenario: Julia object registry replaces Python object imports
    Given a configuration contains a !pyobject-equivalent object reference
    When the configuration is resolved
    Then QuantumGraph resolves the reference through an explicit Julia registry or mapping
    And unresolved registry modules or names fail with clear errors
    And no Python import is required for normal Julia object resolution

  Scenario: Configurable objects can round-trip their configuration metadata
    Given a QuantumGraph object supports construction from configuration
    When the object is serialized back to configuration metadata
    Then the resulting metadata contains enough information to reconstruct the approved behavior
    And implementation-only runtime state is not required for the round trip
