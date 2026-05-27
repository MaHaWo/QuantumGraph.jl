Feature: Common interfaces, registry, and utility contracts
  As a QuantumGraph implementer and downstream extension author
  I want common configuration and registry contracts to be observable through public behavior
  So that config-driven components can be composed without depending on private implementation files

  Background:
    Given QuantumGraph is used as an importable Julia library
    And parser internals are not part of the public behavior contract

  @interfaces @approved
  Scenario: Registry lookup resolves registered Julia objects by stable identifiers
    Given a Julia object has been registered with a stable module-qualified identifier
    When a configuration value references that identifier
    Then QuantumGraph resolves the reference to the registered Julia object
    And the resolved object can be used through QuantumGraph's public Julia API
    And repeated lookups of the same identifier return the same public object binding

  @interfaces @approved
  Scenario: Registry lookup rejects unknown modules or object names clearly
    Given a configuration value references an unknown module or object name
    When QuantumGraph resolves the reference through the registry
    Then resolution fails
    And the error identifies the unresolved module or object name
    And no placeholder object is returned as successful

  @interfaces @approved
  Scenario: Configurable objects expose reconstruction metadata
    Given a public QuantumGraph object was constructed from configuration metadata
    When downstream code asks for the object's configuration metadata
    Then the metadata includes the public type identifier and constructor parameters needed for reconstruction
    And implementation-only runtime state is omitted from the metadata
    And the metadata can be serialized without requiring private source paths

  @interfaces @approved
  Scenario: Nested configuration paths are read and written consistently
    Given a nested configuration contains a value at a multi-segment path
    When QuantumGraph reads and updates that nested configuration path
    Then the original value can be retrieved by the same path
    And the updated value appears only at the requested path
    And missing intermediate path segments are reported with the missing segment name

  @interfaces @approved
  Scenario: Public utility errors preserve user-facing context
    Given a public QuantumGraph utility validates a user-provided path, field name, or object identifier
    When public utility validation fails
    Then the error message includes the invalid value
    And the error message identifies the public operation being attempted
    And the error does not expose private implementation stack details as the primary explanation
