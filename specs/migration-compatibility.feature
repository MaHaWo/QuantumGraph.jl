Feature: Migration compatibility and deferred distributed behavior
  As a downstream user migrating from QuantumGravPy
  I want QuantumGraph to document supported and deferred compatibility boundaries
  So that unsupported distributed or checkpoint behavior is explicit rather than partially broken

  Background:
    Given QuantumGraph is a single-machine Julia library
    And distributed or multi-machine training is outside the current public surface

  Scenario: Distributed training APIs are not advertised as supported
    Given a downstream user reads the migration compatibility documentation
    When they inspect the supported execution modes
    Then single-machine execution is documented as supported
    And DDP, multi-node, or cluster training is documented as deferred or unsupported
    And no public API is described as a supported distributed training entry point

  Scenario: Distributed training requests fail with an intentional deferral error
    Given a workflow configuration requests DDP, multi-node, or distributed training
    When QuantumGraph validates the training configuration
    Then validation fails
    And the error identifies distributed training as outside the current scope
    And the error is not reported as an unrelated missing field or backend crash

  Scenario: Julia-native checkpoints are documented as the supported checkpoint format
    Given a downstream user reads the checkpoint compatibility notes
    When they inspect supported checkpoint artifact types
    Then Julia-native checkpoint artifacts are documented as supported
    And Python Torch checkpoint compatibility is documented as deferred, unsupported, or requiring explicit conversion
    And the documentation identifies which artifacts can be migrated directly

  Scenario: Unsupported legacy checkpoint inputs fail clearly
    Given a workflow is configured to load an unsupported legacy checkpoint artifact
    When QuantumGraph validates the checkpoint input
    Then validation fails
    And the error identifies the unsupported checkpoint artifact type
    And the error points users toward the documented compatibility boundary

  Scenario: Compatibility documentation is reachable from public package documentation
    Given the README or package documentation is available
    When a downstream user looks for migration compatibility information
    Then the documentation links to or includes the compatibility notes
    And supported artifact types are distinguished from deferred artifact types
    And deferred behavior is described as intentional rather than accidental
