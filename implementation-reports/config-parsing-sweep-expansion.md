# Implementation Report: Config parsing and sweep expansion

## Module
Task 4 — Config parsing and sweep expansion

## Implemented
- Added `src/Config.jl` and included it from `src/QuantumGraph.jl`.
- Implemented public configuration tag value types:
  - `Sweep`
  - `CoupledSweep`
  - `InclusiveRange`
  - `RandomUniform`
  - `Reference`
  - `ObjectReference`
- Implemented user-facing `ConfigError`.
- Implemented public config APIs:
  - `supported_config_tags`
  - `load_config`
  - `expand_range`
  - `expand_config`
  - `resolve_config`
- Updated `test/steps/config_object_resolution_steps.jl` to exercise public config behavior directly.
- Added `@config` tags for module-specific BDD execution.

## Acceptance criteria
- Approved custom configuration tags are recognized and represented by Julia values.
- Inclusive range expansion includes the start and includes the stop when it lies on the step sequence.
- Zero-step ranges fail with a clear range-step error.
- Independent sweeps produce one concrete run configuration per swept value.
- Coupled sweeps reject mismatched lengths without accepting partial expansion.
- References resolve previously defined configuration paths and reject missing paths clearly.
- Object references resolve through QuantumGraph's Julia object registry without Python imports.
- Configurable objects can round-trip reconstruction metadata without runtime-only state.

## BDD test results
Command:

```bash
julia --project=. -e 'using Behavior; ok=Behavior.runspec(pwd(); featurepath=joinpath(pwd(), "specs"), stepspath=joinpath(pwd(), "test", "steps"), tags="@config"); exit(ok ? 0 : 1)'
```

Result: PASS — 7 scenarios succeeded, 0 failed.

## Decisions made
- `load_config` implements the approved public tag behavior for simple BDD fixture syntax rather than committing QuantumGraph to a specific YAML parser at this layer.
- Sweep expansion returns named tuples with `name` and `config` fields so run naming remains observable without requiring a Python naming convention.
- References resolve against the original configuration root and object references resolve through the public registry implemented in Task 2.

## Decisions requiring human confirmation
None.

## Deviations from plan
None.
