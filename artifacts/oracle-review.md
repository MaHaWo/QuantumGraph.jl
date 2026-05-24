# Oracle Review: QuantumGraph.jl Discovery Plan

## Summary

The migration plan is broadly aligned with the approved requirements: whole-package Julia migration, library-only delivery, behavior-boundary compatibility, BDD-first testing, YAML/Zarr priority, CUDA required, and DDP deferred. The plan is usable for Step 4, but several items should be tightened before implementation handoff or BDD fanout.

## Blockers / high-priority concerns

1. **BDD and interface ordering is too loose.** Task 3 says BDD specs can run in parallel with interface work if names are stable, but BDD scenarios for config registry, graph sample/batching, model construction, training artifacts, and tuning depend on interface contracts. Freeze `Interfaces.jl`, object registry semantics, graph sample/batch contract, and artifact schemas before module-specific BDD review.

2. **Graph sample/batching contract is a hidden dependency.** Dataset, model components, composite GNN model, training, and CUDA all depend on the exact Julia graph representation. The plan notes this as an open validation in Task 6, but it should be an explicit early design/prototype deliverable before Tasks 6-8.

3. **CUDA validation is too late.** Requirements say CUDA support is important. The plan defers CUDA validation to Task 12 after training integration, but CUDA feasibility may constrain graph representation, batching, model layer choices, and Flux/GraphNeuralNetworks compatibility. Add an early CUDA/GraphNeuralNetworks/Flux feasibility spike before committing the model/data interfaces.

4. **Enzyme.jl is only mentioned late.** Requirements specifically ask whether Enzyme.jl can be the differentiation backend. The plan treats it mostly as a risk and Task 12 validation item. This should be an early feasibility decision because it may affect model implementation style, AD-compatible data structures, mutation choices, and CUDA support.

5. **Artifact compatibility is under-specified.** Requirements say existing artifacts are important, especially config and Zarr, but also checkpoints, Optuna outputs, and training logs/reports. The plan says Torch checkpoints are likely Julia-native unless compatibility is feasible and says Optuna outputs should be read or migrated, but no concrete compatibility matrix is planned until Task 13. Move the artifact support matrix earlier and define per artifact: read existing, write compatible, convert, or explicitly unsupported/deferred.

## Requirements/plan drift

- **Public API/export behavior:** Characterization C001 locks exact Python `__all__` including order, while requirements allow idiomatic Julia APIs if behavior boundaries are preserved. The plan says exact `__all__` ordering is not required unless later required. This is probably acceptable, but BDD must define what “behavior boundary” means for the public Julia surface so C001 is not accidentally ignored.

- **“No redesign candidates” vs mutation flexibility:** Requirements say current design decisions are intentional and not redesign targets, but also allow mutation internals to change if behavior boundaries are preserved. The plan follows this, but BDD must explicitly cover user-visible effects of config name mutation and early-stopping task-state mutation.

- **DDP deferral:** The plan correctly defers DDP, but C026 still needs an explicit BDD/documentation scenario that DDP is intentionally absent/deferred, not silently dropped.

- **Requirements artifact status:** `requirements.md` is approved at the top but still contains a stale final “Approval gate” sentence. Not a planning blocker, but it should be cleaned before presenting final artifacts.

## Missing or underdefined module scope

- **Registry/object resolution:** `!pyobject` equivalent is central but underdefined. The plan should specify registry namespace format, user extension mechanism, error behavior, serialization round-trip behavior, and how old Python fully-qualified paths map to Julia implementations.

- **Training/evaluation schemas:** The plan mentions report schemas and structural artifacts but does not list required column names/keys/files for validator/tester/training reports. These should be derived from `evaluate.py`, `train.py`, existing tests, and characterization C023-C025 before BDD.

- **Criterion/task callable contracts:** Overview notes criterion signatures differ between training (`criterion(outputs, data, trainer)`) and evaluation (`criterion(outputs, data)`). The plan mentions adapters but does not define the callable interface. This is a hidden integration dependency for Evaluation, Training, and GNNModel.

- **Tuning outputs:** `QGTune/tune.py` is included, but characterization says end-to-end tuning and saved best-config YAML need more tests. The plan should either add pre-implementation characterization/fixture work or define BDD from existing `test_tune.py` with explicit gaps.

## Testing gaps

- Add explicit coverage mapping for C001-C028. The plan says all behaviors are converted, but module acceptances visibly cover mainly C004-C025. C001-C003, C026-C028 need explicit destinations: public surface/package metadata, DDP deferral docs, CUDA/environment validation, warnings/incidental behavior policy.

- Add Python-created fixture commitments for Zarr beyond tiny synthetic stores if real research layouts exist. Current characterization warns Zarr compatibility is based on small fixtures only.

- Documentation examples in `docs/training_a_model.md` and `docs/hparam_tuning.md` were not executed. If those examples are part of expected library usage, add BDD or doc-example smoke tests.

- Choose or define the Julia BDD workflow/runner early. The plan says feature files then conversion to tests, but does not decide whether `.feature` files are executable or review-only.

## Output repository/layout issues

- `/Users/hmack/Development/QuantumGraph.jl` currently appears not to be a Julia package yet (`Project.toml` not present); Task 1 handles this.
- There is an existing root-level `plan.md` outside `artifacts/`. Confirm whether it is stale or intentional to avoid confusing future implementers.
- The plan is a single `plan.md` under `artifacts/`. This satisfies the directory requirement, but module-by-module BDD handoff may be easier if later split into separate module files or issues.

## Alternative decomposition to consider

Before implementation, split the early foundation into explicit feasibility/design modules:

1. Package skeleton/test harness.
2. Compatibility matrix and artifact fixture inventory.
3. Object registry/config interface design.
4. Graph sample/batching/device prototype with Flux + GraphNeuralNetworks + CUDA feasibility.
5. AD feasibility decision: Zygote default vs Enzyme viability.

Then proceed with Config/Zarr, Datasets/Models, GNNModel, Evaluation/EarlyStopping, Tuning, Training, CUDA hardening, public docs/exports.

## Recommended next action

Revise the plan lightly before implementation handoff: add early feasibility/design tasks for graph representation, CUDA, Enzyme, registry semantics, and artifact compatibility matrix; add explicit C001-C028 coverage mapping; and clean the stale approval-gate line in `requirements.md`.
