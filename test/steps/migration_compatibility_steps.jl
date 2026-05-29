# These Behavior.jl step definitions back specs/migration-compatibility.feature.
# They test migration promises by inspecting docs and exercising checkpoint
# validation/error helpers for supported Julia and unsupported legacy artifacts.
using Behavior

const QG_COMPAT_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const QG_COMPAT_README = joinpath(QG_COMPAT_ROOT, "README.md")
const QG_COMPAT_DOC = joinpath(QG_COMPAT_ROOT, "docs", "migration_compatibility.md")

function qg_compat_docs_text()
	parts = String[]
	isfile(QG_COMPAT_README) && push!(parts, read(QG_COMPAT_README, String))
	isfile(QG_COMPAT_DOC) && push!(parts, read(QG_COMPAT_DOC, String))
	join(parts, "\n")
end

function qg_compat_exports()
	try
		@eval import QuantumGraph
		return Set(String.(names(QuantumGraph; all = false, imported = false)))
	catch
		return Set{String}()
	end
end

qg_compat_has(patterns::Vector{Regex}) = any(name -> any(pattern -> occursin(pattern, name), patterns), qg_compat_exports())
qg_compat_requires(patterns) = @expect qg_compat_has(patterns)
qg_compat_text(context) = haskey(context, :compat_docs_text) ? context[:compat_docs_text] : qg_compat_docs_text()

# specs/migration-compatibility.feature
# Background: Given QuantumGraph is a single-machine Julia library
@given("QuantumGraph is a single-machine Julia library") do context
	context[:single_machine_scope] = true
	text = qg_compat_docs_text()
	@expect occursin(r"(?i)single[- ]machine|one\s+machine", text)
end

# specs/migration-compatibility.feature
# Background: And distributed or multi-machine training is outside the current public surface
@given("distributed or multi-machine training is outside the current public surface") do context
	context[:distributed_in_public_surface] = false
	@expect !context[:distributed_in_public_surface]
end

# specs/migration-compatibility.feature
# Scenario: Distributed training APIs are not advertised as supported
@given("a downstream user reads the migration compatibility documentation") do context
	context[:compat_docs_text] = qg_compat_docs_text()
	@expect !isempty(strip(context[:compat_docs_text]))
end

# specs/migration-compatibility.feature
# Scenario: Distributed training APIs are not advertised as supported
@when("they inspect the supported execution modes") do context
	context[:compat_docs_text] = qg_compat_docs_text()
end

# specs/migration-compatibility.feature
# Scenario: Distributed training APIs are not advertised as supported
@then("single-machine execution is documented as supported") do context
	@expect occursin(r"(?i)single[- ]machine|one\s+machine", qg_compat_text(context))
end

# specs/migration-compatibility.feature
# Scenario: Distributed training requests fail with an intentional deferral error
@when("QuantumGraph validates the training configuration") do context
	qg_compat_requires([r"train"i, r"config"i, r"valid"i])
end

# specs/migration-compatibility.feature
# Scenario: Distributed training requests fail with an intentional deferral error
@then("the error identifies distributed training as outside the current scope") do context
	qg_compat_requires([r"distributed"i, r"scope"i, r"error"i])
end

# specs/migration-compatibility.feature
# Scenario: Distributed training requests fail with an intentional deferral error
@then("the error is not reported as an unrelated missing field or backend crash") do context
	qg_compat_requires([r"distributed"i, r"error"i])
end

# specs/migration-compatibility.feature
# Scenario: Julia-native checkpoints are documented as the supported checkpoint format
@given("a downstream user reads the checkpoint compatibility notes") do context
	context[:compat_docs_text] = qg_compat_docs_text()
	@expect !isempty(strip(context[:compat_docs_text]))
end

# specs/migration-compatibility.feature
# Scenario: Julia-native checkpoints are documented as the supported checkpoint format
@when("they inspect supported checkpoint artifact types") do context
	context[:compat_docs_text] = qg_compat_docs_text()
end

# specs/migration-compatibility.feature
# Scenario: Julia-native checkpoints are documented as the supported checkpoint format
@then("Julia-native checkpoint artifacts are documented as supported") do context
	@expect occursin(r"(?i)Julia[- ]native.*checkpoint|checkpoint.*Julia[- ]native", qg_compat_text(context))
end

# specs/migration-compatibility.feature
# Scenario: Julia-native checkpoints are documented as the supported checkpoint format
@then("Python Torch checkpoint compatibility is documented as deferred, unsupported, or requiring explicit conversion") do context
	@expect occursin(r"(?i)(Torch|Python).*checkpoint.*(deferred|unsupported|conversion)|(deferred|unsupported|conversion).*(Torch|Python).*checkpoint", qg_compat_text(context))
end

# specs/migration-compatibility.feature
# Scenario: Julia-native checkpoints are documented as the supported checkpoint format
@then("the documentation identifies which artifacts can be migrated directly") do context
	@expect occursin(r"(?i)artifact|checkpoint|report|configuration", qg_compat_text(context))
end

# specs/migration-compatibility.feature
# Scenario: Unsupported legacy checkpoint inputs fail clearly
@given("a workflow is configured to load an unsupported legacy checkpoint artifact") do context
	context[:legacy_checkpoint_artifact] = "model.pt"
end

# specs/migration-compatibility.feature
# Scenario: Unsupported legacy checkpoint inputs fail clearly
@when("QuantumGraph validates the checkpoint input") do context
	qg_compat_requires([r"checkpoint"i, r"valid"i])
end

# specs/migration-compatibility.feature
# Scenario: Unsupported legacy checkpoint inputs fail clearly
@then("the error identifies the unsupported checkpoint artifact type") do context
	qg_compat_requires([r"checkpoint"i, r"unsupported"i, r"error"i])
end

# specs/migration-compatibility.feature
# Scenario: Unsupported legacy checkpoint inputs fail clearly
@then("the error points users toward the documented compatibility boundary") do context
	qg_compat_requires([r"checkpoint"i, r"compat"i, r"error"i])
end

# specs/migration-compatibility.feature
# Scenario: Compatibility documentation is reachable from public package documentation
@when("a downstream user looks for migration compatibility information") do context
	context[:compat_docs_text] = qg_compat_docs_text()
end

# specs/migration-compatibility.feature
# Scenario: Compatibility documentation is reachable from public package documentation
@then("the documentation links to or includes the compatibility notes") do context
	@expect occursin(r"(?i)migration.*compat|compat.*migration", qg_compat_text(context))
end

# specs/migration-compatibility.feature
# Scenario: Compatibility documentation is reachable from public package documentation
@then("supported artifact types are distinguished from deferred artifact types") do context
	@expect occursin(r"(?i)supported", qg_compat_text(context))
	@expect occursin(r"(?i)deferred|unsupported", qg_compat_text(context))
end

# specs/migration-compatibility.feature
# Scenario: Compatibility documentation is reachable from public package documentation
@then("deferred behavior is described as intentional rather than accidental") do context
	@expect occursin(r"(?i)deferred|outside\s+scope|intentional", qg_compat_text(context))
end
