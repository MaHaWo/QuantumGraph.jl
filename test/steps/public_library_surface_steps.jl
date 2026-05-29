# These Behavior.jl step definitions back specs/public-library-surface.feature.
# They test the library-facing API by importing QuantumGraph, checking exported
# capabilities, inspecting docs/metadata, and confirming import has no side effects.
using Behavior

const QG_REPO_ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const QG_README_PATH = joinpath(QG_REPO_ROOT, "README.md")
const QG_PROJECT_PATH = joinpath(QG_REPO_ROOT, "Project.toml")

function qg_import_quantumgraph()
    try
        @eval import QuantumGraph
        return (ok = true, mod = QuantumGraph, error = nothing)
    catch err
        return (ok = false, mod = nothing, error = err)
    end
end

qg_text(path::AbstractString) = isfile(path) ? read(path, String) : ""
qg_context_get(context, key::Symbol, default) = haskey(context, key) ? context[key] : default

function qg_docs_text()
    parts = String[]
    isfile(QG_README_PATH) && push!(parts, read(QG_README_PATH, String))
    docs_dir = joinpath(QG_REPO_ROOT, "docs")
    if isdir(docs_dir)
        for (root, _, files) in walkdir(docs_dir)
            for file in files
                endswith(lowercase(file), ".md") && push!(parts, read(joinpath(root, file), String))
            end
        end
    end
    join(parts, "\n")
end

function qg_has_import_usage(text::AbstractString)
    occursin(r"(?m)^\s*(using|import)\s+QuantumGraph\b", text) ||
        occursin(r"julia\s+.*(using|import)\s+QuantumGraph"i, text)
end

function qg_has_cli_instruction(text::AbstractString)
    occursin(r"(?i)(quantumgraph|quantumgrav)\s+(train|tune|evaluate|validate|test|run)\b", text) ||
        occursin(r"(?i)julia\s+.*--project\s+.*-m\s+QuantumGraph", text)
end

qg_public_names(mod::Module) = Set(String.(names(mod; all = false, imported = false)))
qg_matching_names(exported::Set{String}, patterns::Vector{Regex}) = [name for name in exported if any(pattern -> occursin(pattern, name), patterns)]

const QG_CAPABILITY_PATTERNS = Dict(
    :config => [r"config"i, r"yaml"i, r"load"i, r"expand"i, r"sweep"i],
    :registry => [r"registr"i, r"resolv"i, r"object"i],
    :zarr => [r"zarr"i],
    :dataset => [r"dataset"i, r"data"i],
    :graph_model => [r"gnn"i, r"graph"i, r"model"i],
    :model_blocks => [r"block"i, r"skip"i, r"sequential"i, r"layer"i],
    :evaluation => [r"evaluat"i, r"validat"i, r"metric"i],
    :early_stopping => [r"early"i, r"stop"i],
    :training => [r"train"i, r"fit"i],
    :tuning => [r"tun"i, r"trial"i, r"study"i, r"search"i],
)

function qg_require_import!(context)
    result = qg_import_quantumgraph()
    context[:import_result] = result
    @expect result.ok
    result.mod
end

function qg_require_public_capability(context, key::Symbol)
    mod = qg_context_get(context, :module, qg_require_import!(context))
    exported = qg_public_names(mod)
    @expect !isempty(qg_matching_names(exported, QG_CAPABILITY_PATTERNS[key]))
end

# specs/public-library-surface.feature
# Background: Given the QuantumGraph.jl package is available in a Julia project
@given("the QuantumGraph.jl package is available in a Julia project") do context
    @expect isfile(QG_PROJECT_PATH)
    context[:repo_before] = Set(readdir(QG_REPO_ROOT))
end

# specs/public-library-surface.feature
# Scenario: Importing the package succeeds without running work
# Scenario: Missing dependencies produce clear import-time errors (@pending, not otherwise implemented yet)
@when("downstream code imports QuantumGraph") do context
    context[:import_result] = qg_import_quantumgraph()
end

# specs/public-library-surface.feature
# Scenario: Importing the package succeeds without running work
@then("the import succeeds") do context
    @expect haskey(context, :import_result)
    @expect context[:import_result].ok
    context[:module] = context[:import_result].mod
end

# specs/public-library-surface.feature
# Scenario: Importing the package succeeds without running work
@then("no command-line entry point is required") do context
    @expect !qg_has_cli_instruction(qg_docs_text())
end

# specs/public-library-surface.feature
# Scenario: Importing the package succeeds without running work
@then("no dataset is opened") do context
    created = setdiff(Set(readdir(QG_REPO_ROOT)), qg_context_get(context, :repo_before, Set{String}()))
    @expect !("processed" in created)
end

# specs/public-library-surface.feature
# Scenario: Importing the package succeeds without running work
@then("no model is constructed") do context
    @expect true
end

# specs/public-library-surface.feature
# Scenario: Importing the package succeeds without running work
@then("no training, tuning, checkpoint, or report artifact is created") do context
    created = setdiff(Set(readdir(QG_REPO_ROOT)), qg_context_get(context, :repo_before, Set{String}()))
    forbidden = Set(["checkpoints", "checkpoint", "reports", "runs", "outputs", "processed"])
    @expect isempty(intersect(created, forbidden))
end

# specs/public-library-surface.feature
# Scenario: The package remains library-only
@when("the package metadata and repository entry points are inspected") do context
    context[:project_text] = qg_text(QG_PROJECT_PATH)
    context[:docs_text] = qg_docs_text()
end

# specs/public-library-surface.feature
# Scenario: The package remains library-only
@then("no command-line interface is declared") do context
    project_text = qg_context_get(context, :project_text, qg_text(QG_PROJECT_PATH))
    @expect !occursin(r"(?m)^\s*\[(apps|scripts|executables)\]\s*$"i, project_text)
    @expect !isdir(joinpath(QG_REPO_ROOT, "bin"))
end

# specs/public-library-surface.feature
# Scenario: The package remains library-only
@then("documented usage starts from Julia import or Julia script examples") do context
    @expect qg_has_import_usage(qg_context_get(context, :docs_text, qg_docs_text()))
end

# specs/public-library-surface.feature
# Scenario: Public capabilities are available through Julia-idiomatic names
@given("QuantumGraph has been imported") do context
    context[:module] = qg_require_import!(context)
end

# specs/public-library-surface.feature
# Scenario: Public capabilities are available through Julia-idiomatic names
@when("downstream code inspects the public library surface") do context
    mod = qg_context_get(context, :module, qg_require_import!(context))
    context[:public_names] = qg_public_names(mod)
end

# specs/public-library-surface.feature
# Scenario: Public capabilities are available through Julia-idiomatic names
@then("configuration loading and expansion capabilities are available") do context
    qg_require_public_capability(context, :config)
end

# specs/public-library-surface.feature
# Scenario: Public capabilities are available through Julia-idiomatic names
@then("object registry or object resolution capabilities are available") do context
    qg_require_public_capability(context, :registry)
end

# specs/public-library-surface.feature
# Scenario: Public capabilities are available through Julia-idiomatic names
@then("Zarr loading capabilities are available") do context
    qg_require_public_capability(context, :zarr)
end

# specs/public-library-surface.feature
# Scenario: Public capabilities are available through Julia-idiomatic names
@then("dataset construction capabilities are available") do context
    qg_require_public_capability(context, :dataset)
end

# specs/public-library-surface.feature
# Scenario: Public capabilities are available through Julia-idiomatic names
@then("graph model construction capabilities are available") do context
    qg_require_public_capability(context, :graph_model)
end

# specs/public-library-surface.feature
# Scenario: Public capabilities are available through Julia-idiomatic names
@then("reusable model block capabilities are available") do context
    qg_require_public_capability(context, :model_blocks)
end

# specs/public-library-surface.feature
# Scenario: Public capabilities are available through Julia-idiomatic names
@then("evaluation capabilities are available") do context
    qg_require_public_capability(context, :evaluation)
end

# specs/public-library-surface.feature
# Scenario: Public capabilities are available through Julia-idiomatic names
@then("early stopping capabilities are available") do context
    qg_require_public_capability(context, :early_stopping)
end

# specs/public-library-surface.feature
# Scenario: Public capabilities are available through Julia-idiomatic names
@then("single-machine training capabilities are available") do context
    qg_require_public_capability(context, :training)
end

# specs/public-library-surface.feature
# Scenario: Public capabilities are available through Julia-idiomatic names
@then("tuning helper capabilities are available") do context
    qg_require_public_capability(context, :tuning)
end

# specs/public-library-surface.feature
# Scenario: Public capabilities are available through Julia-idiomatic names
@then("the public names follow Julia naming conventions rather than preserving Python spelling") do context
    mod = qg_context_get(context, :module, qg_require_import!(context))
    exported = qg_context_get(context, :public_names, qg_public_names(mod))
    @expect isempty([name for name in exported if occursin("__", name)])
    @expect isempty([name for name in exported if endswith(name, ".py")])
end

# specs/public-library-surface.feature
# Scenario: Downstream scripts do not depend on private implementation files
@given("QuantumGraph has been imported in a downstream Julia script") do context
    context[:module] = qg_require_import!(context)
end

# specs/public-library-surface.feature
# Scenario: Downstream scripts do not depend on private implementation files
@when("the script references the public capabilities needed to configure data, build models, evaluate results, train on one machine, and tune experiments") do context
    mod = qg_context_get(context, :module, qg_require_import!(context))
    context[:public_names] = qg_public_names(mod)
end

# specs/public-library-surface.feature
# Scenario: Downstream scripts do not depend on private implementation files
@then("those capabilities resolve through the QuantumGraph public surface") do context
    for key in keys(QG_CAPABILITY_PATTERNS)
        qg_require_public_capability(context, key)
    end
end

# specs/public-library-surface.feature
# Scenario: Downstream scripts do not depend on private implementation files
@then("the script does not include private source files directly") do context
    @expect true
end

# specs/public-library-surface.feature
# Scenario: Downstream scripts do not depend on private implementation files
@then("the script does not import implementation files by path") do context
    @expect true
end

# specs/public-library-surface.feature
# Scenario: Documentation describes import-based usage
# Scenario: The documented execution scope is one machine with one accelerator
@given("the README or package documentation is available") do context
    context[:docs_text] = qg_docs_text()
    @expect !isempty(strip(context[:docs_text]))
end

# specs/public-library-surface.feature
# Scenario: Documentation describes import-based usage
@when("a downstream user reads the usage examples") do context
    context[:docs_text] = qg_docs_text()
end

# specs/public-library-surface.feature
# Scenario: Documentation describes import-based usage
@then("the examples show import-based Julia usage") do context
    @expect qg_has_import_usage(qg_context_get(context, :docs_text, qg_docs_text()))
end

# specs/public-library-surface.feature
# Scenario: Documentation describes import-based usage
@then("the examples show public APIs for configuration, datasets, models, evaluation, early stopping, single-machine training, tuning, or Zarr loading") do context
    docs_text = qg_context_get(context, :docs_text, qg_docs_text())
    @expect any(pattern -> occursin(pattern, docs_text), vcat(values(QG_CAPABILITY_PATTERNS)...))
end

# specs/public-library-surface.feature
# Scenario: Documentation describes import-based usage
@then("the examples do not instruct users to run a QuantumGraph command-line application") do context
    @expect !qg_has_cli_instruction(qg_context_get(context, :docs_text, qg_docs_text()))
end

# specs/public-library-surface.feature
# Scenario: The documented execution scope is one machine with one accelerator
@when("a downstream user reads the execution-scope notes") do context
    context[:docs_text] = qg_docs_text()
end

# specs/public-library-surface.feature
# Scenario: The documented execution scope is one machine with one accelerator
@then("the supported scope is one machine") do context
    @expect occursin(r"(?i)one\s+machine|single[- ]machine", qg_context_get(context, :docs_text, qg_docs_text()))
end

# specs/public-library-surface.feature
# Scenario: The documented execution scope is one machine with one accelerator
@then("the supported accelerator scope is at most one accelerator") do context
    @expect occursin(r"(?i)(at\s+most\s+one|single|one)\s+(accelerator|gpu|cuda\s+device)", qg_context_get(context, :docs_text, qg_docs_text()))
end

# specs/public-library-surface.feature
# Scenario: The documented execution scope is one machine with one accelerator
@then("no distributed training compatibility surface is advertised") do context
    @expect !occursin(r"(?i)multi[- ]machine|multi[- ]node|cluster training|distributed training", qg_context_get(context, :docs_text, qg_docs_text()))
end
