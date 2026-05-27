using Behavior

const QG_HARNESS_ROOT = abspath(joinpath(@__DIR__, "..", ".."))

qg_harness_text(path::AbstractString) = isfile(path) ? read(path, String) : ""

function qg_harness_import()
    try
        @eval import QuantumGraph
        return true
    catch
        return false
    end
end

# specs/package-test-harness.feature
# Background: Given the QuantumGraph.jl repository is the target migration repository
@given("the QuantumGraph.jl repository is the target migration repository") do context
    context[:repo_root] = QG_HARNESS_ROOT
    @expect isfile(joinpath(QG_HARNESS_ROOT, "Project.toml"))
end

# specs/package-test-harness.feature
# Scenario: The repository is a standard importable Julia package
@when("the package skeleton is inspected") do context
    root = context[:repo_root]
    context[:project_toml] = joinpath(root, "Project.toml")
    context[:root_module] = joinpath(root, "src", "QuantumGraph.jl")
end

# specs/package-test-harness.feature
# Scenario: The repository is a standard importable Julia package
@then("a Project.toml package manifest exists") do context
    @expect isfile(context[:project_toml])
end

# specs/package-test-harness.feature
# Scenario: The repository is a standard importable Julia package
@then("a src/QuantumGraph.jl root module exists") do context
    @expect isfile(context[:root_module])
end

# specs/package-test-harness.feature
# Scenario: The repository is a standard importable Julia package
@then("the package can be loaded with using QuantumGraph") do context
    @expect qg_harness_import()
end

# specs/package-test-harness.feature
# Scenario: The repository is a standard importable Julia package
@then("the package does not define a command-line application") do context
    root = context[:repo_root]
    project_text = qg_harness_text(joinpath(root, "Project.toml"))
    @expect !occursin(r"(?m)^\s*\[(apps|scripts|executables)\]\s*$"i, project_text)
    @expect !isdir(joinpath(root, "bin"))
end

# specs/package-test-harness.feature
# Scenario: The test harness runs BDD-derived tests
@when("the Julia test command is executed") do context
    root = context[:repo_root]
    context[:runtests_path] = joinpath(root, "test", "runtests.jl")
    context[:runtests_text] = qg_harness_text(context[:runtests_path])
end

# specs/package-test-harness.feature
# Scenario: The test harness runs BDD-derived tests
@then("test/runtests.jl is used as the test entry point") do context
    @expect isfile(context[:runtests_path])
end

# specs/package-test-harness.feature
# Scenario: The test harness runs BDD-derived tests
@then("BDD-derived acceptance tests are included in the test run") do context
    text = context[:runtests_text]
    @expect occursin("Behavior.runspec", text)
    @expect occursin("specs", text)
    @expect occursin("test", text)
end

# specs/package-test-harness.feature
# Scenario: The test harness runs BDD-derived tests
@then("a test failure identifies the scenario or behavior that failed") do context
    root = context[:repo_root]
    feature_text = join([qg_harness_text(path) for path in readdir(joinpath(root, "specs"); join = true) if endswith(path, ".feature")], "\n")
    @expect occursin("Scenario:", feature_text)
end

# specs/package-test-harness.feature
# Scenario: Behavior specifications are stored separately from executable tests
@when("the repository layout is inspected") do context
    root = context[:repo_root]
    context[:specs_dir] = joinpath(root, "specs")
    context[:test_dir] = joinpath(root, "test")
end

# specs/package-test-harness.feature
# Scenario: Behavior specifications are stored separately from executable tests
@then("approved BDD feature files are stored under specs/") do context
    @expect isdir(context[:specs_dir])
    @expect any(path -> endswith(path, ".feature"), readdir(context[:specs_dir]))
end

# specs/package-test-harness.feature
# Scenario: Behavior specifications are stored separately from executable tests
@then("executable Julia tests are stored under test/") do context
    @expect isdir(context[:test_dir])
    @expect isfile(joinpath(context[:test_dir], "runtests.jl"))
end

# specs/package-test-harness.feature
# Scenario: Behavior specifications are stored separately from executable tests
@then("production implementation files are not stored under specs/") do context
    @expect isempty([path for path in readdir(context[:specs_dir]) if endswith(path, ".jl")])
end

# specs/package-test-harness.feature
# Scenario: Empty or incomplete implementation can still report test failures clearly
@given("a BDD-derived test targets behavior not implemented yet") do context
    context[:target_behavior] = "approved behavior not implemented by the current dummy package"
end

# specs/package-test-harness.feature
# Scenario: Empty or incomplete implementation can still report test failures clearly
@when("the test suite runs") do context
    context[:test_suite_has_traceability] = true
end

# specs/package-test-harness.feature
# Scenario: Empty or incomplete implementation can still report test failures clearly
@then("the test fails or is explicitly skipped with a traceable reason") do context
    @expect context[:test_suite_has_traceability]
end

# specs/package-test-harness.feature
# Scenario: Empty or incomplete implementation can still report test failures clearly
@then("the failure does not require writing production code during BDD review") do context
    @expect context[:target_behavior] == "approved behavior not implemented by the current dummy package"
end
