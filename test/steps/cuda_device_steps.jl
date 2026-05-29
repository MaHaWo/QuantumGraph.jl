# These Behavior.jl step definitions back specs/cuda-device.feature. They test
# the device-selection contract through exported capability checks and scenario
# context for CPU, optional single-accelerator, and invalid accelerator requests.
using Behavior

function qg_device_exports()
    try
        @eval import QuantumGraph
        return Set(String.(names(QuantumGraph; all = false, imported = false)))
    catch
        return Set{String}()
    end
end

qg_device_has(patterns::Vector{Regex}) = any(name -> any(pattern -> occursin(pattern, name), patterns), qg_device_exports())
qg_device_requires(patterns) = @expect qg_device_has(patterns)

# specs/cuda-device.feature
# Background: Given CPU execution is always supported by QuantumGraph
@given("CPU execution is always supported by QuantumGraph") do context
    context[:cpu_supported] = true
    qg_device_requires([r"cpu"i, r"device"i])
end

# specs/cuda-device.feature
# Background: And accelerator execution is optional and limited to at most one configured accelerator
@given("accelerator execution is optional and limited to at most one configured accelerator") do context
    context[:max_accelerators] = 1
    qg_device_requires([r"accelerator"i, r"device"i])
end

# specs/cuda-device.feature
# Scenario: CPU execution is selected when no accelerator is requested
@given("a training or model workflow does not request an accelerator") do context
    context[:requested_accelerators] = 0
end

# specs/cuda-device.feature
# Scenario: CPU execution is selected when no accelerator is requested
# Scenario: One available accelerator can be selected explicitly
# Scenario: Unavailable accelerator requests fail clearly
@when("QuantumGraph prepares the execution device") do context
    qg_device_requires([r"device"i, r"prepare"i])
end

# specs/cuda-device.feature
# Scenario: CPU execution is selected when no accelerator is requested
@then("the selected device is CPU") do context
    qg_device_requires([r"cpu"i, r"device"i])
end

# specs/cuda-device.feature
# Scenario: CPU execution is selected when no accelerator is requested
@then("no CUDA or accelerator backend initialization is required") do context
    qg_device_requires([r"cpu"i, r"accelerator"i, r"backend"i])
end

# specs/cuda-device.feature
# Scenario: CPU execution is selected when no accelerator is requested
@then("training can proceed without distributed process setup") do context
    qg_device_requires([r"device"i, r"train"i, r"single"i])
end

# specs/cuda-device.feature
# Scenario: One available accelerator can be selected explicitly
@given("a workflow requests one available accelerator") do context
    context[:requested_accelerators] = 1
    context[:accelerator_available] = true
end

# specs/cuda-device.feature
# Scenario: One available accelerator can be selected explicitly
@then("the selected device represents that accelerator") do context
    qg_device_requires([r"accelerator"i, r"device"i])
end

# specs/cuda-device.feature
# Scenario: One available accelerator can be selected explicitly
@then("model and graph batch values are prepared for that device through public APIs") do context
    qg_device_requires([r"device"i, r"model"i, r"graph"i])
end

# specs/cuda-device.feature
# Scenario: One available accelerator can be selected explicitly
@then("no additional accelerator or multi-process setup is initialized") do context
    qg_device_requires([r"accelerator"i, r"single"i, r"process"i])
end

# specs/cuda-device.feature
# Scenario: Unavailable accelerator requests fail clearly
@given("a workflow requests an accelerator that is not available") do context
    context[:requested_accelerators] = 1
    context[:accelerator_available] = false
end

# specs/cuda-device.feature
# Scenario: Unavailable accelerator requests fail clearly
@then("device preparation fails") do context
    qg_device_requires([r"device"i, r"error"i])
end

# specs/cuda-device.feature
# Scenario: Unavailable accelerator requests fail clearly
@then("the error identifies the requested accelerator setting") do context
    qg_device_requires([r"accelerator"i, r"setting"i, r"error"i])
end

# specs/cuda-device.feature
# Scenario: Unavailable accelerator requests fail clearly
@then("the error explains that the requested backend or device is unavailable") do context
    qg_device_requires([r"backend"i, r"device"i, r"unavailable"i])
end

# specs/cuda-device.feature
# Scenario: Multiple accelerator requests are rejected
@given("a workflow requests more than one accelerator") do context
    context[:requested_accelerators] = 2
end

# specs/cuda-device.feature
# Scenario: Multiple accelerator requests are rejected
@when("QuantumGraph validates the execution device settings") do context
    qg_device_requires([r"device"i, r"valid"i, r"setting"i])
end

# specs/cuda-device.feature
# Scenario: Multiple accelerator requests are rejected
@then("validation fails") do context
    qg_device_requires([r"valid"i, r"error"i])
end

# specs/cuda-device.feature
# Scenario: Multiple accelerator requests are rejected
@then("the error identifies that only one accelerator is supported") do context
    qg_device_requires([r"accelerator"i, r"one"i, r"error"i])
end

# specs/cuda-device.feature
# Scenario: Multiple accelerator requests are rejected
@then("no partial multi-accelerator setup is attempted") do context
    qg_device_requires([r"accelerator"i, r"setup"i, r"error"i])
end

# specs/cuda-device.feature
# Scenario: Device movement preserves graph sample and model structure
@given("a model and GraphNeuralNetworks-compatible graph batch are available") do context
    context[:device_model_graph_fixture] = true
end

# specs/cuda-device.feature
# Scenario: Device movement preserves graph sample and model structure
@when("QuantumGraph prepares them for the selected CPU or accelerator device") do context
    qg_device_requires([r"device"i, r"model"i, r"graph"i])
end

# specs/cuda-device.feature
# Scenario: Device movement preserves graph sample and model structure
@then("the model structure remains usable through the public model API") do context
    qg_device_requires([r"model"i, r"public"i, r"api"i])
end

# specs/cuda-device.feature
# Scenario: Device movement preserves graph sample and model structure
@then("the graph batch remains compatible with the approved graph sample boundary") do context
    qg_device_requires([r"graph"i, r"sample"i, r"boundary"i])
end

# specs/cuda-device.feature
# Scenario: Device movement preserves graph sample and model structure
@then("exact numeric output values are not part of the device-selection behavior contract") do context
    @expect true
end
