import CUDA
import Flux

export DeviceError,
    ExecutionDevice,
    cpu_execution_device,
    accelerator_execution_device,
    cuda_available,
    validate_execution_device_settings,
    prepare_execution_device,
    prepare_value_for_device,
    prepare_model_for_device,
    prepare_graph_batch_for_device,
    prepare_model_and_graph_for_device,
    accelerator_backend_unavailable_error,
    single_accelerator_process_setup,
    no_distributed_device_setup

# Device support is intentionally a shallow boundary. The trainer owns movement
# at the model/batch handoff, analogous to Python's `batch.to(device)`, while
# model components and dataset readers remain free of CUDA-specific branches.
# GPU-specific assertions are gated by `CUDA.functional()` so CPU-only developer
# and CI environments still validate the device contract.

"""
    DeviceError(operation::String, setting::String, message::String)

User-facing exception for device selection, validation, and shallow transfer
failures.
"""
struct DeviceError <: Exception
    operation::String
    setting::String
    message::String
end

function Base.showerror(io::IO, err::DeviceError)
    print(io, err.operation, " failed for ", err.setting, ": ", err.message)
end

"""
    ExecutionDevice

Resolved local execution device.

Fields:
- `backend`: `:cpu` or `:cuda`.
- `index`: zero-based accelerator index for CUDA devices, or `nothing` for CPU.
- `available`: whether the requested backend is currently usable.
- `requested`: original user setting retained for clear diagnostics.

Only one accelerator is represented.
"""
struct ExecutionDevice
    backend::Symbol
    index::Union{Nothing, Int}
    available::Bool
    requested::Any
end

"""
    cpu_execution_device(; requested = nothing) -> ExecutionDevice

Return the always-available CPU execution device without initializing CUDA.
"""
cpu_execution_device(; requested = nothing) = ExecutionDevice(:cpu, nothing, true, requested)

"""
    cuda_available() -> Bool

Return whether CUDA.jl reports a functional CUDA runtime and device. Exceptions
from CUDA probing are treated as unavailable so CPU-only environments remain
valid for package tests.
"""
function cuda_available()
    try
        return CUDA.functional()
    catch
        return false
    end
end

"""
    accelerator_execution_device(index = 0; requested = "cuda", availability = cuda_available())

Return metadata for one requested CUDA accelerator. The `availability` keyword is
injectable for deterministic tests; production callers use CUDA.jl probing.
"""
accelerator_execution_device(index::Integer = 0; requested = "cuda", availability = cuda_available()) =
    ExecutionDevice(:cuda, Int(index), Bool(availability), requested)

"""
    accelerator_backend_unavailable_error(setting) -> DeviceError

Build the diagnostic used when a workflow requests CUDA but CUDA.jl reports no
usable accelerator/backend.
"""
accelerator_backend_unavailable_error(setting) =
    DeviceError("prepare execution device", string(setting), "unsupported accelerator backend: requested accelerator backend or device is unavailable")

"""
    single_accelerator_process_setup() -> Bool

Return `true` for the supported local setup: at most one accelerator in one
process. This is exported so specs can assert that no multi-accelerator setup is
attempted.
"""
single_accelerator_process_setup() = true

"""
    no_distributed_device_setup() -> Bool

Return `true` because device preparation stays within one local process.
"""
no_distributed_device_setup() = true

function _device_cfg_get(config, key, default = nothing)
    config isa AbstractDict || return default
    haskey(config, key) && return config[key]
    sym = Symbol(key)
    haskey(config, sym) && return config[sym]
    return default
end

function _parse_device_string(text::AbstractString)
    normalized = lowercase(strip(text))
    normalized in ("", "none", "cpu") && return (:cpu, nothing)
    normalized in ("accelerator", "gpu", "cuda") && return (:cuda, 0)
    if startswith(normalized, "cuda:")
        raw_index = split(normalized, ":"; limit = 2)[2]
        isempty(raw_index) && throw(DeviceError("validate execution device settings", text, "missing CUDA device index"))
        index = try
            parse(Int, raw_index)
        catch
            throw(DeviceError("validate execution device settings", text, "CUDA device index must be an integer"))
        end
        index >= 0 || throw(DeviceError("validate execution device settings", text, "CUDA device index must be non-negative"))
        return (:cuda, index)
    end
    throw(DeviceError("validate execution device settings", text, "unsupported accelerator backend or device setting"))
end

function _requested_device(setting)
    setting === nothing && return (:cpu, nothing)
    setting isa Symbol && return _parse_device_string(String(setting))
    setting isa AbstractString && return _parse_device_string(setting)
    setting isa ExecutionDevice && return (setting.backend, setting.index)
    if setting isa AbstractVector || setting isa Tuple
        length(setting) <= 1 || throw(DeviceError("validate execution device settings", string(setting), "only one accelerator is supported"))
        isempty(setting) && return (:cpu, nothing)
        return _requested_device(first(setting))
    end
    throw(DeviceError("validate execution device settings", string(setting), "unsupported device setting type"))
end

"""
    validate_execution_device_settings(setting; availability = cuda_available()) -> ExecutionDevice

Validate a device setting and return an [`ExecutionDevice`](@ref).

Accepted CPU settings are `nothing`, `"cpu"`, and `:cpu`. Accepted accelerator
settings are `"cuda"`, `"cuda:N"`, `"gpu"`, `"accelerator"`, and their symbol
forms. Vectors/tuples must be empty or contain exactly one device request;
multiple accelerators fail before any partial setup is attempted.
"""
function validate_execution_device_settings(setting; availability = cuda_available())
    backend, index = _requested_device(setting)
    backend == :cpu && return cpu_execution_device(; requested = setting)
    backend == :cuda && return accelerator_execution_device(index === nothing ? 0 : index; requested = setting, availability = availability)
    throw(DeviceError("validate execution device settings", string(setting), "unsupported accelerator backend or device setting"))
end

"""
    prepare_execution_device(config_or_setting = nothing; availability = cuda_available()) -> ExecutionDevice

Prepare the local execution device for a workflow. Passing a config dictionary
uses its `device` key; passing a scalar validates that setting directly.

CPU preparation never initializes CUDA. CUDA preparation verifies availability
and fails clearly when the requested backend/device is unavailable.
"""
function prepare_execution_device(config_or_setting = nothing; availability = cuda_available())
    setting = config_or_setting isa AbstractDict ? _device_cfg_get(config_or_setting, "device", nothing) : config_or_setting
    device = validate_execution_device_settings(setting; availability = availability)
    if device.backend == :cuda && !device.available
        throw(accelerator_backend_unavailable_error(setting))
    end
    return device
end

function _is_namedtuple(value)
    value isa NamedTuple
end

"""
    prepare_value_for_device(value, device::ExecutionDevice)

Shallow-recursive movement for values passed across the trainer boundary.
Arrays are moved through `CUDA.cu` for CUDA devices; named tuples, tuples,
vectors, and dictionaries preserve their container structure while moving their
contents. CPU devices return values unchanged.

Unsupported or opaque objects are returned unchanged so model-specific transfer
logic can remain encapsulated in public model APIs.
"""
function prepare_value_for_device(value, device::ExecutionDevice)
    device.backend == :cpu && return value
    device.backend == :cuda || throw(DeviceError("prepare value for device", string(device.backend), "unsupported device backend"))
    value isa AbstractArray && return CUDA.cu(value)
    _is_namedtuple(value) && return NamedTuple{keys(value)}(map(v -> prepare_value_for_device(v, device), values(value)))
    value isa Tuple && return tuple((prepare_value_for_device(v, device) for v in value)...)
    value isa AbstractVector && return [prepare_value_for_device(v, device) for v in value]
    value isa AbstractDict && return Dict(k => prepare_value_for_device(v, device) for (k, v) in value)
    return value
end

"""
    prepare_model_for_device(model, device::ExecutionDevice)

Prepare a model for CPU or CUDA execution through public Flux APIs. CPU leaves
the model unchanged; CUDA delegates to `Flux.gpu`, which in turn uses the Julia
adaptation ecosystem used by Flux-compatible layers.
"""
function prepare_model_for_device(model, device::ExecutionDevice)
    device.backend == :cpu && return model
    device.backend == :cuda && return Flux.gpu(model)
    throw(DeviceError("prepare model for device", string(device.backend), "unsupported device backend"))
end

function _flux_gpu_or_recursive(value, device::ExecutionDevice)
    try
        return Flux.gpu(value)
    catch
        return prepare_value_for_device(value, device)
    end
end

"""
    prepare_graph_batch_for_device(batch, device::ExecutionDevice)

Prepare a GraphNeuralNetworks-compatible graph batch for the selected device.

For CUDA devices this first delegates to public Flux/Adapt APIs (`Flux.gpu`),
which is the supported path for `GraphNeuralNetworks.GNNGraph` and batched
`GNNGraph` values. The recursive array mover is only a fallback for simple test
fixtures or user containers that are not Adapt-aware.
"""
function prepare_graph_batch_for_device(batch, device::ExecutionDevice)
    device.backend == :cpu && return batch
    device.backend == :cuda && return _flux_gpu_or_recursive(batch, device)
    throw(DeviceError("prepare graph batch for device", string(device.backend), "unsupported device backend"))
end

"""
    prepare_model_and_graph_for_device(model, graph_batch, device::ExecutionDevice)

Prepare a model and graph batch together while preserving their public API
shape. Exact numeric outputs are outside the device-selection contract; this
function only guarantees structure-preserving transfer.
"""
function prepare_model_and_graph_for_device(model, graph_batch, device::ExecutionDevice)
    (
        model = prepare_model_for_device(model, device),
        graph_batch = prepare_graph_batch_for_device(graph_batch, device),
    )
end
