using Flux

export ModelComponentError,
    ReusableBlock,
    construct_model_component,
    apply_model_block,
    model_component_metadata,
    register_activation!,
    resolve_activation

struct ModelComponentError <: Exception
    operation::String
    field::String
    message::String
end

function Base.showerror(io::IO, err::ModelComponentError)
    print(io, err.operation, " failed for ", err.field, ": ", err.message)
end

const _ACTIVATIONS = Dict{String, Any}(
    "identity" => identity,
    "relu" => Flux.relu,
    "tanh" => tanh,
    "sigmoid" => Flux.sigmoid,
)

function register_activation!(name::AbstractString, activation)
    _ACTIVATIONS[String(name)] = activation
    activation
end

function resolve_activation(name::AbstractString)
    key = String(name)
    haskey(_ACTIVATIONS, key) || throw(ModelComponentError("resolve activation", key, "unknown activation"))
    _ACTIVATIONS[key]
end

struct ReusableBlock
    chain::Flux.Chain
    input_dim::Int
    output_dim::Int
    activation::String
    graph_operator_role::String
    residual::Bool
    projection::Union{Nothing, Flux.Dense}
end

function _require(config::AbstractDict, field::String)
    if haskey(config, field)
        return config[field]
    elseif haskey(config, Symbol(field))
        return config[Symbol(field)]
    end
    throw(ModelComponentError("construct model component", field, "missing configuration field"))
end

function _optional(config::AbstractDict, field::String, default)
    haskey(config, field) && return config[field]
    haskey(config, Symbol(field)) && return config[Symbol(field)]
    default
end

function construct_model_component(config::AbstractDict)
    input_dim = Int(_require(config, "input_dim"))
    output_dim = Int(_require(config, "output_dim"))
    activation_name = String(_require(config, "activation"))
    graph_operator_role = String(_require(config, "graph_operator_role"))
    residual = Bool(_optional(config, "residual", false))
    projection_policy = String(_optional(config, "projection", "none"))

    activation = resolve_activation(activation_name)
    residual && input_dim != output_dim && projection_policy == "none" &&
        throw(ModelComponentError("construct model component", "residual", "dimension compatibility error: skipped-over block input_dim=$input_dim output_dim=$output_dim requires projection"))

    dense = Flux.Dense(input_dim => output_dim, activation)
    projection = residual && input_dim != output_dim ? Flux.Dense(input_dim => output_dim) : nothing
    ReusableBlock(Flux.Chain(dense), input_dim, output_dim, activation_name, graph_operator_role, residual, projection)
end

function _features(input)
    if input isa AbstractArray
        return input
    elseif hasproperty(input, :features)
        return getproperty(input, :features)
    elseif input isa AbstractDict && haskey(input, "features")
        return input["features"]
    elseif input isa AbstractDict && haskey(input, :features)
        return input[:features]
    end
    throw(ModelComponentError("apply model block", "features", "input does not expose node or graph features"))
end

function _with_features(input, features)
    if input isa AbstractArray
        return features
    elseif input isa NamedTuple
        return merge(input, (; features = features))
    elseif input isa AbstractDict
        output = copy(input)
        if haskey(output, "features")
            output["features"] = features
        else
            output[:features] = features
        end
        return output
    end
    return (; features)
end

function apply_model_block(block::ReusableBlock, input)
    x = _features(input)
    size(x, 1) == block.input_dim || throw(ModelComponentError("apply model block", "features", "dimension compatibility error: expected input_dim=$(block.input_dim), got $(size(x, 1))"))
    y = block.chain(x)
    if block.residual
        skip = block.projection === nothing ? x : block.projection(x)
        size(skip) == size(y) || throw(ModelComponentError("apply model block", "residual", "dimension compatibility error: skip path $(size(skip)) cannot be added to block output $(size(y))"))
        y = y .+ skip
    end
    _with_features(input, y)
end

(block::ReusableBlock)(input) = apply_model_block(block, input)

function model_component_metadata(block::ReusableBlock)
    ConfigMetadata(
        "QuantumGraph.ReusableBlock",
        Dict{String, Any}(
            "input_dim" => block.input_dim,
            "output_dim" => block.output_dim,
            "activation" => block.activation,
            "graph_operator_role" => block.graph_operator_role,
            "residual" => block.residual,
            "projection" => block.projection === nothing ? "none" : "linear",
        ),
    )
end

configuration_metadata(block::ReusableBlock) = model_component_metadata(block)
