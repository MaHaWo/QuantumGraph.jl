using Flux
using Statistics: mean
using GraphNeuralNetworks: GraphNeuralNetworks

export GNNModelError,
	CompositeGNNModel,
	construct_gnn_model,
	evaluate_gnn_model,
	gnn_model_outputs,
	gnn_model_embedding,
	gnn_model_pooling_layer,
	active_task_outputs,
	stable_task_identifier,
	gnn_model_metadata,
	save_gnn_model_metadata,
	load_gnn_model_metadata

"""
	GNNModelError(operation::String, field::String, message::String)

User-facing exception for composite GNN model construction and evaluation failures.
"""
struct GNNModelError <: Exception
	operation::String
	field::String
	message::String
end

function Base.showerror(io::IO, err::GNNModelError)
	print(io, err.operation, " failed for ", err.field, ": ", err.message)
end

"""
	CompositeGNNModel

Configurable composite model boundary for graph samples.

The current migration stage preserves the public composition semantics: graph/sample
features are encoded, an embedding is produced through the configured path, and
only active downstream task heads are evaluated. Exact GraphNeuralNetworks.jl
operator mapping remains implementation-defined by the approved BDD spec.
"""
struct CompositeGNNModel
	encoder::Flux.Chain
	task_heads::Dict{Symbol, Flux.Chain}
	active_tasks::Vector{Symbol}
	task_key_mapping::Dict{String, Symbol}
	graph_pool::Any
	input_dim::Int
	embedding_dim::Int
	embedding_path::Symbol
	config_metadata::ConfigMetadata
end

function _gnn_require(config::AbstractDict, field::String)
	haskey(config, field) && return config[field]
	haskey(config, Symbol(field)) && return config[Symbol(field)]
	throw(GNNModelError("construct composite GNN model", field, "missing configuration field"))
end

function _gnn_optional(config::AbstractDict, field::String, default)
	haskey(config, field) && return config[field]
	haskey(config, Symbol(field)) && return config[Symbol(field)]
	default
end

function stable_task_identifier(key)::Symbol
	key isa Symbol && return key
	key isa AbstractString && return Symbol(key)
	return Symbol(string(key))
end

function _task_entries(task_config)
	if task_config isa AbstractDict
		return [(stable_task_identifier(k), v) for (k, v) in task_config]
	end
	entries = Tuple{Symbol, Any}[]
	for item in task_config
		if item isa AbstractString || item isa Symbol
			push!(entries, (stable_task_identifier(item), Dict{String, Any}()))
		elseif item isa AbstractDict
			key = haskey(item, "key") ? item["key"] : haskey(item, :key) ? item[:key] : nothing
			key === nothing && throw(GNNModelError("construct composite GNN model", "task_heads", "task head is missing a task identifier"))
			push!(entries, (stable_task_identifier(key), item))
		else
			throw(GNNModelError("construct composite GNN model", "task_heads", "unsupported task head configuration"))
		end
	end
	entries
end

function _embedding_path(config::AbstractDict)
	path = Symbol(String(_gnn_optional(config, "embedding_path", "pooling")))
	use_pooling = Bool(_gnn_optional(config, "use_pooling", path == :pooling))
	use_latent = Bool(_gnn_optional(config, "use_latent", path == :latent))
	if use_pooling && use_latent
		throw(GNNModelError("construct composite GNN model", "embedding_path", "incompatible pooling and latent configuration"))
	end
	path in (:pooling, :latent) || throw(GNNModelError("construct composite GNN model", "embedding_path", "unsupported embedding path"))
	return path
end

function _metadata_parameters(input_dim, embedding_dim, embedding_path, task_keys, active_tasks)
	Dict{String, Any}(
		"input_dim" => input_dim,
		"embedding_dim" => embedding_dim,
		"embedding_path" => String(embedding_path),
		"task_heads" => [String(key) for key in task_keys],
		"active_tasks" => [String(key) for key in active_tasks],
	)
end

"""
	construct_gnn_model(config::AbstractDict)

Construct a composite GNN model from public configuration.

Required fields are `input_dim`, `embedding_dim`, and non-empty `task_heads`.
Optional fields include `active_tasks` and `embedding_path` (`"pooling"` or
`"latent"`).
"""
function construct_gnn_model(config::AbstractDict)
	input_dim = Int(_gnn_require(config, "input_dim"))
	embedding_dim = Int(_gnn_require(config, "embedding_dim"))
	task_entries = _task_entries(_gnn_require(config, "task_heads"))
	isempty(task_entries) && throw(GNNModelError("construct composite GNN model", "task_heads", "missing downstream task configuration"))

	task_keys = [entry[1] for entry in task_entries]
	active_config = _gnn_optional(config, "active_tasks", task_keys)
	active_tasks = [stable_task_identifier(key) for key in active_config]
	unknown = setdiff(active_tasks, task_keys)
	isempty(unknown) || throw(GNNModelError("construct composite GNN model", "active_tasks", "unknown active task key(s): $(join(string.(unknown), ", "))"))

	embedding_path = _embedding_path(config)
	activation = resolve_activation(String(_gnn_optional(config, "activation", "identity")))
	encoder = Flux.Chain(Flux.Dense(input_dim => embedding_dim, activation))
	graph_pool = GraphNeuralNetworks.GlobalPool(mean)

	task_heads = Dict{Symbol, Flux.Chain}()
	for (key, head_config) in task_entries
		output_dim = head_config isa AbstractDict ? Int(_gnn_optional(head_config, "output_dim", 1)) : 1
		task_heads[key] = Flux.Chain(Flux.Dense(embedding_dim => output_dim))
	end

	mapping = Dict(String(key) => key for key in task_keys)
	metadata = ConfigMetadata("QuantumGraph.CompositeGNNModel", _metadata_parameters(input_dim, embedding_dim, embedding_path, task_keys, active_tasks))
	CompositeGNNModel(encoder, task_heads, active_tasks, mapping, graph_pool, input_dim, embedding_dim, embedding_path, metadata)
end

function _numeric_vector(value)
	if value isa Number
		return Float32[value]
	elseif value isa AbstractArray
		return Float32.(vec(value))
	end
	throw(GNNModelError("evaluate composite GNN model", "features", "non-numeric feature value"))
end

function _sample_features(input)
	if input isa AbstractArray
		return Float32.(input)
	elseif hasproperty(input, :features)
		features = getproperty(input, :features)
		if features isa AbstractArray
			return Float32.(features)
		elseif features isa NamedTuple
			values = reduce(vcat, [_numeric_vector(getproperty(features, key)) for key in keys(features)]; init = Float32[])
			return reshape(values, :, 1)
		elseif features isa AbstractDict
			ordered = sort(collect(keys(features)); by = string)
			values = reduce(vcat, [_numeric_vector(features[key]) for key in ordered]; init = Float32[])
			return reshape(values, :, 1)
		end
	elseif input isa AbstractDict && (haskey(input, "features") || haskey(input, :features))
		return _sample_features((; features = haskey(input, "features") ? input["features"] : input[:features]))
	end
	throw(GNNModelError("evaluate composite GNN model", "features", "compatible graph input does not expose features"))
end

function _graph_for_pooling(input)
	if hasproperty(input, :graph)
		return getproperty(input, :graph)
	elseif input isa AbstractDict && haskey(input, "graph")
		return input["graph"]
	elseif input isa AbstractDict && haskey(input, :graph)
		return input[:graph]
	end
	return nothing
end

function _embedding_from_encoded(model::CompositeGNNModel, input, encoded)
	if model.embedding_path == :pooling
		graph = _graph_for_pooling(input)
		graph === nothing && throw(GNNModelError("evaluate composite GNN model", "graph", "pooling path requires GraphNeuralNetworks-compatible graph input"))
		try
			return model.graph_pool(graph, encoded)
		catch err
			throw(GNNModelError("evaluate composite GNN model", "graph", "GraphNeuralNetworks GlobalPool failed: $(sprint(showerror, err))"))
		end
	elseif model.embedding_path == :latent
		return encoded
	end
	throw(GNNModelError("evaluate composite GNN model", "embedding_path", "unsupported embedding path"))
end

"""
	gnn_model_embedding(model, input)

Return the configured embedding for a compatible graph sample or batch.
"""
function gnn_model_embedding(model::CompositeGNNModel, input)
	features = _sample_features(input)
	size(features, 1) == model.input_dim || throw(GNNModelError("evaluate composite GNN model", "features", "dimension compatibility error: expected input_dim=$(model.input_dim), got $(size(features, 1))"))
	encoded = model.encoder(features)
	_embedding_from_encoded(model, input, encoded)
end

"""
	gnn_model_outputs(model, input)

Evaluate only the active downstream task heads and return a stable task-keyed dictionary.
"""
function gnn_model_outputs(model::CompositeGNNModel, input)
	embedding = gnn_model_embedding(model, input)
	outputs = Dict{Symbol, Any}()
	for task in model.active_tasks
		haskey(model.task_heads, task) || throw(GNNModelError("evaluate composite GNN model", "active_tasks", "unknown active task key '$task'"))
		outputs[task] = model.task_heads[task](embedding)
	end
	outputs
end

evaluate_gnn_model(model::CompositeGNNModel, input) = gnn_model_outputs(model, input)
(model::CompositeGNNModel)(input) = evaluate_gnn_model(model, input)

function gnn_model_pooling_layer(model::CompositeGNNModel)
	model.graph_pool
end

function active_task_outputs(model::CompositeGNNModel)
	copy(model.active_tasks)
end

function gnn_model_metadata(model::CompositeGNNModel)
	model.config_metadata
end

configuration_metadata(model::CompositeGNNModel) = gnn_model_metadata(model)

function save_gnn_model_metadata(model::CompositeGNNModel)
	gnn_model_metadata(model).constructor_parameters
end

function load_gnn_model_metadata(metadata::ConfigMetadata)
	metadata.type_identifier == "QuantumGraph.CompositeGNNModel" || throw(GNNModelError("load composite GNN model metadata", metadata.type_identifier, "unsupported metadata type"))
	construct_gnn_model(metadata.constructor_parameters)
end

function load_gnn_model_metadata(parameters::AbstractDict)
	construct_gnn_model(parameters)
end
