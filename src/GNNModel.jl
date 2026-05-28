using Flux
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
	load_gnn_model_metadata,
	default_dense_encoder,
	default_dense_task_head,
	default_graph_pool

"""
	GNNModelError(operation::String, field::String, message::String)

User-facing exception for composite model construction and evaluation failures.
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

Configurable model orchestrator.

`CompositeGNNModel` wires together configurable components in the same structural
role as the Python model boundary: encoder -> pooling or latent path -> decoder /
task heads. The orchestrator does not construct graph-specific layers directly.
Those are provided by configuration through QuantumGraph's registry. Graph-specific
components are supported by registry-resolved constructors rather than hard-coded
inside the orchestrator.
"""
struct CompositeGNNModel
	encoder::Any
	pooling::Any
	task_heads::Dict{Symbol, Any}
	active_tasks::Vector{Symbol}
	task_key_mapping::Dict{String, Symbol}
	input_dim::Int
	embedding_dim::Int
	embedding_path::Symbol
	component_specs::Dict{String, Any}
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

function _string_key_dict(mapping::AbstractDict)
	Dict{String, Any}(String(k) => v for (k, v) in mapping)
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

function _component_type(spec::AbstractDict, field::String)
	if haskey(spec, "type")
		return spec["type"]
	elseif haskey(spec, :type)
		return spec[:type]
	elseif haskey(spec, "type_identifier")
		return spec["type_identifier"]
	elseif haskey(spec, :type_identifier)
		return spec[:type_identifier]
	end
	throw(GNNModelError("construct composite GNN model", field, "component configuration is missing type"))
end

function _component_args(spec::AbstractDict)
	args = haskey(spec, "args") ? spec["args"] : haskey(spec, :args) ? spec[:args] : Any[]
	args isa Tuple && return collect(args)
	args isa AbstractVector && return collect(args)
	return Any[args]
end

function _component_kwargs(spec::AbstractDict)
	kwargs = haskey(spec, "kwargs") ? spec["kwargs"] : haskey(spec, :kwargs) ? spec[:kwargs] : Dict{String, Any}()
	kwargs isa AbstractDict || throw(GNNModelError("construct composite GNN model", "kwargs", "component kwargs must be a mapping"))
	Dict{Symbol, Any}(Symbol(k) => v for (k, v) in kwargs)
end

function _merge_component_defaults(spec, default_spec::AbstractDict)
	if spec === nothing
		return deepcopy(default_spec)
	elseif spec isa AbstractDict
		merged = deepcopy(_string_key_dict(default_spec))
		provided = _string_key_dict(spec)
		for (key, value) in provided
			if key == "kwargs" && haskey(merged, "kwargs") && value isa AbstractDict
				kw = _string_key_dict(merged["kwargs"])
				merge!(kw, _string_key_dict(value))
				merged["kwargs"] = kw
			else
				merged[key] = value
			end
		end
		return merged
	else
		return spec
	end
end

function _call_component_constructor(target, args, kwargs, field)
	try
		return target(args...; kwargs...)
	catch err
		try
			params = Dict{String, Any}(String(k) => v for (k, v) in kwargs)
			isempty(args) && return target(params)
		catch
			# Preserve the original keyword-construction error below.
		end
		throw(GNNModelError("construct composite GNN model", field, "component construction failed: $(sprint(showerror, err))"))
	end
end

function _construct_configured_component(spec, field::String)
	if spec isa ObjectReference
		return resolve_registered_object(spec.identifier), Dict{String, Any}("type" => spec.identifier, "args" => Any[], "kwargs" => Dict{String, Any}())
	elseif spec isa AbstractString
		return resolve_registered_object(spec), Dict{String, Any}("type" => String(spec), "args" => Any[], "kwargs" => Dict{String, Any}())
	elseif spec isa AbstractDict
		component_type = _component_type(spec, field)
		target = component_type isa ObjectReference ? resolve_registered_object(component_type.identifier) : component_type isa AbstractString ? resolve_registered_object(component_type) : component_type
		args = _component_args(spec)
		kwargs = _component_kwargs(spec)
		component = _call_component_constructor(target, args, kwargs, field)
		normalized = Dict{String, Any}(
			"type" => component_type isa ObjectReference ? component_type.identifier : string(component_type),
			"args" => args,
			"kwargs" => Dict{String, Any}(String(k) => v for (k, v) in kwargs),
		)
		return component, normalized
	else
		return spec, Dict{String, Any}("type" => string(typeof(spec)), "args" => Any[], "kwargs" => Dict{String, Any}())
	end
end

function default_dense_encoder(; input_dim, embedding_dim, activation = "identity")
	act = resolve_activation(String(activation))
	Flux.Chain(Flux.Dense(Int(input_dim) => Int(embedding_dim), act))
end

function default_dense_encoder(params::AbstractDict)
	default_dense_encoder(; input_dim = _gnn_require(params, "input_dim"), embedding_dim = _gnn_require(params, "embedding_dim"), activation = _gnn_optional(params, "activation", "identity"))
end

function default_dense_task_head(; embedding_dim, output_dim = 1)
	Flux.Chain(Flux.Dense(Int(embedding_dim) => Int(output_dim)))
end

function default_dense_task_head(params::AbstractDict)
	default_dense_task_head(; embedding_dim = _gnn_require(params, "embedding_dim"), output_dim = _gnn_optional(params, "output_dim", 1))
end

function default_graph_pool(; aggregation)
	GraphNeuralNetworks.GlobalPool(aggregation)
end

function default_graph_pool(params::AbstractDict)
	default_graph_pool(; aggregation = _gnn_require(params, "aggregation"))
end

function _default_encoder_spec(input_dim, embedding_dim, activation)
	Dict{String, Any}(
		"type" => "QuantumGraph.DenseEncoder",
		"args" => Any[],
		"kwargs" => Dict{String, Any}("input_dim" => input_dim, "embedding_dim" => embedding_dim, "activation" => activation),
	)
end

function _default_task_head_spec(embedding_dim, output_dim)
	Dict{String, Any}(
		"type" => "QuantumGraph.DenseTaskHead",
		"args" => Any[],
		"kwargs" => Dict{String, Any}("embedding_dim" => embedding_dim, "output_dim" => output_dim),
	)
end

function _metadata_parameters(input_dim, embedding_dim, embedding_path, task_keys, active_tasks, component_specs)
	Dict{String, Any}(
		"input_dim" => input_dim,
		"embedding_dim" => embedding_dim,
		"embedding_path" => String(embedding_path),
		"task_heads" => [String(key) for key in task_keys],
		"active_tasks" => [String(key) for key in active_tasks],
		"components" => component_specs,
	)
end

"""
	construct_gnn_model(config::AbstractDict)

Construct a composite model from configuration.

The configuration supplies component `type`, `args`, and `kwargs` entries, which
are resolved through QuantumGraph's public registry. Defaults are registry-backed
components, not hard-coded orchestration logic:

- `encoder`: defaults to `QuantumGraph.DenseEncoder`
- `pooling`: required when `embedding_path == "pooling"`; resolved from config
- each task head: defaults to `QuantumGraph.DenseTaskHead`
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
	activation = String(_gnn_optional(config, "activation", "identity"))

	encoder_spec = _merge_component_defaults(_gnn_optional(config, "encoder", nothing), _default_encoder_spec(input_dim, embedding_dim, activation))
	pooling_config = _gnn_optional(config, "pooling", nothing)
	pooling, normalized_pooling_spec = if pooling_config === nothing
		nothing, nothing
	else
		_construct_configured_component(pooling_config, "pooling")
	end
	encoder, normalized_encoder_spec = _construct_configured_component(encoder_spec, "encoder")

	task_heads = Dict{Symbol, Any}()
	task_head_specs = Dict{String, Any}()
	for (key, head_config) in task_entries
		head_mapping = head_config isa AbstractDict ? head_config : Dict{String, Any}()
		output_dim = Int(_gnn_optional(head_mapping, "output_dim", 1))
		head_spec = _merge_component_defaults(_gnn_optional(head_mapping, "head", nothing), _default_task_head_spec(embedding_dim, output_dim))
		head, normalized_head_spec = _construct_configured_component(head_spec, "task_heads.$key")
		task_heads[key] = head
		task_head_specs[String(key)] = normalized_head_spec
	end

	mapping = Dict(String(key) => key for key in task_keys)
	component_specs = Dict{String, Any}("encoder" => normalized_encoder_spec, "task_heads" => task_head_specs)
	if pooling_config !== nothing
		component_specs["pooling"] = normalized_pooling_spec
	elseif haskey(config, "components") && config["components"] isa AbstractDict && haskey(config["components"], "pooling")
		component_specs["pooling"] = config["components"]["pooling"]
	elseif haskey(config, :components) && config[:components] isa AbstractDict && haskey(config[:components], "pooling")
		component_specs["pooling"] = config[:components]["pooling"]
	end
	metadata = ConfigMetadata("QuantumGraph.CompositeGNNModel", _metadata_parameters(input_dim, embedding_dim, embedding_path, task_keys, active_tasks, component_specs))
	CompositeGNNModel(encoder, pooling, task_heads, active_tasks, mapping, input_dim, embedding_dim, embedding_path, component_specs, metadata)
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

function _call_encoder(encoder, features)
	try
		return encoder(features)
	catch err
		throw(GNNModelError("evaluate composite GNN model", "encoder", "encoder component failed: $(sprint(showerror, err))"))
	end
end

function _call_pooling(pooling, input, encoded)
	graph = _graph_for_pooling(input)
	if graph !== nothing
		try
			return pooling(graph, encoded)
		catch err
			if !(err isa MethodError)
				throw(GNNModelError("evaluate composite GNN model", "pooling", "pooling component failed: $(sprint(showerror, err))"))
			end
		end
	end
	try
		return pooling(encoded)
	catch err
		throw(GNNModelError("evaluate composite GNN model", "pooling", "pooling component failed: $(sprint(showerror, err))"))
	end
end

function _embedding_from_encoded(model::CompositeGNNModel, input, encoded)
	if model.embedding_path == :pooling
		model.pooling === nothing && throw(GNNModelError("evaluate composite GNN model", "pooling", "pooling path requires a configured pooling component"))
		return _call_pooling(model.pooling, input, encoded)
	elseif model.embedding_path == :latent
		return encoded
	end
	throw(GNNModelError("evaluate composite GNN model", "embedding_path", "unsupported embedding path"))
end

"""
	gnn_model_embedding(model, input)

Return the configured embedding for a compatible sample or batch.
"""
function gnn_model_embedding(model::CompositeGNNModel, input)
	features = _sample_features(input)
	size(features, 1) == model.input_dim || throw(GNNModelError("evaluate composite GNN model", "features", "dimension compatibility error: expected input_dim=$(model.input_dim), got $(size(features, 1))"))
	encoded = _call_encoder(model.encoder, features)
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
		try
			outputs[task] = model.task_heads[task](embedding)
		catch err
			throw(GNNModelError("evaluate composite GNN model", "task_heads.$task", "task head failed: $(sprint(showerror, err))"))
		end
	end
	outputs
end

evaluate_gnn_model(model::CompositeGNNModel, input) = gnn_model_outputs(model, input)
(model::CompositeGNNModel)(input) = evaluate_gnn_model(model, input)

function gnn_model_pooling_layer(model::CompositeGNNModel)
	model.pooling
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
