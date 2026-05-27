export ConfigError,
	Sweep,
	CoupledSweep,
	InclusiveRange,
	RandomUniform,
	Reference,
	ObjectReference,
	load_config,
	expand_range,
	expand_config,
	resolve_config,
	supported_config_tags

"""
	ConfigError(operation::String, path::String, message::String)

User-facing exception for configuration loading, expansion, and resolution failures.
"""
struct ConfigError <: Exception
	operation::String
	path::String
	message::String
end

function Base.showerror(io::IO, err::ConfigError)
	print(io, err.operation, " failed at ", err.path, ": ", err.message)
end

abstract type AbstractConfigTag end

"""Configuration value representing a sweep over independent values."""
struct Sweep <: AbstractConfigTag
	values::Vector{Any}
end

"""Configuration value representing one member of a coupled sweep group."""
struct CoupledSweep <: AbstractConfigTag
	group::String
	values::Vector{Any}
end

"""Configuration value representing an inclusive range with nonzero step."""
struct InclusiveRange <: AbstractConfigTag
	start::Any
	stop::Any
	step::Any
end

"""Configuration value representing an approved random uniform sampling request."""
struct RandomUniform <: AbstractConfigTag
	low::Any
	high::Any
end

"""Configuration value referencing another configuration path."""
struct Reference <: AbstractConfigTag
	path::Vector{String}
end

"""Configuration value referencing a Julia object through QuantumGraph's registry."""
struct ObjectReference <: AbstractConfigTag
	identifier::String
end

supported_config_tags() = Set(["!sweep", "!coupled-sweep", "!range", "!random_uniform", "!reference", "!pyobject-equivalent"])

function _parse_scalar(text::AbstractString)
	stripped = strip(text)
	if startswith(stripped, '"') && endswith(stripped, '"') && length(stripped) >= 2
		return stripped[2:(end-1)]
	elseif startswith(stripped, "'") && endswith(stripped, "'") && length(stripped) >= 2
		return stripped[2:(end-1)]
	elseif occursin(r"^-?\d+$", stripped)
		return parse(Int, stripped)
	elseif occursin(r"^-?\d+\.\d+", stripped)
		return parse(Float64, stripped)
	else
		return stripped
	end
end

function _parse_list(text::AbstractString)
	stripped = strip(text)
	startswith(stripped, "[") && endswith(stripped, "]") || throw(ConfigError("load configuration", stripped, "expected list syntax"))
	inner = strip(stripped[2:(end-1)])
	isempty(inner) && return Any[]
	Any[_parse_scalar(part) for part in split(inner, ",")]
end

function _parse_tag_value(tag::AbstractString, rest::AbstractString, path::AbstractString)
	if tag == "!sweep"
		return Sweep(_parse_list(rest))
	elseif tag == "!coupled-sweep"
		pieces = split(strip(rest); limit = 2)
		length(pieces) == 2 || throw(ConfigError("load configuration", String(path), "malformed !coupled-sweep tag"))
		return CoupledSweep(String(pieces[1]), _parse_list(pieces[2]))
	elseif tag == "!range"
		values = _parse_list(rest)
		length(values) == 3 || throw(ConfigError("load configuration", String(path), "!range requires start, stop, and step"))
		return InclusiveRange(values[1], values[2], values[3])
	elseif tag == "!random_uniform"
		values = _parse_list(rest)
		length(values) == 2 || throw(ConfigError("load configuration", String(path), "!random_uniform requires low and high"))
		return RandomUniform(values[1], values[2])
	elseif tag == "!reference"
		return Reference(String.(split(String(_parse_scalar(rest)), ".")))
	elseif tag == "!pyobject-equivalent"
		return ObjectReference(String(_parse_scalar(rest)))
	else
		throw(ConfigError("load configuration", String(path), "unsupported configuration tag $tag"))
	end
end

"""
	load_config(text::AbstractString)

Load a minimal public configuration representation containing approved tags.

This function intentionally specifies QuantumGraph's public tag behavior rather
than a particular YAML parser. It accepts simple `key: !tag value` lines used by
BDD fixtures and returns a dictionary containing QuantumGraph tag values.
"""
function load_config(text::AbstractString)
	config = Dict{String, Any}()
	for (line_number, line) in enumerate(split(text, '\n'))
		stripped = strip(line)
		isempty(stripped) && continue
		startswith(stripped, "#") && continue
		m = match(r"^([^:]+):\s*(![A-Za-z0-9_-]+)\s*(.*)$", stripped)
		if m === nothing
			throw(ConfigError("load configuration", "line $line_number", "unsupported or malformed configuration entry"))
		end
		key, tag, rest = strip(m.captures[1]), m.captures[2], m.captures[3]
		config[key] = _parse_tag_value(tag, rest, key)
	end
	config
end

function load_config(config::AbstractDict)
	Dict{String, Any}(String(k) => v for (k, v) in config)
end

"""
	expand_range(range::InclusiveRange)

Expand an inclusive configuration range.

The start value is always included. The stop value is included when it lies on
the step sequence. A zero step raises `ConfigError`.
"""
function expand_range(range::InclusiveRange)
	range.step == 0 && throw(ConfigError("expand range", "step", "range step cannot be zero"))
	collect(range.start:range.step:range.stop)
end

function _copy_config(config::AbstractDict)
	Dict{String, Any}(String(k) => v for (k, v) in config)
end

function _sweep_entries(config::AbstractDict)
	[(String(k), v) for (k, v) in config if v isa Sweep]
end

function _coupled_entries(config::AbstractDict)
	[(String(k), v) for (k, v) in config if v isa CoupledSweep]
end

function _run_suffix(path::AbstractString, value)
	safe = replace(string(value), r"\s+" => "_")
	string(path, "=", safe)
end

"""
	expand_config(config::AbstractDict)

Expand sweep and coupled-sweep values into concrete run configurations.

Independent sweeps produce one configuration per value. Coupled sweeps with the
same group expand by aligned index and reject mismatched lengths.
"""
function expand_config(config::AbstractDict)
	base = _copy_config(config)
	sweeps = _sweep_entries(base)
	coupled = _coupled_entries(base)

	if !isempty(coupled)
		by_group = Dict{String, Vector{Tuple{String, CoupledSweep}}}()
		for entry in coupled
			push!(get!(by_group, entry[2].group, Tuple{String, CoupledSweep}[]), entry)
		end
		for (group, entries) in by_group
			lengths = [length(entry[2].values) for entry in entries]
			if length(unique(lengths)) != 1
				throw(ConfigError("expand configuration", group, "coupled sweep length mismatch"))
			end
		end
	end

	runs = [Dict{String, Any}(k => v for (k, v) in base if !(v isa Sweep) && !(v isa CoupledSweep))]
	names = ["base"]

	for (path, sweep) in sweeps
		next_runs = Dict{String, Any}[]
		next_names = String[]
		for (run, run_name) in zip(runs, names), value in sweep.values
			updated = copy(run)
			updated[path] = value
			push!(next_runs, updated)
			push!(next_names, run_name == "base" ? _run_suffix(path, value) : string(run_name, "__", _run_suffix(path, value)))
		end
		runs, names = next_runs, next_names
	end

	if !isempty(coupled)
		by_group = Dict{String, Vector{Tuple{String, CoupledSweep}}}()
		for entry in coupled
			push!(get!(by_group, entry[2].group, Tuple{String, CoupledSweep}[]), entry)
		end
		for (group, entries) in by_group
			length_group = length(entries[1][2].values)
			next_runs = Dict{String, Any}[]
			next_names = String[]
			for (run, run_name) in zip(runs, names), i in 1:length_group
				updated = copy(run)
				suffix_parts = String[]
				for (path, coupled_sweep) in entries
					value = coupled_sweep.values[i]
					updated[path] = value
					push!(suffix_parts, _run_suffix(path, value))
				end
				push!(next_runs, updated)
				push!(next_names, string(run_name, "__", group, "[", i, "]_", join(suffix_parts, "_")))
			end
			runs, names = next_runs, next_names
		end
	end

	[(name = names[i], config = runs[i]) for i in eachindex(runs)]
end

function _resolve_value(value, root)
	if value isa Reference
		return get_config_path(root, value.path)
	elseif value isa ObjectReference
		return resolve_registered_object(value.identifier)
	elseif value isa InclusiveRange
		return expand_range(value)
	elseif value isa AbstractDict
		return resolve_config(value)
	else
		return value
	end
end

"""
	resolve_config(config::AbstractDict)

Resolve references and object references in a configuration mapping.

References are resolved against the original configuration root. Object references
are resolved through QuantumGraph's public Julia registry.
"""
function resolve_config(config::AbstractDict)
	root = _copy_config(config)
	resolved = Dict{String, Any}()
	for (key, value) in root
		try
			resolved[key] = _resolve_value(value, root)
		catch err
			err isa PublicInterfaceError && throw(ConfigError("resolve configuration", key, sprint(showerror, err)))
			err isa ConfigError && rethrow()
			throw(ConfigError("resolve configuration", key, sprint(showerror, err)))
		end
	end
	resolved
end
