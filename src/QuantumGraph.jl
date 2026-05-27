module QuantumGraph

include("Interfaces.jl")
include("ZarrLoading.jl")
include("Config.jl")
include("Datasets.jl")
include("Models.jl")
include("GNNModel.jl")

"""
    dummy()

Placeholder function retained while domain modules are implemented.
"""
dummy() = nothing

export dummy

end # module QuantumGraph
