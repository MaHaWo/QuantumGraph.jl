module QuantumGraph

include("Interfaces.jl")
include("ZarrLoading.jl")
include("Config.jl")
include("Datasets.jl")
include("Models.jl")
include("GNNModel.jl")
include("Evaluation.jl")
include("EarlyStopping.jl")
include("Training.jl")
include("Tuning.jl")

function __init__()
    register_object!("QuantumGraph.DenseEncoder", default_dense_encoder)
    register_object!("QuantumGraph.DenseTaskHead", default_dense_task_head)
    register_object!("QuantumGraph.GraphPool", default_graph_pool)
end

"""
    dummy()

Placeholder function retained while domain modules are implemented.
"""
dummy() = nothing

export dummy

end # module QuantumGraph
