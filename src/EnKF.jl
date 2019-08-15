module EnKF

include("state.jl")

include("update.jl")

include("initial.jl")

include("inflation.jl")

include("system.jl")


end # module
