
using Distributions, Statistics, LinearAlgebra

import Base: size, length

import Statistics: mean, var, std


export initialize, initialize!



function initialize(N::Int, Distx0::MultivariateDistribution)
    NS = length(Distx0)

    # pre-allocate space
    ENS = EnsembleState(N, zeros(NS))

    ENS.S .= [rand(Distx0) for i = 1:N]
    return ENS
end
#
function initialize(N::Int, NS::Int)
    # pre-allocate space
    ENS = EnsembleState(N, zeros(NS))
    
    Dist = MvNormal(zeros(NS), I)
    ENS.S .= [rand(Dist) for i = 1:N]
    return ENS
end
#
function initialize!(ENS::EnsembleState{N, TS}, u::TS)  where {N, TS}
    NS = length(u)
    Dist = MvNormal(zeros(NS), I)
    ENS.S .= [rand(Dist) for i = 1:N]
    return ENS
end
