
using Distributions, Statistics, LinearAlgebra

import Base: size, length

import Statistics: mean, var, std


export initialize



function initialize(N::Int, Distx0::MultivariateDistribution)
    NS = length(Distx0)

    # pre-allocate space
    ENS = EnsembleState(N, NS)

    ENS.S .= [rand(Distx0) for i = 1:N]
    return ENS
end

function initialize(N::Int, NS::Int)
    # pre-allocate space
    ENS = EnsembleState(N, NS)
    Dist = MvNormal(zeros(NS), I)
    ENS.S .= [rand(Dist) for i = 1:N]
    return ENS
end

function initialize(ENS::EnsembleState{N, NS, TS})  where {N, NS, TS}
    Dist = MvNormal(zeros(NS), I)
    ENS.S .= [rand(Dist) for i = 1:N]
    return ENS
end
