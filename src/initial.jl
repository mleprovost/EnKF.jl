
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
