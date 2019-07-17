
using Distributions, Statistics, LinearAlgebra

import Base: size, length

import Statistics: mean, var, std

export EnsembleState, deviation

"""
    EnsembleState

An structure  for Ensemble.

Fields:

- 'N' : Number of ensemble members"

- 'NS' : Size of each state vector"

- 'S' : Array of the different ensemble members
"""



mutable struct EnsembleState{N, NS, TS}
    " Array of the different ensemble members"
    S::Array{TS,1}
end

# Return an ensemble of P members where each member is an
# array of type TT and dimension NT
function EnsembleState(N::Int, NS::Int; TS = Array{Float64,1})

    return EnsembleState{N, NS, TS}([zeros(NS) for i = 1:N])
end


function EnsembleState(States::Array{T,1}) where {T}
    N = length(States)
    NS = size(States[1])[1]
    return EnsembleState{N, NS, T}(States)
end

size(ENS::EnsembleState{N, NS, TS})  where {N, NS, TS} = (N, NS)



# Return the mean of all the ensemble member
mean(ENS::EnsembleState{N, NS, TS})  where {N, NS, TS} = mean(ENS.S)


function deviation(ENS::EnsembleState{N, NS, TS})  where {N, NS, TS}
    S̄ = mean(ENS)
     for s in ENS.S
         s .-= S̄
     end
     return ENS
 end
