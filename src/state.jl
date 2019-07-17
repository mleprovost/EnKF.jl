
using Distributions, Statistics, LinearAlgebra

import Base: size, length, hcat, +, -, fill!

import Statistics: mean, var, std

export EnsembleState, deviation!, deviation

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
    S::Array{Array{TS,1},1}
end

# Return an ensemble of P members where each member is an
# array of type TT and dimension NT
function EnsembleState(N::Int, NS::Int; TS = Float64)

    return EnsembleState{N, NS, TS}([zeros(TS, NS) for i = 1:N])
end

function EnsembleState(NT::Tuple{Int, Int}; TS = Float64)
    N, NS = NT
    return EnsembleState{N, NS, TS}([zeros(TS, NS) for i = 1:N])
end


function EnsembleState(States::Array{Array{T,1},1}) where {T}
    N = length(States)
    NS = size(States[1])[1]
    return EnsembleState{N, NS, T}(States)
end

size(ENS::EnsembleState{N, NS, TS})  where {N, NS, TS} = (N, NS)



# Return the mean of all the ensemble member
mean(ENS::EnsembleState{N, NS, TS})  where {N, NS, TS} = mean(ENS.S)


function deviation!(ENS::EnsembleState{N, NS, TS})  where {N, NS, TS}
    S̄ = deepcopy(mean(ENS))
     for s in ENS.S
         s .-= S̄
     end
     return ENS
 end


function deviation(ENSfluc::EnsembleState{N, NS, TS}, ENS::EnsembleState{N, NS, TS})  where {N, NS, TS}
 S̄ = deepcopy(mean(ENS))
 ENSfluc.S .= deepcopy(ENS.S)
  for s in ENSfluc.S
      s .-= S̄
  end
  return ENSfluc
end

function deviation(tabfluc::Array{TS,2}, ENS::EnsembleState{N, NS, TS})  where {N, NS, TS}
 S̄ = deepcopy(mean(ENS))
 tabfluc .= hcat(deepcopy(ENS))
 tabfluc .-= S̄
   return tabfluc
end

function hcat(ENS::EnsembleState{N, NS, TS})  where {N, NS, TS}
   return hcat(ENS.S...)
end


"Fill ensemble state"

function fill!(ENS::EnsembleState{N, NS, TS}, A::Array{Array{TS,1},1})  where {N, NS, TS}
    ENS.S .= deepcopy(A)
    ENS
end

function fill!(ENS::EnsembleState{N, NS, TS}, A::Array{TS,1})  where {N, NS, TS}
    B = deepcopy(A)
    for s in ENS.S
        s .= A
    end
    return ENS
end

"Define addition of two EnsembleState"

function (+)(A::EnsembleState{N, NS, TS}, B::EnsembleState{N, NS, TS}) where {N, NS, TS}
    C = deepcopy(A)
    C.S .+= B.S
    return C
end

"Define substraction of two EnsembleState"

function (-)(A::EnsembleState{N, NS, TS}, B::EnsembleState{N, NS, TS}) where {N, NS, TS}
    C = deepcopy(A)
    C.S .-= B.S
    return C
end
