
using Distributions, Statistics, LinearAlgebra

import Base: size, length, hcat, +, -, fill!

import Statistics: mean, var, std

export EnsembleState!, EnsembleState, deviation!, deviation, cut

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

function EnsembleState!(ENS::EnsembleState{N, NS, T}, States::Array{Array{T,1},1}) where {N, NS, T}
    ENS.S .= deepcopy(States)
    return ENS
end


size(ENS::EnsembleState{N, NS, TS})  where {N, NS, TS} = (N, NS)



"Return the mean of all the ensemble members"
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

" Extend definition of hcat to EnsembleState"
function hcat(ENS::EnsembleState{N, NS, TS})  where {N, NS, TS}
   return hcat(ENS.S...)
end


" Cut an array along the different columns and create an EnsembleState variable with these columns"
function cut(A::Array{TS,2}) where {TS}
    # Get size of A = (length of state vector, number of ensemble members)
        NS, N = size(A)
        B = deepcopy(A)
    # Allocate space
    ENS = EnsembleState(N, NS)
    for i in 1:N
        ENS.S[i] .= B[:,i]
    end
    return ENS
end
"Fill ensemble state"

function fill!(ENS::EnsembleState{N, NS, TS}, A::Array{Array{TS,1},1})  where {N, NS, TS}
    ENS.S .= deepcopy(A)
    ENS
end

function fill!(ENS::EnsembleState{N, NS, TS}, A::Array{TS,1})  where {N, NS, TS}
    B = deepcopy(A)
    for s in ENS.S
        s .= B
    end
    return ENS
end

"Define addition of two EnsembleState"

function (+)(A::EnsembleState{N, NS, TS}, B::EnsembleState{N, NS, TS}) where {N, NS, TS}
    C = deepcopy(A)
    C.S .+= B.S
    return C
end

"Define addition of an Array and an EnsembleState"

function (+)(A::EnsembleState{N, NS, TS}, B::Array{TS,1}) where {N, NS, TS}
    C = deepcopy(A)
    for s in C.S
        s .+=B
    end
    return C
end


"Define substraction of two EnsembleState"

function (-)(A::EnsembleState{N, NS, TS}, B::EnsembleState{N, NS, TS}) where {N, NS, TS}
    C = deepcopy(A)
    C.S .-= B.S
    return C
end

"Define substraction of an Array from an EnsembleState"

function (-)(A::EnsembleState{N, NS, TS}, B::Array{TS,1}) where {N, NS, TS}
    C = deepcopy(A)
    for s in C.S
        s .-=B
    end
    return C
end
