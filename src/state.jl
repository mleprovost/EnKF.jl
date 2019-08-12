
using Distributions, Statistics, LinearAlgebra

import Base: size, length, hcat, +, -, fill!

import Statistics: mean, var, std

export EnsembleState!, EnsembleState, deviation!, deviation, cut, fill!


"""
Define abstract type State for state vector
"""

# abstract type State end



"""
    EnsembleState

An structure  for Ensemble.

Fields:

- 'N' : Number of ensemble members"

- 'S' : Array of the different members
"""



mutable struct EnsembleState{N,TS}
    " Array of the different ensemble members"
    S::Array{TS,1}
end

# Return an ensemble of P members where each member is an
# array of type TS and dimension NS

function EnsembleState(N::Int, u)
    TS = typeof(u)
    return EnsembleState{N,TS}([zero(u) for i = 1:N])
end


function EnsembleState(States::Array{TS,1}) where {TS}
    N = length(States)
    return EnsembleState{N, TS}(States)
end


size(ENS::EnsembleState{N, TS})  where {N, TS} = N, size(ENS.S[1])

length(ENS::EnsembleState{N, TS})  where {N, TS} = N

"Return the mean of all the ensemble members"
mean(ENS::EnsembleState{N, TS})  where {N, TS} = mean(ENS.S)


function deviation!(ENS::EnsembleState{N, TS})  where {N, TS}
    S̄ = deepcopy(mean(ENS))
     for s in ENS.S
         s .-= S̄
     end
     return ENS
 end


function deviation(ENSfluc::EnsembleState{N, TS}, ENS::EnsembleState{N, TS})  where {N, TS}
 S̄ = deepcopy(mean(ENS))
 ENSfluc.S .= deepcopy(ENS.S)
  for s in ENSfluc.S
      s .-= S̄
  end
  return ENSfluc
end

function deviation(tabfluc::Array{TS,2}, ENS::EnsembleState{N, TS})  where {N, TS}
 S̄ = deepcopy(mean(ENS))
 tabfluc .= hcat(deepcopy(ENS))
 tabfluc .-= S̄
   return tabfluc
end

" Extend definition of hcat to EnsembleState"
function hcat(ENS::EnsembleState{N, TS})  where {N, TS}
   return hcat(ENS.S...)
end


" Cut an array along the different columns and create an EnsembleState variable with these columns"
function cut(A::AbstractMatrix{TR}) where {TR}
    # Get size of A = (length of state vector, number of ensemble members)
        NS, N = size(A)
        B = deepcopy(A)
    # Allocate space
    ENS = EnsembleState(N, zeros(NS))
    for i  = 1:N
        ENS.S[i] = B[:,i]
    end
    return ENS
end


function fill!(ENS::EnsembleState{N, TS}, A::TS)  where {N, TS}
    B = deepcopy(A)
    for s in ENS.S
        s .= B
    end
    return ENS
end



"Define addition of two EnsembleState"

function (+)(A::EnsembleState{N, TS}, B::EnsembleState{N, TS}) where {N, TS}
    C = deepcopy(A)
    C.S .+= B.S
    return C
end

"Define addition of an Array and an EnsembleState"

function (+)(A::EnsembleState{N, TS}, B::TS) where {N, TS}
    C = deepcopy(A)
    for s in C.S
        s .+=B
    end
    return C
end


"Define substraction of two EnsembleState"

function (-)(A::EnsembleState{N, TS}, B::EnsembleState{N, TS}) where {N, TS}
    C = deepcopy(A)
    C.S .-= B.S
    return C
end

"Define substraction of an Array from an EnsembleState"

function (-)(A::EnsembleState{N, TS}, B::TS) where {N, TS}
    C = deepcopy(A)
    for s in C.S
        s .-=B
    end
    return C
end
