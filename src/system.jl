
using Distributions, Statistics, LinearAlgebra

import Base: size, length, show

import Statistics: mean, var, std

abstract type PropagationFunction end

abstract type MeasurementFunction end

abstract type RealMeasurementFunction end



""""
Define system ENKF which performs the



Fields:
 - 'f' : propagation function

 - 'A' : inflation

 - 'm' : measurement function based on state

 - 'z' : real measurement function

 - 'ϵ' : measurement noise distribution

 - 'bounds' : bounds on certain states

 - 'isinflated' : Bool = true if state is inflated,
     = false otherwise

 - 'isaugmented' : Bool = true if measurement function is nonlinear,
     = false otherwise
"""



mutable struct ENKF{N, NS, TS, NZ, TZ}

    # "Ensemble of states"
    # ENS::EnsembleState{N, NS, TS}

    "Propagation function"
    f::PropagationFunction

    "Covariance Inflation"
    A::InflationType

    "Measurement function based on state"
    m::MeasurementFunction

    "Real measurement function"
    z::RealMeasurementFunction

    "Measurement noise distribution"
    ϵ::AdditiveInflation{NS}

    "Boolean: is state vector inflated"
    isinflated::Bool

    "Boolean: is state vector augmented"
    isaugmented::Bool
    # "Bounds on certain state"
    # bounds

end

# This is just the default constructeur
# function ENKF(ENS::EnsembleState{N, NS, TS},
#     f::PropagationFunction,
#     A::InflationType,
#     m::MeasurementFunction,
#     z::RealMeasurementFunction,
#     ϵ::AdditiveInflation{NS},
#     isinflated::Bool,
#     isaugmented::Bool) where {N, NS, TS}
#
#     return ENKF{N, NS, TS, NZ, TZ}(f, A, m, z, ϵ, isinflated, isaugmented)
# end


# Define action of PropagationFunction, MeasurementFunction on the states
# function (f::PropagationFunction)(t::Float64, S::Array{TS,1}) where {TS}
#     return f(t,S)
# end
#
#
# function (f::PropagationFunction)(t::Float64, ENS::EnsembleState{N, NS, TS}) where {N, NS, TS}
#     #Allow for size of the output to vary (aggregation and )
#     out = []
#
#     for (k,sk) in enumerate(ENS.S)
#         push!(out, f(t,sk))
#     end
#
#     return EnsembleState(out)
#
# end

# Define action of ENKF on EnsembleState
function (enkf::ENKF{N, NS, TS, NZ, TZ})(t::Float64, Δt::Float64, ENS::EnsembleState{N, NS, TS}) where {N, NS, TS, NZ, TZ}

    "Propagate each ensemble member"
    enkf.f(t, ENS)

    "Covariance inflation if 'isinflated==true' "
    if isinflated ==true
        enkf.A(ENS)
    end

    "Compute mean and deviation"
    Ŝ = deepcopy(mean(ENS))

    ENSfluc = EnsembleState(size(ENS))

    deviation(ENSfluc, ENS)

    "Compute measurement"
    mENS = EnsembleState((N, NZ))
    m(t, mENS, ENS)

    mENS .-= m(t, Ŝ)

    "Get actual measurement"
    zENS = EnsembleState((N, NZ))


















function size()


function Base.show(io::IO, sys::ENKF{N, NS, TS, NZ, TZ})
    where {N, NS, TS, NZ, TZ}
    print(io, "Ensemble Kalman filter with $N members of state with length $NS and measurement vector of length $NZ")
end
