
using Distributions, Statistics, LinearAlgebra

import Base: size, length, show

import Statistics: mean, var, std

export  PropagationFunction,
        MeasurementFunction,
        RealMeasurementFunction,
        ENKF

struct PropagationFunction end

struct MeasurementFunction end

struct RealMeasurementFunction end



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



mutable struct ENKF{N, TS, NZ, TZ}

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
    ϵ::AdditiveInflation{NZ}

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

" Define action of ENKF on EnsembleState "
function (enkf::ENKF{N, TS, NZ, TZ})(t::Float64,
         Δt::Float64,
         ens::EnsembleState{N, NS, TS}) where {N, NS, TS, NZ, TZ}

    "Propagate each ensemble member"
    enkf.f(t, ens)



    "Covariance inflation if 'isinflated==true' "
    if enkf.isinflated ==true
        enkf.A(ens)
    end

    "Compute mean and deviation"
    Ŝ = deepcopy(mean(ens))

    ensfluc = EnsembleState(size(ens))

    deviation(ensfluc, ens)

    A′ = hcat(ensfluc)

    "Compute measurement"
    mens = EnsembleState((N, NZ))
    enkf.m(t, mens, ens)

    Â = hcat(deepcopy(mens))

    "Compute deviation from measurement of the mean"
    Â′  = Â .- enkf.m(t, mean(ens))

    "Get actual measurement"
    zens = EnsembleState((N, NZ))
    enkf.z(t+Δt, zens)

    "Perturb actual measurement"
    ϵ(zens)

    D = hcat(zens)


    "Analysis step with representers, Evensen, Leeuwen et al. 1998"

    if enkf.isaugmented ==true
    "Construct representers"

    b = ((Â′*Â′') + (N-1)*cov(enkf.ϵ)*I) \ (D - Â)

    Bᵀb = (A′*Â′')*b


    "Analysis step"
    ens += cut(Bᵀb)

    end

    return t+Δt, ens

end

# size(enkf::ENKF{N, NS, TS, NZ, TZ}) where {N, NS, TS, NZ, TZ} = (N, NS, NZ)


# function Base.show(io::IO, sys::ENKF{N,, TS, NZ, TZ}) where {N, NS, TS, NZ, TZ}
#     print(io, "Ensemble Kalman filter with $N members of state of length $NS and measurement vector of length $NZ")
# end
