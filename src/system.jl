
using Distributions, Statistics, LinearAlgebra

import Base: size, length, show

import Statistics: mean, var, std

export  PropagationFunction,
        MeasurementFunction,
        FilteringFunction,
        RealMeasurementFunction,
        ENKF

struct PropagationFunction end

struct MeasurementFunction end

struct FilteringFunction end

struct RealMeasurementFunction end



""""
Define system ENKF which performs the



Fields:
 - 'f' : propagation function

 - 'A' : inflation

 - 'G' : filtering function acting on the state

 - 'm' : measurement function based on state

 - 'z' : real measurement function

 - 'ϵ' : measurement noise distribution

 - 'bounds' : bounds on certain states

 - 'isinflated' : Bool = true if state is inflated,
     = false otherwise

 - 'isfiltered' : Bool = true if state has to be filtered,
     = false otherwise

 - 'isaugmented' : Bool = true if measurement function is nonlinear,
     = false otherwise
"""



mutable struct ENKF{N, NZ}

    # "Ensemble of states"
    # ENS::EnsembleState{N, NS, TS}

    "Propagation function"
    f::PropagationFunction

    "Covariance Inflation"
    A::Union{InflationType, RecipeInflation}

    "Filter function"
    G::FilteringFunction

    "Measurement function based on state"
    m::MeasurementFunction

    "Real measurement function"
    z::RealMeasurementFunction

    "Measurement noise distribution"
    ϵ::AdditiveInflation{NZ}

    "Boolean: is state vector inflated"
    isinflated::Bool

    "Boolean: is state vector filtered"
    isfiltered::Bool

    "Boolean: is state vector augmented"
    isaugmented::Bool
    # "Bounds on certain state"
    # bounds

end


" Define action of ENKF on EnsembleState "
function (enkf::ENKF{N, NZ})(t::Float64,
         Δt::Float64,
         ens::EnsembleState{N, TS}) where {N, NZ, TS}

    "Propagate each ensemble member"
    enkf.f(t, ens)


    # println("good prop")
    "Covariance inflation if 'isinflated==true' "
    if enkf.isinflated ==true
        enkf.A(ens)
    end
    # println("good inflation")

    "State filtering if 'isfiltered==true' "
    if enkf.isfiltered ==true
        enkf.G(ens)
    end
    # println("good filtering")



    "Compute mean and deviation"

    ensfluc = EnsembleState(N, ens.S[1])

    deviation(ensfluc, ens)

    A′ = hcat(ensfluc)

    # println("good deviation")

    "Compute measurement"
    mens = EnsembleState(N, zeros(NZ))

        for (i, s) in enumerate(ens.S)
            mens.S[i] = enkf.m(t, deepcopy(s))
        end
    Â = hcat(deepcopy(mens))

    "Define measurement matrix H for linear observations, can be time-varying"
    if enkf.isaugmented == false
        H = deepcopy(enkf.m(t))
    end

    "Compute deviation from measurement of the mean"
    Ŝ = mean(deepcopy(ens))

    Â′  = Â .- enkf.m(t, Ŝ)

    # println("good deviation mean")

    "Get actual measurement"
    zens = EnsembleState(N, zeros(NZ))
    enkf.z(t+Δt, zens)
    # println("good actual measurement")

    "Perturb actual measurement"
    enkf.ϵ(zens)

    D = hcat(zens)


    "Analysis step with representers, Evensen, Leeuwen et al. 1998"

    "Construct representers"
    if enkf.isaugmented ==true

        b = ((Â′*Â′') + (N-1)*cov(enkf.ϵ)*I) \ (D - Â)

        Bᵀb = (A′*Â′')*b

    else

        b = ( H * (A′*A′') * H' + (N-1)*cov(enkf.ϵ)*I) \ (D - Â)

        Bᵀb = (A′*A′')* H' * b
    end

    "Analysis step"
    ens += cut(Bᵀb)

    "State filtering if 'isfiltered==true' "
    if enkf.isfiltered ==true
        enkf.G(ens)
    end

    " Compute a posteriori covariance"
    deviation(ensfluc, ens)

    A′ = hcat(ensfluc)


    return t+Δt, ens, A′*A′'

end


# Create constructor for ENKF

function ENKF(N, NZ, f, A, G, m, z, ϵ, isinflated, isfiltered, isaugmented)
    return ENKF{N, NZ}(f, A, G, m, z, ϵ, isinflated, isfiltered, isaugmented)
end


# size(enkf::ENKF{N, TS, NZ, TZ}) where {N, TS, NZ, TZ} = (N, size, NZ)


# function Base.show(io::IO, sys::ENKF{N, TS, NZ, TZ}) where {N, TS, NZ, TZ}
#     NS = size()
#     print(io, "Ensemble Kalman filter with $N members of state of size $ and measurement vector of length $NZ")
# end
