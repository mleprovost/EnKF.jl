
using Distributions, Statistics, LinearAlgebra

import Base: size, length, show

import Statistics: mean, var, std

export ETKF



""""
Define system ETKF which performs the data assimilation
using the ensemble transform Kalman filter (Bishop, et al. 2001)


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
mutable struct ETKF{N, NZ}

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


"""
    Define action of ETKF on EnsembleState
"""
function (etkf::ETKF{N, NZ})(t::Float64,
         Δt::Float64,
         ens::EnsembleState{N, TS}) where {N, NZ, TS}

    "Propagate each ensemble member"
    etkf.f(t, ens)

    "Covariance inflation if 'isinflated==true' "
    if etkf.isinflated ==true
        etkf.A(ens)
    end

    x̄ᶠ = mean(deepcopy(ens))

    Hx̄ᶠ = etkf.m(t, x̄ᶠ)

    d = etkf.z(t) - Hx̄ᶠ
    "Compute measurement"
    mens = EnsembleState(N, zeros(NZ))

        for (i, s) in enumerate(ens.S)
            mens.S[i] = etkf.m(t, deepcopy(s))
        end
    HXᶠ = hcat(deepcopy(mens))


    "Compute deviation from measurement of the mean"
    Ŝ = mean(deepcopy(ens))

    HX′ᶠ  = HXᶠ.- Hx̄ᶠ

    R = cov(etkf.ϵ)
    C = inv(R)*HX′ᶠ

    A₂ = (N-1)*I + (HX′ᶠ)'*C

    EVDA₂ = Eigen(A₂)
    U = EVDA₂.vectors
    DΣ  = EVDA₂.values
    X′ᶠ = hcat(ens) .- x̄ᶠ
    D = C'*d
    w₁ = U'*D
    w₂  = w₁ ./ DΣ
    w̄ = U*w₂
    W₁′ = zeros(N, N)
    for j=1:N
        W₁′[:, j] .= √(DΣ[j])* U[:,j]
    end
    W′ = W₁′*U'
    W = W′ .+ w̄
    ens += cut(X′ᶠ*W)

    " Compute a posteriori covariance"
    ensfluc = EnsembleState(N, ens.S[1])
    deviation(ensfluc, ens)

    A′ = hcat(ensfluc)

    return t+Δt, ens, A′*A′'

end


# Create constructor for ETKF

function ETKF(N, NZ, f, A, G, m, z, ϵ, isinflated, isfiltered, isaugmented)
    return ETKF{N, NZ}(f, A, G, m, z, ϵ, isinflated, isfiltered, isaugmented)
end
