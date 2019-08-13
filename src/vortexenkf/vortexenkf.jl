using EnKF

using PotentialFlow

export VortexENKF




"""

Create a specific structure for EnKF-based Vortex model,
Darakananda, et al. Data-assimilated low-order vortex modeling of separated flows, APS (2018)


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


mutable struct VortexENKF{N, NZ}

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

end

# 
# " Define action of VortexENKF.f on EnsembleState "
#
# function (venkf::VortexENKF{N, NZ})(t::Float64,
#          Δt::Float64,
#          ens::EnsembleState{N, TS}) where {N, NZ, TS}
#
#
#
#
#








" Define action of VortexENKF on EnsembleState "
function (venkf::VortexENKF{N, NZ})(t::Float64,
         Δt::Float64,
         ens::EnsembleState{N, TS}) where {N, NZ, TS}

    "Propagate each ensemble member"
    venkf.f(t, ens)


    # println("good prop")
    "Covariance inflation if 'isinflated==true' "
    if venkf.isinflated ==true
        venkf.A(ens)
    end
    # println("good inflation")

    "State filtering if 'isfiltered==true' "
    if venkf.isfiltered ==true
        venkf.G(ens)
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
        mens.S[i] = venkf.m(t, deepcopy(s))
    end

    Â = hcat(deepcopy(mens))
    # println("good measurement")

    "Compute deviation from measurement of the mean"
    Ŝ = mean(deepcopy(ens))

    Â′  = Â .- venkf.m(t, Ŝ)

    # println("good deviation mean")

    "Get actual measurement"
    zens = EnsembleState(N, zeros(NZ))
    venkf.z(t+Δt, zens)
    # println("good actual measurement")

    "Perturb actual measurement"
    venkf.ϵ(zens)

    D = hcat(zens)


    "Analysis step with representers, Evensen, Leeuwen et al. 1998"

    if venkf.isaugmented ==true
    "Construct representers"

    b = ((Â′*Â′') + (N-1)*cov(enkf.ϵ)*I) \ (D - Â)

    Bᵀb = (A′*Â′')*b
    # print(size(Bᵀb))
    # print(size(ens))

    "Analysis step"
    ens += cut(Bᵀb)

    "State filtering if 'isfiltered==true' "
    if venkf.isfiltered ==true
        venkf.G(ens)
    end

    end

    return t+Δt, ens

end


# Create constructor for ENKF

function VortexENKF(N, NZ, f, A, G, m, z, ϵ, isinflated, isfiltered, isaugmented)
    return VortexENKF{N, NZ}(f, A, G, m, z, ϵ, isinflated, isfiltered, isaugmented)
end
