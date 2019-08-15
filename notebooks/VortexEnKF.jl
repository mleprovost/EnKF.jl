
# include("/media/mat/HDD/EnKF/src/EnKF.jl")
Pkg.activate("/media/mat/HDD/EnKF/")
using EnKF
using PotentialFlow

# import EnKF

"""
Create a specific structure for EnKF-based Vortex model,
Darakananda, et al. Data-assimilated low-order vortex modeling of separated flows, APS (2018)

"""



"""
    VortexENKFprop

    Define propagation operator VortexENKFprop for VortexENKF
"""

function (::PropagationFunction)(t::Float64, ENS::EnsembleState{N, TS}) where {N, TS}

    empty_inds = Vector{Int}[]
    len = Int[]
    ensnew = []
    # predicted_states (with merging)
    states = deepcopy(ENS.S)
    for (i, state) in enumerate(states)
        new_state = state_equation(state, t, config)
        new_blobs, lesp, tesp = state_to_blobs(new_state, config.δ)
        push!(empty_inds, findall(b -> circulation(b) == 0, new_blobs))
        push!(len, length(new_blobs))
        push!(ensnew, new_state)
    end

    # trim zeros
    @assert all(len .== len[1])
    toremove = intersect(empty_inds...)
    tokeep = filter(i -> i ∉ toremove, 1:len[1])
    for (i, state) in enumerate(ensnew)
        new_blobs, lesp, tesp = state_to_blobs(state, config.δ)
        ensnew[i] = blobs_to_state(new_blobs[tokeep], lesp, tesp)
    end

    # Wrap into an EnsembleState variable
    ENS.S .= deepcopy(ensnew)

    return ENS
end

VortexENKFprop = PropagationFunction()

"""
    VortexENKFinflation

    Define multiplico-additive inflation for VortexENKF
"""

function (RI::RecipeInflation)(ENS::EnsembleState{N, TS}) where {N, TS}
    # unwrap parameters
    αp, αΓ, αlesp, αtesp, β, n = RI.p

    n = ceil(Int, n)

    # get size of a member
    NS = size(ENS.S[1])[1]

    Ŝ = mean(deepcopy(ENS))

    # Covariance inflation
    for (_,s) in enumerate(ENS.S)
        s .= Ŝ .+ β*(s .- Ŝ)
        # don't perturb the state at the 4 first time steps
        if n > 5
        for i in 1:3:size(Ŝ, 1)-3
            s[i:i+1] += αp*randn(2)
            s[i+2]   += αΓ*randn()
        end
            s[end-1] = s[end-1] + αlesp*randn()
            s[end] = s[end] + αtesp*randn()
        end
    end

    return ENS
end



"""
    VortexENKFfiltering

    Define state filtering for VortexENKF
    Keep LESPc of each ensemble member between 0.0 and 3.0

"""

function (::FilteringFunction)(ENS::EnsembleState{N, TS}) where {N, TS}

    for s in ENS.S
        clamp(s[end-1],0.0,3.0)
    end
    return ENS
end

VortexENKFfilter = FilteringFunction()


"""
    Define specific constructor for EnKF

"""


function VortexENKF(N, NZ, ri, m, z, ϵ)

    return ENKF{N, NZ}(VortexENKFprop, ri, VortexENKFfilter, m, z, ϵ, true, true, true)

end
