
using Distributions, Statistics, LinearAlgebra

import Base: size, length

import Statistics: mean, var, std

export InflationType, AdditiveInflation, MultiplicativeInflation,
        MultiAdditiveInflation

"""
    InflationType

An abstract type for Inflation.
"""

abstract type InflationType end



"""
    AdditiveInflation


An type to store additive inflation :

Define additive inflation: x <- x + α with α a N-dimensional vector
drawn from a random distribution

# Fields:
- 'α' : Distribution of the additive inflation

"""


mutable struct AdditiveInflation{NS} <: InflationType

    "Distribution of the additive inflation α"
    α::MultivariateDistribution
end


# By default, the distribution of the additive inflation α is a multivariate
 # normal distribution with zero mean and identity as the covariance matrix

function AdditiveInflation(NS::Int)
    return AdditiveInflation{NS}(MvNormal(zeros(NS), I))
end


# MvNormal(μ, σ) where σ is the standard deviation and not the covariance

"""
    size(A::AdditiveInflation) -> Tuple{Int...}

Return the dimension of the additive inflation

"""
Base.size(A::AdditiveInflation{NS}) where {NS}= size(A.α)

"""
    length(A::AdditiveInflation) -> Int

Return the dimension of the additive inflation

"""
Base.length(A::AdditiveInflation{NS}) where {NS} = length(A.α)



mean(A::AdditiveInflation{NS}) where {NS} = mean(A.α)

var(A::AdditiveInflation{NS}) where {NS} = var(A.α)

# std(A::AdditiveInflation{NS}) where {NS} = std(A.α)




" Define action of AdditiveInflation on an EnsembleState : x <- x + α "

function (A::AdditiveInflation{NS})(ENS::EnsembleState{N, NS, TS}) where {N, NS, TS}
    for s in ENS.S
        s .+= rand(A.α)
    end
    return ENS
end



"""
    MultiplicativeInflation


An type to store multiplicative inflation :

Define multiplicative inflation: x <- x + β*(x - x̂) with β a scalar

# Fields:
- 'β' : multiplicative inflation factor

"""


mutable struct MultiplicativeInflation{NS} <: InflationType

    "Multiplicative inflation factor β"
    β::Real
end

# By default, the multiplicative inflation factor β is set to 1.0

function MultiplicativeInflation(NS::Int)
    return MultiplicativeInflation{NS}(1.0)
end

function MultiplicativeInflation(NS::Int,β::Real)
    return MultiplicativeInflation{NS}(β)
end



"""
    length(A::MultiplicativeInflation) -> Int

Return the dimension of the multiplicative inflation

"""
Base.length(A::MultiplicativeInflation{NS}) where {NS} = NS


"Define action of MultiplicativeInflation : x <- x̂ + β*(x - x̂)"

function (A::MultiplicativeInflation{NS})(ENS::EnsembleState{N, NS, TS}) where {N, NS, TS}
    Ŝ = deepcopy(mean(ENS))
    for s in ENS.S
        s .= Ŝ .+ A.β * (s .- Ŝ)
    end
    return ENS
end




"""
    MultiAdditiveInflation


An type to store multiplico-additive inflation :

Define multiplico-additive inflation: x̃⁻ <- x̂⁻ + β*(x̃⁻ - x̂⁻)  + α with β a scalar

# Fields:
- 'β' : Multiplicative inflation factor
- 'α' : Distribution of the additive inflation

"""


mutable struct MultiAdditiveInflation{NS} <: InflationType

    "Multiplicative inflation factor β"
    β::Real

    "Distribution of the additive inflation α"
    α::MultivariateDistribution
end

# By default, for a Multiplico-additive inflation, the multiplicative inflation
# factor β is set to 1.0, and  α is a  multivariate
 # normal distribution with zero mean and identity as the covariance matrix

function MultiAdditiveInflation(NS::Int)
    return MultiAdditiveInflation{NS}(1.0, MvNormal(zeros(NS), I))
end

function MultiAdditiveInflation(NS::Int,β::Real,α::MultivariateDistribution)
    return MultiAdditiveInflation{NS}(β, α)
end


"""
    size(A::MultiAdditiveInflation) -> Tuple{Int...}

Return the dimension of the additive inflation

"""
Base.size(A::MultiAdditiveInflation{NS}) where {NS}= size(A.α)

"""
    length(A::MultiAdditiveInflation) -> Int

Return the dimension of the additive inflation

"""
Base.length(A::MultiAdditiveInflation{NS}) where {NS} = length(A.α)


mean(A::MultiAdditiveInflation{NS}) where {NS} = mean(A.α)

var(A::MultiAdditiveInflation{NS}) where {NS} = var(A.α)


"Define action of MultiplicativeInflation : x <- x̂ + β*(x - x̂)"

function (A::MultiAdditiveInflation{NS})(ENS::EnsembleState{N, NS, TS}) where {N, NS, TS}
    Ŝ = deepcopy(mean(ENS))
    for s in ENS.S
        s .= Ŝ .+ A.β * (s .- Ŝ) .+ rand(A.α)
    end
    return ENS
end
