
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

Define additive inflation: x̃⁻ <- x̃⁻ + α with α a N-dimensional vector
drawn from a random distribution

# Fields:
- 'α' : Distribution of the additive inflation

"""


mutable struct AdditiveInflation{N} <: InflationType

    "Distribution of the additive inflation α"
    α::MultivariateDistribution
end


# By default, the distribution of the additive inflation α is a multivariate
 # normal distribution with zero mean and identity as the covariance matrix

function AdditiveInflation(N::Int)
    return AdditiveInflation{N}(MvNormal(zeros(N), I))
end


# MvNormal(μ, σ) where σ is the standard deviation and not the covariance

"""
    size(A::AdditiveInflation) -> Tuple{Int...}

Return the dimension of the additive inflation

"""
Base.size(A::AdditiveInflation{N}) where {N}= size(A.α)

"""
    length(A::AdditiveInflation) -> Int

Return the dimension of the additive inflation

"""
Base.length(A::AdditiveInflation{N}) where {N} = length(A.α)



mean(A::AdditiveInflation{N}) where {N} = mean(A.α)

var(A::AdditiveInflation{N}) where {N} = var(A.α)

# std(A::AdditiveInflation{N}) where {N} = std(A.α)

# Define action of AdditiveInflation




"""
    MultiplicativeInflation


An type to store multiplicative inflation :

Define multiplicative inflation: x̃⁻ <- x̂⁻ + β*(x̃⁻ - x̂⁻) with β a scalar

# Fields:
- 'β' : multiplicative inflation factor

"""


mutable struct MultiplicativeInflation{N} <: InflationType

    "Multiplicative inflation factor β"
    β::Real
end

# By default, the multiplicative inflation factor β is set to 1.0

function MultiplicativeInflation(N::Int)
    return MultiplicativeInflation{N}(1.0)
end

function MultiplicativeInflation(N::Int,β::Real)
    return MultiplicativeInflation{N}(β)
end



"""
    length(A::MultiplicativeInflation) -> Int

Return the dimension of the multiplicative inflation

"""
Base.length(A::MultiplicativeInflation{N}) where {N} = N


# Define action of MultiplicativeInflation






"""
    MultiAdditiveInflation


An type to store multiplico-additive inflation :

Define multiplico-additive inflation: x̃⁻ <- x̂⁻ + β*(x̃⁻ - x̂⁻)  + α with β a scalar

# Fields:
- 'β' : Multiplicative inflation factor
- 'α' : Distribution of the additive inflation

"""


mutable struct MultiAdditiveInflation{N} <: InflationType

    "Multiplicative inflation factor β"
    β::Real

    "Distribution of the additive inflation α"
    α::MultivariateDistribution
end

# By default, for a Multiplico-additive inflation, the multiplicative inflation
# factor β is set to 1.0, and  α is a  multivariate
 # normal distribution with zero mean and identity as the covariance matrix

function MultiAdditiveInflation(N::Int)
    return MultiAdditiveInflation{N}(1.0, MvNormal(zeros(N), I))
end

function MultiAdditiveInflation(N::Int,β::Real,α::MultivariateDistribution)
    return MultiAdditiveInflation{N}(β, α)
end


"""
    size(A::MultiAdditiveInflation) -> Tuple{Int...}

Return the dimension of the additive inflation

"""
Base.size(A::MultiAdditiveInflation{N}) where {N}= size(A.α)

"""
    length(A::MultiAdditiveInflation) -> Int

Return the dimension of the additive inflation

"""
Base.length(A::MultiAdditiveInflation{N}) where {N} = length(A.α)


mean(A::MultiAdditiveInflation{N}) where {N} = mean(A.α)

var(A::MultiAdditiveInflation{N}) where {N} = var(A.α)
