
using Distributions, Statistics, LinearAlgebra

import Base: size, length, *

import Statistics: mean, var, std, cov

import Distributions

import Random: AbstractRNG

export InflationType, IdentityInflation, AdditiveInflation, MultiplicativeInflation,
        MultiAdditiveInflation, TupleProduct, Mixed, ParametersInflation, RecipeInflation

"""
    InflationType

An abstract type for Inflation.
"""

abstract type InflationType end

"""
    IdentityInflation


An type to store identity inflation :

Define additive inflation: x <- x



"""


mutable struct IdentityInflation <: InflationType

end


" Define action of IdentityInflation on an EnsembleState : x <- x  "

function (A::IdentityInflation)(ENS::EnsembleState{N, TS}) where {N, TS}
    return ENS
end



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


function AdditiveInflation(α::MultivariateDistribution)
    NS = length(α)
    return AdditiveInflation{NS}(α)
end

# MvNormal(μ, σ) where σ is the standard deviation and not the covariance

"""
    size(A::AdditiveInflation) -> Tuple{Int...}

Return the dimension of the additive inflation

"""
Base.size(A::AdditiveInflation{NS}) where {NS} = size(A.α)

"""
    length(A::AdditiveInflation) -> Int

Return the dimension of the additive inflation

"""
Base.length(A::AdditiveInflation{NS}) where {NS} = length(A.α)



mean(A::AdditiveInflation{NS}) where {NS} = mean(A.α)

var(A::AdditiveInflation{NS}) where {NS} = var(A.α)

cov(A::AdditiveInflation{NS}) where {NS} = cov(A.α)

# std(A::AdditiveInflation{NS}) where {NS} = std(A.α)




" Define action of AdditiveInflation on an EnsembleState : x <- x + α "

function (A::AdditiveInflation{NS})(ENS::EnsembleState{N, TS}) where {N, NS, TS}
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

function (A::MultiplicativeInflation{NS})(ENS::EnsembleState{N, TS}) where {N, NS, TS}
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

cov(A::MultiAdditiveInflation{NS}) where {NS} = cov(A.α)


"Define action of MultiplicativeInflation : x <- x̂ + β*(x - x̂)"

function (A::MultiAdditiveInflation{NS})(ENS::EnsembleState{N, TS}) where {N, NS, TS}
    Ŝ = deepcopy(mean(ENS))
    for s in ENS.S
        s .= Ŝ .+ A.β * (s .- Ŝ) .+ rand(A.α)
    end
    return ENS
end


 ### Code from LowLevelParticleFilters.jl to define tuple of Distributions
"""
 Mixed value support indicates that the distribution is a mix of continuous and discrete dimensions.
 """
 struct Mixed <: ValueSupport end

"""
    TupleProduct(v::NTuple{N,UnivariateDistribution})
Create a product distribution where the individual distributions are stored in a tuple. Supports mixed/hybrid Continuous and Discrete distributions
"""
struct TupleProduct{N,S,V<:NTuple{N,UnivariateDistribution}} <: MultivariateDistribution{S}
    v::V
    function TupleProduct(v::V) where {N,V<:NTuple{N,UnivariateDistribution}}
        all(Distributions.value_support(typeof(d)) == Discrete for d in v) &&
            return new{N,Discrete,V}(v)
        all(Distributions.value_support(typeof(d)) == Continuous for d in v) &&
            return new{N,Continuous,V}(v)
        return new{N,Mixed,V}(v)
    end
end
Base.length(d::TupleProduct{N}) where N = N
Distributions._rand!(rng::AbstractRNG, d::TupleProduct, x::AbstractVector{<:Real}) =     broadcast!(dn->rand(rng, dn), x, d.v)
@generated function Distributions._logpdf(d::TupleProduct{N}, x::AbstractVector{<:Real}) where N
    :(Base.Cartesian.@ncall $N Base.:+ i->logpdf(d.v[i], x[i]))
end
Distributions.mean(d::TupleProduct) = vcat(mean.(d.v)...)
Distributions.var(d::TupleProduct) = vcat(var.(d.v)...)
Distributions.cov(d::TupleProduct) = Diagonal(var(d))
# Distributions.entropy(d::TupleProduct) = sum(entropy, d.v)



"""
    RecipeInflation

A structure to generate covariance inflation distribution from parameters p


# Fields:
- 'p' : Vector{Float64}

"""

mutable struct RecipeInflation
    p::Vector{Real}
end

# function RecipeInflation(p::AbstractVector)
#     return RecipeInflation(p)
# end
