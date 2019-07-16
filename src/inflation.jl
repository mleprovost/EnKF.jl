
using Distributions, Statistics, LinearAlgebra

import Base: size, length

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
- 'N' : Dimension of the state vector
- 'α' : Distribution of the additive inflation

"""


struct AdditiveInflation{N} <: InflationType

    "Distribution of the additive inflation α"
    α::MultivariateDistribution
end



"""
    MultiplicativeInflation


An type to store multiplicative inflation :

Define multiplicative inflation: x̃⁻ <- x̂⁻ + β*(x̃⁻ - x̂⁻) with β a scalar

# Fields:
- 'β' : multiplicative inflation factor

"""


struct MultiplicativeInflation{N} <: InflationType

    "multiplicative inflation factor β"
    β::Real
end
