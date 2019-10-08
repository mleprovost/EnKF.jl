
using Distributions, Statistics, LinearAlgebra

import Base: size, length, show

import Statistics: mean, var, std

export  PropagationFunction,
        MeasurementFunction,
        FilteringFunction,
        RealMeasurementFunction

struct PropagationFunction end

struct MeasurementFunction end

struct FilteringFunction end

struct RealMeasurementFunction end
