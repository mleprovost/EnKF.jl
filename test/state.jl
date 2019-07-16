using Test

using Distributions, LinearAlgebra, Statistics
using EnKF


@testset "EnsembleState" begin

States = [randn(10) for i = 1:4]

EnS = EnsembleState(States)

@test EnS.S == States

@test size(EnS) == (4,10)

@test mean(EnS) - (EnS.S[1]+EnS.S[2]+EnS.S[3]+EnS.S[4])/4 ==zeros(10)

EnS = EnsembleState(4, 5)

@test size(EnS) == (4,5)

@test EnS.S[1] == zeros(5)




end
