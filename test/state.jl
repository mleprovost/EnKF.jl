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


A = MvNormal(ones(5), I)
out = initialize(10, A)
out̄ = mean(out)

out̃1 = out.S[1] .- out̄
out̃2 = out.S[2] .- out̄

fluc = deviation(out)

@test fluc.S[1] == out̃1
@test fluc.S[2] == out̃2
end
