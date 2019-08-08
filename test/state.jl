using Test

using Distributions, LinearAlgebra, Statistics
using EnKF


@testset "EnsembleState" begin

# Test size and initialize
States = [randn(10) for i = 1:4]

EnS = EnsembleState(States)

@test EnS.S == States

@test size(EnS) == (4, (10,))

@test mean(EnS) - (EnS.S[1]+EnS.S[2]+EnS.S[3]+EnS.S[4])/4 ==zeros(10)

EnS = EnsembleState(4, zeros(5))

@test size(EnS) == (4, (5,))

@test EnS.S[1] == zeros(5)


# Test mean and deviation
A = MvNormal(ones(5), I)
out = initialize(10, A)
out̄ = mean(out)

out̃1 = out.S[1] .- out̄
out̃2 = out.S[2] .- out̄

fluc = EnsembleState(10,zeros(5))
deviation(fluc, out)

@test fluc.S[1] == out̃1
@test fluc.S[2] == out̃2

# Test addition and substraction
ENS1 = initialize(10,4)

ENS2 = initialize(10,4)

ENS3 = EnsembleState(10,zeros(4))

ENS3 = ENS1 + ENS2
@test ENS1.S + ENS2.S == ENS3.S

ENS4 = EnsembleState(10,zeros(4))

ENS4 = ENS1- ENS2
@test ENS1.S - ENS2.S == ENS4.S
end
