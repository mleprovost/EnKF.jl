using Test

using Distributions, LinearAlgebra, Statistics
using EnKF


@testset "Additive Inflation" begin

T = AdditiveInflation(10)

@test size(T)==(10,)


@test length(T)==10

@test mean(T)== zeros(10)

A = MvNormal(zeros(10), 1.05*ones(10))

T.α = A

@test var(T) == var(A)

@test var(T) == 1.05^2*ones(10)

end


@testset "Multiplicative Inflation" begin

T = MultiplicativeInflation(10)


@test length(T)==10

@test T.β == 1.0


T.β = 1.05

@test T.β == 1.05

end


@testset "MultiAdditive Inflation" begin

T = MultiAdditiveInflation(10)

@test size(T)== (10,)

@test length(T)==10

@test T.β == 1.0

# @test T.α == MvNormal(zeros(10), I)

@test mean(T) == zeros(10)

@test var(T) ==  ones(10)

A = MvNormal(zeros(10), 1.05*ones(10))


T.β = 1.05

T.α = A

@test T.β == 1.05

@test mean(T) == zeros(10)

@test var(T) == 1.05^2*ones(10)


A = MvNormal(zeros(10), 1.01*ones(10))
T = MultiAdditiveInflation(10, 1.01, A)

@test T.β == 1.01

@test mean(T) == zeros(10)

@test var(T) == 1.01^2*ones(10)

end






@testset "State" begin



end
