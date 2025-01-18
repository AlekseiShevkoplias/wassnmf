using Test
using JuWassNMF
using Random
using NNlib
using LinearAlgebra
using Distances

# Helper function for data generation
function generate_test_data(n_features=100, n_samples=100)
    coord = range(-12, 12, length=n_features)
    f(x, μ) = exp.(-(x .- μ) .^ 2)
    
    Random.seed!(42)
    X = hcat(
        [
            rand() * f(coord, randn() + 6) +
            rand() * f(coord, randn()) +
            rand() * f(coord, randn() - 6) for _ in 1:n_samples
        ]...
    )
    return JuWassNMF.simplex_norm!(X, dims=1)
end

@testset "JuWassNMF.jl Tests" begin
    @testset "Utility Functions" begin
        x = rand(5, 3)
        JuWassNMF.simplex_norm!(x, dims=1)
        @test all(isapprox.(sum(x, dims=1), 1.0, atol=1e-6))
        
        # Test E_star and its gradient
        y = rand(5, 3)
        @test size(JuWassNMF.E_star(y)) == (1, 3)
        @test size(JuWassNMF.E_star_grad(y)) == size(y)
    end

    @testset "Data Generation & Input Validation" begin
        X = generate_test_data(20, 10)
        @test size(X) == (20, 10)
        @test all(isapprox.(sum(X, dims=1), 1.0, atol=1e-6))
        @test all(X .>= 0)
    end

    @testset "Core Functions" begin
        X = generate_test_data(10, 5)
        k = 2
        eps = 0.025
        
        # Test cost matrix computation
        coord = 1:10
        C = pairwise(SqEuclidean(), coord)
        K = exp.(-C / eps)
        @test size(K) == (10, 10)
        @test issymmetric(K)
        
        # Initialize D and Λ
        D = rand(10, k)
        D = JuWassNMF.simplex_norm!(D, dims=1)
        Λ = rand(k, 5)
        Λ = JuWassNMF.simplex_norm!(Λ, dims=1)
        
        # Test dimensions and constraints
        @test size(D) == (10, k)
        @test size(Λ) == (k, 5)
        @test all(isapprox.(sum(D, dims=1), 1.0, atol=1e-6))
        @test all(isapprox.(sum(Λ, dims=1), 1.0, atol=1e-6))
    end

    @testset "Full WNMF with Kernel Matrix" begin
        X = generate_test_data(15, 8)
        k = 3
        coord = 1:15
        C = pairwise(SqEuclidean(), coord)
        K = exp.(-C / 0.025)
        
        # Test with default parameters
        D, Λ = JuWassNMF.wasserstein_nmf(X, K, k, verbose=false)
        @test size(D) == (15, k)
        @test size(Λ) == (k, 8)
        @test all(D .>= 0)
        @test all(Λ .>= 0)
        @test all(isapprox.(sum(D, dims=1), 1.0, atol=1e-6))
        @test all(isapprox.(sum(Λ, dims=1), 1.0, atol=1e-6))
        
        # Test with custom parameters
        D, Λ = JuWassNMF.wasserstein_nmf(X, K, k,
            eps=0.1,
            rho1=1e-1,
            rho2=1e-1,
            n_iter=5,
            verbose=false
        )
        @test size(D) == (15, k)
        @test size(Λ) == (k, 8)
    end

    @testset "Edge Cases with Kernel Matrix" begin
        # Minimal size
        X_tiny = generate_test_data(4, 3)
        coord = 1:4
        C = pairwise(SqEuclidean(), coord)
        K = exp.(-C / 0.025)
        D, Λ = JuWassNMF.wasserstein_nmf(X_tiny, K, 2, verbose=false)
        @test size(D) == (4, 2)
        @test size(Λ) == (2, 3)
        
        # Large regularization
        X_small = generate_test_data(10, 5)
        coord = 1:10
        C = pairwise(SqEuclidean(), coord)
        K = exp.(-C / 1.0)
        D, Λ = JuWassNMF.wasserstein_nmf(X_small, K, 2, verbose=false)
        @test size(D) == (10, 2)
        @test size(Λ) == (2, 5)
    end
end
