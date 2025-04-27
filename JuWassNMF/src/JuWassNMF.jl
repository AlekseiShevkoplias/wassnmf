module JuWassNMF
    using OptimalTransport
    using OptimalTransport.Dual
    using StatsBase
    using Plots
    using LogExpFunctions
    using NNlib: NNlib
    using LinearAlgebra
    using Distances
    using Base.Iterators
    using NMF
    using Optim
    using LineSearches
    using Random  # Add this for RNG control

    export wasserstein_nmf

    function simplex_norm!(x; dims=1)
        return x .= x ./ sum(x; dims=dims)
    end

    function E_star(x; dims=1)
        return logsumexp(x; dims=dims)
    end

    function E_star_grad(x; dims=1)
        return NNlib.softmax(x; dims=1)
    end

    function dual_obj_weights(X, K, ε, D, G, ρ1)
        return sum(Dual.ot_entropic_semidual(X, G, ε, K)) + ρ1 * sum(E_star(-D' * G / ρ1))
    end

    function dual_obj_weights_grad!(∇, X, K, ε, D, G, ρ1)
        ∇ .= Dual.ot_entropic_semidual_grad(X, G, ε, K) - D * E_star_grad(-D' * G / ρ1)
    end

    function dual_obj_dict(X, K, ε, Λ, G, ρ2)
        return sum(Dual.ot_entropic_semidual(X, G, ε, K)) + ρ2 * sum(E_star(-G * Λ' / ρ2))
    end

    function dual_obj_dict_grad!(∇, X, K, ε, Λ, G, ρ2)
        ∇ .= Dual.ot_entropic_semidual_grad(X, G, ε, K) - E_star_grad(-G * Λ' / ρ2) * Λ
    end

    function getprimal_weights(D, G, ρ1)
        return NNlib.softmax(-D' * G / ρ1; dims=1)
    end

    function getprimal_dict(Λ, G, ρ2)
        return NNlib.softmax(-G * Λ' / ρ2; dims=1)
    end

    function solve_weights(X, K, ε, D, ρ1; alg, options)
        opt = optimize(
            g -> dual_obj_weights(X, K, ε, D, g, ρ1),
            (∇, g) -> dual_obj_weights_grad!(∇, X, K, ε, D, g, ρ1),
            zero.(X),
            alg,
            options,
        )
        return getprimal_weights(D, Optim.minimizer(opt), ρ1)
    end

    function solve_dict(X, K, ε, Λ, ρ2; alg, options)
        opt = optimize(
            g -> dual_obj_dict(X, K, ε, Λ, g, ρ2),
            (∇, g) -> dual_obj_dict_grad!(∇, X, K, ε, Λ, g, ρ2),
            zero.(X),
            alg,
            options,
        )
        return getprimal_dict(Λ, Optim.minimizer(opt), ρ2)
    end

    """
    CPU version of Wasserstein NMF with seeding for deterministic results
    
    Parameters:
    -----------
    X : Matrix{Float64}
        Input data matrix
    K : Matrix{Float64}
        Cost matrix
    k : Int
        Number of components
    eps : Float64
        Entropic regularization parameter (default: 0.025)
    rho1 : Float64
        Regularization parameter for weights (default: 0.05)
    rho2 : Float64
        Regularization parameter for dictionary (default: 0.05)
    n_iter : Int
        Number of iterations (default: 10)
    verbose : Bool
        Whether to print progress information (default: true)
    seed : Union{Int, Nothing}
        Random seed for reproducible results (default: nothing)
    
    Returns:
    --------
    D : Matrix{Float64}
        Dictionary matrix
    Λ : Matrix{Float64}
        Weights matrix
    """
    function wasserstein_nmf(
        X::Matrix{Float64}, 
        K::Matrix{Float64},
        k::Int;
        eps::Float64=0.025,
        rho1::Float64=0.05,
        rho2::Float64=0.05,
        n_iter::Int=10,
        verbose::Bool=true,
        seed::Union{Int, Nothing}=1337
    )
        # Set RNG seed if provided
        if seed !== nothing
            rng = Random.MersenneTwister(seed)
        else
            rng = Random.GLOBAL_RNG
        end
        verbose && @info "Starting WassNMF calculation..."

        # Initialize D and Λ with seeded randomness
        D = rand(rng, size(X, 1), k)
        simplex_norm!(D; dims=1)
        Λ = rand(rng, k, size(X, 2))
        simplex_norm!(Λ; dims=1)

        opt_options = Optim.Options(
            iterations=250,
            g_tol=1e-4,
            show_trace=false,
            show_every=10,
        )

        for iter in 1:n_iter
            verbose && @info "Wasserstein-NMF: iteration $iter"
            D .= solve_dict(X, K, eps, Λ, rho2; alg=LBFGS(), options=opt_options)
            Λ .= solve_weights(X, K, eps, D, rho1; alg=LBFGS(), options=opt_options)
        end

        return D, Λ
    end
end
