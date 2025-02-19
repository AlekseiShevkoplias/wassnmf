module JuWassNMF
# from https://juliaoptimaltransport.github.io/OptimalTransport.jl/dev/examples/nmf
    using OptimalTransport
    using OptimalTransport.Dual
    # using MLDatasets: MLDatasets
    using StatsBase
    using Plots;
    default(; palette=:Set1_3)
    using LogExpFunctions
    using NNlib: NNlib
    using LinearAlgebra
    using Distances
    using Base.Iterators
    using NMF
    using Optim
    using LineSearches

    export wasserstein_nmf

    function simplex_norm!(x; dims=1)
        return x .= x ./ sum(x; dims=dims)
    end

    function E_star(x; dims=1)
        return logsumexp(x; dims=dims)
    end;

    function E_star_grad(x; dims=1)
        return NNlib.softmax(x; dims=1)
    end;

    function dual_obj_weights(X, K, ε, D, G, ρ1)
        return sum(Dual.ot_entropic_semidual(X, G, ε, K)) + ρ1 * sum(E_star(-D' * G / ρ1))
    end

    function dual_obj_weights_grad!(∇, X, K, ε, D, G, ρ1)
        return ∇ .= Dual.ot_entropic_semidual_grad(X, G, ε, K) - D * E_star_grad(-D' * G / ρ1)
    end

    function dual_obj_dict(X, K, ε, Λ, G, ρ2)
        return sum(Dual.ot_entropic_semidual(X, G, ε, K)) + ρ2 * sum(E_star(-G * Λ' / ρ2))
    end

    function dual_obj_dict_grad!(∇, X, K, ε, Λ, G, ρ2)
        return ∇ .= Dual.ot_entropic_semidual_grad(X, G, ε, K) - E_star_grad(-G * Λ' / ρ2) * Λ
    end;

    function getprimal_weights(D, G, ρ1)
        return NNlib.softmax(-D' * G / ρ1; dims=1)
    end

    function getprimal_dict(Λ, G, ρ2)
        return NNlib.softmax(-G * Λ' / ρ2; dims=1)
    end;

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
    end;
    function wasserstein_nmf(
        X::Matrix{Float64}, 
        K::Matrix{Float64},  # The kernel matrix
        k::Int; 
        eps::Float64=0.025,
        rho1::Float64=0.05,
        rho2::Float64=0.05,
        n_iter::Int=10,
        verbose::Bool=true
    )
        D = rand(size(X, 1), k)
        simplex_norm!(D; dims=1)
        Λ = rand(k, size(X, 2))
        simplex_norm!(Λ; dims=1)
    
        opt_options = Optim.Options(
            iterations=250,
            g_tol=1e-4,
            show_trace=false,
            show_every=10,

        )
    
        for iter in 1:n_iter
            verbose && @info "Wasserstein-NMF: iteration $iter"
            D .= solve_dict(X, K, eps, Λ, rho2; alg=LBFGS(
                # linesearch = BackTracking(order=2)
                ), options=opt_options)
            Λ .= solve_weights(X, K, eps, D, rho1; alg=LBFGS(
                # linesearch = BackTracking(order=2)
                ), options=opt_options)
        end
    
        return D, Λ
    end
end