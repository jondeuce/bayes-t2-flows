#### Distributions

abstract type MultivariateDist end
abstract type IsotropicDist <: MultivariateDist end

(𝒟::MultivariateDist)(latents::Tuple; kwargs...) = recondition(𝒟, latents; kwargs...)
recondition(𝒟::MultivariateDist, latents::AbstractArray...; kwargs...) = recondition(𝒟, latents; kwargs...)
neglogpdf(𝒟::IsotropicDist, x::AbstractArray) = sum(isotropic_neglogpdf(𝒟, x); dims = 1)
# default to single sample approximation, 𝔼ₚ[-logp] ≈ -logp(x)
function neglogpdf_with_entropy(𝒟::MultivariateDist, x)
    neglogp = neglogpdf(𝒟, x)
    return (neglogp, neglogp)
end

sample_similar(𝒟::MultivariateDist, x::AbstractArray) = sample_similar(𝒟, x, size(x))
sample_similar(𝒟::MultivariateDist, x::AbstractArray, sz1::Int, sz::Int...) = sample_similar(𝒟, x, (sz1, sz...))
sample_similar(𝒟::MultivariateDist, x::AbstractArray, ::Colon, sz::Int...) = sample_similar(𝒟, x, :, sz)
sample_similar(𝒟::MultivariateDist, x::AbstractArray, ::Colon, sz::Tuple{Vararg{Int}}) = sample_similar(𝒟, x, (size(x, 1), sz...)) # override this if first dimension should be inferred
sample_similar(𝒟::MultivariateDist, ::AbstractArray, sz::Tuple{Vararg{Int}}) = error("Sampling not implemented for $(nameof(typeof(𝒟)))") # only need to define this one for most cases

DomainInfo(dim::Int, manifold_dim::Int = dim) = (; dim, manifold_dim) # embedding dimension and intrinsic manifold dimension

struct UnitNormalDist <: IsotropicDist end
@saveable UnitNormalDist

isotropic_neglogpdf(::UnitNormalDist, x) = lib.neglogpdf_unit_gaussian.(x)
sample_similar(::UnitNormalDist, x::AbstractArray, sz::Tuple{Vararg{Int}}) = lib.randn_similar(x, sz...)
neglogpdf_with_entropy(𝒟::UnitNormalDist, x) = (neglogpdf(𝒟, x), lib.fill_similar_row(x, size(x, 1) * lib.entropy_unit_gaussian(eltype(x))))

struct NormalDist{A, B} <: IsotropicDist
    μ::A
    logσ::B
end
@saveable NormalDist

isotropic_neglogpdf(𝒟::NormalDist, x) = lib.neglogpdf_gaussian.(x, 𝒟.μ, 𝒟.logσ)
sample_similar(𝒟::NormalDist, x::AbstractArray, sz::Tuple{Vararg{Int}}) = lib.sample_mv_normal(𝒟.μ, exp.(𝒟.logσ), lib.randn_similar(x, sz...))

struct RicianDist{A, B} <: IsotropicDist
    ν::A
    logσ::B
end
@saveable RicianDist

isotropic_neglogpdf(𝒟::RicianDist, x) = lib.neglogpdf_rician.(x, 𝒟.ν, 𝒟.logσ)
isotropic_neglogpdf_quantized(𝒟::RicianDist, x, Δ) = lib.neglogpdf_qrician.(x, 𝒟.ν, 𝒟.logσ, Δ)
sample_similar(𝒟::RicianDist, x::AbstractArray, sz::Tuple{Vararg{Int}}) = lib.sample_mv_rician(𝒟.ν, exp.(𝒟.logσ), lib.randn_similar(x, sz...), lib.randn_similar(x, sz...))

struct ProductDist{P, R, S, D} <: MultivariateDist
    dists::P
    reducer::R
    splitter::S
    domains::D
end
@saveable ProductDist

ProductDist(dists, domains) = ProductDist(dists, nothing, nothing, domains)
neglogpdf(𝒟::ProductDist, x::AbstractArray) = neglogpdf(𝒟, 𝒟.splitter(x))
neglogpdf(𝒟::ProductDist, xs::Union{<:Tuple, <:NamedTuple}) = sum(map(neglogpdf, 𝒟.dists, xs))
neglogpdf_with_entropy(𝒟::ProductDist, x::AbstractArray) = neglogpdf_with_entropy(𝒟, 𝒟.splitter(x))
function neglogpdf_with_entropy(𝒟::ProductDist, xs::Union{<:Tuple, <:NamedTuple})
    neglogps, entropys = lib.unzip_structs(map(neglogpdf_with_entropy, 𝒟.dists, xs))
    return sum(neglogps), sum(entropys)
end
function sample_similar(𝒟::ProductDist, x::AbstractArray, ::Colon, sz::Tuple{Vararg{Int}})
    ys = map(𝒟.dists, 𝒟.domains) do dist, domain
        return sample_similar(dist, x, (domain.dim, sz...))
    end
    return 𝒟.reducer === nothing ? ys : 𝒟.reducer(ys...)
end
sample_similar(𝒟::ProductDist, x::AbstractArray, sz::Tuple{Vararg{Int}}) = sample_similar(𝒟, x, :, Base.tail(sz))

struct MaskedDist{P <: IsotropicDist, M} <: MultivariateDist
    dist::P
    mask::M
end
@saveable MaskedDist

neglogpdf(𝒟::MaskedDist, x) = sum(lib.apply_mask.(𝒟.mask, isotropic_neglogpdf(𝒟.dist, x)); dims = 1)
sample_similar(𝒟::MaskedDist, x::AbstractArray, sz::Tuple{Vararg{Int}}) = lib.apply_mask.(𝒟.mask, sample_similar(𝒟.dist, x, sz))

struct QuantizedDist{P <: IsotropicDist, TΔ} <: IsotropicDist
    dist::P
    Δ::TΔ
end
@saveable QuantizedDist

isotropic_neglogpdf(𝒟::QuantizedDist, x::AbstractArray) = isotropic_neglogpdf_quantized(𝒟.dist, x, 𝒟.Δ)
function sample_similar(𝒟::QuantizedDist, x::AbstractArray, sz::Tuple{Vararg{Int}})
    y = sample_similar(𝒟.dist, x, sz) # sample from underlying unquantized distribution
    return @. floor(y / 𝒟.Δ) * 𝒟.Δ # quantize samples to bin width Δ
end

struct ConditionalNormalizingFlow{Topts, F, F⁻¹, B, Z} <: MultivariateDist
    opts::Topts
    φ::F # diffeomorphism (defining a normalizing flow)
    φ⁻¹::F⁻¹ # inverse diffeomorphism (shares parameters with φ, if any)
    base::B # base distribution
    latents::Z # conditioning variables
end
@saveable ConditionalNormalizingFlow
lib.Optimisers.trainable(ℱ::ConditionalNormalizingFlow) = (; ℱ.φ, ℱ.φ⁻¹, ℱ.base, ℱ.latents)

Base.show(io::IO, ℱ::ConditionalNormalizingFlow) = print(io, "ConditionalNormalizingFlow(", string(ℱ.opts)[2:(end-1)], ", base = ", nameof(typeof(ℱ.base)), ")")

function ConditionalNormalizingFlow(
    φ, φ⁻¹, base::UnitNormalDist = UnitNormalDist(), latents::Tuple = ();
    udim::Int, xdim::Int,
)
    @ignore_derivatives lib.assert_shared_parameters(φ, φ⁻¹)
    return ConditionalNormalizingFlow((; udim, xdim), φ, φ⁻¹, base, latents)
end

function ConditionalNormalizingFlow(φ, base::UnitNormalDist = UnitNormalDist(), latents::Tuple = (); kwargs...)
    φ⁻¹ = lib.inverse_flow(φ)
    return ConditionalNormalizingFlow(φ, φ⁻¹, base, latents; kwargs...)
end

recondition(ℱ::ConditionalNormalizingFlow, latents::Tuple) = ConditionalNormalizingFlow(ℱ.opts, ℱ.φ, ℱ.φ⁻¹, ℱ.base, latents)

neglogpdf(ℱ::ConditionalNormalizingFlow, x::AbstractArray) = flow_backward(ℱ, x).neglogpx

sample_base_similar(ℱ::ConditionalNormalizingFlow, x::AbstractArray, ::Union{Int, Colon}, sz::Int...) = sample_similar(ℱ.base, x, (ℱ.opts.udim, sz...))
sample_base_similar(ℱ::ConditionalNormalizingFlow, x::AbstractArray) = sample_base_similar(ℱ, x, size(x)...)

function sample_similar(ℱ::ConditionalNormalizingFlow, x::AbstractArray, ::Colon, sz::Tuple{Vararg{Int}})
    u = sample_base_similar(ℱ, x, :, sz...)
    (; x) = flow_forward(ℱ, u, nothing)
    return x
end
sample_similar(ℱ::ConditionalNormalizingFlow, x::AbstractArray, sz::Tuple{Vararg{Int}}) = sample_similar(ℱ, x, :, Base.tail(sz))

flow_forward(ℱ::ConditionalNormalizingFlow, u::AbstractArray) = flow_forward(ℱ, u, lib.zero_logjac(u))
flow_forward(ℱ::ConditionalNormalizingFlow, u::AbstractArray, logjac::Union{AbstractArray, Nothing}) = flow_forward(ℱ, ProbabilityMass(u, logjac, ℱ.latents))
flow_forward(ℱ::ConditionalNormalizingFlow, 𝓂ᵤ::ProbabilityMass) = flow_summary(ℱ, 𝓂ᵤ, ℱ.φ(𝓂ᵤ))

flow_backward(ℱ::ConditionalNormalizingFlow, x::AbstractArray) = flow_backward(ℱ, x, lib.zero_logjac(x))
flow_backward(ℱ::ConditionalNormalizingFlow, x::AbstractArray, logjac::Union{AbstractArray, Nothing}) = flow_backward(ℱ, ProbabilityMass(x, logjac, ℱ.latents))
flow_backward(ℱ::ConditionalNormalizingFlow, 𝓂ₓ::ProbabilityMass) = flow_summary(ℱ, ℱ.φ⁻¹(𝓂ₓ), 𝓂ₓ)

function flow_forward_pathgradient(ℱ::ConditionalNormalizingFlow, u::AbstractArray)
    𝓂ᵤ, 𝓂ₓ = lib.flow_transform_pathgradient(ℱ.φ, u, lib.zero_logjac(u), ℱ.latents)
    return flow_summary(ℱ, 𝓂ᵤ, 𝓂ₓ)
end

function flow_summary(ℱ::ConditionalNormalizingFlow, 𝓂ᵤ::ProbabilityMass, 𝓂ₓ::ProbabilityMass)
    u, x, latents = Base.materialize(𝓂ᵤ.value), Base.materialize(𝓂ₓ.value), 𝓂ₓ.latents

    if 𝓂ᵤ.logjac !== nothing
        # Forward: we have computed x = φ(u) and logjac = log|∂x/∂u| = -log|∂u/∂x|
        # Backward: we have computed u = φ⁻¹(x) and logjac = log|∂u/∂x|
        neglogdudx = 𝓂ₓ.logjac .- 𝓂ᵤ.logjac # negative log jacobian of (forward) transform
        neglogpu, entropyu = neglogpdf_with_entropy(ℱ.base, u) # base distribution likelihood -logp(u)
        neglogpx = neglogpu .+ neglogdudx # -logp(x) = -logp(u) - log|∂u/∂x|
        entropyx = entropyu .+ neglogdudx # single sample for jacobian term
    else
        neglogdudx = neglogpu = neglogpx = entropyx = entropyu = nothing
    end

    return (; u, x, neglogdudx, neglogpu, neglogpx, entropyx, entropyu)
end

"""
    LikelihoodFactory{M}

Likelihood distribution factory.
Simply wraps `model` which produces a likelihood distribution via `𝒟 = sampler(ℒ::LikelihoodFactory, args...)`.
The likelihood is then called as usual via `neglogpdf(𝒟, data)`.
"""
struct LikelihoodFactory{M, D}
    model::M # returns a `like::MultivariateDist` given `latents::Tuple`
    data::D # data for evaluating likelihood `neglogpdf(like::MultivariateDist, data)`
end
@saveable LikelihoodFactory

sampler(ℒ::LikelihoodFactory, xs::AbstractArray...) = sampler(ℒ, xs)
sampler(ℒ::LikelihoodFactory, latents::Tuple) = ℒ.model(latents)

"""
Transformed unnormalized posterior.
"""
struct TransformedPosterior{F, P, L} <: MultivariateDist
    flow::F # ConditionalNormalizingFlow
    prior::P # MultivariateDist
    likelihood::L # LikelihoodFactory
end
@saveable TransformedPosterior

neglogpdf(post::TransformedPosterior, u::AbstractArray) = post(u).neglogpostu
recondition(post::TransformedPosterior, latents::Tuple) = TransformedPosterior(recondition(post.flow, latents), post.prior, post.likelihood)

function (post::TransformedPosterior)(u::AbstractArray; pathgradient = false)
    if pathgradient
        summary = flow_forward_pathgradient(post.flow, u)
    else
        summary = flow_forward(post.flow, u)
    end
    like = sampler(post.likelihood, summary.x)
    negloglike = neglogpdf(like, post.likelihood.data) # negative log likelihood -logp(y|x)
    neglogprior = neglogpdf(post.prior, summary.x) # negative log (possibly deep) prior probability -logp(x)
    return posterior_summary(summary, like, negloglike, neglogprior)
end

function posterior_summary(summary, like, negloglike, neglogprior)
    (; neglogpx, neglogdudx) = summary
    logdxdu = neglogdudx # log jacobian log|∂x/∂u| = -log|∂u/∂x| of the transformation x = φ(u)
    neglogpostu = negloglike .+ neglogprior .- logdxdu # -log p(u|y) up to a constant: -log p(y|x) - log p(x) - log|∂x/∂u|

    # Use sampled log q rather than analytic entropy for the lower-variance ELBO estimator.
    # Roeder et al. (2017), Sticking the Landing
    kldiv = neglogprior .- neglogpx
    negelbo = negloglike .+ kldiv

    return (; summary, like, negloglike, neglogprior, neglogpostu, kldiv, negelbo)
end

#### Physics model

struct InferenceVariables{Txdist, Tlogs, Tlogσ}
    xdist::Txdist
    logs::Tlogs
    logσ::Tlogσ
end
@saveable InferenceVariables

Base.NamedTuple(x::InferenceVariables) = (; x.xdist, x.logs, x.logσ)
Base.merge(x::InferenceVariables, y) = InferenceVariables(merge(NamedTuple(x), NamedTuple(y))...)

function inference_variables(x::AbstractArray)
    xdist, logs, logσ = exp.(x[1:(end-2), ..]), x[(end-1):(end-1), ..], x[end:end, ..]
    return InferenceVariables(xdist, logs, logσ)
end
inference_variables(x̄::NamedTuple) = InferenceVariables(x̄.xdist, x̄.logs, x̄.logσ)
inference_variables(::Type{T}, nT2::Int, sz::Int...) where {T} = inference_variables((; xdist = fill(inv(T(nT2)), nT2, sz...), logs = zeros(T, 1, sz...), logσ = zeros(T, 1, sz...)))
inference_variables(nT2::Int, sz::Int...) = inference_variables(Float32, nT2, sz...)

struct ForwardPhysicsModel{TA, Tmask, Tμ, Tlogσ, Tlogs, Tμb̂} <: IsotropicDist
    A::TA # epg bases
    mask::Tmask # signal mask
    μ::Tμ # normalized predicted signal μ = A * xdist
    logs::Tlogs # log scale
    logσ::Tlogσ # log noise level
    μb̂::Tμb̂ # scaled predicted signal exp(logs) * μ
end
@saveable ForwardPhysicsModel

signal_model(𝒟::ForwardPhysicsModel) = RicianDist(𝒟.μb̂, 𝒟.logσ)

function forward_physics_model(x::InferenceVariables, A::AbstractTensor3D, mask::AbstractMatrix, δb::Union{Nothing, AbstractArray} = nothing)
    (; xdist, logs, logσ) = x

    μ = lib.batched_vec(A, xdist) # predicted signal μ = A * xdist
    μb̂ = @. exp(logs) * μ # scaled predicted signal

    𝒟 = ForwardPhysicsModel(A, mask, μ, logs, logσ, μb̂)

    return δb !== nothing ?
           MaskedDist(QuantizedDist(𝒟, δb), mask) :
           MaskedDist(𝒟, mask)
end

# Inference variables from a flat sample `[logx̄; logs; logσ]`, resampling `logx̄` to the basis grid if needed
function forward_physics_model(x::AbstractArray, A::AbstractTensor3D, args...)
    x′ = size(x, 1) - 2 == size(A, 2) ? inference_variables(x) : InferenceVariables(lib.lift_resample(x[1:(end-2), ..], size(A, 2)), x[(end-1):(end-1), ..], x[end:end, ..])
    return forward_physics_model(x′, A, args...)
end

function forward_physics_logs_logσ_model(logx̄, b, A, mask, δb, logs_prior, logσ_prior)
    μ = lib.apply_mask.(mask, resampled_batched_vec(A, logx̄))
    return neglogpdf_qrician_nuis_laplace(mask, b, μ, δb, logs_prior, logσ_prior)
end

forward_physics_logs_logσ_model(logx̄, b, A, mask, δb) = forward_physics_logs_logσ_model(logx̄, b, A, mask, δb, nothing, nothing)

nuis_prior_dist(priors) = priors
nuis_prior_dist(priors::lib.NotTrainable) = ignore_derivatives(priors.layer)

function forward_physics_logs_logσ_model(priors, latents::Tuple)
    (; logs_prior, logσ_prior) = nuis_prior_dist(priors)
    (x, z, b, A, mask, δb, θ) = latents
    return forward_physics_logs_logσ_model(x, b, A, mask, δb, logs_prior, logσ_prior)
end

function nuis_prior_mode_vjp(prior::NormalDist, ηᵢ, H⁻¹Δᵢ)
    ∂μ = lib.Zygote.unbroadcast(prior.μ, @. exp(-2 * prior.logσ) * H⁻¹Δᵢ)
    ∂logσ = lib.Zygote.unbroadcast(prior.logσ, @. 2 * exp(-2 * prior.logσ) * (ηᵢ - prior.μ) * H⁻¹Δᵢ)
    return lib.ChainRules.Tangent{typeof(prior)}(; μ = ∂μ, logσ = ∂logσ)
end
nuis_prior_mode_vjp(::Nothing, η, w) = lib.NoTangent()

function nuis_prior_hess_vjp(prior::NormalDist, ΔH)
    ∂logσ = lib.Zygote.unbroadcast(prior.logσ, @. -2 * exp(-2 * prior.logσ) * ΔH)
    return lib.ChainRules.Tangent{typeof(prior)}(; μ = lib.ZeroTangent(), logσ = ∂logσ)
end
nuis_prior_hess_vjp(::Nothing, ΔH) = lib.NoTangent()

#### Gaussian-likelihood estimators of `logs` and `logσ`

function gaussian_logs_and_logσ_mle_estimators(mask::AbstractArray{T}, b::AbstractArray{T}, μ::AbstractArray{T}, δb::AbstractArray{T}) where {T <: AbstractFloat}
    n = sum(lib.apply_mask, mask; dims = 1)
    Σbμ = sum(@. lib.apply_mask(mask, (b + δb / 2) * μ); dims = 1)
    Σμ² = sum(@. lib.apply_mask(mask, μ^2); dims = 1)
    Σr² = sum(@. lib.apply_mask(mask, ((b + δb / 2) - (Σbμ / Σμ²) * μ)^2); dims = 1)
    logs = @. log(Σbμ / Σμ²)
    logσ = @. log(Σr² / n) / 2
    log_∂²logs = @. -log(Σμ²) / 2 - logs + logσ
    log_∂²logσ = @. -log(2n) / 2
    return logs, logσ, log_∂²logs, log_∂²logσ
end

gaussian_prior_adjust_logstd(logσ::Number, logσ₀::Number) = -lib.logplusexp(-2 * logσ, -2 * logσ₀) / 2 # 1/σnew² = 1/σ² + 1/σ₀²)
gaussian_prior_adjust_mean(μ::Number, logσ::Number, μ₀::Number, logσ₀::Number) = gaussian_prior_adjust_mean(μ, logσ, μ₀, logσ₀, gaussian_prior_adjust_logstd(logσ, logσ₀))
gaussian_prior_adjust_mean(μ::Number, logσ::Number, μ₀::Number, logσ₀::Number, logσnew::Number) = μ * exp(2 * (logσnew - logσ)) + μ₀ * exp(2 * (logσnew - logσ₀)) # μnew = (μ/σ² + μ₀/σ₀²) / (1/σ² + 1/σ₀²) = μ * (σnew²/σ²) + μ₀ * (σnew²/σ₀²)

function neglogpdf_qrician_nuis_laplace(mask, b, μ, δb, logs_prior, logσ_prior, gradhess_order::Val = Val(8), ∂hess_order = Val(0))
    logs, logσ = neglogpdf_qrician_nuis_mode(mask, b, μ, δb, logs_prior, logσ_prior, gradhess_order)
    Σ⁻¹ = neglogpdf_qrician_nuis_hess(mask, b, μ, δb, logs, logσ, logs_prior, logσ_prior, gradhess_order, ∂hess_order)
    Σ½ = lib.batched_invsqrt_eig_2x2(Σ⁻¹) # Σ = P Pᵀ with P = Q D^½ from the eigendecomposition of Σ⁻¹; more stable than a Cholesky factor
    θ = vcat(logs, logσ)
    return Σ½, θ
end

neglogpdf_qrician_nuis_laplace(mask, b, μ, δb, order::Val = Val(8), ∂order = Val(0)) = neglogpdf_qrician_nuis_laplace(mask, b, μ, δb, nothing, nothing, order, ∂order)
neglogpdf_qrician_nuis_mode(mask, b, μ, δb, order::Val; kwargs...) = neglogpdf_qrician_nuis_mode(mask, b, μ, δb, nothing, nothing, order; kwargs...)

function neglogpdf_qrician_nuis_mode(mask, b, μ, δb, logs_prior, logσ_prior, gradhess_order::Val; tol = 5 * eps(eltype(b)), maxiters = 100, warmup_iters = 2, check_every = 4, verbose = false)

    function loss(_logs, _logσ)
        local loss_order = Val(32)
        local _l = sum(@. neglogpdf_qrician_kernel(mask, b, μ, δb, _logs, _logσ, loss_order); dims = 1)
        return add_neglogprior(_l, _logs, _logσ, logs_prior, logσ_prior)
    end

    # Initial guess: MLE under a Gaussian noise model
    logs, logσ, logstd_logs, logstd_logσ = gaussian_logs_and_logσ_mle_estimators(mask, b, μ, δb)
    if logs_prior isa NormalDist && logσ_prior isa NormalDist
        # Adjust MLE -> approximate MAP solution under log-normal prior
        logs .= gaussian_prior_adjust_mean.(logs, logstd_logs, logs_prior.μ, logs_prior.logσ)
        logσ .= gaussian_prior_adjust_mean.(logσ, logstd_logσ, logσ_prior.μ, logσ_prior.logσ)
    end

    batchsize = size(b, 2)
    θ = lib.svec.(tuple.(logs, logσ))
    Δθ = zero.(θ)
    not_converged_flags = lib.bool_similar(b, true, 1, batchsize)
    local g, H, gH_tuples

    verbose && @info "[iter = 0] l = $(lib.mean(loss(logs, logσ)))"

    for i in 1:maxiters
        # Converged columns are masked out to skip their gradient and Hessian evaluations
        θ_logs = @lazy first(θ)
        θ_logσ = @lazy lib.second(θ)
        not_converged_mask = @lazy lib.apply_mask(not_converged_flags, mask)
        if i == 1
            # Refine initial Gaussian estimator with one step of Newton's method
            gH_tuples = ∂²neglogpdf_gaussian_∂²nuis_withgrad_kernel.(not_converged_mask, b, μ, δb, θ_logs, θ_logσ)
            g = sum(first, gH_tuples; dims = 1)
            H = sum(last, gH_tuples; dims = 1)
        elseif i <= warmup_iters
            # Warmup uses the cheap midpoint rule (order 1), refining the Gaussian-noise initial guess toward the Rician mode
            gH_tuples .= ∂²neglogpdf_qrician_∂²nuis_withgrad_kernel.(not_converged_mask, b, μ, δb, θ_logs, θ_logσ, Val(1))
            g .= sum(first, gH_tuples; dims = 1)
            H .= sum(last, gH_tuples; dims = 1)
        else
            gH_tuples .= mixed_precision_nuis_kernel.(∂²neglogpdf_qrician_∂²nuis_withgrad_kernel, not_converged_mask, b, μ, δb, θ_logs, θ_logσ, gradhess_order)
            g .= sum(first, gH_tuples; dims = 1)
            H .= sum(last, gH_tuples; dims = 1)
        end

        g′ = add_neglogprior_gradient(g, θ_logs, θ_logσ, logs_prior, logσ_prior)
        H′ = add_neglogprior_hessian(H, θ_logs, θ_logσ, logs_prior, logσ_prior)

        # Trust region: limit the Newton step to Δθ_max by requiring λ_min ≥ ||g|| / Δθ_max
        ϵ = eps(eltype(b)) # avoid division by zero when the gradient is tiny
        Δθmax = oftype(ϵ, 0.5)
        @. Δθ = lib.ternary_apply_mask(not_converged_flags, zero(g′), lib.spectral_trust_region, H′, g′, Δθmax)

        # Converged when ½gᵀH⁻¹g < tol; since f = -log p, this is a relative tolerance on p
        i > warmup_iters && (@. not_converged_flags = abs(lib.dot(g′, Δθ) / 2) >= tol)
        @. θ -= Δθ
        verbose && @info "[iter = $i] l = $(lib.mean(loss(first.(θ), lib.second.(θ)))), ||Δθ|| = $(maximum(lib.norm.(Δθ))), |Δlogs| = $(maximum(abs, first.(Δθ))), |Δlogσ| = $(maximum(abs, lib.second.(Δθ))), |Δf| = $(maximum(abs, lib.dot.(g′, Δθ) ./ 2)), % converged = $((batchsize - sum(not_converged_flags))) / $(batchsize) = $(round(100 * (batchsize - sum(not_converged_flags)) / batchsize; digits = 2))%"

        i > warmup_iters && (i - warmup_iters) % check_every == 0 && sum(not_converged_flags) == 0 && break
    end

    return first.(θ), lib.second.(θ)
end

function lib.ChainRules.rrule(::typeof(neglogpdf_qrician_nuis_mode), mask, b, μ, δb, logs_prior, logσ_prior, gradhess_order::Val; kwargs...)
    logs, logσ = neglogpdf_qrician_nuis_mode(mask, b, μ, δb, logs_prior, logσ_prior, gradhess_order; kwargs...)
    function neglogpdf_qrician_nuis_mode_pullback(Δ)
        Δ = lib.ChainRules.unthunk(Δ)
        Δ isa lib.ChainRules.AbstractZero && return (lib.NoTangent(), lib.NoTangent(), Δ, Δ, Δ, logs_prior === nothing ? lib.NoTangent() : Δ, logσ_prior === nothing ? lib.NoTangent() : Δ, lib.NoTangent())
        Δlogs, Δlogσ = Δ
        ΣH = ∂²neglogpdf_qrician_∂²nuis(mask, b, μ, δb, logs, logσ, gradhess_order, Val(0)) # `∂order` is only used in the backward pass
        H = add_neglogprior_hessian!(ΣH, logs, logσ, logs_prior, logσ_prior)
        ∂b, ∂μ, ∂δb, ∂logs_prior, ∂logσ_prior = nuis_mode_vjp(H, mask, b, μ, δb, logs, logσ, logs_prior, logσ_prior, gradhess_order, Δlogs, Δlogσ)
        return (lib.NoTangent(), lib.NoTangent(), ∂b, ∂μ, ∂δb, ∂logs_prior, ∂logσ_prior, lib.NoTangent())
    end
    return (logs, logσ), neglogpdf_qrician_nuis_mode_pullback
end

function nuis_mode_vjp(H, mask, b, μ, δb, logs, logσ, logs_prior, logσ_prior, order, Δlogs, Δlogσ)
    ∇b, ∇μ, ∇δb = ∂²neglogpdf_qrician_∂data∂nuis(mask, b, μ, δb, logs, logσ, order)
    Δlogs, Δlogσ = lib.ChainRules.unthunk(Δlogs), lib.ChainRules.unthunk(Δlogσ)
    Δlogs = Δlogs isa lib.ChainRules.AbstractZero ? zero(eltype(b)) : Δlogs
    Δlogσ = Δlogσ isa lib.ChainRules.AbstractZero ? zero(eltype(b)) : Δlogσ
    Δ = @lazy lib.svec(tuple(Δlogs, Δlogσ))
    H⁻¹Δ = @. H \ Δ # symmetric Hessian ⇒ H⁻ᵀΔ = H⁻¹Δ
    ∂b = @. -lib.dot(H⁻¹Δ, ∇b)
    ∂μ = @. -lib.dot(H⁻¹Δ, ∇μ)
    ∂δb = @. -lib.dot(H⁻¹Δ, ∇δb)
    ∂logs_prior = nuis_prior_mode_vjp(logs_prior, logs, @lazy first.(H⁻¹Δ))
    ∂logσ_prior = nuis_prior_mode_vjp(logσ_prior, logσ, @lazy last.(H⁻¹Δ))
    return ∂b, ∂μ, ∂δb, ∂logs_prior, ∂logσ_prior
end

function neglogpdf_qrician_nuis_hess(mask, b, μ, δb, logs, logσ, logs_prior, logσ_prior, order::Val, ∂order::Val)
    ΣH = ∂²neglogpdf_qrician_∂²nuis(mask, b, μ, δb, logs, logσ, order, Val(0)) # `∂order` is only used in the backward pass
    H = add_neglogprior_hessian!(ΣH, logs, logσ, logs_prior, logσ_prior)
    return lib.as_array(H)
end
neglogpdf_qrician_nuis_hess(mask, b, μ, δb, logs, logσ, order::Val, ∂order::Val) = neglogpdf_qrician_nuis_hess(mask, b, μ, δb, logs, logσ, nothing, nothing, order, ∂order)

@inline vjp(J, Δ::SMatrix{2, 2, T}) where {T} = J' * vec(Δ)
@inline vjp(J, Δ::NTuple{4, T}) where {T} = J' * SVector{4, T}(Δ)

function lib.ChainRules.rrule(::typeof(neglogpdf_qrician_nuis_hess), mask, b, μ, δb, logs, logσ, logs_prior, logσ_prior, order::Val, ∂order::Val)
    function extract_grads(∇, ΔH)
        # Gaussian prior curvature is constant in η but depends on the prior log width
        ∂b = Base.Fix2(getindex, 1).(∇)
        ∂μ = Base.Fix2(getindex, 2).(∇)
        ∂δb = sum(Base.Fix2(getindex, 3), ∇; dims = 1)
        ∂logs = sum(Base.Fix2(getindex, 4), ∇; dims = 1)
        ∂logσ = sum(Base.Fix2(getindex, 5), ∇; dims = 1)
        ∂logs_prior = nuis_prior_hess_vjp(logs_prior, @lazy first.(ΔH))
        ∂logσ_prior = nuis_prior_hess_vjp(logσ_prior, @lazy last.(ΔH))
        return (lib.NoTangent(), lib.NoTangent(), ∂b, ∂μ, ∂δb, ∂logs, ∂logσ, ∂logs_prior, ∂logσ_prior, lib.NoTangent(), lib.NoTangent())
    end

    if ∂order == Val(0)
        # Compute fused Hessian and Jacobian of flattened Hessian
        ΣH_∂H = mixed_precision_nuis_kernel.(∂²neglogpdf_qrician_∂²nuis_withjac_kernel, mask, b, μ, δb, logs, logσ, order)
        ΣH = sum(first, ΣH_∂H; dims = 1)
        function ∂²neglogpdf_qrician_∂²nuis_fused_pullback(ΔH_array)
            ΔH_array = lib.ChainRules.unthunk(ΔH_array)
            # Compute VJP using cached Jacobian from the forward pass
            ΔH = reinterpret(SMatrix{2, 2, eltype(ΔH_array), 4}, reshape(ΔH_array, 4, :))
            ∇ = vjp.(lib.second.(ΣH_∂H), ΔH)
            return extract_grads(∇, ΔH)
        end
        H = add_neglogprior_hessian!(ΣH, logs, logσ, logs_prior, logσ_prior)
        return lib.as_array(H), ∂²neglogpdf_qrician_∂²nuis_fused_pullback
    else
        # Compute Hessian on the forward pass, VJP on the reverse pass
        function ∂²neglogpdf_qrician_∂²nuis_split_pullback(ΔH_array)
            ΔH_array = lib.ChainRules.unthunk(ΔH_array)
            # Compute VJP on the reverse pass
            ΔH = reinterpret(SMatrix{2, 2, eltype(ΔH_array), 4}, reshape(ΔH_array, 4, :))
            ∇ = ∂²neglogpdf_qrician_∂²nuis_vjp_kernel.(ΔH, mask, b, μ, δb, logs, logσ, ∂order)
            return extract_grads(∇, ΔH)
        end
        ΣH = ∂²neglogpdf_qrician_∂²nuis(mask, b, μ, δb, logs, logσ, order, Val(0)) # `∂order` is only used in the backward pass
        H = add_neglogprior_hessian!(ΣH, logs, logσ, logs_prior, logσ_prior)
        return lib.as_array(H), ∂²neglogpdf_qrician_∂²nuis_split_pullback
    end
end

# Helper functions for prior contributions to the joint likelihood, gradient, and Hessian
add_neglogprior(l, logs, logσ, logs_prior::Nothing, logσ_prior::Nothing) = l
add_neglogprior(l, logs, logσ, logs_prior::NormalDist, logσ_prior::NormalDist) = l .+ neglogpdf(logs_prior, logs) .+ neglogpdf(logσ_prior, logσ)
add_neglogprior_gradient(g, logs, logσ, logs_prior::Nothing, logσ_prior::Nothing) = g
add_neglogprior_gradient(g, logs, logσ, logs_prior::NormalDist, logσ_prior::NormalDist) = @lazy g + lib.svec(tuple(exp(-2 * logs_prior.logσ) * (logs - logs_prior.μ), exp(-2 * logσ_prior.logσ) * (logσ - logσ_prior.μ)))

add_neglogprior_hessian!(H, logs, logσ, logs_prior::Nothing, logσ_prior::Nothing) = H
add_neglogprior_hessian(H, logs, logσ, logs_prior::Nothing, logσ_prior::Nothing) = H
add_neglogprior_hessian!(H, logs, logσ, logs_prior::NormalDist, logσ_prior::NormalDist) = H .= add_neglogprior_hessian(H, logs, logσ, logs_prior, logσ_prior)
add_neglogprior_hessian(H, logs, logσ, logs_prior::NormalDist, logσ_prior::NormalDist) = @lazy H + lib.smat_diag(tuple(exp(-2 * logs_prior.logσ), exp(-2 * logσ_prior.logσ)))

#### Kernels for the gradient and Hessian of `neglogpdf_gaussian`

@inline function ∂²neglogpdf_gaussian_∂²nuis_withgrad_kernel(mask, b₊, μ, logs, logσ)
    T = promote_type(typeof(mask), typeof(b₊), typeof(μ), typeof(logs), typeof(logσ))
    if !lib.is_mask_active(mask)
        return zero(SVector{2, T}), zero(SMatrix{2, 2, T, 4})
    else
        σ⁻¹s, σ⁻¹ = exp(logs - logσ), exp(-logσ)
        b̂₊, μ̂ = σ⁻¹ * b₊, σ⁻¹s * μ
        res = b̂₊ - μ̂
        grad = lib.svec(tuple(-μ̂ * res, (1 - res) * (1 + res))) # (∂/∂logs, ∂/∂logσ)
        hess = lib.smat_herm(tuple(μ̂ * (μ̂ - res), 2 * μ̂ * res, 2 * res^2)) # (∂²/∂logs², ∂²/∂logσ∂logs, ∂²/∂logσ²)
        return grad, hess
    end
end
@inline ∂²neglogpdf_gaussian_∂²nuis_withgrad_kernel(mask, b, μ, δb, logs, logσ) = ∂²neglogpdf_gaussian_∂²nuis_withgrad_kernel(mask, b + δb / 2, μ, logs, logσ)

#### Kernels for the gradient, Hessian, and Jacobian of the Hessian for `neglogpdf_qrician`

# Wide bins (δb/σ ≳ 2) cause cancellation in the logσ derivatives at x ≈ ν; evaluate them in Float64
@inline mixed_precision_nuis_kernel(kernel::F, mask, b::Float64, μ, δb, logs, logσ, order::Val) where {F} = kernel(mask, b, μ, δb, logs, logσ, order)
@inline mixed_precision_nuis_kernel(kernel::F, mask, b::Float32, μ, δb, logs, logσ, order::Val) where {F} = δb < 2 * exp(logσ) ? kernel(mask, b, μ, δb, logs, logσ, order) : to32(kernel(Float64(mask), Float64(b), Float64(μ), Float64(δb), Float64(logs), Float64(logσ), order))
@inline to32(x::lib.StaticArrays.SArray) = lib.StaticArrays.similar_type(x, Float32)(x)
@inline to32(x::Tuple) = map(to32, x)
@inline to32(x::AbstractFloat) = Float32(x)

@inline function neglogpdf_qrician_kernel(mask, b, μ, δb, logs, logσ, order::Val)
    T = promote_type(typeof(mask), typeof(b), typeof(μ), typeof(δb), typeof(logs), typeof(logσ))
    if !lib.is_mask_active(mask)
        return zero(T)
    else
        σ⁻¹s, σ⁻¹ = exp(logs - logσ), exp(-logσ)
        return T(lib.neglogpdf_qrician(σ⁻¹ * b, σ⁻¹s * μ, σ⁻¹ * δb, order))
    end
end

# Second-derivative kernels

@inline function ∂²neglogpdf_qrician_∂²nuis_withgrad_kernel(mask, b, μ, δb, logs, logσ, order::Val)
    T = promote_type(typeof(mask), typeof(b), typeof(μ), typeof(δb), typeof(logs), typeof(logσ))
    if !lib.is_mask_active(mask)
        return zero(SVector{2, T}), zero(SMatrix{2, 2, T, 4})
    else
        σ⁻¹s, σ⁻¹ = exp(logs - logσ), exp(-logσ)
        x, ν, δ = σ⁻¹ * b, σ⁻¹s * μ, σ⁻¹ * δb
        (∂x, ∂ν, ∂δ), (∂xx, ∂xν, ∂xδ, ∂νν, ∂νδ, ∂δδ) = lib.FastRicianLikelihoods.∇²neglogpdf_qrician_with_gradient(x, ν, δ, order)

        ∂logs = ν * ∂ν
        ∂logσ = -(x * ∂x + ν * ∂ν + δ * ∂δ)

        ∂logs∂logs = ν * (∂ν + ν * ∂νν)
        ∂logs∂logσ = -ν * (∂ν + (ν * ∂νν + x * ∂xν + δ * ∂νδ))
        ∂logσ∂logσ = (x * ∂x + ν * ∂ν + δ * ∂δ) + ((x^2 * ∂xx + ν^2 * ∂νν + δ^2 * ∂δδ) + 2 * (x * ν * ∂xν + x * δ * ∂xδ + ν * δ * ∂νδ))

        return SVector{2, T}((∂logs, ∂logσ)), SMatrix{2, 2, T, 4}((∂logs∂logs, ∂logs∂logσ, ∂logs∂logσ, ∂logσ∂logσ))
    end
end
@inline ∂²neglogpdf_qrician_∂²nuis_kernel(mask, b, μ, δb, logs, logσ, order::Val) = ∂²neglogpdf_qrician_∂²nuis_withgrad_kernel(mask, b, μ, δb, logs, logσ, order)[2]

@inline function ∂²neglogpdf_qrician_∂data∂nuis_kernel(mask, b, μ, δb, logs, logσ, order::Val)
    T = promote_type(typeof(mask), typeof(b), typeof(μ), typeof(δb), typeof(logs), typeof(logσ))
    if !lib.is_mask_active(mask)
        return zero(SMatrix{3, 2, T, 6})
    else
        σ⁻¹s, σ⁻¹ = exp(logs - logσ), exp(-logσ)
        x, ν, δ = σ⁻¹ * b, σ⁻¹s * μ, σ⁻¹ * δb
        (∂x, ∂ν, ∂δ), (∂xx, ∂xν, ∂xδ, ∂νν, ∂νδ, ∂δδ) = lib.FastRicianLikelihoods.∇²neglogpdf_qrician_with_gradient(x, ν, δ, order)

        ∂b∂logs = σ⁻¹ * ν * ∂xν
        ∂b∂logσ = -σ⁻¹ * (∂x + x * ∂xx + ν * ∂xν + δ * ∂xδ)

        ∂μ∂logs = σ⁻¹s * (∂ν + ν * ∂νν)
        ∂μ∂logσ = -σ⁻¹s * (∂ν + x * ∂xν + ν * ∂νν + δ * ∂νδ)

        ∂δb∂logs = σ⁻¹ * ν * ∂νδ
        ∂δb∂logσ = -σ⁻¹ * (∂δ + x * ∂xδ + ν * ∂νδ + δ * ∂δδ)

        return SMatrix{3, 2, T, 6}((∂b∂logs, ∂μ∂logs, ∂δb∂logs, ∂b∂logσ, ∂μ∂logσ, ∂δb∂logσ))
    end
end

# Third-derivative kernels

@inline function ∂²neglogpdf_qrician_∂²nuis_withjac_kernel(mask, b, μ, δb, logs, logσ, order::Val)
    T = promote_type(typeof(mask), typeof(b), typeof(μ), typeof(δb), typeof(logs), typeof(logσ))
    if !lib.is_mask_active(mask)
        return zero(SMatrix{2, 2, T, 4}), zero(SMatrix{4, 5, T, 20})
    else
        # Rician parameters θ = (x, ν, δ) = (b / σ, s * μ / σ, δb / σ)
        σ⁻¹, σ⁻¹s = exp(-logσ), exp(logs - logσ)
        x, ν, δ = σ⁻¹ * b, σ⁻¹s * μ, σ⁻¹ * δb
        x², ν², δ², xν, xδ, νδ = x^2, ν^2, δ^2, x * ν, x * δ, ν * δ

        # Rician derivatives ∇Ω, ∇²Ω, and ∇(vech(∇²Ω))
        _, g, h, J = lib.FastRicianLikelihoods.∇³neglogpdf_qrician_jacobian_with_primal_gradient_and_hessian(x, ν, δ, order)
        ∂x, ∂ν, ∂δ = g
        hxx, hxν, hxδ, hνν, hνδ, hδδ = h
        Txxx, ____, ____, ____, ____, ____, # ∂h/∂x
        Txxν, Txνν, ____, Tννν, ____, ____, # ∂h/∂ν
        Txxδ, Txνδ, Txδδ, Tννδ, Tνδδ, Tδδδ = Tuple(J) # ∂h/∂δ

        # Nuisance Hessian H = ∂²Ω/∂η² = ∂²Ω/∂[logs, logσ]²
        Hss = ν * muladd(ν, hνν, ∂ν)
        Hsσ = -ν * muladd(x, hxν, muladd(δ, hνδ, muladd(ν, hνν, ∂ν)))
        Hσσ = muladd(x, muladd(x, hxx, ∂x), muladd(δ, muladd(δ, hδδ, ∂δ), muladd(ν, muladd(ν, hνν, ∂ν), 2 * muladd(xν, hxν, muladd(xδ, hxδ, νδ * hνδ)))))
        H = SMatrix{2, 2, T, 4}(Hss, Hsσ, Hsσ, Hσσ)

        # ∂vech(H)/∂θ with vech(H) = (∂²Ω/∂logs², ∂²Ω/∂logs∂logσ, ∂²Ω/∂logσ²)
        ∂Hss∂x = ν * muladd(ν, Txνν, hxν)
        ∂Hss∂ν = muladd(ν, muladd(ν, Tννν, 3 * hνν), ∂ν)
        ∂Hss∂δ = ν * muladd(ν, Tννδ, hνδ)
        ∂Hsσ∂x = -ν * muladd(x, Txxν, muladd(δ, Txνδ, muladd(ν, Txνν, 2 * hxν)))
        ∂Hsσ∂ν = -muladd(x, muladd(ν, Txνν, hxν), muladd(δ, muladd(ν, Tννδ, hνδ), muladd(ν, muladd(ν, Tννν, 3 * hνν), ∂ν)))
        ∂Hsσ∂δ = -ν * muladd(x, Txνδ, muladd(δ, Tνδδ, muladd(ν, Tννδ, 2 * hνδ)))
        ∂Hσσ∂x = muladd(x, muladd(x, Txxx, 3 * hxx), muladd(δ, muladd(δ, Txδδ, 3 * hxδ), muladd(ν, muladd(ν, Txνν, 3 * hxν), muladd(2 * xν, Txxν, muladd(2 * xδ, Txxδ, muladd(2 * νδ, Txνδ, ∂x))))))
        ∂Hσσ∂ν = muladd(x, muladd(x, Txxν, 3 * hxν), muladd(δ, muladd(δ, Tνδδ, 3 * hνδ), muladd(ν, muladd(ν, Tννν, 3 * hνν), muladd(2 * xν, Txνν, muladd(2 * xδ, Txνδ, muladd(2 * νδ, Tννδ, ∂ν))))))
        ∂Hσσ∂δ = muladd(x, muladd(x, Txxδ, 3 * hxδ), muladd(δ, muladd(δ, Tδδδ, 3 * hδδ), muladd(ν, muladd(ν, Tννδ, 3 * hνδ), muladd(2 * xν, Txνδ, muladd(2 * xδ, Txδδ, muladd(2 * νδ, Tνδδ, ∂δ))))))

        # Jacobian ∂H = ∂H/∂[ξ,η] = ∂H/∂[b, μ, δb, logs, logσ]
        ∂Hss∂b, ∂Hsσ∂b, ∂Hσσ∂b = σ⁻¹ * ∂Hss∂x, σ⁻¹ * ∂Hsσ∂x, σ⁻¹ * ∂Hσσ∂x # ∂vech(H)/∂b
        ∂Hss∂μ, ∂Hsσ∂μ, ∂Hσσ∂μ = σ⁻¹s * ∂Hss∂ν, σ⁻¹s * ∂Hsσ∂ν, σ⁻¹s * ∂Hσσ∂ν # ∂vech(H)/∂μ
        ∂Hss∂δb, ∂Hsσ∂δb, ∂Hσσ∂δb = σ⁻¹ * ∂Hss∂δ, σ⁻¹ * ∂Hsσ∂δ, σ⁻¹ * ∂Hσσ∂δ # ∂vech(H)/∂δb
        ∂Hss∂s, ∂Hsσ∂s, ∂Hσσ∂s = ν * ∂Hss∂ν, ν * ∂Hsσ∂ν, ν * ∂Hσσ∂ν # ∂vech(H)/∂s
        ∂Hss∂σ = -muladd(δ, ∂Hss∂δ, muladd(ν, ∂Hss∂ν, x * ∂Hss∂x)) # ∂vech(H)/∂σ
        ∂Hsσ∂σ = -muladd(δ, ∂Hsσ∂δ, muladd(ν, ∂Hsσ∂ν, x * ∂Hsσ∂x))
        ∂Hσσ∂σ = -muladd(δ, ∂Hσσ∂δ, muladd(ν, ∂Hσσ∂ν, x * ∂Hσσ∂x))
        ∂H = SMatrix{4, 5, T, 20}(
            ∂Hss∂b, ∂Hsσ∂b, ∂Hsσ∂b, ∂Hσσ∂b, # ∂H/∂b
            ∂Hss∂μ, ∂Hsσ∂μ, ∂Hsσ∂μ, ∂Hσσ∂μ, # ∂H/∂μ
            ∂Hss∂δb, ∂Hsσ∂δb, ∂Hsσ∂δb, ∂Hσσ∂δb, # ∂H/∂δb
            ∂Hss∂s, ∂Hsσ∂s, ∂Hsσ∂s, ∂Hσσ∂s, # ∂H/∂s
            ∂Hss∂σ, ∂Hsσ∂σ, ∂Hsσ∂σ, ∂Hσσ∂σ, # ∂H/∂σ
        )

        return H, ∂H
    end
end

@inline function ∂²neglogpdf_qrician_∂²nuis_vjp_kernel(ΔH, mask, b, μ, δb, logs, logσ, order::Val)
    T = promote_type(typeof(mask), typeof(b), typeof(μ), typeof(δb), typeof(logs), typeof(logσ))
    if !lib.is_mask_active(mask)
        return zero(SVector{5, T})
    else
        σ⁻¹s, σ⁻¹ = exp(logs - logσ), exp(-logσ)
        x, ν, δ = σ⁻¹ * b, σ⁻¹s * μ, σ⁻¹ * δb

        ΔH₁₁, ΔH₂₁, ΔH₁₂, ΔH₂₂ = ΔH
        ΔHc₁ = ΔH₁₁ + ΔH₂₂ - (ΔH₁₂ + ΔH₂₁)
        ΔHc₂ = 2 * ΔH₂₂ - (ΔH₁₂ + ΔH₂₁)
        ΔHc₃ = ΔHc₁ + ΔHc₂ # ΔH₁₁ - 4 * ΔH₁₂ + 3 * ΔH₂₂
        ΔHc₄ = ΔH₂₂ + ΔHc₂ # 3 * ΔH₂₂ - 2 * ΔH₁₂

        Δh = SVector{6, T}(ΔH₂₂ * x^2, ΔHc₂ * x * ν, 2 * ΔH₂₂ * x * δ, ΔHc₁ * ν^2, ΔHc₂ * δ * ν, ΔH₂₂ * δ^2)
        _, g, h, JᵀΔh = lib.FastRicianLikelihoods._∇³neglogpdf_qrician_vjp_with_primal_gradient_and_hessian(Δh, x, ν, δ, order)

        (∂x, ∂ν, ∂δ), (∂xx, ∂xν, ∂xδ, ∂νν, ∂νδ, ∂δδ), (Tx, Tν, Tδ) = g, h, JᵀΔh
        jvpx = muladd(ΔH₂₂, ∂x, muladd(3 * ΔH₂₂, muladd(x, ∂xx, δ * ∂xδ), muladd(ΔHc₃, ν * ∂xν, Tx)))
        jvpν = muladd(ΔHc₁, ∂ν, muladd(3 * ΔHc₁, ν * ∂νν, muladd(ΔHc₄, muladd(x, ∂xν, δ * ∂νδ), Tν)))
        jvpδ = muladd(ΔH₂₂, ∂δ, muladd(3 * ΔH₂₂, muladd(x, ∂xδ, δ * ∂δδ), muladd(ΔHc₃, ν * ∂νδ, Tδ)))

        ∂b = σ⁻¹ * jvpx
        ∂μ = σ⁻¹s * jvpν
        ∂δb = σ⁻¹ * jvpδ
        ∂logs = ν * jvpν
        ∂logσ = -muladd(δ, jvpδ, muladd(ν, jvpν, x * jvpx))

        return SVector{5, T}(∂b, ∂μ, ∂δb, ∂logs, ∂logσ)
    end
end
∂²neglogpdf_qrician_∂²nuis(mask, b, μ, δb, logs, logσ, order::Val, ∂order::Val) = sum(mixed_precision_nuis_kernel.(∂²neglogpdf_qrician_∂²nuis_kernel, mask, b, μ, δb, logs, logσ, order); dims = 1) # `∂order` is only used in the backward pass

function ∂²neglogpdf_qrician_∂data∂nuis(mask, b, μ, δb, logs, logσ, order::Val)
    ∇ = ∂²neglogpdf_qrician_∂data∂nuis_kernel.(mask, b, μ, δb, logs, logσ, order)
    ∂b = (x -> x[1, :]).(∇)
    ∂μ = (x -> x[2, :]).(∇)
    ∂δb = sum((x -> x[3, :]).(∇); dims = 1)
    return ∂b, ∂μ, ∂δb
end

recondition(𝒟::ForwardPhysicsModel, latents::Tuple) = forward_physics_model(only(latents), 𝒟.A, 𝒟.mask)

# Likelihood of data under the forward model
isotropic_neglogpdf(𝒟::ForwardPhysicsModel, b::AbstractArray) = isotropic_neglogpdf(signal_model(𝒟), b)
isotropic_neglogpdf_quantized(𝒟::ForwardPhysicsModel, b::AbstractArray, Δ::AbstractArray) = isotropic_neglogpdf_quantized(signal_model(𝒟), b, Δ)

# Sample homoscedastic Rician noise
sample_similar(𝒟::ForwardPhysicsModel, x::AbstractArray, ::Colon, sz::Tuple{Vararg{Int}}) = sample_similar(𝒟, x, (size(𝒟.μb̂, 1), sz...))
sample_similar(𝒟::ForwardPhysicsModel, x::AbstractArray, sz::Tuple{Vararg{Int}}) = sample_similar(signal_model(𝒟), x, sz)

# Multiply the transposed basis by the signal.
transposed_batched_vec(A, b) = lib.batched_vec(ignore_derivatives(lib.batched_transpose(A)), b)

# Basis multiplication with on-the-fly resampling when `A` and `logx̄` differ in T2 resolution
resampled_batched_vec(A, logx̄) = lib.batched_vec(A, lib.lift_resample(logx̄, size(A, 2)))

#### AbstractModel

"""
    AbstractModel{opts, model}

Interface for ephemeral model structs which are just light wrappers around a named tuple `opts` and an inner model `model`.
Default constructor: `M(model; kwargs...) = M((; kwargs...), model)`.
"""
abstract type AbstractModel{opts, model} end

lib.Optimisers.trainable(m::M) where {M <: AbstractModel} = (; m.model)

(::Type{M})(model; kwargs...) where {M <: AbstractModel} = M((; kwargs...), model)

Base.show(io::IO, m::M) where {M <: AbstractModel} = print(io, "$(nameof(M))$(m.opts)")

lib.rebuild_types(::Type{M}, fields::NamedTuple) where {M <: AbstractModel} = M(fields.opts, fields.model)

function build_models(cfg::Config, dl_train)
    enc = Encoder(cfg)
    prior = Prior(cfg)
    genatr = Generator(cfg, prior)

    # Move all models to the GPU together to preserve weight sharing (e.g. genatr/prior)
    models = (; genatr, enc, prior) |> lib.safegpu
    models = cleanup_models(cfg, models)

    models = share_window_filters(cfg, models)

    models = dynamic_init_batchnorm!(cfg, models, dl_train; nsamples = cfg.train.initialization.batchnorm_samples::Int)
    models = dynamic_init_tokenizers!(cfg, models, dl_train; nsamples = cfg.train.initialization.tokenizer_samples::Int)

    models = dynamic_init_flows_variance!(cfg, models)

    return models
end

function build_optimizers(cfg::Config, models)
    β = cfg.opt.psgd.beta1::Float64
    rule = lib.KronPSGD(;
        eta = cfg.opt.lr.start::Float64,
        beta = β, beta_sf = β, lambda = 0.0,
        rmsclip = cfg.opt.psgd.rmsclip::Float64, eta_Q = cfg.opt.psgd.eta_Q::Float64,
        kron_Q = true, diag_Q = false,
        whiten_grad = true,
        schedule_free = true,
    )
    wdecay = cfg.opt.wdecay.start::Float64
    weight_rule = wdecay > 0 ? lib.Optimisers.adjust(rule; lambda = wdecay) : rule
    optimizers = lib.OptimisersSetup.setup(rule, models) do layer
        return layer isa lib.Flux.Dense ? (; weight = weight_rule) : nothing
    end
    return optimizers
end

function cleanup_models(cfg, models)

    begin
        @info "Checking weight initialization"
        gain = 1.0 # output head calibration follows this orthogonality check
        for m in models
            lib.fmapstructure_with_path(m; exclude = (k, l) -> l isa lib.Flux.Dense) do k, l
                # Weights must be (semi-)orthogonal, scaled by gain * sqrt(fan_out / fan_in)
                fan_out, fan_in = size(l.weight)
                Wnorm = gain * sqrt(fan_out / fan_in) # spectral norm at init
                WtW = fan_out <= fan_in ? l.weight * l.weight' : l.weight' * l.weight # inner product
                WtW_target = lib.arr_similar(WtW, lib.eye(Wnorm^2, min(fan_out, fan_in))) # should be Diag(Wnorm)^2
                WtW_meanerr = lib.mean(abs, WtW - WtW_target) # mean deviation from target
                iszero(WtW) || @assert WtW_meanerr <= 1e-6 "Matrix not correctly initialized: k = $k, fan_out = $fan_out, fan_in = $fan_in, target = $Wnorm, actual = $(sqrt(lib.mean(abs, lib.diag(WtW))))" # WtW = 0 is also allowed
                return nothing
            end
        end
    end

    return models
end

"""
    share_window_filters(cfg, models; verbose = false)

Share a single window filter between the genatr and prior to avoid impedence mismatch.
That is, the last layer of the genatr is shared with the first (and only) layer of the prior.
"""
function share_window_filters(cfg::Config, models; verbose::Bool = false)
    filter_count = Ref(0)
    filters = Any[]
    models′ = lib.fmap_with_path(models; exclude = (_, l) -> is_window_filter(l)) do k, l
        if (filter_count[] += 1) == 1
            verbose && @info "Found window filter: $k"
            push!(filters, l.model)
        else
            verbose && @info "Replacing window filter: $k"
            @reset l.model = only(filters)
        end
        return l
    end
    @assert filter_count[] == 4 "Expected 4 window filters; found $(filter_count[])" # two for genatr/prior fwd flows, two for genatr/prior inv flows

    return models′
end

is_window_filter(l) = l isa lib.AffineFlow && l.model isa lib.LowPassFlowModel

function dynamic_init_batchnorm!(cfg, models, dl_train; nsamples)
    @info "Initializing BatchNorm"
    (; θ_embedder) = models.enc.model
    θ_embedder isa lib.Flux.Chain || (@warn "θ_embedder is not a Chain"; return models)
    bn_index = findfirst(l -> l isa lib.NotTrainable{<:lib.Flux.BatchNorm}, θ_embedder.layers)
    bn_index === nothing && (@warn "θ_embedder has no frozen BatchNorm"; return models)
    bn = θ_embedder.layers[bn_index].layer

    nbatches = nsamples ÷ cfg.train.batchsize::Int
    lib.Flux.trainmode!(bn)
    with_initialization_data(dl_train) do dl # initialization must not advance the caller's stream
        for (i, batch) in enumerate(device_batches(dl))
            bn.momentum = Float32(1 / i)
            bn(batch.θ)
            i >= nbatches && break
        end
    end

    @reset models.enc.model.θ_embedder.layers[bn_index] = lib.BatchNorm2Scale(bn; freeze = true)

    return models
end

function flow_init_sample(cfg::Config; batchsize = 32_768, passlatents = true)
    device = lib.USE_GPU[] ? lib.CUDA : lib.Random
    u = device.randn(Float32, cfg.data.nT2::Int - 1, batchsize)
    z = passlatents ? (lib.unitrms(device.randn(Float32, cfg.arch.nlatent::Int, batchsize); dims = 1),) : ()
    return (; u, z)
end

# Scaling the terminal weights and biases scales the affine perturbation to first order.
flow_output_parameters(::Any) = AbstractArray[]
flow_output_parameters(ℱ::lib.AffineFlow) = flow_output_parameters(ℱ.model)
flow_output_parameters(ℱ::lib.InverseFlow) = flow_output_parameters(ℱ.flow)
flow_output_parameters(l::lib.Flux.Dense) = l.bias isa AbstractArray ? AbstractArray[l.weight, l.bias] : AbstractArray[l.weight]
flow_output_parameters(l::lib.GatedLinearUnit) = flow_output_parameters(l.output)
flow_output_parameters(l::lib.Flux.Parallel) = mapreduce(flow_output_parameters, vcat, l.layers; init = AbstractArray[])
flow_output_parameters(::lib.Flux.SkipConnection) = error("identity skip connection bypasses the conditioner output head")

function flow_output_parameters(c::lib.Flux.Chain)
    for l in Iterators.reverse(c.layers)
        ps = flow_output_parameters(l)
        isempty(ps) || return ps
    end
    return AbstractArray[]
end

# The post-head SqueezeRange bias contributes to the same affine perturbation.
flow_squeeze_biases(::Any) = AbstractArray[]
flow_squeeze_biases(ℱ::lib.AffineFlow) = flow_squeeze_biases(ℱ.model)
flow_squeeze_biases(ℱ::lib.InverseFlow) = flow_squeeze_biases(ℱ.flow)
flow_squeeze_biases(l::lib.SqueezeRange) = l.bias isa AbstractArray ? AbstractArray[l.bias] : AbstractArray[]
flow_squeeze_biases(c::lib.Flux.Chain) = mapreduce(flow_squeeze_biases, vcat, c.layers; init = AbstractArray[])

isconditionalaffine(::Any) = false
isconditionalaffine(::lib.AffineFlow) = true

function flow_identity_collect(f, φ)
    ps = AbstractArray[]
    lib.fmapstructure(ℱ -> (append!(ps, f(ℱ)); nothing), φ; exclude = isconditionalaffine)
    return unique(objectid, ps) # `φ` and `φ⁻¹` share parameters
end

# All affine output parameters, including post-head biases.
flow_head_parameters(φ) = unique(objectid, vcat(flow_identity_collect(flow_output_parameters, φ), flow_identity_collect(flow_squeeze_biases, φ)))

# KL[𝒩(0, (1+εσ)²I) || 𝒩(0, I)]: the KL budget of an isotropic standard deviation perturbation `εσ`
flow_kl_budget(εσ, d::Int) = d * (εσ * (2 + εσ) / 2 - log1p(εσ)) # εσ * (2 + εσ) = (1 + εσ)^2 - 1

# Average conditional KL: E[(|φ(u;z)|² - |u|²)/2 - log|det ∂φ|]
function flow_init_kl(φ, (; u, z))
    d = size(u, 1)
    𝓂 = lib.flow_transform(φ, u, lib.zero_logjac(u), z)
    x, logjac = Base.materialize(𝓂.value), Base.materialize(𝓂.logjac)
    K = lib.mean((sum(abs2, x; dims = 1) .- sum(abs2, u; dims = 1)) ./ 2 .- logjac) |> Float64
    μ = lib.mean(x; dims = 2)
    s² = lib.mean(abs2, x .- μ) |> Float64
    return (; K, K_per_dim = K / d, pooled_std = √s², Displacement = lib.mean(abs2, x .- u) |> Float64)
end
flow_init_kl(cfg::Config, φ; kwargs...) = flow_init_kl(φ, flow_init_sample(cfg; kwargs...))

# Calibrate near identity using K(g) ≈ K(0) + A g² and common random numbers.
function calibrate_flow_heads!(cfg, φ; εσ, εcalib = 0.05, passlatents = true, maxiter = 8)
    @assert εσ > 0 && εcalib > 0
    d = cfg.data.nT2::Int - 1
    εfit = max(εσ, εcalib) # the KL estimator's relative error grows as `1/(εσ√N)`
    κ_target = flow_kl_budget(εσ, d)
    κ = flow_kl_budget(εfit, d)
    ps = flow_head_parameters(φ)
    @assert !isempty(ps) "No conditional affine output heads found"
    p₀ = map(copy, ps)
    setgain!(g) = foreach((p, p′) -> p .= eltype(p)(g) .* p′, ps, p₀)
    uz = flow_init_sample(cfg; passlatents, batchsize = cfg.train.initialization.flow_samples::Int)

    setgain!(0)
    K_identity = flow_init_kl(φ, uz).K
    @assert 0 <= K_identity < κ_target "Backbone KL at zero head gain exceeds the target: $(K_identity) ≥ $(κ_target)"
    g, K₀ = 1.0, 0.0
    for i in 1:maxiter
        setgain!(g)
        K = flow_init_kl(φ, uz).K
        i == 1 && (K₀ = K)
        @assert isfinite(K) && K > K_identity "Flow KL did not increase with the head gain: K = $(K), K_identity = $(K_identity)"
        r = (κ - K_identity) / (K - K_identity)
        g *= √r
        abs(r - 1) < 0.01 && break
    end
    g *= √((κ_target - K_identity) / (κ - K_identity))
    setgain!(g)

    stats = flow_init_kl(φ, uz)
    @debug "Flow head calibration" εσ εfit κ_target κ_fit = κ K₀ K_identity gain = g stats

    return g
end

function calibrate_embedding!(E, counts)
    @assert all(>=(0), counts) && isfinite(sum(counts)) && sum(counts) > 0 "Echo-count calibration requires nonnegative frequencies with positive finite total"
    p = Float32.(counts ./ sum(counts))
    p = lib.arr_similar(E.weight, p)
    E.weight .-= E.weight * p
    if count(>(0), counts) > 1 # A single observed category has no variation after centering.
        q = sum(lib.mean(abs2, E.weight; dims = 1) .* reshape(p, 1, :))
        @assert isfinite(q) && q > 0 "Centered embedding second moment must be finite and positive"
        E.weight ./= √q
    end
    return E
end

# Tokenizer inputs have E[x²] = 1; the nTE embedding has zero mean and unit mean square
# Second moments rather than variances, since zero is meaningful for `b` and `A`.
function dynamic_init_tokenizers!(cfg, models, dl_train; nsamples = 2^16)
    (; b_embedder, Aᵀb_embedder, A_embedder, nTE_embedder) = models.enc.model
    embedders = (b_embedder, Aᵀb_embedder, A_embedder)
    scale_indices = map(embedders) do e
        i = findfirst(l -> l isa lib.ConstantScale, e.layers)
        @assert i !== nothing "Tokenizer has no ConstantScale layer"
        return i
    end
    countmask, toindex, E = nTE_embedder.layers[1], nTE_embedder.layers[2], nTE_embedder.layers[end]
    @assert E isa lib.Flux.Embedding "nTE tokenizer does not end with an Embedding"

    q, counts = zeros(Float64, 3), zeros(Float64, size(E.weight, 2))
    with_initialization_data(dl_train) do dl # calibration must not advance the caller's stream
        for (i, batch) in enumerate(device_batches(dl))
            (; b, A, mask) = batch
            Aᵀb = transposed_batched_vec(A, b)
            q .= lib.running_mean.(i, q, Float64[lib.mean(abs2, b), lib.mean(abs2, Aᵀb), lib.mean(abs2, A)])
            for k in vec(Array(toindex(countmask(mask))))
                counts[k] += 1
            end
            i >= cld(nsamples, cfg.train.batchsize::Int) && break
        end
    end

    @assert all(qᵢ -> isfinite(qᵢ) && qᵢ > 0, q) "Tokenizer second moments must be finite and positive; got $(q)"
    foreach((e, i, qᵢ) -> e.layers[i].scale .= Float32(inv(√qᵢ)), embedders, scale_indices, q)
    calibrate_embedding!(E, counts)
    @info "Calibrated tokenizer"
    @debug "Tokenizer calibration" rms = (.√q...,) nTE_support = count(>(0), counts)

    return models
end

function dynamic_init_flows_variance!(cfg, models)
    function check_backbone(φ, nflows)
        cond_every = 1
        @assert φ isa lib.Flux.Chain && mod(length(φ), nflows ÷ max(cond_every, 1)) == 0 && !(φ[end] isa lib.SimplexFlow) "Invalid flow backbone: $(length(φ)), nflows = $nflows"
        return φ
    end

    εσ = cfg.arch.flows.initialization.perturbation::Float64
    # Conditioner bodies keep the standard initialization; only the output heads carry the gain `g`
    init_backbone!(φ; passlatents) = calibrate_flow_heads!(cfg, φ; εσ, passlatents)

    @info "Initializing flow backbone"
    @assert models.genatr.model.base isa UnitNormalDist "Prior model does not have a standard normal base distribution"
    φ = check_backbone(models.genatr.model.φ.layers[1].flows.xdist_flow.layers[1].flow, cfg.arch.flows.nflows::Int) # flow backbone for xdist_flow
    cfg.arch.flows.initialization.gain = init_backbone!(φ; passlatents = true) # derived, not user-set: the head gain realizing `εσ`, reused by `adjust_learning_rate!`

    return models
end

#### Utils

wrapscalars(m) = lib.fmap(x -> [x], m; exclude = x -> x isa AbstractFloat)

function orthogonal_init(fan_out, fan_in)
    return Float32(sqrt(fan_out / fan_in)) .* lib.Flux.orthogonal(fan_out, fan_in)
end

function FlowConditioner(arch::Config; insize, outsize)
    hdim = arch.hdim::Int
    return TypedChain(
        lib.Flux.Dense(insize => hdim, lib.NNlib.swish; init = orthogonal_init, bias = false),
        [lib.Flux.Dense(hdim => hdim, lib.NNlib.swish; init = orthogonal_init, bias = false) for _ in 1:2],
        lib.Flux.Dense(hdim => outsize; init = orthogonal_init, bias = false),
    )
end

function EncoderMLP(arch::Config, insize)
    model = lib.GatedLinearUnit(insize => arch.hdim::Int => insize, lib.NNlib.swish;
        prenorm = lib.RMSNorm(insize), skip = false, init = orthogonal_init, bias = false)
    return lib.Flux.SkipConnection(model, +)
end

#### Flows

function ElementwiseFlowBlock(arch::Config; insize, nlatent)
    model = TypedChain(lib.Concat(1), FlowConditioner(arch; insize = nlatent, outsize = 2 * insize),
        lib.SqueezeRange(2 * insize; bias = true),
        lib.GetIndex((1:insize, ..), ((insize+1):(2*insize), ..)))
    return lib.AffineFlow(model, lib.squareplus)
end

function SplitCouplingFlowBlock(arch::Config; insize, split_at, flip, nlatent)
    condsize = flip ? insize - split_at : split_at
    return lib.SplitCouplingFlow(ElementwiseFlowBlock(arch; insize = insize - condsize, nlatent = condsize + nlatent); insize, split_at, flip)
end

#### Simplex flows

function WindowFlow(; insize, noisefloor)
    logw = lib.LowPassFlowModel(Float32, insize; noisefloor, time_scale = 2 * log(1.0f0 / 10.0f-3))
    return lib.LatentFlow(lib.AffineFlow(logw), tuple ∘ log_relative_t2_range ∘ last)
end

#### Deep simplex flow

function SimplexFlowBlock(cfg::Config; nlatent)
    udim = cfg.data.nT2::Int - 1
    arch = cfg.arch.flows
    A = lib.ExpAffineFlow(Float32, udim)
    C1 = SplitCouplingFlowBlock(arch; insize = udim, split_at = udim ÷ 2, flip = false, nlatent)
    C2 = SplitCouplingFlowBlock(arch; insize = udim, split_at = udim ÷ 2, flip = true, nlatent)
    E = ElementwiseFlowBlock(arch; insize = udim, nlatent)
    return AnyChain(A, AnyChain(C1, C2), lib.inverse_flow(A), E)
end

function SimplexFlowBackbone(cfg::Config; nflows::Int, kwargs...)
    # Deep flow in Aitchison space from ℝ^{nT2-1} --> ℝ^{nT2-1}
    return AnyChain([SimplexFlowBlock(cfg; kwargs...) for _ in 1:nflows])
end

#### Generator

struct Generator{O, M} <: AbstractModel{O, M}
    opts::O
    model::M
end
@saveable Generator

function Generator(cfg::Config, prior)
    nTE = cfg.data.nTE::Int
    nT2 = cfg.data.nT2::Int
    arch = cfg.arch.flows
    nflows = arch.nflows::Int
    nlatent = cfg.arch.nlatent::Int

    udim = nT2 - 1 + 2 # `nT2 - 1` Aitchison coefficients + log noise level + log scale
    xdim = nT2 + 2 # `nT2` T2-components + nuisance variables

    # Deep flow from ℝ^{nT2-1} --> Δ_{nT2} conditional on latents:
    #   p(xdist | latents = (z, ...))
    xdist_inner_flow = lib.LatentFlow(
        SimplexFlowBackbone(cfg; nflows, nlatent),
        tuple ∘ first, # extract (z,) from latents = (z, b, A, mask, δb, θ)
    )
    xdist_flow = AnyChain(xdist_inner_flow,
        WindowFlow(; insize = nT2 - 1, noisefloor = cfg.arch.prior.xdist.noisefloor::Float64 |> Float32),
        lib.SimplexFlow(; insize = nT2 - 1))

    # Laplacian approximation flow from ℝ² --> ℝ² conditional on T2 dist, latents:
    #   p(logs, logσ | latents = (xdist, z, ...))
    logs_logσ_priors = (; logs_prior = prior.model.logs, logσ_prior = prior.model.logσ)
    logs_logσ_flow = lib.Generic2x2AffineFlow(@call(forward_physics_logs_logσ_model(logs_logσ_priors, _)))

    # Stack into joint distribution over T2 dist, scale, noise level:
    #   p(xdist, logs, logσ | z) = p(logs, logσ | latents = (xdist, z, ...)) * p(xdist | latents = (z, ...))
    #                            = p(logs, logσ | latents = (xdist, ...)) * p(xdist | latents = (z, ...))
    xdist_logs_logσ_flow = lib.StackedFlow(
        (; xdist_flow, logs_logσ_flow),
        (; xdist_flow = nT2 - 1 => nT2, logs_logσ_flow = 2 => 2);
        flip = false,
    )

    flow = xdist_logs_logσ_flow
    base = UnitNormalDist()
    model = ConditionalNormalizingFlow(flow, base; udim, xdim)

    return Generator(model; nTE, nT2, nlatent, udim, xdim)
end

function (𝒢::Generator)(u::Union{AbstractArray, Nothing}, latents::Tuple; deterministic = false)
    z = first(latents)
    (; nTE, nT2, nlatent, udim, xdim) = 𝒢.opts
    @assert_ad size(z, 1) == nlatent && (u === nothing || size(u, 1) == udim)

    if u === nothing
        u = deterministic ?
            lib.zeros_similar(z, udim, Base.tail(size(z))...) : # compute (pseudo-)mode of distribution
            sample_base_similar(𝒢.model, z, :, Base.tail(size(z))...) # sample from distribution
    end

    return 𝒢(ProbabilityMass(u, lib.zero_logjac(u), latents))
end
(𝒢::Generator)(z::AbstractArray, args::AbstractArray...; kwargs...) = 𝒢(nothing, (z, args...); kwargs...)

function (𝒢::Generator)(𝓂ᵤ::ProbabilityMass)
    u, z = 𝓂ᵤ.value, first(𝓂ᵤ.latents)
    (; nTE, nT2, nlatent, udim, xdim) = 𝒢.opts
    @assert_ad size(u, 1) == udim && size(z, 1) == nlatent

    summary = flow_forward(𝒢.model, 𝓂ᵤ)
    x = inference_variables(summary.x)

    return (; x, z, summary)
end

#### Generator utils

function marginal_posterior(𝒢::Generator, latents::Tuple = ())
    # Marginal posterior q(x̄|z) of the generator q(x̄, η|z) = q(x̄|z) q(η|x̄,z), with the same base distribution
    (; udim, xdim) = 𝒢.opts
    @assert_ad 𝒢.model.base isa UnitNormalDist
    base = UnitNormalDist() # base distribution π(u) = N(0,I)
    flow = fwd_backbone(𝒢) # extract backbone flow φ(u | z)
    flow⁻¹ = inv_backbone(𝒢) # extract inverse backbone flow φ⁻¹(x̄ | z)
    opts = (; udim = udim - 2, xdim = xdim - 2)
    return ConditionalNormalizingFlow(opts, flow, flow⁻¹, base, latents) # q = φ ∘ π
end

function with_fwd_backbone(f, 𝒢::Generator)
    𝒢′ = set_fwd_backbone(𝒢, lib.AnyChain(f(fwd_backbone(𝒢)))) # not a deepcopy, 𝒢′ shares parameters with 𝒢
    return @set 𝒢′.model.φ⁻¹ = lib.inverse_flow(𝒢′.model.φ)
end
prepend_fwd_backbone(𝒢::Generator, ℱ) = with_fwd_backbone(flow -> lib.AnyChain(ℱ, flow), 𝒢) # prepend flow to forward backbone
identity_backbone(𝒢::Generator) = with_fwd_backbone(flow -> flow isa lib.SimplexFlow ? flow : flow[end], 𝒢) # keep only final SimplexFlow

@inline function check_fwd_backbone(flow)
    @assert_ad flow isa lib.SimplexFlow || flow isa lib.Flux.Chain && flow.layers[end] isa lib.SimplexFlow "Last layer of forward backbone must be SimplexFlow"
    return flow
end

@inline function check_inv_backbone(flow)
    @assert_ad flow isa lib.InverseFlow{<:lib.SimplexFlow} || flow isa lib.Flux.Chain && flow.layers[1] isa lib.InverseFlow{<:lib.SimplexFlow} "First layer of inverse backbone must be InverseFlow{SimplexFlow}"
    return flow
end

@inline fwd_backbone(𝒢::Generator) = check_fwd_backbone(𝒢.model.φ.layers[1].flows.xdist_flow)
@inline inv_backbone(𝒢::Generator) = check_inv_backbone(𝒢.model.φ⁻¹.layers[2].flows.xdist_flow)
@inline set_fwd_backbone(𝒢::Generator, flow) = @set 𝒢.model.φ.layers[1].flows.xdist_flow = check_fwd_backbone(flow)

# Replace deep-learned backbone with identity, keep learned low-pass window filter + final SimplexFlow
identity_backbone_with_window_flow(𝒢::Generator) = with_fwd_backbone(drop_deep_flow, 𝒢)

# The deep flow is the first `LatentFlow` of the forward backbone.
function deep_flow_index(flow)
    i = findfirst(l -> l isa lib.LatentFlow, flow.layers)
    @assert i !== nothing "Forward backbone has no deep flow"
    return i
end
drop_deep_flow(flow) = flow isa lib.SimplexFlow ? flow : (i = deep_flow_index(flow); lib.AnyChain(flow.layers[begin:(i-1)]..., flow.layers[(i+1):end]...))

function window_filter(𝒢::Generator)
    flow = fwd_backbone(𝒢)
    i = findfirst(is_window_flow, flow.layers)
    @assert i !== nothing "Forward backbone has no window filter"
    return flow.layers[i]
end
is_window_flow(l) = l isa lib.LatentFlow && is_window_filter(l.flow)

#### Prior

struct Prior{O, M} <: AbstractModel{O, M}
    opts::O
    model::M
end
@saveable Prior

function Prior(cfg::Config)
    nTE = cfg.data.nTE::Int
    nT2 = cfg.data.nT2::Int

    mean_logσ = -lib.snr_to_log_noise_level(cfg.arch.prior.noiselevel.mean::Float64) |> Float32
    std_logσ = lib.snr_to_log_noise_level(cfg.arch.prior.noiselevel.std::Float64) |> Float32 # SNR standard deviation in dB

    std_s = cfg.arch.prior.logscale.std::Float64 # distribution of s = exp(logs) is approximately N(1, σₛ²)
    std_logs = √(log1p(std_s^2)) |> Float32 # log-normal parameters with mean 1 and variance σₛ²: σ² = log(1 + σₛ²), μ = -σ²/2
    mean_logs = -std_logs^2 / 2 |> Float32 # mean correction to log-normal distribution, E[s] = 1 = exp(μ + σ²/2) <=> μ = -σ²/2

    θ = (; nTE, nT2, mean_logσ, std_logσ, mean_logs, std_logs)

    flow = AnyChain(WindowFlow(; insize = nT2 - 1, noisefloor = cfg.arch.prior.xdist.noisefloor::Float64 |> Float32),
        lib.SimplexFlow(; insize = nT2 - 1))
    xdist = ConditionalNormalizingFlow(flow, UnitNormalDist(); udim = nT2 - 1, xdim = nT2)
    logs = NormalDist(θ.mean_logs, log(θ.std_logs)) |> wrapscalars # logs ~ 𝒩(-σₛ²/2, σₛ²), i.e. unit mean scale
    logσ = NormalDist(θ.mean_logσ, log(θ.std_logσ)) |> wrapscalars # Gaussian prior on log noise level

    model = (; xdist, logs, logσ)

    return Prior(model; θ...)
end

nuisance(x) = (; logs = x.logs, logσ = x.logσ)
nuisance_models(𝒫::Prior) = nuisance(𝒫.model)
nuisance_domains(::Prior) = (; logs = DomainInfo(1), logσ = DomainInfo(1))
nuisance_prior(𝒫::Prior) = ProductDist(nuisance_models(𝒫), nuisance_domains(𝒫))

xdist_domain(𝒫::Prior) = (; xdist = DomainInfo(𝒫.opts.nT2, 𝒫.opts.nT2 - 1))
xdist_prior(𝒫::Prior) = 𝒫.model.xdist

function recondition(𝒫::Prior, latents::Tuple)
    @assert_ad 𝒫.model isa NamedTuple && 𝒫.model.xdist isa ConditionalNormalizingFlow
    𝒫 = Prior(𝒫.opts, merge(𝒫.model, (; xdist = recondition(𝒫.model.xdist, latents))))
    return 𝒫
end

function composite_prior(𝒫::Prior, latents::Tuple)
    𝒫 = recondition(𝒫, latents)
    (; nT2) = 𝒫.opts

    priors = (; xdist = 𝒫.model.xdist, logs = 𝒫.model.logs, logσ = 𝒫.model.logσ)
    splitter = lib.GetIndex(; xdist = (1:nT2, ..), logs = ((nT2+1):(nT2+1), ..), logσ = ((nT2+2):(nT2+2), ..))
    domains = merge(xdist_domain(𝒫), nuisance_domains(𝒫))

    return ProductDist(priors, vcat, splitter, domains)
end

sample_similar_nuisance(𝒫::Prior, x::InferenceVariables) = sample_similar_nuisance(𝒫, x.xdist) # doesn't matter which array is passed, only need type and Base.tail(size(x))
sample_similar_nuisance(𝒫::Prior, x::AbstractArray) = sample_similar(nuisance_prior(𝒫), x, :, Base.tail(size(x))...)
sample_similar_nuisance(𝒫::Prior, x::AbstractArray, ::Colon, sz::Int...) = sample_similar(nuisance_prior(𝒫), x, 2, sz...)

sample_similar_xdist(𝒫::Prior, x::InferenceVariables) = sample_similar_xdist(𝒫, x.xdist) # doesn't matter which array is passed, only need type and Base.tail(size(x))
sample_similar_xdist(𝒫::Prior, x::AbstractArray) = sample_similar_xdist(𝒫, x, :, Base.tail(size(x))...)
sample_similar_xdist(𝒫::Prior, x::AbstractArray, ::Colon, sz::Int...) = (; xdist = exp.(sample_similar(xdist_prior(𝒫), x, 𝒫.opts.nT2, sz...)))

sample_similar(𝒫::Prior, x::InferenceVariables) = merge(sample_similar_xdist(𝒫, x), sample_similar_nuisance(𝒫, x))
sample_similar(𝒫::Prior, x::AbstractArray) = sample_similar(𝒫, x, :, Base.tail(size(x))...)
sample_similar(𝒫::Prior, x::AbstractArray, ::Colon, sz::Int...) = vcat(sample_similar(xdist_prior(𝒫), x, 𝒫.opts.nT2, sz...), sample_similar(nuisance_prior(𝒫), x, :, Base.tail(size(x))...)...)

#### Cross-resolution lifting

# Rebuilds generator/prior on `Δⁿ′⁻¹` reusing the trained Aitchison backbone and shared window filter.
# Cross-resolution lifting preserves the shallow Gaussian spectrum prior.

lift_window_flow(wf::lib.LatentFlow, n′::Int) = @set wf.flow.model = lib.lift_lowpass_window(wf.flow.model, n′)

function lift_simplex_flow(sf::lib.SimplexFlow, n′::Int)
    sf′ = lib.SimplexFlow(Float32; insize = n′ - 1)
    return @set sf′.logβ = sf.logβ
end

# `[u; v] ↦ [backbone.flow(u); v]` — pads the trained `(n-1)`-dim backbone with `(n′-n)` fresh DCT
# coefficients passing through unchanged.
function lift_aitchison_backbone(backbone::lib.LatentFlow, n::Int, n′::Int)
    @assert n′ > n "cross-resolution lifting requires n′ > n; got $n, $n′"
    return @set backbone.flow = lib.ProductFlow(
        (xdist = backbone.flow, new = identity),
        (xdist = (n - 1) => (n - 1), new = (n′ - n) => (n′ - n)),
    )
end

# Extract `(backbone, window, trained_simplex)` from the trained generator's xdist flow.
function trained_xdist_layers(𝒢::Generator)
    @assert 𝒢.model isa ConditionalNormalizingFlow && 𝒢.model.base isa UnitNormalDist
    layers = fwd_backbone(𝒢).layers
    @assert length(layers) == 3 "expected [backbone, window, simplex]; got $(length(layers))"
    backbone, window, simplex = layers
    return backbone, window, simplex isa lib.SimplexFlow ? simplex : simplex.layers[end]
end

# Wrap a lifted xdist subchain `ℝⁿ′⁻¹ → Δⁿ′⁻¹` into a `Generator` at resolution `n′`.
function wrap_lifted_generator(𝒢::Generator, n′::Int, lifted_xdist)
    (; nTE, nlatent) = 𝒢.opts
    udim, xdim = n′ - 1 + 2, n′ + 2
    flow = lib.StackedFlow(
        (; xdist_flow = lifted_xdist, logs_logσ_flow = 𝒢.model.φ.layers[2].flow),
        (; xdist_flow = (n′ - 1) => n′, logs_logσ_flow = 2 => 2);
        flip = false,
    )
    return Generator((; nTE, nT2 = n′, nlatent, udim, xdim), ConditionalNormalizingFlow(flow, 𝒢.model.base; udim, xdim))
end

# Wrap a lifted xdist subchain `ℝⁿ′⁻¹ → Δⁿ′⁻¹` into a `Prior` at resolution `n′`.
function wrap_lifted_prior(𝒫::Prior, n′::Int, lifted_xdist)
    cnf = 𝒫.model.xdist
    @assert cnf isa ConditionalNormalizingFlow "lift_prior requires ConditionalNormalizingFlow xdist; got $(typeof(cnf))"
    cnf′ = ConditionalNormalizingFlow(lifted_xdist, cnf.base, cnf.latents; udim = n′ - 1, xdim = n′)
    𝒫′ = @set 𝒫.model = merge(𝒫.model, (; xdist = cnf′))
    return @reset 𝒫′.opts.nT2 = n′
end

# Extract the trained simplex flow from the prior's xdist `[window, simplex]` chain.
function prior_trained_simplex(𝒫::Prior)
    layers = 𝒫.model.xdist.φ.layers
    @assert length(layers) == 2 "expected [window, simplex]; got $(length(layers))"
    sf = layers[2]
    return sf isa lib.SimplexFlow ? sf : sf.layers[end]
end

function lift_generator(𝒢::Generator, n′::Int, lifted_window::lib.LatentFlow)
    backbone, _, simplex = trained_xdist_layers(𝒢)
    return wrap_lifted_generator(𝒢, n′, lib.AnyChain(
        lift_aitchison_backbone(backbone, 𝒢.opts.nT2, n′),
        lifted_window,
        lift_simplex_flow(simplex, n′),
    ))
end

lift_prior(𝒫::Prior, n′::Int, lifted_window::lib.LatentFlow) =
    wrap_lifted_prior(𝒫, n′, lib.AnyChain(lifted_window, lift_simplex_flow(prior_trained_simplex(𝒫), n′)))

function lift_models(models, n′::Int)
    @assert models.genatr.opts.nT2 == models.prior.opts.nT2 "generator and prior must share nT2"
    n′ == models.genatr.opts.nT2 && return models
    lifted_window = lift_window_flow(window_filter(models.genatr), n′)
    return merge(models, (;
        genatr = lift_generator(models.genatr, n′, lifted_window),
        prior = lift_prior(models.prior, n′, lifted_window))
    ) |> lib.gpu
end

#### Encoder

struct Encoder{O, M} <: AbstractModel{O, M}
    opts::O
    model::M
end
@saveable Encoder

function Encoder(cfg::Config)
    nTE = cfg.data.nTE::Int
    nT2 = cfg.data.nT2::Int
    arch = cfg.arch.encoder
    init = orthogonal_init
    bias = false
    nlatent = cfg.arch.nlatent::Int
    tokensize = arch.tokensize::Int

    # Signal tokenizer. Input: b [nTEmax x batchsize]
    b_embedder = AnyChain(
        lib.Flux.unsqueeze(; dims = 2), # reshape: [nTEmax x 1 x batchsize]
        lib.ConstantScale(lib.Flux.ones32(1)), # data-dependent frozen scaling layer; tuned to rescale inputs to `E[b²] = 1`
        lib.Flux.Dense(nTE => tokensize; init, bias), # linear upsample to [tokensize x 1 x batchsize]; no activation before the normalized residual
        EncoderMLP(arch, tokensize),
    )

    # Signal tokenizer. Input: Aᵀb [nT2 x batchsize]
    Aᵀb_embedder = AnyChain(
        lib.Flux.unsqueeze(; dims = 2), # reshape: [nT2 x 1 x batchsize]
        lib.ConstantScale(lib.Flux.ones32(1)), # data-dependent frozen scaling layer; tuned to rescale inputs to `E[(Aᵀb)²] = 1`
        lib.Flux.Dense(nT2 => tokensize; init, bias), # linear upsample to [tokensize x 1 x batchsize]; no activation before the normalized residual
        EncoderMLP(arch, tokensize),
    )

    # EPG bases tokenizer. Input: A [nTEmax x nT2 x batchsize]
    A_embedder = AnyChain(
        lib.ConstantScale(lib.Flux.ones32(1)), # data-dependent frozen scaling layer; tuned to rescale inputs to `E[A²] = 1`
        lib.Flux.Dense(nTE => tokensize; init, bias), # linear upsample to [tokensize x nT2 x batchsize]; no activation before the normalized residual
        EncoderMLP(arch, tokensize),
    )

    # Scalars tokenizer. Input: θ [nθ x batchsize]
    nθ = num_scalar_features()
    θ_embedder = AnyChain(
        lib.Flux.unsqueeze(; dims = 2), # reshape: [nθ x 1 x batchsize]
        lib.NotTrainable(lib.Flux.BatchNorm(nθ; momentum = 1.0f0, affine = true, track_stats = true, eps = 0.0f0)), # rescale to mean=0, std=1: [nθ x batchsize]
        lib.Flux.Dense(nθ => tokensize; init, bias), # linear upsample to [tokensize x 1 x batchsize]; no activation before the normalized residual
        EncoderMLP(arch, tokensize),
    )

    # Num. echoes tokenizer. Input: mask [nTEmax x batchsize]
    nTEmin = cfg.train.augment.nTEmin::Int # minimum allowable masked signal length
    nTE_embedder = AnyChain(
        lib.count_mask_active, # count nonzero mask entries: [1 x batchsize]
        lib.broadcast_function(Base.Fix2(-, Int32(nTEmin - 1))), # map `nTEᵢ` to token index `nTEᵢ - nTEmin + 1 ∈ {1, ..., nTE - nTEmin + 1}`
        lib.Flux.Embedding(nTE - nTEmin + 1 => tokensize; init = (sz...,) -> 0.02f0 .* lib.Flux.randn32(sz...)), # gather embeddings: [tokensize x 1 x batchsize]
    )

    xformer = lib.TuplifyInputs(
        AnyChain(
            lib.@call(lib.SinusoidalPositionalEncoding(_, Float32.((4, 4 + nT2)), Float32(1 / √2))),
            [lib.Transformer(arch.tokensize::Int, arch.nheads::Int, arch.headsize::Int, arch.hdim::Int;
                init, scale = Float32(1 / arch.headsize::Int))
             for _ in 1:(arch.ntransformers::Int)],
        ),
    )

    post = AnyChain(
        lib.Flux.flatten, # [tokensize x 4 x batchsize] -> [4 * tokensize x batchsize]
        EncoderMLP(arch, 4 * tokensize), # MLPs
        lib.Flux.Dense(4 * tokensize => nlatent; init, bias = false), # dense downsample to latent space dimension; bias = false since we want mean(z) = 0
        lib.RMSNorm(nlatent; init = Returns(0.0f0), eps = 0.0f0), # rescale to unit rms along the first dimension
    )

    model = (; b_embedder, Aᵀb_embedder, A_embedder, nTE_embedder, θ_embedder, xformer, post)
    return Encoder(model; nTE, nT2, nlatent)
end

function (ℰ::Encoder)(b::AbstractMatrix, A::AbstractTensor3D, mask::AbstractMatrix, θ::AbstractMatrix)
    enc_cond = encode_conditionals(ℰ, A, mask, θ)
    return encode_signal(ℰ, b, A, enc_cond)
end

function encode_conditionals(ℰ::Encoder, A::AbstractTensor3D, mask::AbstractMatrix, θ::AbstractMatrix)
    (; A_embedder, nTE_embedder, θ_embedder) = ℰ.model
    A_emb = A_embedder(ignore_derivatives(A)) # EPG bases embedding
    θ_emb = θ_embedder(ignore_derivatives(θ)) # normalized scalar parameters embedding
    nTE_emb = nTE_embedder(ignore_derivatives(mask)) # nTE is derived from mask then mapped to discrete embedding
    return (; A_emb, θ_emb, nTE_emb)
end

function encode_signal(ℰ::Encoder, b::AbstractMatrix, A::AbstractTensor3D, enc_cond::NamedTuple)
    (; b_embedder, Aᵀb_embedder, xformer, post) = ℰ.model
    (; A_emb, θ_emb, nTE_emb) = enc_cond
    Aᵀb = transposed_batched_vec(ignore_derivatives(A), ignore_derivatives(b)) # dual signal A'b
    b_emb = b_embedder(ignore_derivatives(b)) # signal embedding
    Aᵀb_emb = Aᵀb_embedder(ignore_derivatives(Aᵀb)) # dual signal embedding
    Q = hcat(b_emb, Aᵀb_emb, θ_emb, nTE_emb) # query tokens concatenated, `hcat(b, Aᵀb, θ, nTE)`
    KV = hcat(Q, A_emb) # key/value tokens concatenated, `hcat(b, Aᵀb, θ, nTE, A)`
    X, _ = xformer(Q, KV) # cross-attention
    z = post(X) # final mlp; include scalar embeddings for extra information
    return z
end

#### Posterior sampling

function ForwardPhysicsTransformedPosterior(
    𝒢::Generator,
    𝒫::Prior,
    z::AbstractArray{T},
    b::AbstractMatrix{T},
    A::AbstractTensor3D{T},
    mask::AbstractMatrix{T},
    δb::AbstractArray{T},
    θ::AbstractArray{T},
) where {T}
    flow = recondition(𝒢.model, (z, b, A, mask, δb, θ))
    prior = composite_prior(𝒫, (θ,))
    model = @call(forward_physics_model(_, A, mask, δb)) ∘ only
    likelihood = LikelihoodFactory(model, b)
    return TransformedPosterior(flow, prior, likelihood)
end
