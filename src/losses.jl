#### Encoder moment loss

function wasserstein2_unit_multivariate_gaussian(μ, Σ; α::Real = eps(eltype(Σ)))
    λ = eigvals_herm_cpu(Σ) # eigenvalues of Σ (computed on cpu in Float64)
    @. λ = (1 - α) * λ + α # shrink eigenvalues: λ' = (1-α)λ + α
    return sum(abs2, μ) / 2 + sum(abs2, @. √λ - 1) / 2 # ∑ᵢ μᵢ²/2 + (√λᵢ - 1)²/2, with λᵢ the eigenvalues of Σ
end

function ChainRules.rrule(::typeof(wasserstein2_unit_multivariate_gaussian), μ, Σ; α::Real = eps(eltype(Σ)))
    # Gradient of Wasserstein-2 distance:
    #   W = ||μ||² / 2 + (tr(Σ) + n - 2 * tr(√Σ)) / 2 = ∑ᵢ μᵢ² / 2 + (√λᵢ - 1)² / 2
    # where λᵢ are the eigenvalues of Σ. The gradients are:
    #   ∂W/∂μ = μ
    #   ∂W/∂Σ = (I - √Σ⁻¹) / 2 = V * diag(1 - 1/√λᵢ) * V' / 2
    # where λ, V are the eigenvalues and eigenvectors of Σ, respectively.
    λ, V = eigen_herm_cpu(Σ) # eigendecomposition of Σ (computed on cpu in Float64)
    @. λ = (1 - α) * max(λ, zero(λ)) + α # shrink eigenvalues: λ -> (1-α)λ + α
    W = sum(abs2, μ) / 2 + ofeltypefloat(Σ, sum(abs2, @. √λ - 1)) / 2
    function ∇wasserstein2_unit_multivariate_gaussian_inner(Δ)
        ΔΣ = V * LinearAlgebra.Diagonal(@. Δ * (1 - α) * (√λ - 1) / (2 * √λ)) * V' # (1 - α) * Δ * (I - √Σ⁻¹) / 2
        return (NoTangent(), Δ * μ, device_similar(Σ, symmetrize!(ΔΣ)))
    end
    return W, ∇wasserstein2_unit_multivariate_gaussian_inner
end

#### Wasserstein/Earth Mover's distance between discrete probability distributions

# 1st Wasserstein (Earth Mover's) distance between distributions on a common grid
wasserstein1(x, y; agg = mean, dims = 1) = agg(sum(abs, cumsum(x .- y; dims); dims))

#### Gaussian negative log-likelihood

@inline neglogpdf_gaussian(x, μ, logσ) = ((exp(-logσ) * (x - μ))^2 + oftypefloat(x, log2π)) / 2 + logσ
@inline function ∇neglogpdf_gaussian(x, μ, logσ)
    ∂x = exp(-2logσ) * (x - μ)
    ∂μ = -∂x
    ∂logσ = 1 - ∂x * (x - μ)
    return (∂x, ∂μ, ∂logσ)
end
@scalar_rule neglogpdf_gaussian(x, μ, logσ) (∇neglogpdf_gaussian(x, μ, logσ)...,)
@define_ternary_dual_scalar_rule neglogpdf_gaussian ∇neglogpdf_gaussian

@inline neglogpdf_unit_gaussian(x) = (x^2 + oftypefloat(x, log2π)) / 2
@inline ∇neglogpdf_unit_gaussian(x) = x
@scalar_rule neglogpdf_unit_gaussian(x) ∇neglogpdf_unit_gaussian(x)
@define_unary_dual_scalar_rule neglogpdf_unit_gaussian ∇neglogpdf_unit_gaussian

neglogpdf_isotropic_gaussian(x::AbstractArray; dims = 1) = sum(neglogpdf_unit_gaussian, x; dims)

#### Gaussian entropy

@inline entropy_unit_gaussian(::Type{T}) where {T <: AbstractFloat} = (one(T) + T(log2π)) / 2

#### Rician negative log-likelihood

neglogpdf_rician(x, ν, logσ) = FastRicianLikelihoods.neglogpdf_rician(x, ν, logσ)
neglogpdf_rician(x, ν) = FastRicianLikelihoods.neglogpdf_rician(x, ν)

neglogpdf_qrician(x, ν, logσ, δ, order::Val = Val(8)) = FastRicianLikelihoods.neglogpdf_qrician(x, ν, logσ, δ, order)
neglogpdf_qrician(x, ν, δ, order::Val = Val(8)) = FastRicianLikelihoods.neglogpdf_qrician(x, ν, δ, order)

# Fit generalized Pareto distribution to data using the algorithm of Zhang et al. (2009)
# (Zhang & Stephens, 2009; Vehtari et al., 2022)
struct GPDFitWorkspace{T <: AbstractFloat}
    n::Int
    m::Int
    x::Vector{T}
    L::Vector{T}
end

function GPDFitWorkspace(::Type{T}, n::Int) where {T <: AbstractFloat}
    m = 20 + isqrt(n)
    @assert 1 ≤ m ≤ n "m must be between 1 and n, but m = $m and n = $n"
    x = zeros(T, n)
    L = zeros(T, m)
    return GPDFitWorkspace{T}(n, m, x, L)
end

function load_samples!(work::GPDFitWorkspace{T}, x::AbstractVector{T}; sorted = false) where {T <: AbstractFloat}
    @assert length(x) == work.n "number of samples must match workspace size"
    copy!(work.x, x)
    !sorted && sort!(work.x)
    return work
end

function gpdfit(x::AbstractVector{T}; sorted = false) where {T <: AbstractFloat}
    work = GPDFitWorkspace(T, length(x))
    load_samples!(work, x; sorted)
    return gpdfit(work)
end

function gpdfit(x::AbstractMatrix{T}; sorted = false) where {T <: AbstractFloat}
    n, batchsize = size(x)
    ntasks = Threads.nthreads()
    chunksize = cld(batchsize, ntasks)

    σ = zeros(T, 1, batchsize)
    k = zeros(T, 1, batchsize)
    @sync GC.@preserve x for tid in 1:ntasks
        Threads.@spawn begin
            work = GPDFitWorkspace(T, n)
            @inbounds for j in (1+(tid-1)*chunksize):min(tid*chunksize, batchsize)
                load_samples!(work, uview(x, :, j); sorted)
                σ[j], k[j] = gpdfit(work)
            end
        end
    end

    return (; σ, k)
end

function gpdfit(work::GPDFitWorkspace{T}) where {T <: AbstractFloat}
    (; n, m, x, L) = work

    xₙ = @inbounds x[end]
    x₄ = @inbounds x[floor(Int, n / 4 + 0.5)]

    θ_node(i) = inv(xₙ) + (1 - √(m / T(i - 0.5))) / 3x₄

    @inbounds for i in 1:m
        θᵢ = θ_node(i)
        kᵢ = zero(T)
        @simd for j in 1:n
            kᵢ += log1p(-θᵢ * x[j]) / n
        end
        L[i] = -n * (log(-kᵢ / θᵢ) + kᵢ + 1)
    end

    θ = zero(T)
    @inbounds for i in 1:m
        θᵢ = θ_node(i)
        Lᵢ = L[i]
        wᵢ⁻¹ = zero(T)
        @simd for j in 1:m
            wᵢ⁻¹ += exp(L[j] - Lᵢ) # Compute the weights w
        end
        θ += θᵢ / wᵢ⁻¹
    end

    k = zero(T)
    @simd for i in 1:n
        k += log1p(-θ * x[i]) / n
    end
    σ = -k / θ

    return (; σ, k)
end

"""
Fit a generalized Pareto tail to sorted importance weights; regularization shrinks the tail index toward 0.5.
"""
function psis_diagnostic(rₛ::AbstractVecOrMat{<:AbstractFloat})
    S = size(rₛ, 1) # number of samples
    M = psis_tail_size(S) # number of tail samples
    û = rₛ isa AbstractVector ? @inbounds(rₛ[end-M]) : @views rₛ[(end-M):(end-M), ..] # cutpoint (scalar or 1×batchsize)
    x = @views rₛ[(end-M+1):end, ..] .- û # Generalized Pareto tail samples
    (; σ, k) = gpdfit(x; sorted = true) # fit generalized Pareto distribution

    # Relative scale σ′ = σ/û, comparable across cutpoints
    σ′ = rₛ isa AbstractVector ? σ / û : (σ ./= û)

    M₀ = 10 # weakly informative prior worth 10 tail observations, shrinking toward k = 0.5
    k₀ = ofeltypefloat(k, 0.5)
    k′ = @. (M * k + M₀ * k₀) / (M + M₀)
    return (; k = k′, σ = σ′)
end

"""
    psis_tail_size(S::Int) -> Int

Compute the PSIS tail size M for S samples: M = min(S/5, 3√S).
"""
psis_tail_size(S::Int) = floor(Int, min(S / 5, 3 * √S))

# Inverse DCT basis; `D` is the Aitchison dimension and `D+1` the simplex dimension
idct_simplex_basis(D::Int) = idct_simplex_basis(Float32, D)
idct_simplex_basis(::Type{T}, D::Int) where {T} = T.(idct!(Float64.((1:(D+1)) .== (2:(D+1))'), 1))
idct_simplex_basis(::Type{<:BigFloat}, D::Int) = idct_simplex_basis_brute(BigFloat, D)

function idct_simplex_basis_brute(::Type{T}, D::Int) where {T}
    N = D + 1 # simplex dimension
    k = (1:(N-1))' # frequencies (columns)
    j = 1:N # DCT nodes (rows)
    kθⱼ = @. T((k * (2j - 1)) // 2N) # `N x N-1` matrix of angles relative to π
    nrm = sqrt(2 / T(N)) # column normalization
    return @. cospi(kθⱼ) * nrm # N x N-1 DCT basis cos(kθⱼ), θⱼ = π(j-1/2)/N, k = 1..N-1
end
idct_simplex_basis_brute(N::Int) = idct_simplex_basis_brute(Float32, N)
