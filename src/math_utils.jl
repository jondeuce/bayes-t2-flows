@inline abs4(x) = abs2(x)^2

snr_to_log_noise_level(snr) = log(oftypefloat(snr, 10)) * (snr / 20) # equivalently: log(exp10(snr / 20))

# Smooth maximum with sharpness parameter α: https://en.wikipedia.org/wiki/Smooth_maximum
# smoothmax(x₁, x₂) = log(exp(x₁) + exp(x₂))
function smoothmax(x₁, x₂)
    m = max(x₁, x₂)
    return m + log(exp(x₁ - m) + exp(x₂ - m))
end

# smoothmin(x₁, x₂) = -log(exp(-x₁) + exp(-x₂)) = -smoothmax(-x₁, -x₂)
function smoothmin(x₁, x₂)
    m = min(x₁, x₂)
    return m - log(exp(m - x₁) + exp(m - x₂))
end

smoothmax(x₁, x₂, α) = smoothmax(α * x₁, α * x₂) / α # smoothmax(x₁, x₂, α) = log(exp(α * x₁) + exp(α * x₂)) / α
smoothmin(x₁, x₂, α) = smoothmin(α * x₁, α * x₂) / α # smoothmin(x₁, x₂, α) = -log(exp(-α * x₁) + exp(-α * x₂)) / α

@inline ∇smoothmax(x₁, x₂) = (Flux.sigmoid_fast(x₁ - x₂), Flux.sigmoid_fast(x₂ - x₁))
@scalar_rule smoothmax(x₁, x₂) (∇smoothmax(x₁, x₂)...,)
@define_binary_dual_scalar_rule smoothmax ∇smoothmax

@inline ∇smoothmin(x₁, x₂) = (Flux.sigmoid_fast(x₂ - x₁), Flux.sigmoid_fast(x₁ - x₂))
@scalar_rule smoothmin(x₁, x₂) (∇smoothmin(x₁, x₂)...,)
@define_binary_dual_scalar_rule smoothmin ∇smoothmin

"""
    squareplus(x) = (x + √(x² + 4)) / 2

Smooth algebraic rectifier from ℝ to ℝ⁺ (Barron, 2021), with b = 4 so that squareplus(0) = 1 and
squareplus(x) * squareplus(-x) = 1; its slope and curvature at 0 match softplus.
"""
@inline function squareplus(x)
    s = sqrt(x^2 + 4)
    return x >= 0 ? (x + s) / 2 : 2 / (s - x) # squareplus(x) * squareplus(-x) = 1
end
@inline function ∇squareplus(x)
    s = sqrt(x^2 + 4)
    return x >= 0 ? (x + s) / 2s : 2 / (s * (s - x)) # ∇squareplus(x) = squareplus(x) / √(x² + 4) = (1 + x / √(x² + 4)) / 2
end
@scalar_rule squareplus(x) ∇squareplus(x)
@define_unary_dual_scalar_rule squareplus ∇squareplus

# Log squareplus: log(squareplus(x)) computed accurately for |x| < 1
@inline function logsquareplus(x)
    if abs(x) < 1
        return log1p(x * (1 + x / (sqrt(x^2 + 4) + 2)) / 2)
    else
        return log(squareplus(x))
    end
end
@inline ∇logsquareplus(x) = inv(sqrt(x^2 + 4))
@scalar_rule logsquareplus(x) ∇logsquareplus(x)
@define_unary_dual_scalar_rule logsquareplus ∇logsquareplus

# Squaremax: softmax with squareplus instead of exp
function squaremax!(y::AbstractArray, x::AbstractArray; dims)
    @. y = squareplus(x)
    y .= y ./ sum(y; dims)
    return y
end
squaremax(x::AbstractArray; dims) = squaremax!(similar(x, eltypefloat(x)), x; dims)

function ChainRules.rrule(::typeof(squaremax), x; dims)
    # Minimal closed-over variables (recompute y in pullback)
    function squaremax_pullback(dy)
        y = squareplus.(x)
        Σtmp = sum(y; dims)
        y ./= Σtmp
        dx = y .* dy
        sum!(Σtmp, dx)
        @. dx = (dx - y * Σtmp) / sqrt(x^2 + 4)
        return (NoTangent(), dx)
    end
    y = squareplus.(x)
    y ./= sum(y; dims)

    return y, squaremax_pullback
end

logplusexp(x₁, x₂) = smoothmax(x₁, x₂) # log(exp(x₁) + exp(x₂))
# log(|exp(x₁) - exp(x₂)|)
function logminusexp(x₁, x₂)
    (x₁, x₂) = minmax(x₁, x₂)
    return x₂ + log1p(-exp(x₁ - x₂))
end

@inline ∇logminusexp(x₁, x₂) = (inv(-expm1(x₂ - x₁)), inv(-expm1(x₁ - x₂)))
@scalar_rule logminusexp(x₁, x₂) (∇logminusexp(x₁, x₂)...,)
@define_binary_dual_scalar_rule logminusexp ∇logminusexp

running_mean(i::Int, x̄, xnew) = i == 1 ? xnew : x̄ .+ (xnew .- x̄) ./ i
running_mean(i::Int) = (x̄, xnew) -> running_mean(i, x̄, xnew)

running_neglogmeanexp(i::Int, x̄, xnew) = i == 1 ? xnew : smoothmin.(x̄ .- log1p(ofeltypefloat(x̄, -1 // i)), xnew .+ log(ofeltypefloat(xnew, i)))
running_neglogmeanexp(i::Int) = (x̄, xnew) -> running_neglogmeanexp(i, x̄, xnew)

running_neglogmeanexp(i::Int, x̄, xnew, α) = i == 1 ? xnew : α == 0 ? running_mean(i, x̄, xnew) : smoothmin.(x̄ .- log1p(ofeltypefloat(x̄, -1 // i)) ./ α, xnew .+ log(ofeltypefloat(xnew, i)) ./ α, α)
running_neglogmeanexp(i::Int, α) = (x̄, xnew) -> running_neglogmeanexp(i, x̄, xnew, α)

# Mask test using the value part, for ForwardDiff v1 compatibility
@inline is_mask_active(mask::ForwardDiff.Dual) = is_mask_active(ForwardDiff.value(mask))
@inline is_mask_active(mask::Bool) = mask
@inline is_mask_active(mask::Real) = mask == true
@non_differentiable is_mask_active(mask)

# Count active mask entries along dimension `dims`
count_mask_active(mask::AbstractArray; dims = 1) = sum(is_mask_active, mask; dims, init = zero(Int32))
@non_differentiable count_mask_active(mask::AbstractArray)

# Mask a value `x`: if mask == true, return the value `x`, else return `zero(x)`.
# Explicit gradient needed because Zygote differentiates broadcasts via ForwardDiff, which treats `==` specially
#   [1] https://github.com/FluxML/Zygote.jl/issues/1570
#   [2] https://discourse.julialang.org/t/and-inside-zygote-gradient-are-inaccurate-on-gpu/128269
#   [3] https://github.com/JuliaDiff/ForwardDiff.jl/pull/481
@inline apply_mask(mask) = apply_mask(mask, one(mask))
@inline apply_mask(mask, x) = ifelse(is_mask_active(mask), x, zero(x))
@inline apply_mask(mask, x::Union{Tuple, NamedTuple}) = ifelse(is_mask_active(mask), x, map(zero, x))

@inline ∇apply_mask(mask, x) = (zero(mask), ifelse(is_mask_active(mask), one(x), zero(x)))
@inline ∇apply_mask(mask, x::Union{Tuple, NamedTuple}) = (zero(mask), ifelse(is_mask_active(mask), map(one, x), map(zero, x)))
@scalar_rule apply_mask(mask, x) (∇apply_mask(mask, x)...,)
@define_binary_dual_scalar_rule apply_mask ∇apply_mask

# Apply function `f` to `args...` if `mask == true`, else return `default`
@inline ternary_apply_mask(mask, default, f, args...) = is_mask_active(mask) ? f(args...) : default

"""
Langevin function `L(y) = coth(y) - 1/y`.

`coth(y) - 1/y` loses `≈ eps/y` to cancellation near zero, where the odd Taylor series `y/3 - y³/45 + 2y⁵/945 - y⁷/4725 + 2y⁹/93555` is used instead.
The switch `y⁎ = (eps · 638512875/1382)^(1/12)` equates the rounding error with the series truncation error `1382y¹¹/638512875` (`y⁎ ≈ 0.15` for `Float64`, `0.79` for `Float32`).
"""
function langevin(y::Real)
    y = float(y)
    T = typeof(y)
    abs(y) >= langevin_series_cutoff(T) && return coth(y) - inv(y)
    return y * evalpoly(y * y, (T(1 // 3), T(-1 // 45), T(2 // 945), T(-1 // 4725), T(2 // 93555)))
end
langevin_series_cutoff(::Type{T}) where {T} = (eps(T) * 638512875 / 1382)^inv(T(12))

"""
Derivative of the Langevin function, `L′(y) = 1/y² - csch²(y)`.

As for `langevin`, the even Taylor series `1/3 - y²/15 + 2y⁴/189 - y⁶/675 + 2y⁸/10395 - 1382y¹⁰/58046625` replaces the direct form near zero,
switching at `y⁎ = (eps · 18243225/52)^(1/14)`, where the rounding error `≈ eps/y²` equals the truncation error `52y¹²/18243225` (`y⁎ ≈ 0.19` for `Float64`, `0.80` for `Float32`).
"""
function dlangevin(y::Real)
    y = float(y)
    T = typeof(y)
    abs(y) >= dlangevin_series_cutoff(T) && return inv(y * y) - csch(y)^2
    return evalpoly(y * y, (T(1 // 3), T(-1 // 15), T(2 // 189), T(-1 // 675), T(2 // 10395), T(-1382 // 58046625)))
end
dlangevin_series_cutoff(::Type{T}) where {T} = (eps(T) * 18243225 / 52)^inv(T(14))

"""
Mean of the truncated geometric distribution `p(d) ∝ exp(-λd)` on `d ∈ 0:m`, for `λ ∈ [0, ∞]`.

The closed form `1/expm1(λ) - (m+1)/expm1((m+1)λ)` cancels as `λ → 0`, where the equivalent `m/2 - [(m+1)L((m+1)λ/2) - L(λ/2)]/2`, with `L = langevin`, is used instead.
For `λ > 1` the closed form is stable, since its second term is at most `2/(e+1) ≈ 0.54` times the first.
"""
function truncgeom_mean(λ::Real, m::Integer)
    λ = float(λ)
    λ <= 1 && return (m - (m + 1) * langevin((m + 1) * λ / 2) + langevin(λ / 2)) / 2
    return inv(expm1(λ)) - (m + 1) / expm1((m + 1) * λ)
end

"""
Variance of the truncated geometric distribution `p(d) ∝ exp(-λd)` on `d ∈ 0:m`, for `λ ∈ [0, ∞]`; equal to `-∂λ truncgeom_mean(λ, m)`.

The closed form `csch²(λ/2)/4 - (m+1)² csch²((m+1)λ/2)/4` cancels as `λ → 0`, where the equivalent `[(m+1)² L′((m+1)λ/2) - L′(λ/2)]/4`, with `L′ = dlangevin`, is used instead.
"""
function truncgeom_var(λ::Real, m::Integer)
    λ = float(λ)
    λ <= 1 && return ((m + 1)^2 * dlangevin((m + 1) * λ / 2) - dlangevin(λ / 2)) / 4
    return (csch(λ / 2)^2 - (m + 1)^2 * csch((m + 1) * λ / 2)^2) / 4
end

"""
Rate `λ ∈ [0, ∞]` of the truncated geometric distribution `p(d) ∝ exp(-λd)` on `d ∈ 0:m` with mean `d̄ ∈ [0, m/2]`.

The mean `E(λ) = 1/expm1(λ) - (m+1)/expm1((m+1)λ)` is decreasing and convex on `λ ≥ 0`, with `E′ = -Var` and `E″ = κ₃ ≥ 0`.
Dropping its second term gives the `m → ∞` solution `λ₀ = log1p(1/d̄)`, an upper bound on the root with error `≈ (m+1)d̄ᵐ`, hence exact to rounding for small `d̄`.
By convexity, one Newton step from `λ₀` lands left of the root, and Newton's method then increases monotonically to it.
If that first step does not move left to a finite point, `λ₀` is already at rounding level (this includes `Var` underflowing for tiny `d̄`).
The iteration stops once a step no longer increases `λ`; each continuing step strictly increases `λ` over finitely many floats, so it terminates.
"""
function truncgeom_rate(d̄::Real, m::Integer)
    d̄ = float(d̄)
    T = typeof(d̄)
    @assert 0 <= d̄ <= m / 2 "Truncated geometric mean $d̄ must lie in [0, m/2] = [0, $(m / 2)]"
    d̄ == 0 && return T(Inf)
    newton_step(λ) = (truncgeom_mean(λ, m) - d̄) / truncgeom_var(λ, m)
    λ₀ = log1p(d̄) - log(d̄) # log1p(1/d̄) without overflow for subnormal d̄
    λ = λ₀ + newton_step(λ₀)
    -T(Inf) < λ < λ₀ || return λ₀ # also catches NaN
    λ = max(λ, zero(T))
    while true
        λ′ = λ + newton_step(λ)
        λ′ > λ || return λ # also catches NaN
        λ = λ′
    end
end

"""
Quantile function `d = floor(-log(1 - u(1 - exp(-(m+1)λ))) / λ)` of the truncated geometric distribution `p(d) ∝ exp(-λd)` on `d ∈ 0:m`, for `u ∈ [0, 1)` and `λ ∈ [0, ∞]`.

`expm1` and `log1p` keep full relative precision as `λ → 0`; `min` absorbs rounding as `u → 1`.
"""
function truncgeom_quantile(λ::Real, m::Integer, u::Real)
    λ, u = promote(float(λ), float(u))
    λ == 0 && return floor(Int, u * (m + 1))
    c = -expm1(-(m + 1) * λ)
    return min(floor(Int, -log1p(-u * c) / λ), Int(m))
end

#### Array utils

const LowerOrUpperTriangular{T, A} = Union{LowerTriangular{T, A}, UpperTriangular{T, A}}

# Eager identity matrix
eye(::Type{T}, n::Int) where {T} = eye(one(T), n)
eye(s::Number, n::Int) = collect(s * LinearAlgebra.I(n))

# Sample multivariate normal
@inline sample_mv_normal(μ, σ, z) = @. μ + σ * z

# Sample multivariate rician
@inline sample_mv_rician(ν, σ, zR, zI) = @. √((ν + σ * zR)^2 + (σ * zI)^2)

tr_eager(Σ::AbstractMatrix) = tr(Σ)
ChainRules.rrule(::typeof(tr_eager), Σ::AbstractMatrix) = tr(Σ), Δ -> (NoTangent(), diag!(zero(Σ), Δ)) # the default tr cotangent is a Diagonal, which hits CPU fallbacks

function eigen_herm_cpu(Σ::AbstractMatrix)
    if !all(isfinite, Σ)
        λ = fill(ofeltypefloat(Σ, NaN), size(Σ, 1))
        V = fill(ofeltypefloat(Σ, NaN), size(Σ, 1), size(Σ, 1))
    else
        λ, V = LinearAlgebra.eigen(Hermitian(Σ))
    end
    return (; values = λ, vectors = V)
end
eigen_herm_cpu(Σ::CuMatrix) = map(Base.Fix1(arr_similar, Σ), eigen_herm_cpu(cpu64(Σ))) # no differentiable eigendecomposition on gpu + extra accuracy on cpu is worth it

eigvals_herm_cpu(Σ::CuMatrix) = arr_similar(Σ, eigvals_herm_cpu(cpu64(Σ))) # no differentiable eigvals on gpu + extra accuracy on cpu is worth it
eigvals_herm_cpu(Σ::AbstractMatrix, ϵ = zero(eltypefloat(Σ))) = !all(isfinite, Σ) ? fill(ofeltypefloat(Σ, NaN), size(Σ, 1)) : max.(LinearAlgebra.eigvals(Hermitian(Σ)), ϵ) # ensure nonnegative eigenvalues

"""
    matrix_exp!(X, M; W, Y, Z, tol)

Compute `X = exp(M)` in-place via scaling-and-squaring with Taylor series.
See: Bader et al., "Computing the Matrix Exponential with an Optimized Taylor Polynomial Algorithm", https://arxiv.org/abs/2007.09651.
"""
function matrix_exp!(X::AbstractMatrix{T}, M::AbstractMatrix{T}, W::AbstractMatrix{T} = similar(X), Y::AbstractMatrix{T} = similar(X), Z::AbstractMatrix{T} = similar(X); tol::Real = eps(real(T))) where {T}
    n = LinearAlgebra.checksquare(M)
    @assert size(X) == size(W) == size(Y) == size(Z) == (n, n) "Buffer shapes must match input size $(size(M))"
    nrm = maximum(sum(abs, M; dims = 1)) # ||M||_1
    s = nrm > 0 ? max(0, Base.ceil(Int, log2(2 * nrm))) : 0
    σ = ldexp(one(real(T)), -s)
    W .= M .* σ
    fill!(X, zero(T))
    diag_view(X) .= one(T)
    term_bound = nrm * σ
    k = 2
    Ycur = W
    Ytmp = Y
    while term_bound > tol
        X .+= Ycur
        mul!(Ytmp, W, Ycur, inv(T(k)), zero(T))
        term_bound *= nrm * σ / k
        Ycur, Ytmp = Ytmp, Ytmp === Y ? Z : Y
        k += 1
    end
    Xcur = X
    Xtmp = Ytmp
    for _ in 1:s
        mul!(Xtmp, Xcur, Xcur)
        Xcur, Xtmp = Xtmp, Xcur
    end
    Xcur === X || copyto!(X, Xcur)
    return X
end

"""
    matrix_exp(M::AbstractMatrix)

Non-mutating matrix exponential; see [`matrix_exp!`](@ref).
"""
matrix_exp(M::AbstractMatrix) = matrix_exp!(similar(M), M)

function ChainRules.rrule(::typeof(matrix_exp), M::AbstractMatrix)
    Y = matrix_exp(M)
    function ∇matrix_exp(Ȳ)
        Δ = ChainRules.unthunk(Ȳ)
        # The Fréchet derivative is linear in Δ; normalize its block before exponentiation.
        scale = maximum(abs, Δ)
        iszero(scale) && return (NoTangent(), zeros_similar(M))
        isfinite(scale) || return (NoTangent(), fill!(similar(M), oftype(scale, NaN)))
        n = size(M, 1)
        B = zeros_similar(M, 2n, 2n)
        @views B[1:n, 1:n] .= M'
        @views B[1:n, (n+1):(2n)] .= Δ ./ scale
        @views B[(n+1):(2n), (n+1):(2n)] .= M'
        E = matrix_exp(B)
        M̄ = E[1:n, (n+1):(2n)] .* scale
        return (NoTangent(), M̄)
    end
    return Y, ∇matrix_exp
end

function spectral_trust_region(H::AbstractMatrix{T}, g::AbstractVector{T}, Δ::T; maxiters = 0) where {T}
    λ, Q = LinearAlgebra.eigen(Hermitian(H))
    λ₁ = first(λ)
    Qᵀg = Q'g
    if λ₁ > 0
        # Check Newton step for feasibility
        p = Q * (Qᵀg ./ λ)
        if norm(p) < Δ
            return p # Newton step is small enough
        end
    end

    # Trust region: limit the Newton step to Δ by requiring λ₁ ≥ ||g|| / Δ
    λ₀ = max(norm(g) / Δ - λ₁, 0) # λ₀ ≥ -λ₁
    p = Q * (Qᵀg ./ (λ .+ λ₀))
    np = norm(p)

    # Refine damping parameter using algorithm 4.3 from Nocedal & Wright (2006):
    # (Nocedal & Wright, 2006)
    p_best, np_best = p, np
    for i in 1:maxiters
        # Newton's method update for λ, for the secular equation φ(λ) = 1/||p(λ)|| - 1/Δ = 0
        F = LinearAlgebra.cholesky(Hermitian(H + λ₀ * LinearAlgebra.I); check = false)
        if i > 1
            p = F \ g
            np = norm(p)
        end
        if abs(np - Δ) < abs(np_best - Δ)
            p_best, np_best = p, np
        end
        if abs(np_best - Δ) < √eps(T) * max(Δ, one(T))
            return p_best
        end
        q = F.U' \ p
        nq = norm(q)
        λ₀ = λ₀ + (np / nq)^2 * (np - Δ) / Δ
        if λ₀ < -λ₁ + eps(T)
            return p_best # Newton iterations diverged; return safe initial guess
        end
    end

    return p_best
end

symmetrize!(x::AbstractMatrix) = x .= (x + x') ./ 2

"""
    batched_diag_view(A::AbstractArray, k::Int = 0)

View the diagonal of a batch of matrices `A` with optional offset `k`.
"""
function batched_diag_view(A::AbstractArray, k::Int = 0)
    Idiag = LinearAlgebra.diagind(size(A, 1), size(A, 2), k)
    return reshape(view(reshape(A, size(A, 1) * size(A, 2), :), Idiag, :), size(A, 1) - abs(k), size(A)[3:end]...)
end
diag_view(A::AbstractMatrix, k::Int = 0) = view(A, LinearAlgebra.diagind(A, k))

batched_diag_view(A::AbstractMatrix, k::Int = 0) = diag_view(A, k)
batched_diag_view(A::AbstractVector, k::Int = 0) = error("Cannot take the diagonal of a vector")

"""
    batched_diag!(A::AbstractArray, d::Union{AbstractArray, Number}, k::Int = 0)

Set the diagonal of a batch of matrices `A` with optional offset `k` to `d`.
"""
function batched_diag!(A::AbstractArray, d::Union{AbstractArray, Number}, k::Int = 0)
    @assert size(A, 1) == size(A, 2) && (d isa Number || size(d, 1) + abs(k) == size(A, 1)) "Dimension mismatch; got size(A) = $(size(A)) and size(d) = $(size(d))"
    batched_diag_view(A, k) .= d
    return A
end
diag!(A::AbstractMatrix, d::Union{Number, AbstractVector}, k::Int = 0) = batched_diag!(A, d, k)

function batched_diag_add!(A::AbstractArray, d::Union{AbstractArray, Number}, k::Int = 0)
    batched_diag_view(A, k) .+= d
    return A
end

# In-place square solves for the low-rank PSGD preconditioner.
function batched_ldiv!(A::AbstractArray{T, 3}, b::AbstractMatrix{T}) where {T}
    batched_ldiv!(A, reshape(b, size(b, 1), 1, size(b, 2)))
    return b
end

function batched_ldiv!(A::AbstractArray{T, 3}, B::AbstractArray{T, 3}) where {T}
    @assert size(A, 1) == size(A, 2) == size(B, 1) && size(A, 3) == size(B, 3)
    Threads.@threads for k in axes(A, 3)
        @views LinearAlgebra.ldiv!(LinearAlgebra.qr!(A[:, :, k]), B[:, :, k])
    end
    return B
end

function batched_ldiv!(A::CuArray{T, 3}, B::CuArray{T, 3}) where {T <: Union{Float32, Float64}}
    @assert size(A, 1) == size(A, 2) == size(B, 1) && size(A, 3) == size(B, 3)
    matrices = [view(A, :, :, k) for k in axes(A, 3)]
    rhs = [view(B, :, :, k) for k in axes(B, 3)]
    CUDA.CUBLAS.gels_batched!('N', matrices, rhs)
    return B
end

#### Index streams

# Infinite index streams into `1:nrow`, consumed `chunksize` at a time; `copy` continues the stream

struct ContiguousIndices{R <: AbstractRNG}
    rng::R
    nrow::Int
end

struct UniformIndices{R <: AbstractRNG}
    rng::R
    nrow::Int
end

# Contiguous source blocks drawn without replacement, including the final partial block.
struct BlockMixIndices{R <: AbstractRNG}
    rng::R
    nrow::Int
    blocklen::Int # contiguous rows per source block
    nmix::Int # source blocks combined into one mixed pool
    order::Vector{Int}
    pos::Base.RefValue{Int}
end

function BlockMixIndices(rng::AbstractRNG, nrow::Int, blocklen::Int, nmix::Int)
    @assert nrow > 0 && blocklen > 0 && nmix > 0 "Block sampling requires positive sizes"
    nblocks = cld(nrow, blocklen)
    return BlockMixIndices(rng, nrow, blocklen, min(nmix, nblocks), randperm(rng, nblocks), Ref(1))
end

Base.copy(s::ContiguousIndices) = ContiguousIndices(copy(s.rng), s.nrow)
Base.copy(s::BlockMixIndices) = BlockMixIndices(copy(s.rng), s.nrow, s.blocklen, s.nmix, copy(s.order), Ref(s.pos[]))
Base.copy(s::UniformIndices) = UniformIndices(copy(s.rng), s.nrow)

# Draw a contiguous run, which loads much faster from a memory-mapped table
next_indices!(s::ContiguousIndices, chunksize::Int) = rand(s.rng, 0:(s.nrow-chunksize)) .+ (1:chunksize)

# Next contiguous source block, without replacement within a sampling cycle
function next_block!(s::BlockMixIndices)
    if s.pos[] > length(s.order)
        randperm!(s.rng, s.order)
        s.pos[] = 1
    end
    block = s.order[s.pos[]]
    s.pos[] += 1
    return ((block-1)*s.blocklen+1):min(block*s.blocklen, s.nrow)
end

# Do not mix across cycle boundaries: each row must be emitted before the next cycle begins.
function next_blocks!(s::BlockMixIndices)
    remaining = s.pos[] > length(s.order) ? length(s.order) : length(s.order) - s.pos[] + 1
    return [next_block!(s) for _ in 1:min(s.nmix, remaining)]
end

function next_indices!(s::UniformIndices, chunksize::Int)
    I = zeros(Int, chunksize)
    sample!(s.rng, 1:s.nrow, I; replace = false, ordered = false)
    return sort!(I) # sorted indices load faster
end

# Per-voxel PI adaptation of the log step size during sampler warmup.
struct AcceptanceController{T, A}
    setpoint::T
    Kp::T
    Ki::T
    lrmin::T
    lrmax::T
    lr::A
    total_error::A
end
function AcceptanceController(lr; setpoint, Kp, Ki, lrmin, lrmax)
    return AcceptanceController(setpoint, Kp, Ki, lrmin, lrmax, lr, zeros_similar(lr))
end

function update!(ctrl::AcceptanceController, acceptance_ratio)
    error = ctrl.setpoint .- acceptance_ratio
    Δloglr = ctrl.Kp .* error
    if ctrl.Ki > 0
        integral = ctrl.total_error .+= error
        Δloglr .+= ctrl.Ki .* integral
    end
    ctrl.lr .= clamp.(ctrl.lr .* exp.(.-Δloglr), ctrl.lrmin, ctrl.lrmax)
    return ctrl.lr
end

"""
OnlineMetropolisSampler
"""
@with_kw_noshow struct OnlineMetropolisSampler{T}
    θ                      :: Array{T, 3} # parameter values
    ntheta::Int            = size(θ, 1) # number of parameters per data point
    ndata::Int             = size(θ, 2) # number of data points, i.e. each column of θ represents estimates for a separate datum Y
    nsamples::Int          = size(θ, 3) # length of the Metropolis-Hastings MCMC chain which is recorded
    i::Vector{Int}         = ones(Int, ndata) # current sample index in cyclical chain buffer θ
    accept::Array{Bool, 3} = zeros(Bool, 1, ndata, nsamples) # records whether proposal was accepted or not
    neglogPXθ::Array{T, 3} = fill(T(Inf), 1, ndata, nsamples) # negative log likelihoods; initialize with Inf to guarantee acceptance of first sample
    neglogPθ::Array{T, 3}  = zeros(T, 1, ndata, nsamples) # negative log priors; initialization is moot due to neglogPXθ initialized to Inf
    neglogQθ::Array{T, 3}  = zeros(T, 1, ndata, nsamples) # independent proposal Q(θ|θ′) ≡ Q(θ); initial value is irrelevant since neglogPXθ starts at Inf
end
Base.show(io::IO, s::OnlineMetropolisSampler{T}) where {T} = print(io, "OnlineMetropolisSampler{$(T)}(ntheta = $(s.ntheta), ndata = $(s.ndata), nsamples = $(s.nsamples))")

buffer_index(s::OnlineMetropolisSampler, j::Int) = CartesianIndex(j, mod1(s.i[j], s.nsamples))
random_index(s::OnlineMetropolisSampler, j::Int) = CartesianIndex(j, rand(1:s.nsamples))
random_indices(s::OnlineMetropolisSampler, J = 1:s.ndata) = random_index.((s,), J)
function Random.rand(s::OnlineMetropolisSampler, J = 1:s.ndata)
    idx = random_indices(s, J)
    return s.θ[:, idx], s.neglogPXθ[:, idx], s.neglogPθ[:, idx], s.neglogQθ[:, idx]
end

# c.f. https://stats.stackexchange.com/a/163790
function update!(s::OnlineMetropolisSampler, θ′::AbstractMatrix, neglogPXθ′::AbstractMatrix, neglogPθ′::AbstractMatrix, neglogQθ′::AbstractMatrix, J = 1:s.ndata)
    @assert size(θ′, 1) == s.ntheta
    @assert size(θ′, 2) == size(neglogPXθ′, 2) == size(neglogPθ′, 2) == size(neglogQθ′, 2)
    @assert size(neglogPXθ′, 1) == size(neglogPθ′, 1) == size(neglogQθ′, 1) == 1

    Threads.@threads for j in eachindex(J)
        @inbounds begin
            col = J[j]
            curr = buffer_index(s, col)
            s.i[col] += 1
            next = buffer_index(s, col)

            # Metropolis-Hastings acceptance ratio:
            #        α = min(1, (PXθ′ * Pθ′ * Qθ) / (PXθ * Pθ * Qθ′))
            # ==> logα = min(0, logPXθ′ + logPθ′ + logQθ - logPXθ - logPθ - logQθ′)
            logα   = min(0, s.neglogPXθ[1, curr] + s.neglogPθ[1, curr] + neglogQθ′[1, j] - neglogPXθ′[1, j] - neglogPθ′[1, j] - s.neglogQθ[1, curr])
            accept = logα > log1p(-rand())

            # Keep accepted points, otherwise the current points
            s.accept[1, next] = accept
            @inbounds for i in 1:size(θ′, 1)
                s.θ[i, next] = accept ? θ′[i, j] : s.θ[i, curr]
            end
            s.neglogPXθ[1, next] = accept ? neglogPXθ′[1, j] : s.neglogPXθ[1, curr]
            s.neglogPθ[1, next] = accept ? neglogPθ′[1, j] : s.neglogPθ[1, curr]
            s.neglogQθ[1, next] = accept ? neglogQθ′[1, j] : s.neglogQθ[1, curr]
        end
    end
end

#### Autocorrelation utils

"""
    autocorr_fft(x::AbstractArray; dims::Int)

Compute normalized autocorrelation function using the fast Fourier transform and the convolution theorem.
Samples are stacked along dimension `dims`. Returns ρ(τ) for τ = 1, ..., N-1, with ρ(0) = 1 omitted.
"""
function autocorr_fft(x::AbstractArray; dims::Int)
    # Pad to at least 2N-1 to prevent wrap-around effects from the circular convolution
    N = size(x, dims)
    Npad = nextpow(2, 2N - 1)
    x_padded = zeros_similar(x, ntuple(i -> i == dims ? Npad : size(x, i), ndims(x)))
    view(x_padded, ntuple(i -> i == dims ? (1:N) : Colon(), ndims(x))...) .= x .- mean(x; dims)

    # Convolution theorem: autocorrelation is the inverse FFT of the power spectral density
    x̂ = rfft(x_padded, dims)
    S = abs2.(x̂) # power spectral density
    ρ = irfft(S, Npad, dims)
    ρ = view(ρ, ntuple(i -> i == dims ? (2:N) : Colon(), ndims(x))...) ./ # normalize and truncate to the original length N
        view(ρ, ntuple(i -> i == dims ? (1:1) : Colon(), ndims(x))...)

    return ρ
end

# ρ(τ)/ρ(0) ≈ mean over j of ∏ᵢ₌₁^τ 𝟙_reject(i+j), from runs of rejections across chains (Albergo et al., 2019)
function autocorr_from_acceptance(acc::AbstractVecOrMat, τ::Int = size(acc, 1))
    # acc[i] = 𝟙_{i-1 -> i accepted}, so it has size N - 1 in the first dimension
    N = size(acc, 1) + 1
    counts = count_runs(acc, τ)
    return counts ./ (N .- (1:τ))
end

# τ_int sums ρ(τ) up to the smallest M with M ≥ C τ_int(M), C ≥ 6 (Sokal, 1997)
function autocorr_time(ρ::AbstractArray, C = 6.0; dims::Int)
    # Create mask that is true for τ <= M
    ρ_mask = (1:size(ρ, dims)) .<= ofeltypefloat(ρ, C) .* (1 // 2 .+ cumsum(ρ; dims))
    ρ_mask = accumulate!(&, ρ_mask, ρ_mask; dims, init = true) # ensures that ρ_mask[τ] = false for τ > M

    # Compute integrated autocorrelation time: τ_int = 1/2 + Σ_{τ=1}^{M} ρ(τ)/ρ(0)
    τ_int = 1 // 2 .+ sum(ρ_mask .* ρ; dims)
    return τ_int
end

# Fast O(N)
function count_runs!(work::NamedTuple, acc::AbstractVector, τ::Int)
    @assert τ <= size(acc, 1)
    (; counts, freq, tmp1, tmp2) = work
    N = length(acc)
    Lmax = τ
    i = 1
    @inbounds while i <= N
        if acc[i] == 0
            j = i + 1
            while j <= N && acc[j] == 0
                j += 1
            end
            L = j - i
            freq[L] += 1
            Lmax = max(Lmax, L)
            i = j
        else
            i += 1
        end
    end

    @inbounds for t in Lmax:-1:1
        tmp1[t] = tmp1[t+1] + freq[t]
        tmp2[t] = tmp2[t+1] + freq[t] * t
    end

    @inbounds for t in 1:τ
        counts[t] = tmp2[t] - (t - 1) * tmp1[t]
    end

    return counts
end

function count_runs(acc::AbstractMatrix, τ::Int)
    @assert τ <= size(acc, 1)
    counts = zeros(Int, τ, size(acc, 2))
    @sync @views for J in DECAES.split_indices(; length = size(acc, 2), minchunksize = 1024, maxpartitions = Threads.nthreads())
        work = count_runs_workspace(size(acc, 1), τ)
        Threads.@spawn @inbounds for j in J
            reset_count_runs_workspace!(work)
            count_runs!(work, acc[:, j], τ)
            copyto!(counts[:, j], work.counts)
        end
    end
    return counts
end
count_runs(acc::AbstractVector, τ::Int) = count_runs!(count_runs_workspace(length(acc), τ), acc, τ)

function count_runs_workspace(n::Int, τ::Int)
    counts = zeros(Int, τ)
    freq = zeros(Int, n)
    tmp1 = zeros(Int, n + 1)
    tmp2 = zeros(Int, n + 1)
    return (; counts, freq, tmp1, tmp2)
end

function reset_count_runs_workspace!(work::NamedTuple)
    (; counts, freq, tmp1, tmp2) = work
    fill!(counts, 0)
    fill!(freq, 0)
    fill!(tmp1, 0)
    fill!(tmp2, 0)
    return work
end

"""
    broadcast!!(f, x)

Apply `f` to each element of `x` in-place if x isa an AbstractArray.
"""
broadcast!!(f, x::AbstractArray, y...) = broadcast!(f, x, x, y...)
broadcast!!(f, x::StaticArray, y...) = broadcast(f, x, y...)
broadcast!!(f, x::Number, y...) = f(x, y...)

"""
    rms(x; dims = :) = sqrt.(mean(abs2, x; dims))
"""
rms(x; dims = :) = broadcast!!(sqrt, mean(abs2, x; dims))

unitrms(x; dims = :) = x ./ sqrt.(mean(abs2, x; dims))

"""
    unitsum(x; dims = :) = x ./ sum(x; dims)
"""
unitsum(x; dims = :) = unitsum!(copy(x); dims)
unitsum!(x; dims = :) = broadcast!!(/, x, sum(x; dims))
unitsum(; dims = :) = x -> unitsum(x; dims)
unitsum!(; dims = :) = x -> unitsum!(x; dims)

"""
    snr(x, n)

Signal-to-noise ratio of the signal `x` relative to the noise `n`.
"""
snr(x, n; dims = 1) = 10 .* log10.(sum(abs2, x; dims) ./ sum(abs2, n; dims))

"""
    SymmetricRealNDFT

Plan for exact trigonometric interpolation of real signals with known
even or odd symmetry, sampled on a uniform grid.

Even symmetry reconstructs f(t) via the cosine series:

```math
f(t) = c_0 + 2 \\sum_{k=1}^{N/2-1} c_k \\cos(kt) + c_{N/2} \\cos(Nt/2)
```

Odd symmetry reconstructs f(t) via the sine series:

```math
f(t) = \\sum_{k=1}^{K} s_k \\sin(kt)
```

Complexity: O(CMK) for evaluation of C channels at M points with K coefficients.
"""
struct SymmetricRealNDFT{T, D, Sym, V <: AbstractVector{T}, M <: AbstractMatrix{T}}
    fsize::NTuple{D, Int} # size of the function f(t) being approximated
    sym::Val{Sym} # symmetry type (:even or :odd)
    c0::V # DC terms (C x 1 vector)
    ck::M # scaled cosine/sine coeffs (C x (K + 1) matrix)
end

Base.show(io::IO, plan::SymmetricRealNDFT{T, D, Sym}) where {T, D, Sym} = print(io, "SymmetricRealNDFT{", T, ", ", D, ", ", Sym, "}(fsize=", plan.fsize, ", nsamples=", 2 * size(plan.ck, 2), ")")

function Base.chop(plan::SymmetricRealNDFT{T}; rtol::Real = 25 * eps(T), atol::Real = zero(T)) where {T}
    (; fsize, sym, c0, ck) = plan
    thresh = max(maximum(abs, ck) * T(rtol), T(atol))
    jlast = findlast(vec(any(c -> abs(c) > T(thresh), ck; dims = 1:(ndims(ck)-1))))
    jlast !== nothing && (ck = ck[.., 1:jlast]) # truncate to keep only the significant coefficients
    @. ck = ifelse(abs(ck) < thresh, zero(T), ck) # zero out small coefficients
    return SymmetricRealNDFT(fsize, sym, c0, ck)
end

struct SymmetricRealNDFTDerivative{T, D, Sym, P <: SymmetricRealNDFT{T, D, Sym}}
    plan::P
end
Base.adjoint(plan::SymmetricRealNDFT) = SymmetricRealNDFTDerivative(plan)

function SymmetricRealNDFT(f::AbstractArray{T, D}, sym::Val{Sym}) where {T, D, Sym}
    @assert Sym === :even || Sym === :odd "symmetry must be :even or :odd"

    fsize..., B = size(f)
    C = prod(fsize)
    @assert C >= 1 "At least one channel is required."

    N, K = iseven(B) ? (B, B ÷ 2 - 1) : Sym === :even ? (2 * (B - 1), B - 2) : (2 * (B + 1), B)
    @assert N >= 4 && mod(N, 4) == 0 "N must be even for valid Nyquist definition and at least 4 samples are required."

    F = if iseven(B)
        # rfft returns N/2 + 1 = K + 2 coefficients for k = 0..N/2 = 0..K+1
        rfft(reshape(f, C, N), 2) ./ N
    elseif Sym === :even
        # compute DCT-I via FFT by reflecting samples: [f, reverse(f)]
        rfft(hcat(reshape(f, C, B), reshape(f[.., (B-1):-1:2], C, B - 2)), 2) ./ N
    else # Sym === :odd
        # compute DST-I via FFT by reflecting samples: [0, f, 0, -reverse(f)]
        rfft(hcat(zeros_similar(f, C), reshape(f, C, B), zeros_similar(f, C), -reshape(f[.., B:-1:1], C, B)), 2) ./ N
    end

    if Sym === :even
        # even: use real parts -> cosine coefficients = 2 * Re(F_k) for k = 1..K+1
        ℜ𝔢F = real.(F) # C × (K + 2)
        ℜ𝔢F[:, 2:(K+1)] .*= 2 # double the cosine coefficients k = 1..K
        c0 = ℜ𝔢F[:, 1] # extract DC coefficient (c0: C x 1 vector)
        ck = ℜ𝔢F[:, 2:(K+2)] # extract remaining coefficients k = 1..K+1 (ck: C x (K+1) matrix)
        return SymmetricRealNDFT(fsize, sym, c0, ck)
    else
        # odd: use imaginary parts -> sine coefficients = -2 * Im(F_m) for m=1..K
        ℑ𝔪F = imag.(F) # C × (K + 2)
        ℑ𝔪F[:, 2:(K+1)] .*= -2 # double and negate the sine coefficients m = 1..K
        ℑ𝔪F[:, K+2] .= 0 # Nyquist column K+2 is zero for odd symmetry
        c0 = zeros_similar(ℑ𝔪F, 0) # DC component is zero and unused
        ck = ℑ𝔪F[:, 2:(K+2)] # extract remaining coefficients m = 1..K+1 (ck: C x (K+1) matrix)
        return SymmetricRealNDFT(fsize, sym, c0, ck)
    end
end

function (plan::SymmetricRealNDFT{T, D, :even})(t::Union{Tf, AbstractVector{Tf}}) where {T, D, Tf <: Union{T, ForwardDiff.Dual{<:Any, T}}}
    (; fsize, c0, ck) = plan
    𝒞 = cos.((one(T):T(size(ck, 2))) .* t') # construct cosine matrix 𝒞 (K+1 x M)
    Y = repeat(c0, 1, length(t)) .+ ck * 𝒞 # add DC terms: Y = c0 + ck * 𝒞 where c0 (C x 1) x 𝒞 (K+1 x M) -> (C x M)
    return reshape(Y, fsize..., size(t)...)
end

function (plan::SymmetricRealNDFT{T, D, :odd})(t::Union{Tf, AbstractVector{Tf}}) where {T, D, Tf <: Union{T, ForwardDiff.Dual{<:Any, T}}}
    (; fsize, ck) = plan
    𝒮 = sin.((one(T):T(size(ck, 2))) .* t') # construct sine matrix 𝒮 (K+1 x M)
    Y = ck * 𝒮 # Y = ck * 𝒮 where ck (C x K+1) x 𝒮 (K+1 x M) -> (C x M)
    return reshape(Y, fsize..., size(t)...)
end

function (∂plan::SymmetricRealNDFTDerivative{T, D, :even})(t::Union{Tf, AbstractVector{Tf}}) where {T, D, Tf <: Union{T, ForwardDiff.Dual{<:Any, T}}}
    (; fsize, ck) = ∂plan.plan
    ks = one(T):T(size(ck, 2)) # vector of frequencies k = 1..K+1
    ∂𝒞 = (-ks) .* sin.(ks .* t') # construct cosine derivative matrix ∂𝒞 (K+1 x M)
    Y = ck * ∂𝒞 # Y = ck * ∂𝒞 where ck (C x K+1) x ∂𝒞 (K+1 x M) -> (C x M)
    return reshape(Y, fsize..., size(t)...)
end

function (∂plan::SymmetricRealNDFTDerivative{T, D, :odd})(t::Union{Tf, AbstractVector{Tf}}) where {T, D, Tf <: Union{T, ForwardDiff.Dual{<:Any, T}}}
    (; fsize, ck) = ∂plan.plan
    ks = one(T):T(size(ck, 2)) # vector of frequencies k = 1..K+1
    ∂𝒮 = ks .* cos.(ks .* t') # construct sine derivative matrix ∂𝒮 (K+1 x M)
    Y = ck * ∂𝒮 # Y = ck * ∂𝒮 where ck (C x K+1) x ∂𝒮 (K+1 x M) -> (C x M)
    return reshape(Y, fsize..., size(t)...)
end

#### Resampling distributions between grid resolutions

# Column-stochastic lift L (n′ x n) by linear interpolation on the normalized log-T2 grid; L = I when n′ = n
function linear_lift_matrix(n::Int, n′::Int)
    n == n′ && return Matrix{Float64}(LinearAlgebra.I(n))
    t  = (0:(n-1)) ./ (n - 1)
    t′ = (0:(n′-1)) ./ (n′ - 1)
    Δ  = inv(n - 1) # input grid pitch
    L  = max.(0, 1 .- abs.(t′ .- t') ./ Δ)
    return L ./ sum(L; dims = 1)
end

function lift_resample(logx̄::AbstractMatrix, n′::Int)
    n = size(logx̄, 1)
    x̄ = exp.(logx̄)
    n == n′ && return x̄
    L = @ignore_derivatives arr_similar(logx̄, linear_lift_matrix(n, n′))
    return L * x̄
end

#### Polynomial field fitting

"""
    PolynomialField

Least-squares polynomial fit of a scalar field f(r⃗) over a point cloud in D dimensions.

Given sample points {r⃗ᵢ} and corresponding values {f(r⃗ᵢ)}, constructs a polynomial approximation

```math
f(r⃗) ≈ φ⃗(r̃)ᵀ w⃗
```

where r̃ = (r⃗ - shift) ./ scale ∈ [-1,1]ᴰ are normalized coordinates and φ⃗(r̃) is the monomial
basis vector up to the specified total degree.

Construction uses QR factorization for numerical stability. Evaluation at new points is
GPU-friendly (broadcast-based monomial evaluation over vectors of `SVector`) and differentiable
with respect to the stored weights.

Number of basis functions: M = binomial(degree + D, D); e.g. D=3: degree=0 → 1, degree=1 → 4, degree=2 → 10, degree=3 → 20.
"""
struct PolynomialField{T, D, E <: AbstractVector{NTuple{D, Int}}, W <: AbstractVector{T}}
    degree::Int # total polynomial degree
    shift::SVector{D, T} # normalization center: r̃ = (r⃗ - shift) ./ scale
    scale::SVector{D, T} # normalization half-width
    exponents::E # length-M vector of monomial multi-indices
    weights::W # M-vector of polynomial coefficients
end

Base.show(io::IO, fit::PolynomialField{T, D}) where {T, D} = print(io, "PolynomialField{$T, $D}(degree=$(fit.degree), nbasis=$(length(fit.weights)))")

# Constructor: fit polynomial to D × N coordinate matrix and N-vector of values
function PolynomialField(points::AbstractVector{SVector{D, T}}, values::AbstractVector{T}; degree::Int) where {D, T}
    N = length(points)
    @assert length(values) == N "Expected $N values, got $(length(values))"
    @assert degree >= 0 "Polynomial degree must be non-negative"

    # Normalization: map bounding box to [-1, 1]^D
    lo = reduce(broadcast_function(min), points; init = SVector{D, T}(ntuple(_ -> T(+Inf), D)))
    hi = reduce(broadcast_function(max), points; init = SVector{D, T}(ntuple(_ -> T(-Inf), D)))
    center = (lo .+ hi) ./ 2
    halfwidth = (hi .- lo) ./ 2
    shift = SVector{D, T}(center)
    scale = SVector{D, T}(halfwidth)

    # Build design matrix on normalized coordinates
    exponents = device_similar(values, _monomial_exponents(Val(D), degree))
    M = length(exponents)
    @assert N >= M "Underdetermined: need ≥$M points for degree-$degree in $(D)D (got $N)"

    # Solve via QR for numerical stability
    weights = LinearAlgebra.qr(_polynomial_design_matrix(points, shift, scale, exponents)) \ values
    return PolynomialField(degree, shift, scale, exponents, weights)
end

const IntegerTupleLike{D} = Union{CartesianIndex{D}, NTuple{D, <:Integer}}

function PolynomialField(::Val{D}, points::AbstractMatrix{T}, values::AbstractVector{T}; degree::Int) where {D, T}
    return PolynomialField(_standardize(T, Val(D), points), values; degree)
end

function PolynomialField(points::AbstractArray{TI}, values::AbstractArray{T}; degree::Int) where {D, T, TI <: IntegerTupleLike{D}}
    @assert size(points) == size(values)
    return PolynomialField(vec(_standardize(T, Val(D), points)), vec(values); degree)
end

# Evaluate polynomial field at vector of SVector{D}. Differentiable w.r.t. fit.weights.
function (fit::PolynomialField{T, D})(points::Union{SVector{D, T}, AbstractVector{SVector{D, T}}}) where {T, D}
    Φ = polynomial_design_matrix(fit, points) # N × M
    w = fit.weights # M-vector
    return Φ * w # N-vector
end
(fit::PolynomialField{T, D})(points::AbstractArray{SVector{D, T}}) where {T, D} = reshape(fit(vec(points)), size(points))
(fit::PolynomialField{T, D})(points) where {T, D} = fit(_standardize(T, Val(D), points))

# Public API: design matrix Φ (N × M) at given points (vector of SVector{D}).
function polynomial_design_matrix(fit::PolynomialField{T, D}, points::Union{SVector{D, T}, AbstractVector{SVector{D, T}}}) where {T, D}
    return _polynomial_design_matrix(points, fit.shift, fit.scale, fit.exponents) # N × M matrix
end
polynomial_design_matrix(fit::PolynomialField{T, D}, points) where {T, D} = polynomial_design_matrix(fit, _standardize(T, Val(D), points))

# Build design matrix Φ (N × M) via broadcasting monomial evaluation.
# points: length-N vector of SVector{D}; exponents: length-M vector of NTuple{D, Int}.
function _polynomial_design_matrix(points::AbstractVector{SVector{D, T}}, shift::SVector{D, T}, scale::SVector{D, T}, exponents::AbstractVector{NTuple{D, Int}}) where {D, T}
    return _evaluate_monomial.(_box_normalize.((shift,), (scale,), points), reshape(exponents, 1, :)) # N × M matrix
end
function _polynomial_design_matrix(point::SVector{D, T}, shift::SVector{D, T}, scale::SVector{D, T}, exponents::AbstractVector{NTuple{D, Int}}) where {D, T}
    return _evaluate_monomial.((_box_normalize(shift, scale, point),), exponents)' # length-M adjoint vector
end

# Standardize coordinates to vector of SVector{D, T}
function _standardize(::Type{T}, ::Val{D}, coords::AbstractMatrix{T}) where {T, D}
    @assert size(coords, 1) == D "Expected $D-dimensional coordinates, got $(size(coords, 1))"
    return vec(reinterpret(SVector{D, T}, coords))
end
_standardize(::Type{T}, ::Val{D}, points::AbstractArray{TI}) where {T, D, TI <: IntegerTupleLike{D}} = _standardize.(T, Val(D), points)
@inline _standardize(::Type{T}, ::Val{D}, point::CartesianIndex{D}) where {T, D} = SVector{D, T}(Tuple(point))
@inline _standardize(::Type{T}, ::Val{D}, point::NTuple{D, Ti}) where {T, D, Ti <: Integer} = SVector{D, T}(point)

# Normalize points to [-1, 1]^D
@inline _box_normalize(shift::SVector{D}, scale::SVector{D}, point::SVector{D}) where {D} = @. (point - shift) / scale

# Evaluate single monomial: prod_d r[d]^e[d]. Fully unrolled at compile time for small D.
@inline _evaluate_monomial(r::SVector{D}, e::NTuple{D, Int}) where {D} = prod(r .^ e)

# Enumerate monomial multi-indices {(k₁,...,k_D) : k₁+...+k_D ≤ maxdeg} in lexicographic order
# (same order as nested loops over k₁, then k₂, … with kⱼ ≤ maxdeg − ∑_{i<j} kᵢ).
# Returns a length-M vector of D-tuples, M = binomial(maxdeg + D, D).
# `D` is a `Val` so tuple lengths are concrete; `maxdeg` is an Int
function _monomial_exponents(::Val{D}, maxdeg::Int) where {D}
    M = binomial(maxdeg + D, D)
    exps = Vector{NTuple{D, Int}}(undef, M)
    e = zero(MVector{D, Int})
    @inbounds for idx in 1:M
        exps[idx] = Tuple(e)
        idx == M && break
        j = D
        while j >= 1
            prefix_sum = 0
            for n in 1:(j-1)
                prefix_sum += e[n]
            end
            max_ej = maxdeg - prefix_sum
            if e[j] < max_ej
                e[j] += 1
                for n in (j+1):D
                    e[n] = 0
                end
                break
            end
            j -= 1
        end
    end
    return exps
end

"""
    CachedPolynomialField

Cached evaluation of a `PolynomialField` at a fixed set of points.

Stores the QR-factorized design matrix `Q` (materialized as a dense matrix) and reparameterized
weights `w̃ = R * w` so that evaluation is `Q * w̃`. This is equivalent to `Φ * w` but
optimizing `w̃` via gradient descent is well-conditioned since `Q'Q = I`.

Only `w̃` is trainable (via `Optimisers.trainable`), so `Q` is treated as a fixed constant
during optimization.
"""
struct CachedPolynomialField{M, W}
    Φ::M # dense Q factor (N × M), orthonormal columns
    w::W # reparameterized weights w̃ = R * w (M-vector)
end

Base.show(io::IO, cpf::CachedPolynomialField) = print(io, "CachedPolynomialField(npoints=$(size(cpf.Φ, 1)), nbasis=$(size(cpf.Φ, 2)))")

Optimisers.trainable(cpf::CachedPolynomialField) = (; w = cpf.w)

(cpf::CachedPolynomialField)() = cpf.Φ * cpf.w

function CachedPolynomialField(fit::PolynomialField{T, D}, points::AbstractVector{SVector{D, T}}) where {T, D}
    Φ = polynomial_design_matrix(fit, points) # N × M
    F = LinearAlgebra.qr(Φ)
    Q = typeof(Φ)(F.Q) # materialize Q as dense matrix
    w̃ = F.R * fit.weights # reparameterized weights in orthonormal basis
    return CachedPolynomialField(Q, w̃)
end
CachedPolynomialField(fit::PolynomialField{T, D}, points) where {T, D} = CachedPolynomialField(fit, _standardize(T, Val(D), points))

# Flip angles `αs` are in degrees, as elsewhere in the pipeline, and `∇A` is therefore per degree
function ∇EPGBasis_with_primal(αs::Union{T, AbstractVector{T}}, T2s::AbstractVector{T}; ETL::Int, TE = 10e-3, T1 = 1.0) where {T}
    nT2 = length(T2s)
    A = zeros(T, ETL, nT2, size(αs)...)
    ∇A = zeros(T, ETL, nT2, size(αs)...)
    return ∇EPGBasis_with_primal!(∇A, A, αs, T2s; TE, T1)
end

function ∇EPGBasis_with_primal!(∇A::AbstractArray{T}, A::AbstractArray{T}, αs::Union{T, AbstractVector{T}}, T2s::AbstractVector{T}; TE = 10e-3, T1 = 1.0) where {T}
    ETL, nT2, nα = size(A, 1), size(A, 2), size(A, 3)
    @assert size(A) == size(∇A) && nT2 == length(T2s) && nα == length(αs) "Size mismatch"

    # Differentiate EPG impulse response w.r.t. α via ForwardDiff
    opts = Ref(DECAES.EPGConstantFlipAngleOptions(ETL, T(NaN), TE, T(NaN), T1))
    caches = DECAES.EPGWorkCacheDict(opts[])
    function f!(y, x)
        θ = DECAES.restructure(opts[], (; α = deg2rad(x))) # `EPGOptions` stores radians; differentiating through `deg2rad` keeps `∇A` per degree
        DECAES.epg_impulse_response!(y, caches[eltype(y)], θ)
        return y
    end

    # Preallocate buffers for ForwardDiff.derivative!
    Acol, ∇Acol = zeros(T, ETL), zeros(T, ETL)
    res = DiffResults.DiffResult(Acol, ∇Acol)
    cfg = ForwardDiff.DerivativeConfig(f!, Acol, zero(T))
    for (iα, α) in enumerate(αs)
        for (jT2, T2) in enumerate(T2s)
            opts[] = DECAES.restructure(opts[], (; T2))
            ForwardDiff.derivative!(res, f!, Acol, α, cfg)
            A[:, jT2, iα] .= DiffResults.value(res)
            ∇A[:, jT2, iα] .= DiffResults.derivative(res)
        end
    end

    return ∇A, A
end

#### Batched gradients

# See: https://discourse.julialang.org/t/generating-finite-difference-stencils/85876/5
function stencil(x::AbstractVector{<:Real}, x₀::Real, m::Integer)
    ℓ = 0:(length(x)-1)
    m in ℓ || throw(ArgumentError("order $m ∉ $ℓ"))
    A = @. (x' - x₀)^ℓ / factorial(ℓ)
    w = A \ (ℓ .== m) # vector of weights w
    sum(w) == 0 || @warn "Weights do not sum to zero due to numerical error"
    return w
end

function batched_hessian_from_grad(∇f, x::AbstractArray{T}, δ::T = cbrt(eps(T))) where {T}
    n = size(x, 1)
    xs = repeat(reshape(x, n, 1, 1, Base.tail(size(x))...), 1, n, 2)
    xs .+= eye_similar(x, n) .* reshape(arr_similar(x, T[-δ, δ]), 1, 1, :)
    gradients = reshape(∇f(reshape(xs, n, :)), n, n, 2, Base.tail(size(x))...)
    gradients .*= reshape(arr_similar(x, T[-1//2, 1//2]), 1, 1, :)
    H = dropdims(sum(gradients; dims = 3); dims = 3) ./ δ
    perm = ntuple(i -> i == 1 ? 2 : i == 2 ? 1 : i, ndims(H))
    H .= (H .+ permutedims(H, perm)) ./ 2
    return H
end
