#### Chains
const VectorChain = Flux.Chain{Vector{Any}}

@inline TupleChain(layers::T) where {T <: Tuple} = Flux.Chain{T}(layers)
@inline TupleChain(layers::AbstractVector) = TupleChain((layers...,))

#### Params

idset(x::AbstractArray) = IdDict{Any, Nothing}(vec(x) .=> nothing)
isequal_idset(x1::IdDict{<:Any, Nothing}, x2::IdDict{<:Any, Nothing}) = length(x1) == length(x2) && all(haskey(x2, k) for k in keys(x1)) && all(haskey(x1, k) for k in keys(x2))

allnumeric(m; unique = true, exclude = isnumeric) = unique ? filter(exclude, Functors.fcollect(m)) : Functors.fleaves(m; exclude) # `fcollect` returns unique nodes by object id; `fleaves` returns all numeric nodes
alltrainable(m; path = false) = Optimisers.trainables(m; path)

function assert_shared_parameters(model₁, model₂)
    for trainable in [true, false]
        θ₁, θ₂ = trainable ? (alltrainable(model₁), alltrainable(model₂)) : (allnumeric(model₁; unique = false), allnumeric(model₂; unique = false))
        @assert length(θ₁) == length(θ₂) string("Number of ", trainable ? "trainable" : "parameter", " arrays mismatch")
        @assert isequal_idset(idset(θ₁), idset(θ₂)) string(trainable ? "Trainable" : "Parameter", " arrays mismatch")
    end
    return nothing
end

function assert_shared_optimizer_state(model, optimizers; error_on_mismatch = false, verbose = false)
    leafs = filter(l -> l isa Optimisers.Leaf, Functors.fcollect(optimizers)) # `fcollect` returns unique nodes by object id
    params = alltrainable(model)
    if length(leafs) != length(params)
        msg = "Number of optimizer states does not match number of trainable parameters: $(length(leafs)) ≠ $(length(params))"
        error_on_mismatch ? error(msg) : @warn msg
    end

    # Detailed check: for every tied parameter, the optimizer state should also be tied
    cache = IdDict()
    pass = Ref(true)
    optimizers′ = Functors.fmap_with_path(optimizers; exclude = (k, l) -> l isa Optimisers.Leaf, cache = nothing) do k, leaf # `cache = nothing` ensures all `Leaf`s are visited
        param = Functors.getkeypath(model, k)
        if haskey(cache, param)
            k_cache, leaf_cache = cache[param]
            if leaf !== leaf_cache
                if leaf != leaf_cache
                    @error "Optimizer state mismatch:" k_curr = k k_cache
                    pass[] = false
                    return leaf
                else
                    @warn "Replacing optimizer state with cached state:" k_curr = k k_cache
                    return leaf_cache
                end
            else
                verbose && @info "Optimizer state matched:" k_curr = k k_cache
                return leaf
            end
        else
            cache[param] = k => leaf
            return leaf
        end
    end
    error_on_mismatch && !pass[] && error("Optimizer state mismatch")

    return optimizers′
end

# Walk trainable parameters of `x` together with an auxiliary structure `aux`
#   See: https://github.com/FluxML/Optimisers.jl/issues/143#issuecomment-1608487174
fmap_trainable_biwalk(f, x, aux; exclude = isnumeric, kwargs...) = fmap(f, x, aux; exclude, walk = Optimisers._Trainable_biwalk(), kwargs...)

optimizers_trainmode!(model, optimizers) = fmap_trainable_biwalk(optimizers_trainmode!, model, optimizers) # recurse
optimizers_trainmode!(x, l::Optimisers.Leaf) = l.frozen ? x : optimizers_trainmode!(l.rule, l.state, x) # base case
optimizers_trainmode!(::AbstractRule, state, x) = x # default to no-op
# apply to all rules in chain, in order
function optimizers_trainmode!(rule::Optimisers.OptimiserChain, state, x)
    foreach((r, s) -> optimizers_trainmode!(r, s, x), rule.opts, state)
    return x
end

optimizers_testmode!(model, optimizers) = fmap_trainable_biwalk(optimizers_testmode!, model, optimizers) # recurse
optimizers_testmode!(x, l::Optimisers.Leaf) = l.frozen ? x : optimizers_testmode!(l.rule, l.state, x) # base case
optimizers_testmode!(::AbstractRule, state, x) = x # default to no-op
# apply to all rules in chain, in order
function optimizers_testmode!(rule::Optimisers.OptimiserChain, state, x)
    foreach((r, s) -> optimizers_testmode!(r, s, x), rule.opts, state)
    return x
end

function all_trainmode!(model, optimizers)
    Flux.trainmode!(model)
    optimizers_trainmode!(model, optimizers)
    return model
end

function all_testmode!(model, optimizers)
    Flux.testmode!(model)
    optimizers_testmode!(model, optimizers)
    return model
end

function with_trainmode!(f, model)
    try
        Flux.trainmode!(model)
        f(model)
    finally
        Flux.testmode!(model)
    end
end

function with_testmode!(f, model)
    try
        Flux.testmode!(model)
        f(model)
    finally
        Flux.trainmode!(model)
    end
end

function with_trainmode!(f, model, optimizer)
    try
        all_trainmode!(model, optimizer)
        f(model, optimizer)
    finally
        all_testmode!(model, optimizer)
    end
end

function with_testmode!(f, model, optimizer)
    try
        all_testmode!(model, optimizer)
        f(model, optimizer)
    finally
        all_trainmode!(model, optimizer)
    end
end

rebuild_types(::Type{T}, fields::NamedTuple) where {T <: AbstractRule} = constructorof(T)(; fields...)

#### LRA PSGD

Base.@kwdef struct LRAPSGD <: AbstractRule
    eta::Float64 = 0.001 # learning rate
    beta::Float64 = 0.9 # momentum parameters
    beta_sf::Float64 = beta # momentum parameters for schedule-free interpolation
    epsilon::Float64 = 1.0e-8 # epsilon for numerical stability
    lambda::Float64 = 0.0 # decoupled relative weight decay
    rmsclip::Float64 = 0.0 # clip rms norm of preconditioned gradient
    eta_Q::Float64 = 0.1 # preconditioner learning rate
    max_rank::Int = 10 # max rank of LRA factors
    init_d_scale::Float64 = 0.0 # init scale for diagonal d; if zero, d scale is initialized dynamically
    init_UV_scale::Float64 = √0.1 # init scale for U and V factors
    whiten_grad::Bool = true # whiten gradients if true, otherwise whiten momentum
    schedule_free::Bool = false # use schedule-free momentum
end
@saveable LRAPSGD leaf = true

function Optimisers.init(o::LRAPSGD, y::AbstractArray{T}) where {T}
    @assert o.whiten_grad || o.beta > 0 "LRAPSGD: whitening momentum requires β₁ > 0"
    @assert o.beta >= 0 "LRAPSGD: β₁ must be non-negative"
    N = size(y, 1) # rows in LRA approximation
    B = (length(y) ÷ N,) # batch size
    r = clamp(o.max_rank, 1, N)
    m = !o.whiten_grad ? zeros_similar(y, N, B...) : zeros_similar(y, 0, map(zero, B)...)
    U, V = [randn_similar(y, N, r, B...) for _ in 1:2]
    U .*= T(o.init_UV_scale) ./ sqrt.(sum(abs2, U; dims = (1, 2)))
    V .*= T(o.init_UV_scale) ./ sqrt.(sum(abs2, V; dims = (1, 2)))
    d = fill_similar(y, T(o.init_d_scale), N, B...)
    L₀, Lᵤ, Lᵥ = [zeros_similar(y, 1, 1, B...) for _ in 1:3]
    if o.schedule_free
        @assert o.whiten_grad "Schedule-free LRA requires whiten_grad = true"
        xₜ₊₁ = copy(reshape(y, N, B...)) # momentum buffer is used for xₜ₊₁ in schedule-free mode
        zₜ₊₁ = copy(reshape(y, N, B...))
    else
        xₜ₊₁ = zeros_similar(y, 0, map(zero, B)...) # unused
        zₜ₊₁ = zeros_similar(y, 0, map(zero, B)...) # unused
    end
    return (; m, U, V, d, L₀, Lᵤ, Lᵥ, xₜ₊₁, zₜ₊₁, γₜmax = zero(T), Σγₜmax² = zero(T), t = 0)
end

Optimisers.apply!(o::LRAPSGD, state, y::AbstractArray, dy::AbstractArray) = Optimisers.apply!(o, state, y, dy, nothing)
Optimisers.apply!(o::LRAPSGD, state, y::AbstractArray, dy::AbstractArray, h::AbstractArray, v::AbstractArray) = Optimisers.apply!(o, state, y, dy, (h, v))

function Optimisers.apply!(o::LRAPSGD, state, y::AbstractArray{T, N}, dy::AbstractArray{T, N}, hv_pair::Union{Nothing, NTuple{2, AbstractArray{T, N}}}) where {T, N}
    @assert N > 1 # should always hold
    y_batched, dy_batched = map(y -> reshape(y, size(y, 1), :), (y, dy))
    hv_pair_batched = hv_pair === nothing ? nothing : map(y -> reshape(y, size(y, 1), :), hv_pair)
    state′, dy_batched = batched_psgd_lra_step!(o, state, y_batched, dy_batched, hv_pair_batched)
    dy = N == 2 ? dy_batched : reshape(Base.materialize(dy_batched), size(y))
    return state′, dy
end
function batched_psgd_lra_step!(o::LRAPSGD, state, y::AbstractMatrix{T}, dy::AbstractMatrix{T}, hv_pair::Union{Nothing, NTuple{2, AbstractMatrix{T}}}) where {T}
    nrm(X; dims = (1, 2)) = sqrt.(sum(abs2, X; dims))
    unsq(X) = Flux.unsqueeze(X; dims = 2)
    flat(X) = dropdims(X; dims = 2)
    ᵀ(X) = batched_transpose(X)

    η, β, β_sf, ϵ, λ, ηQ = T(o.eta), T(o.beta), T(o.beta_sf), T(o.epsilon), T(o.lambda), T(o.eta_Q)
    (; m, U, V, d, L₀, Lᵤ, Lᵥ, xₜ₊₁, zₜ₊₁, γₜmax, Σγₜmax², t) = state
    ϵtiny = T(1e-30)
    βLQ = T(0.1)

    β₁ₜ = (β - β^(t + 1)) / (1 - β^(t + 1)) # bias correction absorbed into β₁ₜ = β₁(1 - β₁ᵗ⁻¹)/(1 - β₁ᵗ)

    # Whitening or HVP pair (h, v)
    h, v = hv_pair === nothing ? (dy, randn_similar(y)) : hv_pair

    if t == 0 && o.init_d_scale <= 0
        if hv_pair !== nothing # Newton mode
            d .= mean(abs2.(v); dims = 1) .^ (T(1 // 4)) .* mean(abs4.(h); dims = 1) .^ (T(-1 // 8))
        else # Whitening mode
            d .= mean(abs4.(h); dims = 1) .^ (T(-1 // 8))
        end
    end

    # Gradient or momentum for preconditioner and/or update
    if !o.whiten_grad
        @. m = β₁ₜ * m + (1 - β₁ₜ) * dy
    end
    ĝ = o.whiten_grad ? dy : m

    if hv_pair === nothing
        h̄, v̄ = dy, v
    else
        h̄, v̄ = h, v
    end

    # Compute a = Q * h = (I + UVᵀ)(d ⊙ h) and b = Q⁻ᵀ * v = (I + VUᵀ)⁻¹(v ./ d)
    IpUᵀV = batched_diag_add!(batched_mul(ᵀ(U), V), one(T))
    a = unsq(d .* h̄ .+ batched_vec(U, batched_vec(ᵀ(V), d .* h̄)))
    b = unsq(v̄ ./ d .- batched_vec(V, batched_ldiv!(IpUᵀV, batched_vec(ᵀ(U), v̄ ./ d))))

    # Balance dynamic ranges of U and V (per batch item)
    nrmU, nrmV = nrm(U), nrm(V)
    ρ = @. ifelse((nrmU > ϵ) && (nrmV > ϵ), sqrt(nrmU / nrmV), one(T))
    U ./= ρ
    V .*= ρ

    if rand() < 0.5
        aᵀV, bᵀV = batched_mul(ᵀ(a), V), batched_mul(ᵀ(b), V)
        aᵀVVᵀ, bᵀVVᵀ = batched_mul(aᵀV, ᵀ(V)), batched_mul(bᵀV, ᵀ(V))
        aᵀVVᵀU, bᵀVVᵀU = batched_mul(aᵀVVᵀ, U), batched_mul(bᵀVVᵀ, U)

        ℓᵤ = nrm(a) .* nrm(aᵀVVᵀ) .+ nrm(b) .* nrm(bᵀVVᵀ)
        @. Lᵤ = max((1 - βLQ) * Lᵤ + βLQ * ℓᵤ, ℓᵤ, ϵtiny)
        @. U -= (ηQ / Lᵤ) * ((a * (aᵀV + aᵀVVᵀU)) - (b * (bᵀV + bᵀVVᵀU)))
    else
        Uᵀa, Uᵀb = batched_mul(ᵀ(U), a), batched_mul(ᵀ(U), b)
        UUᵀa, UUᵀb = batched_mul(U, Uᵀa), batched_mul(U, Uᵀb)
        VUᵀa, VUᵀb = batched_mul(V, Uᵀa), batched_mul(V, Uᵀb)
        aᵀU, bᵀU = reshape(Uᵀa, 1, size(Uᵀa, 1), :), reshape(Uᵀb, 1, size(Uᵀb, 1), :)

        ℓᵥ = nrm(a) .* nrm(UUᵀa) .+ nrm(b) .* nrm(UUᵀb)
        @. Lᵥ = max((1 - βLQ) * Lᵥ + βLQ * ℓᵥ, ℓᵥ, ϵtiny)
        @. V -= (ηQ / Lᵥ) * (((a + VUᵀa) * aᵀU) - ((b + VUᵀb) * bᵀU))
    end

    ℓ₀ = maximum(@. a^2 + b^2; dims = 1)
    @. L₀ = max((1 - βLQ) * L₀ + βLQ * ℓ₀, ℓ₀, ϵtiny)

    s = @. 1 - (ηQ / L₀) * (a^2 - b^2)
    U .*= s
    V ./= s
    d .*= flat(s)

    # Precondition current gradient g with updated P = D(I+VUᵀ)(I+UVᵀ)D
    Qĝ = d .* ĝ .+ batched_vec(U, batched_vec(ᵀ(V), d .* ĝ))
    Pĝ = d .* (Qĝ .+ batched_vec(V, batched_vec(ᵀ(U), Qĝ)))

    # Optionally clip preconditioned gradient, which should have rms norm around 1
    if o.rmsclip > 0
        Pĝ_rms, Pĝ_clip = rms(Pĝ; dims = 1), T(o.rmsclip)
        @. Pĝ *= ifelse(Pĝ_rms > Pĝ_clip, Pĝ_clip / Pĝ_rms, one(T))
    end

    if o.schedule_free
        Δ, γₜmax, Σγₜmax² = schedule_free_sgd_update!(xₜ₊₁, zₜ₊₁, y, Pĝ, β_sf, λ, η, γₜmax, Σγₜmax²)
    else
        Δ = @lazy η * (Pĝ + λ * y)
    end

    return (; m, U, V, d, L₀, Lᵤ, Lᵥ, xₜ₊₁, zₜ₊₁, γₜmax, Σγₜmax², t = t + 1), Δ
end

function optimizers_trainmode!(o::LRAPSGD, state, θ::AbstractArray{T}) where {T}
    !o.schedule_free && return θ # no-op if schedule-free
    β_sf = T(o.beta_sf)
    x, z = state.xₜ₊₁, state.zₜ₊₁ # momentum buffer is used for xₜ₊₁ in schedule-free mode
    y = @lazy x + (1 - β_sf) * (z - x) # β₁ does not need bias correction since yₜ is not zero-initialized
    reshape(θ, size(x)) .= y # train using y = interpolant between x and z
    return θ
end

function optimizers_testmode!(o::LRAPSGD, state, θ::AbstractArray{T}) where {T}
    !o.schedule_free && return θ # no-op if schedule-free
    x, z = state.xₜ₊₁, state.zₜ₊₁ # momentum buffer is used for xₜ₊₁ in schedule-free mode
    reshape(θ, size(x)) .= x # evalute using x = running average of z
    return θ
end

#### Kronecker PSGD

@with_kw struct KronPSGD <: AbstractRule
    eta::Float64 = 1.0e-3 # learning rate for weight
    beta::Float64 = 0.9 # momentum parameters
    beta_sf::Float64 = beta # momentum parameters for schedule-free interpolation
    epsilon::Float64 = 1.0e-8 # epsilon for numerical stability
    lambda::Float64 = 0.0 # decoupled relative weight decay
    rmsclip::Float64 = 0.0 # clip rms norm of preconditioned gradient
    eta_Q::Float64 = 0.1 # learning rate for preconditioners Q1, Q2
    kron_Q::Bool = true # use Kronecker preconditioners
    diag_Q::Bool = false # use diagonal preconditioners
    balance_Q::Bool = true # occasionally balance Q1, Q2
    init_Q_scale::Float64 = 0.0 # init scale for Q1, Q2 preconditioners; if zero, scale is initialized dynamically
    whiten_grad::Bool = true # whiten gradients if true, otherwise whiten momentum
    schedule_free::Bool = false # use schedule-free momentum
end
@saveable KronPSGD leaf = true

function Optimisers.init(o::KronPSGD, y::AbstractMatrix{T}) where {T}
    @assert o.kron_Q || o.diag_Q "KronPSGD: can't have non-diagonal, non-Kronecker preconditioners"
    @assert o.rmsclip >= 0 "KronPSGD: rmsclip must be zero to indicate no clipping or else positive to indicate a clipping threshold"
    @assert o.whiten_grad || o.beta > 0 "KronPSGD: whitening momentum requires β₁ > 0"
    @assert o.beta >= 0 "KronPSGD: β₁ must be non-negative"
    m, n = size(y)
    init_scale = o.init_Q_scale <= 0 ? one(T) : T(o.init_Q_scale) # use fixed scale if provided, otherwise initialize to 1.0 and update dynamically on first step
    Q₁ = o.diag_Q ? fill_similar(y, init_scale, o.kron_Q ? (m, 1) : (m, n)) : eye_similar(y, init_scale, m)
    Q₂ = o.diag_Q ? fill_similar(y, init_scale, o.kron_Q ? (1, n) : (0, 0)) : eye_similar(y, init_scale, n)
    m̂ = !o.whiten_grad ? zeros_similar(y) : zeros_similar(y, 0, 0)
    LQ₁ = LQ₂ = zero(T)
    xₜ₊₁ = o.schedule_free ? copy(y) : zeros_similar(y, 0, 0) # xₜ₊₁ is used for schedule-free updates
    zₜ₊₁ = o.schedule_free ? copy(y) : zeros_similar(y, 0, 0) # zₜ₊₁ is used for schedule-free updates
    return (; Q₁, Q₂, m̂, LQ₁, LQ₂, xₜ₊₁, zₜ₊₁, γₜmax = zero(T), Σγₜmax² = zero(T), t = 0)
end
Optimisers.init(o::KronPSGD, y::AbstractArray) = Optimisers.init(o, reshape(y, size(y, 1), :))

# If whitening, `hv_pair` is nothing; otherwise, `hv_pair` is a tuple of the form `(h, v)`
Optimisers.apply!(o::KronPSGD, state, y::AbstractArray, dy::AbstractArray) = Optimisers.apply!(o, state, y, dy, nothing)
Optimisers.apply!(o::KronPSGD, state, y::AbstractArray, dy::AbstractArray, h::AbstractArray, v::AbstractArray) = Optimisers.apply!(o, state, y, dy, (h, v))

function Optimisers.apply!(o::KronPSGD, state, y::AbstractVecOrMat, dy, hv_pair::Union{Nothing, NTuple{2, AbstractVecOrMat}})
    if length(y) == 1 || (!o.kron_Q && o.diag_Q)
        return diag_psgd_update!(o, state, y, dy, hv_pair)
    else
        return kron_psgd_update!(o, state, y, dy, hv_pair)
    end
end

function Optimisers.apply!(o::KronPSGD, state, y::AbstractArray, dy, hv_pair::Union{Nothing, NTuple{2, AbstractArray}})
    state′, dy′ = Optimisers.apply!(
        o, state,
        reshape(y, size(y, 1), :),
        reshape(Base.materialize(dy), size(y, 1), :),
        hv_pair === nothing ? nothing : map(y -> reshape(Base.materialize(y), size(y, 1), :), hv_pair),
    )
    return state′, reshape(Base.materialize(dy′), size(y))
end

function diag_psgd_update!(o::KronPSGD, state, y::AbstractArray{T}, dy, hv_pair::Union{Nothing, NTuple{2, AbstractArray{T}}}) where {T}
    # Diagonal preconditioning, treating `y` as a flattened 1D vector
    η, β, β_sf, ϵ, λ, ηQ = T(o.eta), T(o.beta), T(o.beta_sf), T(o.epsilon), T(o.lambda), T(o.eta_Q)
    (; Q₁, Q₂, m̂, LQ₁, LQ₂, xₜ₊₁, zₜ₊₁, γₜmax, Σγₜmax², t) = state # Q₂ is unused for 1D vectors
    ϵtiny = T(1e-30)
    βLQ = T(0.1)

    # Whitening or HVP pair (h, v)
    h, v = hv_pair === nothing ? (dy, randn_similar(y)) : hv_pair

    # Dynamic scale initialization on first step
    if t == 0 && o.init_Q_scale <= 0
        if hv_pair !== nothing # Newton mode
            Q₁ .*= mean(abs2.(v))^T(1 // 4) * mean(abs4.(h))^T(-1 // 8)
        else # Whitening mode
            Q₁ .*= mean(abs4.(h))^T(-1 // 8)
        end
    end

    β₁ₜ = (β - β^(t + 1)) / (1 - β^(t + 1)) # bias correction absorbed into β₁ₜ = β₁(1 - β₁ᵗ⁻¹)/(1 - β₁ᵗ)

    # Gradient or momentum for preconditioner and/or update
    if !o.whiten_grad
        @. m̂ = β₁ₜ * m̂ + (1 - β₁ₜ) * dy
    end
    g_whiten = o.whiten_grad ? dy : m̂
    ĝ_whiten = g_whiten

    rand() # Preconditioner and auxiliary draws share a reproducible RNG stream.
    if hv_pair === nothing
        h̄, v̄ = dy, v
    else
        h̄, v̄ = h, v
    end
    Qh = Q₁ .* h̄
    Q⁻ᵀv = v̄ ./ Q₁
    ℓQ₁ = max(maximum(@. Qh^2 + Q⁻ᵀv^2), ϵtiny)
    LQ₁ = max((1 - βLQ) * LQ₁ + βLQ * ℓQ₁, ℓQ₁)
    @. Q₁ -= (ηQ / LQ₁) * (Qh^2 - Q⁻ᵀv^2) * Q₁ # Q₁ => (1 - μ * Δ₁) * Q₁

    Δ = Q₁ .^ 2 .* ĝ_whiten

    # Optionally clip preconditioned gradient, which should have rms norm around 1
    if o.rmsclip > 0 && (Δ_rms = rms(Δ)) > (Δ_clip = o.rmsclip * (hv_pair === nothing ? one(T) : max(rms(y), one(T))))
        Δ .*= T(Δ_clip / Δ_rms)
    end

    if o.schedule_free
        # Schedule-free update
        dy′, γₜmax, Σγₜmax² = schedule_free_sgd_update!(xₜ₊₁, zₜ₊₁, y, Δ, β_sf, λ, η, γₜmax, Σγₜmax²)
    else
        # Scaled preconditioned gradient with weight decay
        dy′ = @lazy η * (Δ + λ * y)
    end

    return (; Q₁, Q₂, m̂, LQ₁, LQ₂, xₜ₊₁, zₜ₊₁, γₜmax, Σγₜmax², t = t + 1), dy′
end

function kron_psgd_update!(o::KronPSGD, state, y::AbstractVector{T}, dy, hv_pair::Union{Nothing, NTuple{2, AbstractVector{T}}}) where {T}
    η, β, β_sf, ϵ, λ, ηQ = T(o.eta), T(o.beta), T(o.beta_sf), T(o.epsilon), T(o.lambda), T(o.eta_Q)
    (; Q₁, Q₂, m̂, LQ₁, LQ₂, xₜ₊₁, zₜ₊₁, γₜmax, Σγₜmax², t) = state # Q₂ is unused for 1D vectors
    ϵtiny = T(1e-30)
    βLQ = T(0.1)

    # Whitening or HVP pair (h, v)
    h, v = hv_pair === nothing ? (dy, randn_similar(y)) : hv_pair

    # Dynamic scale initialization on first step (similar to LRAPSGD)
    if t == 0 && o.init_Q_scale <= 0
        if hv_pair !== nothing # Newton mode
            Q₁ .*= mean(abs2.(v))^T(1 // 4) * mean(abs4.(h))^T(-1 // 8)
        else # Whitening mode
            Q₁ .*= mean(abs4.(h))^T(-1 // 8)
        end
    end

    β₁ₜ = (β - β^(t + 1)) / (1 - β^(t + 1)) # bias correction absorbed into β₁ₜ = β₁(1 - β₁ᵗ⁻¹)/(1 - β₁ᵗ)

    # Gradient or momentum for preconditioner and/or update
    if !o.whiten_grad
        @. m̂ = β₁ₜ * m̂ + (1 - β₁ₜ) * dy
    end
    g_whiten = o.whiten_grad ? dy : m̂
    ĝ_whiten = g_whiten

    rand() # Preconditioner and auxiliary draws share a reproducible RNG stream.
    if hv_pair === nothing
        h̄, v̄ = dy, v
    else
        h̄, v̄ = h, v
    end
    if o.diag_Q
        Qh = Q₁ .* h̄
        Q⁻ᵀv = v̄ ./ Q₁
        ℓQ₁ = max(maximum(@. Qh^2 + Q⁻ᵀv^2), ϵtiny)
        LQ₁ = max((1 - βLQ) * LQ₁ + βLQ * ℓQ₁, ℓQ₁)
        @. Q₁ -= (ηQ / LQ₁) * (Qh^2 - Q⁻ᵀv^2) * Q₁ # Q₁ => (1 - μ * Δ₁) * Q₁
    else
        Tri = UpperTriangular
        TriProj = UpperTriangular # QR approximation [I + A]_R ≈ I + triu(A) + triu(A, 1), accurate for ||A|| < 0.25 (Li, PSGD)
        TriQ₁ = Tri(Q₁)
        Qh = Q₁ * h̄ # dense * dense faster than triangular * dense on GPU
        Q⁻ᵀv = TriQ₁' \ v̄
        hᵀQᵀ, vᵀQ⁻¹ = Qh', Q⁻ᵀv'
        ℓQ₁ = max(dot(Qh, Qh) + dot(Q⁻ᵀv, Q⁻ᵀv), ϵtiny)
        LQ₁ = max((1 - βLQ) * LQ₁ + βLQ * ℓQ₁, ℓQ₁)
        ΔQ₁ = TriProj(@. (ηQ / LQ₁) * (Qh * hᵀQᵀ - Q⁻ᵀv * vᵀQ⁻¹)) * Q₁ # triangular * triangular fails on GPU
        Q₁ .-= ΔQ₁ # Q₁ => triu(I - μ * Δ₁) * Q₁ = Q₁ - μ * triu(Δ₁) * Q₁
    end

    if o.diag_Q
        Δ = vec(Q₁) .^ 2 .* ĝ_whiten
    else
        Δ = Q₁' * (Q₁ * ĝ_whiten) # dense * dense faster than triangular * dense on GPU
    end

    # Optionally clip preconditioned gradient, which should have rms norm around 1
    if o.rmsclip > 0 && (Δ_rms = rms(Δ)) > (Δ_clip = o.rmsclip * (hv_pair === nothing ? one(T) : max(rms(y), one(T))))
        Δ .*= T(Δ_clip / Δ_rms)
    end

    if o.schedule_free
        # Schedule-free update
        dy′, γₜmax, Σγₜmax² = schedule_free_sgd_update!(xₜ₊₁, zₜ₊₁, y, Δ, β_sf, λ, η, γₜmax, Σγₜmax²)
    else
        # Scaled preconditioned gradient with weight decay
        dy′ = @lazy η * (Δ + λ * y)
    end

    return (; Q₁, Q₂, m̂, LQ₁, LQ₂, xₜ₊₁, zₜ₊₁, γₜmax, Σγₜmax², t = t + 1), dy′
end

function kron_psgd_update!(o::KronPSGD, state, y::AbstractMatrix{T}, dy, hv_pair::Union{Nothing, NTuple{2, AbstractMatrix{T}}}) where {T}
    η, β, β_sf, ϵ, λ, ηQ = T(o.eta), T(o.beta), T(o.beta_sf), T(o.epsilon), T(o.lambda), T(o.eta_Q)
    (; Q₁, Q₂, m̂, LQ₁, LQ₂, xₜ₊₁, zₜ₊₁, γₜmax, Σγₜmax², t) = state
    ϵtiny = T(1e-30)
    βLQ = T(0.1)

    # Whitening or HVP pair (h, v)
    h, v = hv_pair === nothing ? (dy, randn_similar(y)) : hv_pair

    # Dynamic scale initialization on first step (similar to LRAPSGD)
    if t == 0 && o.init_Q_scale <= 0
        if hv_pair !== nothing # Newton mode
            Q_scale = mean(abs2.(v))^T(1 // 4) * mean(abs4.(h))^T(-1 // 8)
        else # Whitening mode
            Q_scale = mean(abs4.(h))^T(-1 // 8)
        end
        Q₁ .*= √Q_scale # Q -> Q_scale * Q where Q = Q₂ ⊗ Q₁
        Q₂ .*= √Q_scale
    end

    β₁ₜ = (β - β^(t + 1)) / (1 - β^(t + 1)) # bias correction absorbed into β₁ₜ = β₁(1 - β₁ᵗ⁻¹)/(1 - β₁ᵗ)

    # Gradient or momentum for preconditioner and/or update
    if !o.whiten_grad
        @. m̂ = β₁ₜ * m̂ + (1 - β₁ₜ) * dy
    end
    g_whiten = o.whiten_grad ? dy : m̂
    ĝ_whiten = g_whiten

    rand() # Preconditioner and auxiliary draws share a reproducible RNG stream.
    if hv_pair === nothing
        h̄, v̄ = dy, v
    else
        h̄, v̄ = h, v
    end
    if o.diag_Q
        Qh = Q₁ .* h̄ .* Q₂
        Q⁻ᵀv = v̄ ./ (Q₁ .* Q₂)
        A₁, B₁ = sum(abs2, Qh; dims = 2), sum(abs2, Q⁻ᵀv; dims = 2)
        A₂, B₂ = sum(abs2, Qh; dims = 1), sum(abs2, Q⁻ᵀv; dims = 1)
        ℓQ₁ = max(maximum(@. A₁ + B₁), ϵtiny)
        ℓQ₂ = max(maximum(@. A₂ + B₂), ϵtiny)
        LQ₁ = max((1 - βLQ) * LQ₁ + βLQ * ℓQ₁, ℓQ₁)
        LQ₂ = max((1 - βLQ) * LQ₂ + βLQ * ℓQ₂, ℓQ₂)
        @. Q₁ -= (ηQ / LQ₁) * (A₁ - B₁) * Q₁ # Q₁ => (1 - μ * Δ₁) * Q₁
        @. Q₂ -= (ηQ / LQ₂) * (A₂ - B₂) * Q₂ # Q₂ => (1 - μ * Δ₂) * Q₂
    else
        Tri = UpperTriangular
        TriProj = UpperTriangular # QR approximation [I + A]_R ≈ I + triu(A) + triu(A, 1), accurate for ||A|| < 0.25 (Li, PSGD)
        TriQ₁, TriQ₂ = Tri(Q₁), Tri(Q₂)
        A = (Q₁ * h̄) * Q₂' # dense * dense faster than triangular * dense on GPU
        B = (TriQ₂' \ v̄') / TriQ₁ # B = Q₂⁻ᵀ * vᵀ * Q₁⁻¹
        AAᵀ, AᵀA, BBᵀ, BᵀB = A * A', A' * A, B * B', B' * B
        ℓQ₁ = max(psgd_opnorm_lower_bound!(AAᵀ + BᵀB), ϵtiny)
        ℓQ₂ = max(psgd_opnorm_lower_bound!(AᵀA + BBᵀ), ϵtiny)
        LQ₁ = max((1 - βLQ) * LQ₁ + βLQ * ℓQ₁, ℓQ₁)
        LQ₂ = max((1 - βLQ) * LQ₂ + βLQ * ℓQ₂, ℓQ₂)
        ΔQ₁ = TriProj(@. (ηQ / LQ₁) * (AAᵀ - BᵀB)) * Q₁ # triangular * triangular fails on GPU
        ΔQ₂ = TriProj(@. (ηQ / LQ₂) * (AᵀA - BBᵀ)) * Q₂ # triangular * triangular fails on GPU
        Q₁ .-= ΔQ₁ # Q₁ => triu(I - μ₁ * Δ₁) * Q₁ = Q₁ - μ₁ * triu(Δ₁) * Q₁
        Q₂ .-= ΔQ₂ # Q₂ => triu(I - μ₂ * Δ₂) * Q₂ = Q₂ - μ₂ * triu(Δ₂) * Q₂
    end
    if o.balance_Q && rand() < 0.01
        psgd_balance_Q!(Q₁, Q₂)
    end

    if o.diag_Q
        Δ = Q₁ .^ 2 .* ĝ_whiten .* Q₂ .^ 2 # Q₁, Q₂ represent diagonal preconditioners; Q₁ is a column vector, Q₂ is a row vector
    elseif size(ĝ_whiten, 1) < size(ĝ_whiten, 2)
        Δ = (((Q₁' * Q₁) * ĝ_whiten) * Q₂') * Q₂ # dense * dense faster than triangular * dense on GPU
    else # size(ĝ_whiten, 1) >= size(ĝ_whiten, 2)
        Δ = Q₁' * (Q₁ * (ĝ_whiten * (Q₂' * Q₂))) # dense * dense faster than triangular * dense on GPU
    end

    # Optionally clip preconditioned gradient, which should have rms norm around 1
    if o.rmsclip > 0 && (Δ_rms = rms(Δ)) > (Δ_clip = o.rmsclip * (hv_pair === nothing ? one(T) : max(rms(y), one(T))))
        Δ .*= T(Δ_clip / Δ_rms)
    end

    if o.schedule_free
        # Schedule-free update
        dy′, γₜmax, Σγₜmax² = schedule_free_sgd_update!(xₜ₊₁, zₜ₊₁, y, Δ, β_sf, λ, η, γₜmax, Σγₜmax²)
    else
        # Scaled preconditioned gradient with weight decay
        dy′ = @lazy η * (Δ + λ * y)
    end

    return (; Q₁, Q₂, m̂, LQ₁, LQ₂, xₜ₊₁, zₜ₊₁, γₜmax, Σγₜmax², t = t + 1), dy′
end

function optimizers_trainmode!(o::KronPSGD, state, θ::AbstractArray{T}) where {T}
    !o.schedule_free && return θ # no-op if not schedule-free
    β_sf = T(o.beta_sf)
    (; xₜ₊₁, zₜ₊₁) = state
    yₜ₊₁ = @lazy xₜ₊₁ + (1 - β_sf) * (zₜ₊₁ - xₜ₊₁) # β₁ does not need bias correction since yₜ is not zero-initialized
    reshape(θ, size(xₜ₊₁)) .= yₜ₊₁ # train using y = interpolant between x and z
    return θ
end

function optimizers_testmode!(o::KronPSGD, state, θ::AbstractArray{T}) where {T}
    !o.schedule_free && return θ # no-op if not schedule-free
    (; xₜ₊₁, zₜ₊₁) = state
    reshape(θ, size(xₜ₊₁)) .= xₜ₊₁ # evaluate using x = running average of z
    return θ
end

#### PSGD helper functions

function psgd_opnorm_lower_bound!(A::AbstractMatrix)
    if length(A) == 1 || size(A, 1) == 1 || size(A, 2) == 1
        return norm(A)
    end
    max_abs = maximum(abs, A)
    max_abs == 0 && return zero(eltype(A))
    A ./= max_abs
    col_max, i = findmax(dropdims(sum(abs2, A; dims = 1); dims = 1))
    row_max, j = findmax(dropdims(sum(abs2, A; dims = 2); dims = 2))
    if col_max > row_max
        y = A[:, i]
        x = A' * y
        x ./= norm(x)
        mul!(y, A, x)
        return max_abs * norm(y)
    else
        y = conj!(A[j, :])
        x = A * y
        x ./= norm(x)
        mul!(y, A', x)
        return max_abs * norm(y)
    end
end

function psgd_balance_Q!(Q₁::AbstractMatrix, Q₂::AbstractMatrix)
    # Rescales Q₁, Q₂ so their ∞-norms are on a similar scale
    nrm₁, nrm₂ = maximum(abs, Q₁), maximum(abs, Q₂)
    if nrm₁ > 1e-12 && nrm₂ > 1e-12
        ρ = sqrt(nrm₁ / nrm₂)
        @. Q₁ = Q₁ / ρ
        @. Q₂ = Q₂ * ρ
    end
    return nothing
end

function schedule_free_sgd_update!(xₜ₊₁, zₜ₊₁, yₜ, g, β, λ, γₜ, γₜmax, Σγₜmax²)
    cₜ₊₁, γₜmax, Σγₜmax² = schedulefree_update_lr_stats(γₜ, γₜmax, Σγₜmax²)
    λₜ = λ * (γₜmax <= 0 ? one(λ) : γₜ / γₜmax) # Defazio (2025), Why gradients rapidly increase near the end of training
    @. zₜ₊₁ = zₜ₊₁ - γₜ * (g + λₜ * yₜ)
    @. xₜ₊₁ = (1 - cₜ₊₁) * xₜ₊₁ + cₜ₊₁ * zₜ₊₁
    yₜ₊₁ = @lazy xₜ₊₁ + (1 - β) * (zₜ₊₁ - xₜ₊₁) # β₁ does not need bias correction since yₜ is not zero-initialized
    dyₜ = @lazy yₜ - yₜ₊₁ # Optimisers.jl expects a delta, not the updated value
    return dyₜ, γₜmax, Σγₜmax²
end

function schedulefree_update_lr_stats(γₜ, γₜmax, Σγₜmax²)
    if γₜ > 0
        γₜmax = max(γₜmax, γₜ) # the maximum learning rate follows the reference implementation
        Σγₜmax² += γₜmax^2
        cₜ₊₁ = Σγₜmax² == 0 ? one(γₜmax) : γₜmax^2 / Σγₜmax²
    else
        cₜ₊₁ = zero(γₜmax) # γₜ = 0 is special-cased to ensure xₜ, yₜ, zₜ are not updated
    end
    return cₜ₊₁, γₜmax, Σγₜmax²
end
