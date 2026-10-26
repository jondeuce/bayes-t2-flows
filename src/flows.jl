"""
    ProbabilityMass{X, dX, Z}(x::X, logjac::dX, latents::Z)

A `ProbabilityMass` tracks the infinitesimal volume `dx` at a transformed point `x = φ(u | latents)` relative to the volume `du` in the original space, where `φ` is a diffeomorphism.
The log–Jacobian determinant `logjac` is `log |∂φ(u) / ∂u| = -log |∂φ⁻¹(x) / ∂x|`, and is accumulated over successive transformations.

If `logjac` is initialized to `-logpᵤ(u)`, where `pᵤ` is a base distribution over `u`, then the flow `φ` transforms the base density into `-logpₓ(x) = -logpᵤ(u) + log |∂φ(u) / ∂u|` at `x = φ(u)`, and `logjac` so tracks `pₓ(x)`.
Thus, we think of the struct as a whole as tracking the density of probability mass at `x`, hence the name `ProbabilityMass`.

The change-of-variables formula states that

    ∫_X h(x) dx = ∫_U h(φ(u)) |∂φ(u) / ∂u| du,

where `h(x)` is a measurable function, and `X = φ(U)` where `U` is a region in `u`-space.
The infinitesimal volume element `du` at `u` is thus scaled into `dx = |∂φ(u) / ∂u| du`.

If `u` has density `pᵤ(u)` and `x = φ(u)`, then the pushforward density is given by

    pₓ(x) = pᵤ(u) |∂φ(u) / ∂u|⁻¹

where u = φ⁻¹(x); by the change-of-variables formula, this ensures that probability mass is conserved, i.e. `∫_X pₓ(x) dx = ∫_U pᵤ(u) du`.

On Riemannian manifolds, the relative volume `|∂φ(u) / ∂u|` is equivalent to the volume density `√|g(x)|`, where `g(x)` is the metric tensor.
"""
struct ProbabilityMass{X, dX, Z}
    value::X
    logjac::dX
    latents::Z
end
@saveable ProbabilityMass

getvalue(𝓂::ProbabilityMass) = 𝓂.value
getlogjac(𝓂::ProbabilityMass) = 𝓂.logjac

Base.NamedTuple(𝓂::ProbabilityMass) = (; 𝓂.value, 𝓂.logjac, 𝓂.latents)
Base.Tuple(𝓂::ProbabilityMass) = (𝓂.value, 𝓂.logjac, 𝓂.latents)
Base.iterate(𝓂::ProbabilityMass, state...) = Base.iterate(NamedTuple(𝓂), state...)

ProbabilityMass(x::AbstractArray, latents::Tuple = ()) = ProbabilityMass(x, nothing, latents)
ProbabilityMass(x::AbstractArray, logjac::Union{AbstractArray, Nothing}) = ProbabilityMass(x, logjac, ())
ProbabilityMass(𝓂::NamedTuple) = ProbabilityMass(𝓂.value, 𝓂.logjac, 𝓂.latents)

flow_transform(ℱ, x::AbstractArray, args...) = flow_transform(ℱ, ProbabilityMass(x, args...))
flow_transform(ℱ, 𝓂::ProbabilityMass) = ℱ(𝓂)

function Base.isapprox(𝓂₁::ProbabilityMass, 𝓂₂::ProbabilityMass; kwargs...)
    ≈(x, y) = isapprox(x, y; kwargs...)
    return (𝓂₁.value ≈ 𝓂₂.value) &&
           (𝓂₁.logjac === 𝓂₂.logjac === nothing || (𝓂₁.logjac ≈ 𝓂₂.logjac)) &&
           fmap_all(≈, 𝓂₁.latents, 𝓂₂.latents; exclude = isnumeric)
end

zero_logjac(x) = zeros_similar(x, 1, Base.tail(size(x))...)

maybe_checkpointed(f, x, checkpoint::Bool) = checkpoint ? Zygote.checkpointed(f, x) : f(x)
maybe_checkpointed(f, x, ::Val{true}) = Zygote.checkpointed(f, x)
maybe_checkpointed(f, x, ::Val{false}) = f(x)

# Path gradients via a custom rrule: only the inverse flow is needed, avoiding error-prone roundtrip values
function flow_transform_pathgradient(flow, u, logjacu, latents)
    mu = ProbabilityMass(u, logjacu, latents)
    mx = flow_transform(flow, mu)
    return mu, mx
end

function ChainRules.rrule(cfg::ChainRules.RuleConfig{>:ChainRules.HasReverseMode}, ::typeof(flow_transform_pathgradient), flow, u, logjacu, latents)
    # Semantically, a flow which computes both x and logjacx in one pass via
    #   x, logjacx = flow(u, logjacu, latents)
    # is equivalent to a flow which first computes x and then computes logjacx given x via
    #   x = flow(u, nothing, latents)[1] <-- logjacu is not needed to compute x
    #   logjacx = -inverse_flow(flow)(x, -logjacu, latents)[2].
    # The extra backward pass is needed for the path derivative anyway, and separates the score terms to drop

    # returns (x, logjacx)
    function flow_transform_fwd(_flow, _u, _latents)
        mx = flow_transform(_flow, _u, logjacu, _latents)
        return (mx.value, mx.logjac)
    end

    rev_flow = inverse_flow(flow) # keep only ∂x; flow and latent tangents are score-function terms and are dropped
    # returns (u, logjacx)
    function flow_transform_rev(_x)
        mu = flow_transform(rev_flow, _x, -logjacu, latents)
        return (mu.value, -mu.logjac)
    end

    # Run forward flow transform as usual
    (x, logjacx), Jfwd = ChainRules.rrule_via_ad(cfg, flow_transform_fwd, flow, u, latents)
    mu = ProbabilityMass(u, logjacu, latents)
    mx = ProbabilityMass(x, logjacx, latents)

    project_flow = ChainRules.ProjectTo(flow)
    project_u = ChainRules.ProjectTo(u)
    project_logjac = ChainRules.ProjectTo(logjacu)
    project_latents = ChainRules.ProjectTo(latents)

    function path_derivative((Δmu, Δmx))
        # Run pullback over reverse flow then forward flow to get the path derivative contributions
        _, Jrev = ChainRules.rrule_via_ad(cfg, flow_transform_rev, x) # u′ ≈ u and logjacx′ ≈ logjacx up to numerical error
        _, Δx_logjacx = Jrev((Δmu.value, Δmx.logjac)) # Δθ_logjacx = (Δflow_logjacx, Δlatents_logjacx) would be the score function contributions
        _, Δflow_x, Δu_x, Δlatents_x = Jfwd((Δmx.value + Δx_logjacx, ZeroTangent())) # Δmx.logjac enters through Jrev as a path derivative, not through Jfwd (a score term)
        return (NoTangent(), project_flow(Δflow_x), project_u(Δu_x), project_logjac(Δmu.logjac) + project_logjac(Δmx.logjac), project_latents(Δmu.latents) + project_latents(Δmx.latents) + Δlatents_x) # add the Δmx.latents contribution; score-function terms are dropped
    end

    return (mu, mx), path_derivative
end

struct InverseFlow{F}
    flow::F
end
@saveable InverseFlow

function Flux.trainmode!(ℱ::InverseFlow, mode = false)
    Flux.trainmode!(ℱ.flow, mode)
    return ℱ
end
function Flux.testmode!(ℱ::InverseFlow, mode = true)
    Flux.testmode!(ℱ.flow, mode)
    return ℱ
end

# Rebuild a flow similar to `ℱ₀` from `ℱ` using `functor`.
# E.g. rebuild_flow(Flux.Chain, (; layers = (identity, abs2))) builds a Chain from a NamedTuple
rebuild_flow(ℱ₀, ℱ) = rebuild_flow(typeof(ℱ₀), ℱ) # rebuild ℱ based on type of ℱ₀
rebuild_flow(::Type{F}, ℱ::F) where {F} = ℱ # same type, no need to rebuild
function rebuild_flow(::Type{F}, ℱ) where {F}
    (func, re) = functor(F, ℱ)
    return re(func)
end

# Similar to `functor`, the flow `ℱ₀` guides how the inverse flow of `ℱ` should be constructed.
# The instance is needed as well as the type, since type information can be lost (e.g. `VectorChain`)
inverse_flow(ℱ) = inverse_flow(ℱ, ℱ) # fallback to self guidance
inverse_flow(ℱ₀, ℱ) = InverseFlow(rebuild_flow(ℱ₀, ℱ)) # default method: rebuild and wrap in `InverseFlow`

# Inverse of `InverseFlow` and other trivial wrapper types
inverse_flow(ℱ₀::InverseFlow, ℱ) = rebuild_flow(ℱ₀.flow, ℱ.flow) # rebuild inner flow
inverse_flow(ℱ₀::NotTrainable, ℱ) = NotTrainable(inverse_flow(ℱ₀.layer, ℱ.layer))
inverse_flow(ℱ₀::ComposedFunction, ℱ) = inverse_flow(ℱ₀.inner, ℱ.inner) ∘ inverse_flow(ℱ₀.outer, ℱ.outer)
inverse_flow(::typeof(identity), ℱ) = ℱ

# Inverse `Chain`s
inverse_flow(ℱ₀::Flux.Chain{<:AbstractVector}, ℱ) = Flux.Chain(inverse_flow.(Base.reverse(ℱ₀.layers), Base.reverse(ℱ.layers)))
inverse_flow(ℱ₀::Flux.Chain{<:Tuple}, ℱ) = Flux.Chain(inverse_flow.(Base.reverse(ℱ₀.layers), Base.reverse(ℱ.layers)))

"""
Latent flow. Preprocess the latent variables before passing them to an inner flow.
Return a probability mass with the original latent variables.
"""
struct LatentFlow{F, P}
    flow::F
    preprocess::P
end
@saveable LatentFlow

inverse_flow(ℱ₀::LatentFlow, ℱ) = LatentFlow(inverse_flow(ℱ₀.flow, ℱ.flow), ℱ.preprocess)

function (ℱ::LatentFlow)(𝓂ₓ::ProbabilityMass)
    𝓂ₓ′ = ProbabilityMass(𝓂ₓ.value, 𝓂ₓ.logjac, ℱ.preprocess(𝓂ₓ.latents)) # preprocessed latents
    𝓂ᵤ′ = ℱ.flow(𝓂ₓ′) # flow using preprocessed latents
    𝓂ᵤ = ProbabilityMass(𝓂ᵤ′.value, 𝓂ᵤ′.logjac, 𝓂ₓ.latents) # return original latents
    return 𝓂ᵤ
end

struct ProductFlow{F, P, P⁻¹, C}
    flows::F
    partitioner::P
    inv_partitioner::P⁻¹
    combiner::C
end
@saveable ProductFlow

function ProductFlow(flows::Union{Tuple, NamedTuple}, sizes::Union{Tuple, NamedTuple})
    sizes = map(product_flow_size, sizes)
    @assert length(flows) == length(sizes) "Number of flows and sizes must match"
    @assert keys(flows) == keys(sizes) "Flow and size keys must match"
    insizes, outsizes = map(first, sizes), map(second, sizes)
    inoffsets, outoffsets = cumsum((0, insizes...)), cumsum((0, outsizes...))
    inds = mapenumerated((i, _) -> ((inoffsets[i]+1):inoffsets[i+1], ..), insizes)
    inv_inds = mapenumerated((i, _) -> ((outoffsets[i]+1):outoffsets[i+1], ..), outsizes)
    partitioner = GetIndex(inds)
    inv_partitioner = GetIndex(inv_inds)
    combiner = Concat(1)
    return ProductFlow(flows, partitioner, inv_partitioner, combiner)
end

product_flow_size(sz::Int) = sz => sz
product_flow_size(sz::Pair{Int, Int}) = sz

function inverse_flow(ℱ₀::ProductFlow, ℱ)
    ℱ⁻¹ = map(inverse_flow, ℱ₀.flows, ℱ.flows)
    return ProductFlow(ℱ⁻¹, ℱ.inv_partitioner, ℱ.partitioner, ℱ.combiner)
end

function (ℱ::ProductFlow)(𝓂ₓ::ProbabilityMass)
    x = Base.materialize(𝓂ₓ.value)
    xs = ℱ.partitioner(x)
    ∂x = 𝓂ₓ.logjac === nothing ? nothing : zero_logjac(x)
    𝓂ᵤs = map(ℱ.flows, xs) do ℱᵢ, xᵢ
        flow_transform(ℱᵢ, xᵢ, ∂x, 𝓂ₓ.latents) # flow xᵢ forward
    end
    u = ℱ.combiner(map(getvalue, 𝓂ᵤs)) # concatenate uᵢ
    ∂u = 𝓂ₓ.logjac === nothing ? nothing : lazy.((+).(𝓂ₓ.logjac, map(getlogjac, 𝓂ᵤs)...)) # accumulate ∂uᵢ
    return ProbabilityMass(u, ∂u, 𝓂ₓ.latents)
end

struct CouplingFlow{F, P, P⁻¹, C}
    flow::F
    partitioner::P
    inv_partitioner::P⁻¹
    combiner::C
end
@saveable CouplingFlow


function SplitCouplingFlow(
    flow;
    insize::Int,
    outsize::Int = insize,
    split_at::Int = insize ÷ 2,
    inv_split_at::Int = split_at,
    flip::Bool = false,
    combiner = Concat(1),
)
    inds = ((1:split_at, ..), ((split_at+1):insize, ..))
    inv_inds = ((1:inv_split_at, ..), ((inv_split_at+1):outsize, ..))
    partitioner = !flip ? GetIndex(inds) : GetIndex(reverse(inds))
    inv_partitioner = !flip ? GetIndex(inv_inds) : GetIndex(reverse(inv_inds))
    combiner = !flip ? combiner : Base.Fix1(flip_combiner, combiner)
    return CouplingFlow(flow, partitioner, inv_partitioner, combiner)
end
flip_combiner(combiner, (a, b)::NTuple{2, Any}) = combiner((b, a))

function inverse_flow(ℱ₀::CouplingFlow, ℱ)
    ℱ⁻¹ = inverse_flow(ℱ₀.flow, ℱ.flow)
    return CouplingFlow(ℱ⁻¹, ℱ.inv_partitioner, ℱ.partitioner, ℱ.combiner)
end

function (ℱ::CouplingFlow)(𝓂ₓ::ProbabilityMass)
    xᵃ, xᵇ = ℱ.partitioner(𝓂ₓ.value)
    𝓂ᵤᵇ = flow_transform(ℱ.flow, xᵇ, 𝓂ₓ.logjac, (xᵃ, 𝓂ₓ.latents...)) # flow xᵇ forward, conditioned on xᵃ
    u, ∂u = ℱ.combiner((xᵃ, 𝓂ᵤᵇ.value)), 𝓂ᵤᵇ.logjac # xᵃ remains unchanged
    return ProbabilityMass(u, ∂u, 𝓂ₓ.latents)
end

function StackedFlow(
    flows::NamedTuple{Ks, <:NTuple{2, Any}},
    sizes::NamedTuple{Ks, NTuple{2, Pair{Int, Int}}};
    flip::Bool = false,
) where {Ks}
    @assert length(flows) == length(sizes) == 2 "StackedFlow only supports two flows"
    (flow1, flow2), ((in1, out1), (in2, out2)) = flows, sizes

    if !flip
        TypedChain(
            # Flow 1 is applied to the first `in1` elements, and the last `in2` are left unchanged
            ProductFlow(
                NamedTuple{Ks}((flow1, identity)),
                NamedTuple{Ks}((in1 => out1, in2 => in2)),
            ),
            # Flow 2 is applied to the last `in2` elements, conditioned on the first `out1` elements
            SplitCouplingFlow(
                flow2;
                insize = out1 + in2,
                outsize = out1 + out2,
                split_at = out1,
                inv_split_at = out1,
                flip = false,
            ),
        )
    else
        TypedChain(
            # Flow 2 is applied to the last `in2` elements, and the first `in1` are left unchanged
            ProductFlow(
                NamedTuple{Ks}((identity, flow2)),
                NamedTuple{Ks}((in1 => in1, in2 => out2)),
            ),
            # Flow 1 is applied to the first `in1` elements, conditioned on the last `out2` elements
            SplitCouplingFlow(
                flow1;
                insize = in1 + out2,
                outsize = out1 + out2,
                split_at = in1,
                inv_split_at = out1,
                flip = true,
            ),
        )
    end
end

struct ExpAffineFlow{TM, Tb}
    M::TM
    b::Tb
end
@saveable ExpAffineFlow

Base.show(io::IO, ℱ::ExpAffineFlow) = print(io, "ExpAffineFlow(size(M) = ", size(ℱ.M), ")")

ExpAffineFlow(::Type{T}, n::Int) where {T} = ExpAffineFlow(zeros(T, n, n), zeros(T, n))

function (ℱ::ExpAffineFlow)(𝓂ₓ::ProbabilityMass)
    (; M, b) = ℱ
    u = matrix_exp(M) * Base.materialize(𝓂ₓ.value)
    u = @lazy u + b
    ∂u = 𝓂ₓ.logjac === nothing ? nothing : lazy.(𝓂ₓ.logjac .+ tr_eager(M))
    return ProbabilityMass(u, ∂u, 𝓂ₓ.latents)
end

function (ℱ::InverseFlow{<:ExpAffineFlow})(𝓂ᵤ::ProbabilityMass)
    (; M, b) = ℱ.flow
    u = 𝓂ᵤ.value .- b
    x = matrix_exp(-M) * u
    ∂x = 𝓂ᵤ.logjac === nothing ? nothing : lazy.(𝓂ᵤ.logjac .- tr_eager(M))
    return ProbabilityMass(x, ∂x, 𝓂ᵤ.latents)
end

struct ConstantDenseAffineFlow{TA, TI, Tb, TJ}
    A::TA
    A⁻¹::TI
    b::Tb
    logabsdetA::TJ
end
@saveable ConstantDenseAffineFlow
Optimisers.trainable(::ConstantDenseAffineFlow) = NamedTuple()

ConstantDenseAffineFlow(ℱ::ExpAffineFlow) = ConstantDenseAffineFlow(matrix_exp(ℱ.M), matrix_exp(-ℱ.M), ℱ.b, tr_eager(ℱ.M))

# The trained matrix is fixed during inference; only the input cotangent is needed.
constant_affine_apply(A, x) = A * x
function ChainRules.rrule(::typeof(constant_affine_apply), A, x)
    function constant_affine_pullback(Δy)
        Δ = ChainRules.unthunk(Δy)
        Δx = Δ isa ChainRules.AbstractZero ? Δ : A' * Base.materialize(Δ)
        return ChainRules.NoTangent(), ChainRules.NoTangent(), Δx
    end
    return constant_affine_apply(A, x), constant_affine_pullback
end

function (ℱ::ConstantDenseAffineFlow)(𝓂ₓ::ProbabilityMass)
    u = constant_affine_apply(ℱ.A, Base.materialize(𝓂ₓ.value))
    u = @lazy u + ℱ.b
    ∂u = 𝓂ₓ.logjac === nothing ? nothing : lazy.(Base.broadcasted(Base.Fix2(+, ℱ.logabsdetA), 𝓂ₓ.logjac))
    return ProbabilityMass(u, ∂u, 𝓂ₓ.latents)
end

function (ℱ::InverseFlow{<:ConstantDenseAffineFlow})(𝓂ᵤ::ProbabilityMass)
    ℱ₀ = ℱ.flow
    u = 𝓂ᵤ.value .- ℱ₀.b
    x = constant_affine_apply(ℱ₀.A⁻¹, u)
    ∂x = 𝓂ᵤ.logjac === nothing ? nothing : lazy.(Base.broadcasted(Base.Fix2(-, ℱ₀.logabsdetA), 𝓂ᵤ.logjac))
    return ProbabilityMass(x, ∂x, 𝓂ᵤ.latents)
end

function fix_affine_matrices(model)
    isaffine(x) = x isa Union{ExpAffineFlow, InverseFlow{<:ExpAffineFlow}}
    fixed(x) = x
    fixed(x::ExpAffineFlow) = ConstantDenseAffineFlow(x)
    fixed(x::InverseFlow) = InverseFlow(fixed(x.flow))
    return fmap(fixed, model; exclude = isaffine)
end

"""
Constant translation flow `x -> x + b`.
"""
struct ShiftFlow{Tb}
    b::Tb
end
@saveable ShiftFlow
Optimisers.trainable(::ShiftFlow) = NamedTuple()

Base.show(io::IO, ℱ::ShiftFlow) = print(io, "ShiftFlow(size(b) = ", size(ℱ.b), ")")

(ℱ::ShiftFlow)(𝓂ₓ::ProbabilityMass) = ProbabilityMass((@lazy 𝓂ₓ.value + ℱ.b), 𝓂ₓ.logjac, 𝓂ₓ.latents)
(ℱ::InverseFlow{<:ShiftFlow})(𝓂ᵤ::ProbabilityMass) = ProbabilityMass((@lazy 𝓂ᵤ.value - ℱ.flow.b), 𝓂ᵤ.logjac, 𝓂ᵤ.latents)

struct SimplexFlow{A, F}
    Ψ::A
    logβ::F
end
@saveable SimplexFlow
Optimisers.trainable(ℱ::SimplexFlow) = (; logβ = ℱ.logβ) # simplex basis is not trainable

Base.show(io::IO, ℱ::SimplexFlow) = print(io, "SimplexFlow(simplexdim = ", size(ℱ.Ψ, 1), ", logβ = ", ℱ.logβ, ")")

SimplexFlow(Ψ::AbstractMatrix{T}, logβ = zero(T)) where {T} = SimplexFlow{typeof(Ψ), typeof(logβ)}(Ψ, logβ)

SimplexFlow(args...; kwargs...) = SimplexFlow(Float32, args...; kwargs...)
SimplexFlow(::Type{T}, args...; kwargs...) where {T} = SimplexFlow(cpu_device(), T, args...; kwargs...)
SimplexFlow(dev::AbstractDevice, ::Type{T}, args...; insize::Int) where {T} = SimplexFlow(arr_similar(dev, T, idct_simplex_basis(T, insize)), args...)

SimplexFlow_unpack(ℱ::SimplexFlow, ::Tuple) = (; logβ = ℱ.logβ)

function (ℱ::SimplexFlow)(𝓂ᵤ::ProbabilityMass)
    # Flow from Aitchison coefficients `u ∈ ℝⁿ⁻¹` to normalized distribution `x̄ ∈ Δⁿ⁻¹`
    T = eltypefloat(ℱ.Ψ)
    n = size(ℱ.Ψ, 1) # Ψ ∈ ℝⁿˣⁿ⁻¹
    logβ₀ = T(log(π^2 / 6) / 2)
    logsqrtn = T(log(n) / 2)

    (; logβ) = SimplexFlow_unpack(ℱ, 𝓂ᵤ.latents) # log inverse temperature
    βu = @. exp(logβ + logβ₀) * 𝓂ᵤ.value # βu is a distribution representated in the ILR basis
    logx̄ = logsoftmax(ℱ.Ψ * βu; dims = 1) # log normalized distribution
    ∂x̄ = 𝓂ᵤ.logjac === nothing ? nothing : lazy.(simplex_logjac_update(𝓂ᵤ.logjac, sum(logx̄; dims = 1), +, logβ, n, logβ₀, logsqrtn))

    return ProbabilityMass(logx̄, ∂x̄, 𝓂ᵤ.latents)
end

function (ℱ::InverseFlow{<:SimplexFlow})(𝓂ₓ::ProbabilityMass)
    # Flow from normalized distribution `x̄ ∈ Δⁿ⁻¹` to Aitchison coefficients `u ∈ ℝⁿ⁻¹`
    T = eltypefloat(ℱ.flow.Ψ)
    n = size(ℱ.flow.Ψ, 1) # Ψ ∈ ℝⁿˣⁿ⁻¹
    logβ₀ = T(log(π^2 / 6) / 2)
    logsqrtn = T(log(n) / 2)

    # No need to centre logx̄: Ψᵀ𝟙 = 0, so Ψᵀ(z + c𝟙) = Ψᵀz
    (; logβ) = SimplexFlow_unpack(ℱ.flow, 𝓂ₓ.latents) # log inverse temperature
    logx̄ = Base.materialize(𝓂ₓ.value) # logx̄ is the log of a normalized distribution
    u = ℱ.flow.Ψ' * (@. exp(-(logβ + logβ₀)) * logx̄) # ILR basis representation
    ∂u = 𝓂ₓ.logjac === nothing ? nothing : lazy.(simplex_logjac_update(𝓂ₓ.logjac, sum(logx̄; dims = 1), -, logβ, n, logβ₀, logsqrtn))

    return ProbabilityMass(u, ∂u, 𝓂ₓ.latents)
end

# Fused simplex log-Jacobian update; scalar `logβ` is captured as a constant
@inline simplex_logjac_update(∂, s, op::O, logβ::Number, n::Int, logβ₀, logsqrtn) where {O} = Base.broadcasted((a, b) -> op(op(a, b), (n - 1) * (logβ + logβ₀) + logsqrtn), ∂, s)
@inline simplex_logjac_update(∂, s, op::O, logβ, n::Int, logβ₀, logsqrtn) where {O} = Base.broadcasted((a, b, β) -> op(op(a, b), (n - 1) * (β + logβ₀) + logsqrtn), ∂, s, logβ)

#### Aitchison geometry transformations

# DCT and inverse DCT in Aitchison space
simplex_idct(u::AbstractVecOrMat, Ψ = idct_simplex_basis_similar(u)) = Ψ * u # Aitchison coefficients u ∈ ℝᴰ⁻¹ to logits logx̄ ∈ Log(Δᴰ⁻¹) ⊂ ℝᴰ

# Aitchison DCT helpers
idct_simplex_basis_similar(u::AbstractVecOrMat) = arr_similar(u, idct_simplex_basis(float(eltype(u)), size(u, 1))) # transposed DCT basis, size `D x (D-1)`
@non_differentiable idct_simplex_basis_similar(u)

# Aitchison transform and inverse transform
aitchison_inv_transform(u::AbstractVecOrMat, Ψ = idct_simplex_basis_similar(u), β₀ = ofeltypefloat(u, π / √6)) = softmax(simplex_idct(β₀ .* u, Ψ); dims = 1) # transform from Aitchison coefficients `u ∈ ℝⁿ⁻¹` to normalized distribution `x̄ ∈ Δⁿ⁻¹`
aitchison_inv_transform(u::AbstractArray, args...) = reshape(aitchison_inv_transform(reshape(u, size(u, 1), :), args...), :, Base.tail(size(u))...)

"""
Low-pass flow model

Input vectors are assumed to represent batches of 1-D DCT spectra,
where the coefficients are ordered by increasing frequency.
"""
struct LowPassFlowModel{W, T, S}
    w::W
    t::T
    logτ::S
end
@saveable LowPassFlowModel
Optimisers.trainable(model::LowPassFlowModel) = (; w = model.w, logτ = model.logτ)

# The time scale τ is rescaled by the learned scale τ₀: log τ -> log τ - log τ₀
(m::LowPassFlowModel)(logτ) = log_weights(m.w, m.t, logτ .- m.logτ)
(m::LowPassFlowModel)(args::Tuple) = (m(args...),)

function LowPassFlowModel(::Type{T}, insize::Int; noisefloor, time_scale) where {T}
    @assert noisefloor > 0 && time_scale > 0
    t = collect((1:insize) ./ T(insize))
    gaussian = ScaleLowPassWeights(GaussianLowPassWeights(), zeros(T, 1))
    weights = PositiveLowPassWeights(gaussian, [log(T(noisefloor))]) # learned floor, initialized at `noisefloor`
    return LowPassFlowModel(weights, t, [log(T(time_scale))])
end

"""
Resample `m` onto a length-`(n′-1)` time grid `t_k = k / (n - 1)`, with post-floor amplitude factor `½log(n′/n)` so that `w′(t) = √(n′/n) w(t)`.
Here `n = length(m.t) + 1` is the full dimension including the DC component implied by `m`.
"""
function lift_lowpass_window(m::LowPassFlowModel, n′::Int)
    T = eltype(m.t)
    n = length(m.t) + 1
    t′ = collect((1:(n′-1)) ./ T(n - 1))
    w′ = ScaleLowPassWeights(m.w, T(log(n′ / n) / 2))
    return LowPassFlowModel(w′, t′, m.logτ)
end

"""
Gaussian window: w(t) = exp(-t^2 / 2τ²)
"""
struct GaussianLowPassWeights end
@saveable GaussianLowPassWeights

log_weights(::GaussianLowPassWeights, t, logτ) = @lazy -(t * exp(-logτ))^2 / 2

"""
Global weight scale: w₀ * w(t)
"""
struct ScaleLowPassWeights{W, T}
    w::W
    logw₀::T
end
@saveable ScaleLowPassWeights

function log_weights(w::ScaleLowPassWeights, t, logτ)
    logw = log_weights(w.w, t, logτ)
    return @lazy logw + w.logw₀ # add global scale
end

"""
Learned variance floor: w₊(t) = √(w(t)² + w₀²).
"""
struct PositiveLowPassWeights{W, T}
    w::W
    logw₀::T
end
@saveable PositiveLowPassWeights

function log_weights(w::PositiveLowPassWeights, t, logτ)
    logw = log_weights(w.w, t, logτ)
    α = eltype(w.logw₀)(2)
    return @lazy smoothmax(logw, w.logw₀, α)
end

struct AffineFlow{Topts, Trect, Tmodel}
    opts::Topts
    rect::Trect # rectifier f : ℝ → ℝ⁺ assumed to satisfy rect(-x) = 1 / rect(x)
    model::Tmodel # model returning one of: array θ = [logs, t], tuple (logs, t), or tuple (logs,) implying t = 0
end
@saveable AffineFlow

function AffineFlow(model, rectifier = exp; checkpoint::Bool = false)
    @assert rectifier ∈ (exp, squareplus) "Invalid rectifier: $(rectifier)"
    return AffineFlow((; checkpoint = Val(checkpoint)), rectifier, model)
end

AffineFlow_unpack(logs, t, logjac) = (; logs, t, logjac)
AffineFlow_unpack((logs, t, logjac)::NTuple{3, Any}) = AffineFlow_unpack(logs, t, logjac)
AffineFlow_unpack((logs, t)::NTuple{2, Any}) = AffineFlow_unpack(logs, t, nothing)
AffineFlow_unpack((logs,)::NTuple{1, Any}) = AffineFlow_unpack(logs, false, nothing)
AffineFlow_unpack(θ::AbstractArray) = @views AffineFlow_unpack((θ[1:(end÷2), ..], θ[(end÷2+1):end, ..]))

# Log Jacobian for scaling by `logs` with size `[D1 x D2 x ...]`:
#   - if `logs` is scalar, lazily multiply by `n`
#   - if `logs` has `D1 = 1`, multiply by `n`
#   - if `logs` has `D1 > 1`, sum along the first dimension
scale_logjac(::typeof(exp), n::Int, logs::Number) = n * logs
scale_logjac(::typeof(exp), n::Int, logs::Union{AbstractArray, Base.Broadcast.Broadcasted}) = size(logs)[1] == 1 ? Base.broadcasted(*, n, logs) : Base.broadcasted(*, 1, sum(Base.materialize(logs); dims = 1)) # multiplying by 1 for type stability, else output type depends on the size
scale_logjac(::typeof(squareplus), n::Int, logs::Number) = n * logsquareplus(logs)
scale_logjac(::typeof(squareplus), n::Int, logs::Union{AbstractArray, Base.Broadcast.Broadcasted}) = size(logs)[1] == 1 ? Base.broadcasted(*, n, logsquareplus.(logs)) : Base.broadcasted(*, 1, sum(logsquareplus.(logs); dims = 1)) # multiplying by 1 for type stability, else output type depends on the size
scale_logjac(rect, n::Int, logs::Number) = n * log(rect(logs))
scale_logjac(rect, n::Int, logs::Union{AbstractArray, Base.Broadcast.Broadcasted}) = size(logs)[1] == 1 ? Base.broadcasted(*, n, log.(rect.(logs))) : Base.broadcasted(*, 1, sum(log.(rect.(logs)); dims = 1)) # multiplying by 1 for type stability, else output type depends on the size

function (ℱ::AffineFlow)(𝓂ₓ::ProbabilityMass)
    (; opts, rect, model) = ℱ
    n = size(𝓂ₓ.value)[1]

    (; logs, t, logjac) = AffineFlow_unpack(maybe_checkpointed(model, 𝓂ₓ.latents, getfieldstatic(opts, Val(:checkpoint), Val(false))))
    u = @lazy rect(logs) * 𝓂ₓ.value + t
    ∂u = 𝓂ₓ.logjac === nothing ? nothing : logjac === nothing ? lazy.(𝓂ₓ.logjac .+ scale_logjac(rect, n, logs)) : @lazy 𝓂ₓ.logjac + logjac

    return ProbabilityMass(u, ∂u, 𝓂ₓ.latents)
end

function (ℱ::InverseFlow{<:AffineFlow})(𝓂ᵤ::ProbabilityMass)
    (; opts, rect, model) = ℱ.flow
    n = size(𝓂ᵤ.value)[1]

    (; logs, t, logjac) = AffineFlow_unpack(maybe_checkpointed(model, 𝓂ᵤ.latents, getfieldstatic(opts, Val(:checkpoint), Val(false))))
    x = @lazy rect(-logs) * (𝓂ᵤ.value - t) # Requires rect(-x) = 1 / rect(x).
    ∂x = 𝓂ᵤ.logjac === nothing ? nothing : logjac === nothing ? lazy.(𝓂ᵤ.logjac .- scale_logjac(rect, n, logs)) : @lazy 𝓂ᵤ.logjac - logjac

    return ProbabilityMass(x, ∂x, 𝓂ᵤ.latents)
end

"""
Generic affine flow over ℝ².

Internal `model` should return a tuple `(A, b)` where `A` is a [2 x 2 x batchsize] array of linear transformations and `b` is a [2 x batchsize] array of biases.
"""
struct Generic2x2AffineFlow{M}
    model::M
end
@saveable Generic2x2AffineFlow

function (ℱ::Generic2x2AffineFlow)(𝓂ₓ::ProbabilityMass)
    A, b = ℱ.model(𝓂ₓ.latents)
    u = batched_vec(A, 𝓂ₓ.value) .+ b
    ∂u = 𝓂ₓ.logjac === nothing ? nothing : 𝓂ₓ.logjac .+ log.(abs.(Flux.flatten(batched_det_2x2(A))))
    return ProbabilityMass(u, ∂u, 𝓂ₓ.latents)
end

function (ℱ::InverseFlow{<:Generic2x2AffineFlow})(𝓂ᵤ::ProbabilityMass)
    A, b = ℱ.flow.model(𝓂ᵤ.latents)
    x = batched_vec(batched_inv_2x2(A), 𝓂ᵤ.value .- b)
    ∂x = 𝓂ᵤ.logjac === nothing ? nothing : 𝓂ᵤ.logjac .- log.(abs.(Flux.flatten(batched_det_2x2(A))))
    return ProbabilityMass(x, ∂x, 𝓂ᵤ.latents)
end

as_svector(::Type{SVector{n, T}}, x::AbstractArray{T}) where {n, T} = SVector{n, T}.(tuple.(ntuple(i -> x[i:i, ..], Val(n))...))
as_smatrix(::Type{SMatrix{m, n, T, L}}, x::AbstractArray{T}) where {m, n, T, L} = SMatrix{m, n, T, L}.(Tuple.(as_svector(SVector{L, T}, reshape(x, m * n, size(x)[3:end]...))))
as_array(x::AbstractArray{S}) where {n, S <: SVector{n}} = maybevcat(ntuple(i -> Base.Fix2(getindex, i).(x), Val(n))...)
as_array(x::AbstractArray{S}) where {m, n, S <: SMatrix{m, n}} = reshape(as_array(vec.(x)), m, n, :)
ChainRules.rrule(::typeof(as_svector), ::Type{SVector{n, T}}, x) where {n, T} = as_svector(SVector{n, T}, x), Δ -> (NoTangent(), NoTangent(), as_array(Δ))
ChainRules.rrule(::typeof(as_smatrix), ::Type{SMatrix{m, n, T, L}}, x) where {m, n, T, L} = as_smatrix(SMatrix{m, n, T, L}, x), Δ -> (NoTangent(), NoTangent(), as_array(Δ))
ChainRules.rrule(::typeof(as_array), x::AbstractArray{SVector{n, T}}) where {n, T} = as_array(x), Δ -> (NoTangent(), as_svector(SVector{n, T}, Δ))
ChainRules.rrule(::typeof(as_array), x::AbstractArray{SMatrix{m, n, T, L}}) where {m, n, T, L} = as_array(x), Δ -> (NoTangent(), as_smatrix(SMatrix{m, n, T, L}, Δ))

@inline svec(x::NTuple{n, T}) where {n, T} = SVector{n, T}(x)
@inline smat_diag(x::NTuple{n, T}) where {n, T} = smat_diag(SDiagonal{n, T}(x))
@inline smat_diag(x::LinearAlgebra.Diagonal{T, SVector{n, T}}) where {n, T} = SMatrix{n, n, T, n * n}(x)
@inline smat_herm(x::NTuple{1, T}) where {T} = SMatrix{1, 1, T, 1}(x)
@inline smat_herm(x::NTuple{3, T}) where {T} = SMatrix{2, 2, T, 4}((x[1], x[2], x[2], x[3])) # [a b; b c]
@inline smat_herm(x::NTuple{6, T}) where {T} = SMatrix{3, 3, T, 9}((x[1], x[2], x[3], x[2], x[4], x[5], x[3], x[5], x[6])) # [a b c; b d e; c e f]

# Split batched 2x2 matrices [a b; c d] into (a, b, c, d), each [1 x 1 x ..]
function batched_parts_2x2(A)
    @assert_ad size(A, 1) == size(A, 2) == 2
    return @views (A[1:1, 1:1, ..], A[1:1, 2:2, ..], A[2:2, 1:1, ..], A[2:2, 2:2, ..])
end

# determinants of batches of 2x2 matrices: [2 x 2 x ..] => [1 x 1 x ..]
function batched_det_2x2(A)
    (a, b, c, d) = batched_parts_2x2(A)
    return @. a * d - b * c
end

# inverses of batches of 2x2 matrices: [2 x 2 x ..] => [2 x 2 x ..]
function batched_inv_2x2(A)
    (a, b, c, d) = batched_parts_2x2(A)
    return reshape([d; -c; -b; a], 2, 2, size(A)[3:end]...) ./ batched_det_2x2(A)
end

batched_apply_2x2(f::F, A::AbstractArray) where {F} = unflatten_from_smat_2x2(f.(flatten_to_smat_2x2(A)))
batched_invsqrt_eig_2x2(A::AbstractArray) = batched_apply_2x2(eiginvsqrt_psd_2x2, A)

function batched_apply_2x2_inner_pullback(f::F, a::SMatrix{2, 2, T, 4}, δ::SMatrix{2, 2, T, 4}) where {F, T}
    J = ForwardDiff.jacobian(f, a)::SMatrix{4, 4, T, 16}
    return SMatrix{2, 2, T, 4}(J' * vec(δ))
end

function ChainRules.rrule(::typeof(batched_apply_2x2), f::F, A::AbstractArray{T}) where {F, T}
    function batched_apply_2x2_pullback(Δ)
        Δ = ChainRules.unthunk(Δ)
        ΔA = batched_apply_2x2_inner_pullback.(f, flatten_to_smat_2x2(A), flatten_to_smat_2x2(Δ))
        return (NoTangent(), NoTangent(), unflatten_from_smat_2x2(ΔA))
    end
    return batched_apply_2x2(f, A), batched_apply_2x2_pullback
end

flatten_to_smat_2x2(A::AbstractArray{T}) where {T} = reinterpret(SMatrix{2, 2, T, 4}, reshape(A, 4, size(A)[3:end]...))
unflatten_from_smat_2x2(B::AbstractArray{SMatrix{2, 2, T, 4}}) where {T} = reshape(reinterpret(T, B), 2, 2, size(B)[2:end]...)
ChainRules.rrule(::typeof(flatten_to_smat_2x2), A::AbstractArray{T}) where {T} = flatten_to_smat_2x2(A), Δ -> (NoTangent(), unflatten_from_smat_2x2(ChainRules.unthunk(Δ)))
ChainRules.rrule(::typeof(unflatten_from_smat_2x2), B::AbstractArray{SMatrix{2, 2, T, 4}}) where {T} = unflatten_from_smat_2x2(B), Δ -> (NoTangent(), flatten_to_smat_2x2(ChainRules.unthunk(Δ)))

function eigen_psd_2x2(A::NTuple{4, T}) where {T}
    a11, a21, a12, a22 = A
    a12 = (a12 + a21) / 2

    λ1 = ((a11 + a22) + sign(a11 + a22) * sqrt((a11 - a22)^2 + 4 * a12^2)) / 2
    if iszero(a12) && a11 == a22
        return (a11, a22), (one(T), zero(T), zero(T), one(T))
    end
    λ2 = (a11 * a22 - a12^2) / λ1

    if abs(a11 - λ1) > abs(a22 - λ1)
        v11 = -a12
        v21 = a11 - λ1
    else
        v11 = -(a22 - λ1)
        v21 = a12
    end
    v1nrm = sqrt(v11^2 + v21^2)
    v11 /= v1nrm
    v21 /= v1nrm

    if abs(a11 - λ2) > abs(a22 - λ2)
        v12 = -a12
        v22 = a11 - λ2
    else
        v12 = -(a22 - λ2)
        v22 = a12
    end
    v2nrm = sqrt(v12^2 + v22^2)
    v12 /= v2nrm
    v22 /= v2nrm

    return (λ1, λ2), (v11, v21, v12, v22)
end
function eigen_psd_2x2(A::SMatrix{2, 2, T}) where {T}
    D, V = eigen_psd_2x2(Tuple(A))
    return SVector{2, T}(D), SMatrix{2, 2, T, 4}(V)
end

function eiginvsqrt_psd_2x2(A::SMatrix{2, 2, T}) where {T}
    D, V = eigen_psd_2x2(A)
    return V ./ sqrt.(ifelse.(D' .< 0, T(NaN), max.(D', D[1] * eps(T)))) # floor unresolved eigenvalues in the inverse only; negative curvature gives NaN
end
