broadcast_function(f) = Base.BroadcastFunction(f)
broadcast_function(::typeof(identity)) = identity
broadcast_function(f::Base.BroadcastFunction) = f

"""
catchain(chain)

Concatenation of `c1` and `c2` into a `VectorChain`.
"""
const ChainOrVec = Union{Flux.Chain, AbstractVector}
catchain(c1::ChainOrVec, c2::ChainOrVec) = VectorChain(Any[c1..., c2...])
catchain(c1::ChainOrVec, c2) = VectorChain(Any[c1..., c2])
catchain(c1, c2::ChainOrVec) = VectorChain(Any[c1, c2...])
catchain(c1, c2) = VectorChain(Any[c1, c2])

"""
flattenchain(chain)

Recursively flattens `chain`, removing redundant `Chain` wrappers.
"""
function flattenchain(T, chain::ChainOrVec)
    length(chain) == 1 && return flattenchain(T, chain[1]) # trivial chain
    layers = reduce(catchain, flattenchain.(T, chain); init = Any[]) # concatenate flattened chains
    layers = filter(f -> f !== identity, Any[layers...]) # drop trivial identity layers
    isempty(layers) && return identity # all layers were trivial; return identity
    length(layers) == 1 && return layers[1] # reduced to single layer
    return T(layers)
end
flattenchain(T, chain) = chain

"""
AnyChain(layers...)

Like `Flux.Chain`, but recursively flattens redundant layers using `flattenchain`.
Returns a `VectorChain`.
"""
AnyChain(layers...) = flattenchain(VectorChain, Any[layers...])

"""
TypedChain(layers...)

Like `AnyChain`, but layers are stored as `Tuple`s.
Returns a `TupleChain`.
"""
TypedChain(layers...) = flattenchain(TupleChain, Any[layers...])

"""
NotTrainable(layer)

Wraps the callable `layer` such any that parameters internal to `layer`
are treated as constant by Flux during training.
"""
struct NotTrainable{F}
    layer::F
end
@saveable NotTrainable
Optimisers.trainable(::NotTrainable) = NamedTuple() # no trainable parameters

NotTrainable(l::NotTrainable) = l

Base.show(io::IO, l::NotTrainable) = print(io, "NotTrainable(h = ", l.layer isa Number ? l.layer : summary(l.layer), ")")

@inline (l::NotTrainable)(xs...) = whitebox_apply(l.layer, xs...)

@inline whitebox_apply(f, xs...) = ignore_derivatives(f)(xs...)

"""
    ConstantScale(scale[, shift])

Elementwise affine map `scale .* x .+ shift` with constant parameters; `shift = nothing` scales only.
"""
struct ConstantScale{S, T}
    scale::S
    shift::T
end
@saveable ConstantScale
Optimisers.trainable(::ConstantScale) = NamedTuple()

ConstantScale(scale) = ConstantScale(scale, nothing)
ConstantScale(l::NotTrainable) = ConstantScale(l.layer)

function ConstantScale(l::Flux.Scale)
    @assert l.σ === identity "ConstantScale requires an identity activation; got $(l.σ)"
    return ConstantScale(l.scale, l.bias === false ? nothing : l.bias)
end

Base.show(io::IO, l::ConstantScale) = print(io, "ConstantScale(", summary(l.scale), l.shift === nothing ? "" : ", " * summary(l.shift), ")")

@inline (l::ConstantScale)(x::AbstractArray) = constant_scale(l.scale, l.shift, x)

@inline constant_scale(s, ::Nothing, x) = s .* x
@inline constant_scale(s, t, x) = s .* x .+ t

function ChainRules.rrule(::typeof(constant_scale), s, t, x)
    constant_scale_pullback(Δ) = (NoTangent(), NoTangent(), NoTangent(), s .* ChainRules.unthunk(Δ))
    return constant_scale(s, t, x), constant_scale_pullback
end

function BatchNorm2Scale(bn::Flux.BatchNorm; freeze = true)
    @assert bn.track_stats "BatchNorm layer must track statistics"
    scale, bias = batchnorm_affine_transform(bn)
    freeze && bn.λ === identity && return ConstantScale(scale, bias)
    l = Flux.Scale(scale, bias, bn.λ)
    return freeze ? NotTrainable(l) : l
end

function batchnorm_affine_transform(bn::Flux.BatchNorm)
    (; λ, γ, β, μ, σ², ϵ) = bn
    μ, σ = μ, sqrt.(σ² .+ ofeltypefloat(σ², ϵ))
    if Flux.hasaffine(bn)
        # x <- λ(γ * (x - μ) / σ + β) = λ(γ / σ * x + β - γ * μ / σ)
        return γ ./ σ, β .- γ .* μ ./ σ
    else
        # x <- λ((x - μ) / σ) = λ(x / σ - μ / σ)
        return inv.(σ), .-μ ./ σ
    end
end

"""
Concatenate inputs along a dimension.
"""
struct Concat{N}
    N::Val{N}
end
@saveable Concat
Optimisers.trainable(::Concat) = NamedTuple() # no trainable parameters

Base.show(io::IO, ::Concat{N}) where {N} = print(io, "Concat(", N, ")")

@inline Concat(N::Int) = Concat(Val(N))

@inline (c::Concat)(xs...) = c(xs)
@inline (::Concat{1})(xs::Union{<:Tuple, <:NamedTuple}) = maybevcat(xs)
@inline (::Concat{2})(xs::Union{<:Tuple, <:NamedTuple}) = maybehcat(xs)
@inline (::Concat{N})(xs::Union{<:Tuple, <:NamedTuple}) where {N} = maybecat(xs; dims = Val(N))

@inline maybevcat(xs::Union{<:Tuple, <:NamedTuple}) = maybevcat(xs...)
@inline maybehcat(xs::Union{<:Tuple, <:NamedTuple}) = maybehcat(xs...)
@inline maybecat(xs::Union{<:Tuple, <:NamedTuple}; dims) = maybecat(xs...; dims)

@inline maybevcat(x, xs...) = isempty(xs) ? x : cat(map(Base.materialize, (x, xs...))...; dims = 1)
@inline maybehcat(x, xs...) = isempty(xs) ? x : cat(map(Base.materialize, (x, xs...))...; dims = 2)
@inline maybecat(x, xs...; dims) = isempty(xs) ? x : cat(map(Base.materialize, (x, xs...))...; dims)

struct TuplifyInputs{F}
    f::F
end
@saveable TuplifyInputs

TuplifyInputs(fs...) = TuplifyInputs(AnyChain(fs...))

@inline (m::TuplifyInputs)(xs...) = call_unsplatted(m.f, xs...)

"""
GetIndex(inds...)

Non-trainable layer which indexes into input argument `x` as `getindex(x, inds...)`.
"""
struct GetIndex{inds}
    inds::Val{inds}
end
@saveable GetIndex

Base.show(io::IO, ::GetIndex{inds}) where {inds} = print(io, "GetIndex$(inds)")

GetIndex(args...) = GetIndex(args)
GetIndex(; kwargs...) = GetIndex(NamedTuple(kwargs))
function GetIndex(inds::Tuple)
    @assert(_check_getindex(inds))
    return GetIndex(Val(inds))
end

function GetIndex(inds::NamedTuple)
    @assert(_check_getindex(inds))
    return GetIndex(Val(inds))
end

_check_getindex(inds) = _check_getindex_error(inds)
_check_getindex(inds::Tuple) = !any(i -> i isa Tuple, inds) || _check_getindex_error(inds)
_check_getindex(inds::Tuple{Vararg{Tuple}}) = all(_check_getindex, inds)
_check_getindex(inds::NamedTuple) = all(_check_getindex, inds)
function _check_getindex_error(inds)
    error("Expected either a tuple of tuples, a named tuple of tuples, or a tuple of indices. Received inds = $inds")
end

@inline (g::GetIndex{inds})(x) where {inds} = _getindex(x, inds)
@inline _getindex(x, inds::Tuple) = _getindex(x, inds...)
@inline _getindex(x, inds::Tuple{Vararg{Tuple}}) = map(i -> _getindex(x, i), inds)
@inline _getindex(x, inds::NamedTuple) = map(i -> _getindex(x, i), inds)
@inline _getindex(x, I::Union{Int, AbstractRange{Int}, Colon, Ellipsis}...) = view(Base.materialize(x), I...)

"""
    RMSNorm(size::Int...; init = zeros, eps = 1e-8)

Implements Root Mean Square Normalization introduced in https://arxiv.org/pdf/1910.07467.pdf.
"""
struct RMSNorm{F, W, T, N}
    f::F
    γ::W
    eps::T
    size::NTuple{N, Int}
end
@saveable RMSNorm

RMSNorm(size::Int...; init = zeros, eps = 1e-8) = RMSNorm(squareplus, Float32.(init(size)), Float32(eps), size)

(rms::RMSNorm)(x::AbstractArray) = rms.f.(rms.γ) .* x ./ sqrt.(mean(abs2, x; dims = 1:length(rms.size)) .+ ofeltypefloat(x, rms.eps))

"""
    GatedLinearUnit(insize => hdim [=> outsize], σ = Flux.swish; kwargs...)

Implements the Gated Linear Unit (GLU) introduced in https://arxiv.org/abs/2002.05202v1.
"""
struct GatedLinearUnit{Topts, N, G, L, O}
    opts::Topts
    prenorm::N
    gate::G
    linear::L
    output::O
end
@saveable GatedLinearUnit
Optimisers.trainable(m::GatedLinearUnit) = (; m.prenorm, m.gate, m.linear, m.output)

function GatedLinearUnit((insize, (hdim, outsize))::Pair{Int, Pair{Int, Int}}, σ = Flux.swish; skip::Bool, prenorm, kwargs...)
    gate = Flux.Dense(insize => hdim, σ; kwargs...)
    linear = Flux.Dense(insize => hdim, identity; kwargs...)
    output = Flux.Dense(hdim => outsize, identity; kwargs...)
    return GatedLinearUnit((; skip), prenorm, gate, linear, output)
end
GatedLinearUnit((insize, hdim)::Pair{Int, Int}, args...; kwargs...) = GatedLinearUnit(insize => hdim => insize, args...; kwargs...)

function (m::GatedLinearUnit)(x::AbstractVecOrMat)
    (; opts, prenorm, gate, linear, output) = m
    xn = prenorm(x)
    h = gate.σ.(gate.weight * xn .+ gate.bias) .* (linear.weight * xn .+ linear.bias) # fusing the broadcast is faster than two bias_act! calls
    y = bias_act!(identity, output.weight * h, output.bias)
    return opts.skip ? y .+ x : y
end
(m::GatedLinearUnit)(x::AbstractArray) = reshape(m(reshape(x, size(x, 1), :)), :, Base.tail(size(x))...)

"""
SqueezeRange

Smoothly squeeze unconstrained input into learnable range.
Maps `x -> scale * f(x / scale) + bias` where `scale = rect(logs)`.
"""
struct SqueezeRange{F, R, S, B}
    f::F # squeezer, e.g. tanh, such that f(x) is bounded
    rect::R # rectifier, e.g. squareplus, mapping ℝ to [0, ∞)
    logs::S # argument to rectifier
    bias::B # bias term
end
@saveable SqueezeRange

Base.show(io::IO, l::SqueezeRange) = print(io, "SqueezeRange(size = ", size(l.logs), ", f = ", l.f, ", rect = ", l.rect, ", bias = ", l.bias != 0, ")")

SqueezeRange(sz::Int...; init = Flux.zeros32, bias = false) = SqueezeRange(Flux.tanh_fast, squareplus, init(sz), bias ? init(sz) : false)

function (l::SqueezeRange)(x)
    # Smooth squeeze: approximately identity for |x| << scale, saturating below scale for |x| >> scale
    (; f, rect, logs, bias) = l
    scale = rect.(logs)
    return @. scale * f(x / scale) + bias
end
