"""
Sinusoidal positional encoding

    A. Vaswani et al., "Attention Is All You Need," http://arxiv.org/abs/1706.03762
"""
function SinusoidalPositionalEncoding(x::AbstractArray{T}, θ::AbstractFloat = T(10000.0), scale::AbstractFloat = one(T)) where {T <: AbstractFloat}
    tokensize, seqlength = Int32(size(x, 1)), Int32(size(x, 2)) # `x` assumed to have shape `tokensize x seqlength[ x batchsize]`
    return T(scale) .* x .+ sincos_pos_encoding_kernel.(T(θ), tokensize, zero(Int32):(tokensize-one(Int32)), (zero(Int32):(seqlength-one(Int32)))')
end
SinusoidalPositionalEncoding(xs::Tuple{Vararg{AbstractArray}}, θs::Union{AbstractFloat, Tuple} = ofeltypefloat(first(xs), 10000.0), scale::AbstractFloat = ofeltypefloat(first(xs), 1)) = SinusoidalPositionalEncoding.(xs, θs, scale)

# Frequency and content scale are fixed hyperparameters.
ChainRules.rrule(::typeof(SinusoidalPositionalEncoding), x::AbstractArray, θ) = SinusoidalPositionalEncoding(x, θ), Δ -> (NoTangent(), Δ, NoTangent())
ChainRules.rrule(::typeof(SinusoidalPositionalEncoding), x::AbstractArray, θ, scale::AbstractFloat) = SinusoidalPositionalEncoding(x, θ, scale), Δ -> (NoTangent(), eltype(x)(scale) .* ChainRules.unthunk(Δ), NoTangent(), NoTangent())

# Sinusoidal positional encoding kernel:
# out = { sin(pos / 10000^(2i / d)), if k = 2i
#       { cos(pos / 10000^(2i / d)), if k = 2i + 1
function sincos_pos_encoding_kernel(θ::T, d::Int32, k::Int32, pos::Int32) where {T <: AbstractFloat}
    two = Int32(2)
    i = k ÷ two
    sinθ, cosθ = sincos(pos * θ^(T(-two * i) / d))
    return ifelse(k % two == zero(Int32), sinθ, cosθ)
end

struct MultiHeadAttention{Topts, Q, K, V, O}
    opts::Topts
    qproj::Q
    kproj::K
    vproj::V
    oproj::O
end
@saveable MultiHeadAttention

Base.show(io::IO, mh::MultiHeadAttention) = print(io, "MultiHeadAttention$(mh.opts)")

function MultiHeadAttention(insize::Int, nheads::Int, headsize::Int; init, scale)
    return MultiHeadAttention(
        (; nheads, scale),
        Flux.Dense(insize, headsize * nheads; init, bias = false),
        Flux.Dense(insize, headsize * nheads; init, bias = false),
        Flux.Dense(insize, headsize * nheads; init, bias = false),
        Flux.Dense(headsize * nheads, insize; init, bias = false),
    )
end

function (mh::MultiHeadAttention)(Q, K, V)
    (; nheads, scale) = mh.opts
    QP, KP, VP = mh.qproj(Q), mh.kproj(K), mh.vproj(V)
    return mh.oproj(attention(QP, KP, VP; nheads, scale))
end

#### Dot product attention

function attention(Q::AbstractMatrix, K::AbstractMatrix, V::AbstractMatrix; kwargs...)
    batched = map(x -> reshape(x, size(x)..., 1), (Q, K, V))
    return reshape(attention(batched...; kwargs...), size(V, 1), size(Q, 2))
end

function attention(Q::AbstractTensor3D{T}, K::AbstractTensor3D{T}, V::AbstractTensor3D{T}; nheads::Int, scale = nothing) where {T}
    @assert size(Q, 1) == size(K, 1) && size(K, 2) == size(V, 2) && size(Q, 3) == size(K, 3) == size(V, 3)
    qseqlength, kvseqlength, batchsize = size(Q, 2), size(K, 2), size(K, 3)
    qkheadsize = size(Q, 1) ÷ nheads
    vheadsize = size(V, 1) ÷ nheads

    Q = reshape(Q, qkheadsize, nheads, qseqlength, batchsize)
    K = reshape(K, qkheadsize, nheads, kvseqlength, batchsize)
    V = reshape(V, vheadsize, nheads, kvseqlength, batchsize)

    S = reshape(
        batched_mul(
            reshape(permutedims(K, (3, 1, 2, 4)), kvseqlength, qkheadsize, nheads * batchsize),
            reshape(permutedims(Q, (1, 3, 2, 4)), qkheadsize, qseqlength, nheads * batchsize),
        ),
        kvseqlength, qseqlength, nheads, batchsize,
    )

    S = S .* T(something(scale, 1 / sqrt(qkheadsize)))

    A = softmax(S; dims = 1)

    X = permutedims(
        reshape(
            batched_mul(
                reshape(permutedims(V, (1, 3, 2, 4)), vheadsize, kvseqlength, nheads * batchsize),
                reshape(A, kvseqlength, qseqlength, nheads * batchsize),
            ),
            vheadsize, qseqlength, nheads, batchsize,
        ),
        (1, 3, 2, 4),
    )
    X = reshape(X, vheadsize * nheads, qseqlength, batchsize)

    return X
end

struct Transformer{MH, MHNQ, MHNK, FF, FFN}
    mh::MH
    mhnq::MHNQ
    mhnk::MHNK
    ff::FF
    ffn::FFN
end
@saveable Transformer

function Transformer(insize::Int, nheads::Int, headsize::Int, hiddensize::Int; init, scale)
    mh = MultiHeadAttention(insize, nheads, headsize; init, scale)
    mhnq, mhnk = RMSNorm(insize), RMSNorm(insize)
    ff = GatedLinearUnit(insize => hiddensize => insize, NNlib.swish; skip = false, prenorm = identity, init, bias = false)
    return Transformer(mh, mhnq, mhnk, ff, RMSNorm(insize))
end

function (xf::Transformer)((Q, KV)::Tuple{Vararg{AbstractArray, 2}})
    Q̄, K̄ = xf.mhnq(Q), xf.mhnk(KV)
    X = Q .+ xf.mh(Q̄, K̄, K̄)
    X = X .+ xf.ff(xf.ffn(X))
    return (X, KV)
end
