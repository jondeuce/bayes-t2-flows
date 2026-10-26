#### Extensible saving/loading

# Store named fields without concrete type parameters; rebuild checkpoints using the current types.
struct TypeErased
    type::Any
    data::Any
end

should_erase_types(x) = false
should_rebuild_types(x) = false
should_rebuild_types(x::TypeErased) = true

erase_types(x) = fmap(erase_types, x; exclude = should_erase_types)
erase_types(x::NamedTuple{Fs}) where {Fs} = NamedTuple{Fs, NTuple{length(Fs), Any}}(Tuple(map(erase_types, x)))
rebuild_types(x) = fmap(rebuild_types, x; exclude = should_rebuild_types)
rebuild_types(x::TypeErased) = rebuild_types(x.type, map(rebuild_types, x.data))
rebuild_types(T::Type, x::NamedTuple) = constructorof(T)(x...)

# Wrap `constructorof` from ConstructionBase.jl in a generated function
@generated constructorof(::Type{T}) where {T} = Accessors.ConstructionBase.constructorof(T)

"""
    @saveable T [leaf = true]

Store a type's named fields in checkpoints, optionally marking it as a Functors leaf.
"""
macro saveable(T, options...)
    @assert isempty(options) || options == (:(leaf = true),)
    ex = quote
        $(@__MODULE__).should_erase_types(x::$T) = true
        function $(@__MODULE__).erase_types(x::$T)
            local fs = fieldnames($T)
            return $TypeErased($T, NamedTuple{fs}(map(f -> $erase_types(getfield(x, f)), fs)))
        end
    end
    isempty(options) || push!(ex.args, :($Functors.@leaf $T))
    return esc(ex)
end

#### Function call capturing

struct FunctionCaller{F, Tpre, Tpost, Tkw}
    f::F
    preargs::Tpre
    postargs::Tpost
    kwargs::Tkw
end
@saveable FunctionCaller

@inline (c::FunctionCaller{F, Tpre, Tpost, Tkw})(x...) where {F, Tpre, Tpost, Tkw} = c.f(c.preargs..., x..., c.postargs...; c.kwargs...)
@inline (c::FunctionCaller{F, Tpre, Nothing, Tkw})(x...) where {F, Tpre, Tkw} = c.f(c.preargs...; c.kwargs...)

"""
Extracts function arguments from a call expression.

Usage:

    @call f(preargs..., _, postargs...; kwargs...)

Example:

    @call sum(abs2, _; dims = 1)

Returns a `FunctionCaller` instance capturing `f`, `preargs`, `postargs`, and `kwargs`.
"""
macro call(ex)
    return callm(ex)
end

function callm(ex)
    local f, args, kwargs
    if !MacroTools.@capture(ex, f_(args__; kwargs__))
        !MacroTools.@capture(ex, f_(args__)) && error("Expected `call` expression")
        kwargs = pairs((;))
    end
    inds = findall(args .=== :_)
    length(inds) > 1 && error("Expected at most one `_` positional argument in `call` expression")
    i = isempty(inds) ? length(args) + 1 : only(inds) # no `_` found; treat as empty postargs
    return quote
        $FunctionCaller(
            $(esc(f)), # function
            ($([esc(x) for x in args[begin:(i-1)]]...),), # positional arguments before `_`
            $(i > length(args) ? nothing : Expr(:tuple, [esc(x) for x in args[(i+1):end]]...)), # positional arguments after `_`
            (; $([esc(x) for x in kwargs]...)), # keyword arguments
        )
    end
end

#### Convenience

Base.@propagate_inbounds second(x) = x[2]

@inline call_unsplatted(f, xs...) = f(xs)
@inline call_unsplatted(f, xs::Tuple) = f(xs)

macro assert_ad(ex, msgs...)
    # Assert expression within a `ChainRulesCore.ignore_derivatives` block
    msg = isempty(msgs) ? string(ex) : esc(msgs[1]) # message uses unescaped expression for prettier printing
    return :($ignore_derivatives() do
        @assert $(esc(ex)) $(msg)
    end)
end

#### Array

const AbstractTensor3D{T} = AbstractArray{T, 3}
const AbstractTensor4D{T} = AbstractArray{T, 4}
const CuTensor3D{T} = CuArray{T, 3}
const CuTensor4D{T} = CuArray{T, 4}

"""
Move objects between CPU and GPU. Ideally, we would just define

    cpu(x) = Flux.cpu(x)
    gpu(x) = Flux.gpu(x) # opinionated conversion, e.g. Float64 -> Float32

But using `fmap` directly gives more control over the recursion.
"""
todevice(to, x) = fmap(to, x; exclude = isleafnode)
isleafnode(x) = Functors.isleaf(x)
isleafnode(x::AbstractArray) = !(x isa Adapt.WrappedArray) && isbitstype(eltype(x)) # `Adapt.WrappedArray` must be recursed into

adaptgpu(x) = Adapt.adapt(CuArray, x)
adaptcpu(x) = Adapt.adapt(Array, x)

const USE_GPU = Ref(false)

cpu(x) = todevice(adaptcpu, x)
gpu(x) = USE_GPU[] ? todevice(CUDA.cu, x) : cpu(x) # opinionated conversion, e.g. Float64 -> Float32

cpu(::Type{Float32}, x) = todevice(Flux.f32 ∘ adaptcpu, x)
gpu(::Type{Float32}, x) = USE_GPU[] ? todevice(Flux.f32 ∘ adaptgpu, x) : cpu(Float32, x)
cpu(::Type{Float64}, x) = todevice(Flux.f64 ∘ adaptcpu, x)
gpu(::Type{Float64}, x) = USE_GPU[] ? todevice(Flux.f64 ∘ adaptgpu, x) : cpu(Float64, x)

cpu64(x) = cpu(Float64, x)
cpu32(x) = cpu(Float32, x)
gpu64(x) = gpu(Float64, x)
gpu32(x) = gpu(Float32, x)

ChainRules.rrule(::typeof(cpu), x::AbstractArray) = cpu(x), Δ -> (NoTangent(), arr_similar(x, Δ))
ChainRules.rrule(::typeof(gpu), x::AbstractArray) = gpu(x), Δ -> (NoTangent(), arr_similar(x, Δ))
ChainRules.rrule(::typeof(cpu), ::Type{T}, x::AbstractArray) where {T} = cpu(T, x), Δ -> (NoTangent(), NoTangent(), arr_similar(x, Δ))
ChainRules.rrule(::typeof(gpu), ::Type{T}, x::AbstractArray) where {T} = gpu(T, x), Δ -> (NoTangent(), NoTangent(), arr_similar(x, Δ))

cpu(::Type{A}) where {A <: AbstractArray} = Array{eltype(A), ndims(A)}
gpu(::Type{A}) where {A <: AbstractArray} = USE_GPU[] ? CuArray{eltype(A), ndims(A), CUDA.DeviceMemory} : cpu(A)

device_similar(x, y) = device_similar(get_device(x), y)
device_similar(::CPUDevice, y) = cpu(y)
device_similar(::CUDADevice, y) = gpu(y)

function safedevice(x, device)
    θx = alltrainable(x; path = true)
    θx = map(((k, x),) -> (k, copy(x)), θx)
    y = device(x)
    θy = alltrainable(y; path = true)
    @assert length(θx) == length(θy) "Mismatched number trainable parameters: $(length(θx)) != $(length(θy))"
    for ((kx, xi), (ky, yi)) in zip(θx, θy)
        @assert kx == ky "Mismatched trainable parameter key: $(kx) != $(ky)"
        @assert size(xi) == size(yi) "Mismatched trainable parameter size at k = $(kx): $(size(xi)) != $(size(yi))"
    end
    return y
end
safegpu(x) = safedevice(x, gpu)

# Recursively map over `AbstractArray`s, preserving type
function fmap_array(f, x, args...; kwargs...)
    x′ = fmap(x, args...; exclude = isnumeric, kwargs...) do y, ys...
        return f(y, ys...)::typeof(y) # inference through fmap is problematic, but here we restrict to type-preseving array ops `f`
    end
    return convert(typeof(x), x′) # keep the original container type; inference can narrow typeof(x′), e.g. Vector{Any} -> Vector{Array}
end

# Map reduce over nested structure
function fmap_reduce(f, op, xs...; init, kwargs...)
    acc = Ref(init)
    fmapstructure(xs...; kwargs...) do x...
        y = f(x...)
        acc[] = op(acc[], y)
        return nothing
    end
    return acc[]
end
fmap_all(f, xs...; kwargs...) = fmap_reduce(f, &, xs...; init = true, kwargs...)

# Repeat each slice along the last dimension `rep` times consecutively
repeatbatch(x::AbstractArray{<:Number}, rep::Int) = rep == 1 ? x : reshape(repeat(reshape(x, size(x)[1:(end-1)]..., 1, size(x)[end]), ntuple(_ -> 1, ndims(x) - 1)..., rep, 1), size(x)[1:(end-1)]..., size(x)[end] * rep)
repeatbatch(rep::Int) = Base.Fix2(repeatbatch, rep)

for f in [:zeros, :ones, :rand, :randn, :randexp]
    f_similar = Symbol(f, :_similar)
    f_similar_row = Symbol(f, :_similar_row)
    @eval $f_similar(x::AbstractArray, sz::Int...) = $f_similar(x, sz)
    @eval $f_similar(x::AbstractArray, sz::Tuple = size(x)) = $f_similar(get_device(x), eltype(x), sz)
    @eval $f_similar(dev::AbstractDevice, ::Type{T}, sz::Int...) where {T} = $f_similar(dev, T, sz)
    if f === :randexp
        @eval $f_similar(x::Number) = Random.randexp(float(typeof(x)))
        @eval $f_similar(::CPUDevice, ::Type{T}, sz::NTuple{N, Int}) where {T, N} = @ignore_derivatives Random.randexp(T, sz...)::Array{T, N} # fallback
        @eval $f_similar(::CUDADevice, ::Type{T}, sz::NTuple{N, Int}) where {T, N} = @ignore_derivatives (x = CUDA.zeros(T, sz...); map!($f_similar, x, x); x)::CuArray{T, N} # CUDA
    else
        @eval $f_similar(::CPUDevice, ::Type{T}, sz::NTuple{N, Int}) where {T, N} = @ignore_derivatives Base.$f(T, sz...)::Array{T, N} # fallback
        @eval $f_similar(::CUDADevice, ::Type{T}, sz::NTuple{N, Int}) where {T, N} = @ignore_derivatives CUDA.$f(T, sz...)::CuArray{T, N} # CUDA
    end
    @eval $f_similar_row(x::AbstractArray) = $f_similar(x, 1, Base.tail(size(x))...)
end

fill_similar(x::AbstractArray, v, sz::Int...) = fill_similar(x, v, sz)
fill_similar(x::AbstractArray, v, sz::Tuple = size(x)) = fill_similar(get_device(x), eltype(x), v, sz)
fill_similar(dev::AbstractDevice, ::Type{T}, sz::Int...) where {T} = fill_similar(dev, T, sz)
fill_similar(::CPUDevice, ::Type{T}, v, sz::Tuple) where {T} = @ignore_derivatives Base.fill(T(v), sz...) # fallback
fill_similar(::CUDADevice, ::Type{T}, v, sz::Tuple) where {T} = @ignore_derivatives CUDA.fill(T(v), sz...) # CUDA
fill_similar_row(x::AbstractArray, v) = fill_similar(x, v, 1, Base.tail(size(x))...)

# Array of similar device type
arr_similar(x::AbstractArray, y::AbstractArray) = arr_similar(get_device(x), eltype(x), y)
arr_similar(::CPUDevice, ::Type{T}, y::AbstractArray) where {T} = Adapt.adapt(Array{T}, y) # fallback
arr_similar(::CUDADevice, ::Type{T}, y::AbstractArray) where {T} = Adapt.adapt(CuArray{T}, y)
arr_similar(x::AbstractArray, Δ::ChainRules.AbstractThunk) = arr_similar(x, ChainRules.unthunk(Δ))
arr_similar(::AbstractArray, Δ::ChainRules.AbstractZero) = Δ
ChainRules.rrule(::typeof(arr_similar), x::AbstractArray, y::AbstractArray) = arr_similar(x, y), Δ -> (NoTangent(), NoTangent(), arr_similar(y, Δ)) # preserve input type on backward pass

gpu_similar(x::AbstractArray) = gpu_similar(get_device(x), x)
gpu_similar(::CPUDevice, x::AbstractArray{T}) where {T} = gpu(T, x) # adapt cpu array to gpu
gpu_similar(::CUDADevice, x) = x # identity for gpu arrays
ChainRules.rrule(::typeof(gpu_similar), x::AbstractArray) = gpu_similar(x), Δ -> (NoTangent(), arr_similar(x, Δ)) # adapt Δ to type of x
with_gpu(f, args::AbstractArray...) = device_similar(args[1], f(map(gpu_similar, args)...))

eye_similar(x::AbstractArray, diag::Number, sz::Tuple) = batched_diag!(zeros_similar(x, first(sz), first(sz), Base.tail(sz)...), diag)
eye_similar(x::AbstractArray, sz::Tuple) = eye_similar(x, one(eltype(x)), sz)
eye_similar(x::AbstractArray, diag::AbstractFloat, sz1::Int = size(x, 1), sz::Int...) = eye_similar(x, diag, (sz1, sz...)) # need `diag::AbstractFloat` to disambiguate with `sz1::Int`
eye_similar(x::AbstractArray, sz1::Int = size(x, 1), sz::Int...) = eye_similar(x, (sz1, sz...))
@non_differentiable eye_similar(x, diag::Number, sz::Tuple)

bool_similar(x::AbstractArray, v::Bool, sz::Int...) = bool_similar(x, v, sz)
bool_similar(x::AbstractArray, v::Bool, sz::Tuple = size(x)) = bool_similar(get_device(x), x, v, sz)
bool_similar(::CPUDevice, ::AbstractArray, v::Bool, sz::Tuple) = fill(v, sz)
bool_similar(::CUDADevice, ::AbstractArray, v::Bool, sz::Tuple) = CUDA.fill(v, sz)

@inline ofeltypefloat(x::Number, y) = oftype(float(x), y)
@inline ofeltypefloat(x, y) = convert(float(eltype(x)), y)
@inline oftypefloat(x, y) = oftype(float(x), y)
@inline eltypefloat(x) = float(eltype(x))

# Recursively convert all `AbstractFloat`s to `T`
struct ConvertAllFloat{T} end
@inline ConvertAllFloat(::Type{T}) where {T} = ConvertAllFloat{T}()
@inline (f::ConvertAllFloat{T})(x) where {T} = convert_leaf_float(T, x)

@inline convert_leaf_float(::Type{T}, x) where {T} = x # fallback
@inline convert_leaf_float(::Type{T}, x::AbstractFloat) where {T} = T(x)
@inline convert_leaf_float(::Type{T}, x::AbstractArray{<:AbstractFloat}) where {T} = T.(x)

@inline convert_all_float(::Type{T}, x) where {T} = fmap(ConvertAllFloat(T), x)
@inline getfieldstatic(x::T, ::Val{F}, default) where {T, F} = hasfield(T, F) ? getfield(x, F) : default

dict() = OrderedDict{String, Any}()
dict(x, xs...) = dict(OrderedDict(x, xs...))
dict(d::OrderedDict{String, Any}) = d
dict(d::AbstractDict) = OrderedDict{String, Any}(string(k) => v for (k, v) in pairs(d))

# Unzip array/tuple/named tuple of structs into struct of arrays/tuples/named tuples
@inline unzip_structs(x::AbstractArray{T}) where {T} = NamedTuple{fieldnames(T)}(map(f -> getfield.(x, f), fieldnames(T)))
@inline unzip_structs(x::AbstractArray{T}) where {N, T <: NTuple{N, Any}} = ntuple(i -> getindex.(x, i), Val(N))
@inline unzip_structs(x::AbstractArray{T}) where {K, T <: NamedTuple{K}} = NamedTuple{K}(ntuple(i -> getindex.(x, i), Val(length(K))))
@inline unzip_structs(x::NTuple{<:Any, NTuple{N, Any}}) where {N} = ntuple(i -> getindex.(x, i), Val(N))
@inline unzip_structs(x::NamedTuple{<:Any, <:NTuple{<:Any, NTuple{N, Any}}}) where {N} = unzip_structs(Tuple(x))

# Map over an iterable, passing the enumeration index first; preserves `NamedTuple` keys
function mapenumerated(f, xs)
    local i = Ref(0) # must be `Ref` for type stability, else it will get boxed
    fenumerated(x) = f(i[] += 1, x)
    return map(fenumerated, xs)
end

dict_like(x::AbstractDict, args...) = constructorof(typeof(x))(args...)

map_dict(f, x::AbstractDict, ys...) = dict_like(x, k => f(v, map(y -> y[k], ys)...) for (k, v) in x)
map_dict(f) = (xs...,) -> map_dict(f, xs...)

#### Chunked apply

function chunked_apply(f, x; chunksize::Union{Int, Nothing} = nothing, kwargs...)
    bigchunksize = chunkdim(x)
    if chunksize === nothing || chunksize >= bigchunksize
        f(x; kwargs...)
    else
        _chunked_apply(f, x; bigchunksize, chunksize, kwargs...)
    end
end

function chunked_apply(f, x...; chunksize::Union{Int, Nothing} = nothing, kwargs...)
    return chunked_apply(y -> f(y...; kwargs...), x; chunksize)
end

function _chunked_apply(f, x; bigchunksize::Int, chunksize::Int, kwargs...)
    out = _init_chunk_apply(f, x, 1:chunksize, bigchunksize; kwargs...)
    for i in (chunksize+1):chunksize:bigchunksize
        _chunk_apply!(out, f, x, i:min(i+chunksize-1, bigchunksize); kwargs...)
    end
    return out
end

function _chunk_apply!(out, f, x, chunkinds; kwargs...)
    chunk = fmap(_x -> _x[.., chunkinds], x; exclude = isnumeric)
    y = f(chunk; kwargs...)
    fmap(out, y; exclude = isnumeric) do _out, _y
        _fill_chunk!(_out, _y, chunkinds)
    end
end

function _init_chunk_apply(f, x, chunkinds, bigchunksize; kwargs...)
    chunk = fmap(_x -> _x[.., chunkinds], x; exclude = isnumeric)
    y = f(chunk; kwargs...)
    fmap(y; exclude = isnumeric) do _y
        _fill_chunk!(_init_chunk(_y, bigchunksize), _y, chunkinds)
    end
end

@inline _init_chunk(y, N) = zeros_similar(y, datadims(y)..., N)
@inline function _fill_chunk!(out, y, I)
    @inbounds(out[.., I] .= y)
    return out
end

function chunkdim(x)
    dim = Ref(-1)
    fmapstructure(x; exclude = isnumeric) do _x
        @assert dim[] == -1 || dim[] == chunkdim(_x) "Inconsistent chunk dimensions"
        dim[] = chunkdim(_x)
        return nothing
    end
    return dim[]
end
@inline chunkdim(x::AbstractArray{<:Any, N}) where {N} = size(x, N)
@inline datadims(x::AbstractArray{<:Any, N}) where {N} = ntuple(i -> size(x, i), N - 1)

function ChainRules.rrule(::typeof(_chunked_apply), f, x; bigchunksize::Int, chunksize::Int, kwargs...)
    @not_implemented("rrule not implemented for _chunked_apply")
end

@non_differentiable chunkdim(x)
@non_differentiable datadims(x)

# Convenience path constructors
projectdir(args...) = normpath(pkgdir(@__MODULE__, args...))
scriptsdir(args...) = projectdir("scripts", args...)

# Large numerical results are cached in `.cache/`; repository paths map to the same relative path there.
cachedir(args...) = projectdir(".cache", args...)
isinside(path, root) = (relative = relpath(path, root); relative != ".." && !startswith(relative, "../"))
function cachepath(filename)
    path = abspath(filename)
    isinside(path, cachedir()) && return path
    return isinside(path, projectdir()) ? cachedir(relpath(path, projectdir())) : path
end

# Relative paths are taken to be relative to the project directory; absolute paths are taken directly
function expand_path(paths::AbstractString...)
    path = expanduser(joinpath(paths...))
    return isabspath(path) ? path : projectdir(path)
end

clean_normpath(paths::AbstractString...) = normpath(expand_path(paths...))
clean_realpath(paths::AbstractString...) = realpath(expand_path(paths...))

#### Saving and loading

safe_save(filename, data::AbstractDict) = JLD2.save(filename, dict(erase_types(cpu32(data))))
safe_load(filename) = rebuild_types(JLD2.load(filename))
