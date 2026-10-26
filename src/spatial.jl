#### Spatial regularization utils

# Full-neighbourhood kernels: 3×3 (2D) and 3×3×3 (3D) with center 0. Chebyshev distance 1.
function uniform_kernel(::Val{N}, ::Type{T}) where {N, T}
    K = ones(T, ntuple(_ -> 3, Val(N)))
    K[ntuple(Returns(2), Val(N))...] = zero(T)
    return K
end

# Distance-weighted kernel: w(a) = 1/‖a‖
function distance_kernel(::Val{N}, ::Type{T}) where {N, T}
    K = uniform_kernel(Val(N), T)
    center = CartesianIndex(ntuple(_ -> 2, N))
    for I in CartesianIndices(K)
        I == center && continue
        d = sqrt(T(sum(abs2, Tuple(I - center))))
        K[I] = K[I] / d
    end
    return K
end

# Enumerate full-neighbourhood offset tuples as CartesianIndex.
@inline function _stencil_offsets(::Val{N}) where {N}
    zero_idx = CartesianIndex(ntuple(_ -> 0, Val(N)))
    return [I for I in CartesianIndex(ntuple(Returns(-1), Val(N))):CartesianIndex(ntuple(Returns(1), Val(N))) if I != zero_idx]
end

# Look up kernel weight at a stencil offset (offset ∈ {-1,0,1}^N, center = (2,2,...)).
@inline _kernel_weight(kernel::AbstractArray{T, N}, offset::CartesianIndex{N}) where {T, N} = kernel[CartesianIndex(ntuple(Returns(2), N))+offset]

# Convert kernel to StaticArray for compile-time optimized lookups.
@inline _static_kernel(kernel::AbstractArray{T, N}) where {T, N} = StaticArrays.SArray{Tuple{ntuple(_ -> 3, Val(N))...}, T}(kernel)

# O(1) array lookup for stencil membership: offset components ∈ {-1,0,1}^N.
# Returns (true, weight) if offset is in the stencil with nonzero weight, (false, zero) otherwise.
@inline function _in_stencil(kernel::AbstractArray{T, N}, a::CartesianIndex{N}) where {T, N}
    all(ntuple(d -> -1 <= a[d] <= 1, Val(N))) || return (false, zero(T))
    @inbounds w = kernel[CartesianIndex(ntuple(Returns(2), Val(N)))+a]
    return w > zero(T) ? (true, w) : (false, zero(T))
end

# Check if a kernel is radially symmetric around its center.
function _is_radially_symmetric(kernel::AbstractArray{T, N}) where {T, N}
    @assert size(kernel) == ntuple(_ -> 3, N) "Kernel must have size 3^N"
    center = CartesianIndex(ntuple(_ -> 2, N))
    for offset in _stencil_offsets(Val(N))
        kernel[center+offset] == kernel[center-offset] || return false
    end
    return true
end

"""
    SpatialAdjacency{T, N}

Masked voxel adjacency with symmetric normalization ``W = D^{-1/2} A D^{-1/2}``.
Full Chebyshev-1 neighbourhood: ``3^N - 1`` neighbours (8 for 2D, 26 for 3D).
The weighted adjacency ``A_{ij} = w(\\text{offset}(i,j))`` uses a radially symmetric
nonneg kernel ``w`` (default: uniform unit weights).

# Fields

  - `shape::NTuple{N, Int}`: spatial dimensions of the dense embedding grid
  - `kernel`: ``N``-dimensional neighbour weight array (size ``3^N``, center = 0)
  - `fixed_dim::Int`: for 2D slices, the dimension held constant (0 for 3D)
  - `fixed_index::Int`: index value along `fixed_dim` (0 for 3D)
  - `total_degree::T`: sum of degrees over all active voxels, ``\\sum_i d_i``
  - `inv_sqrt_degree_dense`: ``D^{-1/2}`` array over dense grid (zero off-mask)
  - `linear_idx`: maps active voxels to linear indices in the dense grid
  - `tr_adj_powers`: cached ``(\\operatorname{tr}(W), \\operatorname{tr}(W^2), \\operatorname{tr}(W^3), \\operatorname{tr}(W^4), ...)``
  - `tr_lap_inv_powers`: cached ``(\\operatorname{tr}_+(L^0), \\operatorname{tr}_+(L^{-1}), \\operatorname{tr}_+(L^{-2}), \\operatorname{tr}_+(L^{-3}), ...)``
"""
struct SpatialAdjacency{T, N, Tkernel <: AbstractArray{T, N}, Tlinear <: AbstractVector{<:Integer}, Tinvsqrt <: AbstractArray{T}, Ttradj <: AbstractVector{T}, Ttrlap <: AbstractVector{T}}
    shape::NTuple{N, Int} # spatial dimensions of dense grid
    kernel::Tkernel # neighbour weight kernel (3^N, center=0)
    fixed_dim::Int # slice dimension (0 for 3D)
    fixed_index::Int # slice index (0 for 3D)
    total_degree::T # sum of degrees over all active voxels, Σᵢ dᵢ
    inv_sqrt_degree_dense::Tinvsqrt # D^{-1/2} on dense grid
    linear_idx::Tlinear # active voxel → dense linear index
    tr_adj_powers::Ttradj # cached invariants: (tr(W¹), tr(W²), tr(W³), tr(W⁴), ...)
    tr_lap_inv_powers::Ttrlap # cached invariants: (tr(L^0), tr(L^{-1}), tr(L^{-2}), tr(L^{-3}), ...)

    function SpatialAdjacency(shape::NTuple{N, Int}, kernel::Tkernel, fixed_dim::Int, fixed_index::Int, total_degree::AbstractFloat, inv_sqrt_degree_dense::Tinvsqrt, linear_idx::Tlinear, tr_adj_powers::Ttradj, tr_lap_inv_powers::Ttrlap) where {N, T, Tkernel <: AbstractArray{T, N}, Tlinear <: AbstractVector{<:Integer}, Tinvsqrt <: AbstractArray{T}, Ttradj <: AbstractVector{T}, Ttrlap <: AbstractVector{T}}
        return new{T, N, Tkernel, Tlinear, Tinvsqrt, Ttradj, Ttrlap}(
            shape, kernel, fixed_dim, fixed_index, total_degree, inv_sqrt_degree_dense, linear_idx, tr_adj_powers, tr_lap_inv_powers,
        )
    end
end

Base.length(op::SpatialAdjacency) = length(op.linear_idx)
Base.size(op::SpatialAdjacency) = op.shape
Base.eltype(::SpatialAdjacency{T, N}) where {T, N} = T
Base.show(io::IO, op::SpatialAdjacency{T, N}) where {T, N} = print(io, "SpatialAdjacency($(N)D, shape=$(op.shape), nactive=$(length(op)), fixed_dim=$(op.fixed_dim), fixed_index=$(op.fixed_index))")

"""
    SpatialAdjacencyOperator(kernel, idx; slice_dim = nothing, k_max = 4)
    SpatialAdjacencyOperator([T,] Val(N), idx; slice_dim = nothing, k_max = 4)

Construct a [`SpatialAdjacency`](@ref) from 3D index tuples `idx`.
For `N = 2`, infers or validates the fixed slice dimension.

`kernel` is an optional ``N``-dimensional nonneg weight array (size ``3^N``, center = 0).
Defaults to the uniform Chebyshev-1 kernel (all ones, zero center).

`k_max` (≤ 6) sets the number of cached trace slots: `tr(W^k)` are computed exactly via neighborhood walks,
and the `tr_lap_inv_powers` slots are initialized to `NaN` and filled by [`estimate_stochastic_traces!`](@ref).
"""
function SpatialAdjacencyOperator(
    kernel::AbstractArray{T, N},
    idx::AbstractArray{<:NTuple{3, <:Integer}};
    slice_dim::Union{Int, Nothing} = nothing,
    k_max::Int = 4,
) where {T, N}
    @assert N in (2, 3) "spatial dimension must be 2 or 3"
    @assert 2 <= k_max <= 6 "k_max must be in [2, 6]"
    @assert !isempty(idx) "Cannot build SpatialAdjacency from an empty index set"
    @assert ndims(kernel) == N && all(==(3), size(kernel)) "kernel must be $(N)D with size $(ntuple(_ -> 3, N))"
    @assert kernel[ntuple(Returns(2), N)...] == zero(T) "kernel center must be zero"
    @assert all(>=(zero(T)), kernel) "kernel must be nonnegative"
    @assert _is_radially_symmetric(kernel) "kernel must be radially symmetric"

    varying_dims, fixed_dim, fixed_index = _resolve_spatial_axes(Val(N), idx; slice_dim)
    shape, mask, linear_idx = _build_local_coords_and_mask(Val(N), idx, varying_dims)
    total_degree, inv_sqrt_degree_dense, tr_adj_powers = _compute_degree_and_traces(mask, kernel; k_max)
    tr_lap_inv_powers = fill(T(NaN), k_max)

    return SpatialAdjacency(shape, kernel, fixed_dim, fixed_index, total_degree, inv_sqrt_degree_dense, linear_idx, tr_adj_powers, tr_lap_inv_powers)
end
SpatialAdjacencyOperator(::Type{T}, ::Val{N}, args...; kwargs...) where {T, N} = SpatialAdjacencyOperator(distance_kernel(Val(N), T), args...; kwargs...)
SpatialAdjacencyOperator(::Val{N}, args...; kwargs...) where {N} = SpatialAdjacencyOperator(Float64, Val(N), args...; kwargs...)

"""
    estimate_stochastic_traces!(op; deflation_kwargs = (;))

Fill the `tr_+(L^{-k})` slots of `op.tr_lap_inv_powers` in place via [`trace_laplacian_pseudoinverse_deflated`](@ref).
"""
function estimate_stochastic_traces!(op::SpatialAdjacency{T}; deflation_kwargs = (;)) where {T}
    k_inv = length(op.tr_lap_inv_powers)
    result = trace_laplacian_pseudoinverse_deflated(op; k_max = k_inv, deflation_kwargs...)
    copyto!(op.tr_lap_inv_powers, result.traces)
    return op
end

function _build_local_coords_and_mask(::Val{N}, idx::AbstractArray{<:NTuple{3, <:Integer}}, varying_dims) where {N}
    coords_raw = map(vec(idx)) do I
        ntuple(k -> Int(I[varying_dims[k]]), N)
    end

    lo = ntuple(k -> minimum(c -> c[k], coords_raw), N)
    hi = ntuple(k -> maximum(c -> c[k], coords_raw), N)
    shape = ntuple(k -> hi[k] - lo[k] + 1, N)

    local_coords = map(coords_raw) do c
        CartesianIndex(ntuple(k -> c[k] - lo[k] + 1, N))
    end

    mask = fill(false, shape...)
    mask[local_coords] .= true

    linear_idx = collect(LinearIndices(shape)[local_coords])

    return shape, mask, linear_idx
end

"""
    _compute_degree_and_traces(mask, kernel; k_max = 4)

Compute total degree ``\\sum d(x)``, inverse square root degree ``D^{-1/2}``, and traces
``\\operatorname{tr}(W^k)`` for ``k \\in \\{2, \\ldots, k_{\\max}\\}``.

Let ``M \\in \\{0,1\\}^{|V|}`` be the binary mask and ``w(a)`` the kernel weights for offsets ``a \\in \\mathcal{O}``.

  - Degree: ``d(x) = \\sum_{a \\in \\mathcal{O}} w(a) M(x+a)``

  - Normalized weights: ``r(x) = d(x)^{-1}`` for ``x \\in \\operatorname{supp}(M)``

  - Traces:

      + ``\\operatorname{tr}(W^1) = 0``
      + ``\\operatorname{tr}(W^2) = \\sum_{x,a} r(x) r(x+a) w(a)^2``
      + ``\\operatorname{tr}(W^3) = \\sum_{x,a,b : b-a \\in \\mathcal{O}} r(x) r(x+a) r(x+b) w(a)w(b)w(b-a)``
      + ``\\operatorname{tr}(W^4) = \\sum_{x,a,b,c : -(a+b+c) \\in \\mathcal{O}} r(x) r(x+a) r(x+a+b) r(x+a+b+c) w(a)w(b)w(c)w(-(a+b+c))``
"""
function _compute_degree_and_traces(mask::AbstractArray{Bool, N}, kernel_array::AbstractArray{T, N}; k_max::Int = 4) where {T, N}
    @assert 2 <= k_max <= 6 "k_max must be in 2:6"
    kernel = _static_kernel(kernel_array)
    offsets = _stencil_offsets(Val(N))
    sz = size(mask)
    r = zeros(T, sz)
    r_strides = Base.strides(r)

    # Precompute linear offsets and weights for degree calculation
    O_la, O_w = Int[], T[]
    for a in offsets
        wa = _kernel_weight(kernel, a)
        wa == 0 && continue
        push!(O_la, sum(d -> Tuple(a)[d] * r_strides[d], 1:N))
        push!(O_w, wa)
    end

    # Compute degree, skipping interior nodes (all neighbours present → d = d_max)
    d_max = sum(kernel)
    r_min = one(T) / d_max
    mask_int = _erode_mask(mask, offsets)

    total_degree = zero(T)
    lin_inds = LinearIndices(mask)
    n_interior = 0

    for x in CartesianIndices(mask)
        !mask[x] && continue
        x_lin = lin_inds[x]

        if mask_int[x]
            @inbounds r[x_lin] = r_min
            n_interior += 1
        else
            dx = zero(T)
            is_safe = true
            for d in 1:N
                if x[d] < 2 || x[d] > sz[d] - 1
                    is_safe = false
                    break
                end
            end
            if is_safe
                @inbounds @simd ivdep for j in eachindex(O_w)
                    dx += O_w[j] * mask[x_lin+O_la[j]]
                end
            else
                for a in offsets
                    if checkbounds(Bool, mask, x + a)
                        @inbounds dx += _kernel_weight(kernel, a) * mask[x+a]
                    end
                end
            end
            total_degree += dx
            @inbounds r[x_lin] = dx > 0 ? inv(dx) : zero(T)
        end
    end
    total_degree += n_interior * d_max

    # tr(W²): walk x → x+a → x
    slow_paths_2 = Tuple{CartesianIndex{N}, T}[]
    fast_la_2, fast_w_2 = Int[], T[]
    C₂ = zero(T)
    for a in offsets
        wa² = _kernel_weight(kernel, a)^2
        wa² == 0 && continue
        C₂ += wa²
        push!(slow_paths_2, (a, wa²))
        push!(fast_la_2, _linear_offset(a, r_strides))
        push!(fast_w_2, wa²)
    end

    # tr(W³): walk x → x+a → x+b → x (requires b-a ∈ O)
    slow_paths_3 = Tuple{CartesianIndex{N}, CartesianIndex{N}, T}[]
    fast_la_3, fast_lb_3, fast_w_3 = Int[], Int[], T[]
    C₃ = zero(T)
    if k_max >= 3
        for a in offsets, b in offsets
            found, w_ba = _in_stencil(kernel, b - a)
            !found && continue
            wprod = _kernel_weight(kernel, a) * _kernel_weight(kernel, b) * w_ba
            wprod == 0 && continue
            C₃ += wprod
            push!(slow_paths_3, (a, b, wprod))
            push!(fast_la_3, _linear_offset(a, r_strides))
            push!(fast_lb_3, _linear_offset(b, r_strides))
            push!(fast_w_3, wprod)
        end
    end

    # tr(W⁴): walk x → x+a → x+a+b → x+a+b+c → x (requires -(a+b+c) ∈ O)
    slow_paths_4 = Tuple{CartesianIndex{N}, CartesianIndex{N}, CartesianIndex{N}, T}[]
    fast_la_4, fast_lab_4, fast_labc_4, fast_w_4 = Int[], Int[], Int[], T[]
    C₄ = zero(T)
    if k_max >= 4
        for a in offsets, b in offsets, c in offsets
            found, w_neg = _in_stencil(kernel, -(a + b + c))
            !found && continue
            wprod = _kernel_weight(kernel, a) * _kernel_weight(kernel, b) * _kernel_weight(kernel, c) * w_neg
            wprod == 0 && continue
            C₄ += wprod
            ab, abc = a + b, a + b + c
            push!(slow_paths_4, (a, ab, abc, wprod))
            push!(fast_la_4, _linear_offset(a, r_strides))
            push!(fast_lab_4, _linear_offset(ab, r_strides))
            push!(fast_labc_4, _linear_offset(abc, r_strides))
            push!(fast_w_4, wprod)
        end
    end

    # tr(W⁵): walk x → ... → x with 4 offsets; closure -(a+b+c+d) ∈ O
    slow_paths_5 = Tuple{CartesianIndex{N}, CartesianIndex{N}, CartesianIndex{N}, CartesianIndex{N}, T}[]
    fast_offsets_5 = NTuple{4, Int}[]
    fast_w_5 = T[]
    C₅ = zero(T)
    if k_max >= 5
        for a in offsets, b in offsets, c in offsets, d in offsets
            found, w_neg = _in_stencil(kernel, -(a + b + c + d))
            !found && continue
            wprod = _kernel_weight(kernel, a) * _kernel_weight(kernel, b) * _kernel_weight(kernel, c) * _kernel_weight(kernel, d) * w_neg
            wprod == 0 && continue
            C₅ += wprod
            ab, abc, abcd = a + b, a + b + c, a + b + c + d
            push!(slow_paths_5, (a, ab, abc, abcd, wprod))
            push!(fast_offsets_5, (_linear_offset(a, r_strides), _linear_offset(ab, r_strides), _linear_offset(abc, r_strides), _linear_offset(abcd, r_strides)))
            push!(fast_w_5, wprod)
        end
    end

    # tr(W⁶): walk x → ... → x with 5 offsets; closure -(a+b+c+d+e) ∈ O
    slow_paths_6 = Tuple{CartesianIndex{N}, CartesianIndex{N}, CartesianIndex{N}, CartesianIndex{N}, CartesianIndex{N}, T}[]
    fast_offsets_6 = NTuple{5, Int}[]
    fast_w_6 = T[]
    C₆ = zero(T)
    if k_max >= 6
        for a in offsets, b in offsets, c in offsets, d in offsets, e in offsets
            found, w_neg = _in_stencil(kernel, -(a + b + c + d + e))
            !found && continue
            wprod = _kernel_weight(kernel, a) * _kernel_weight(kernel, b) * _kernel_weight(kernel, c) * _kernel_weight(kernel, d) * _kernel_weight(kernel, e) * w_neg
            wprod == 0 && continue
            C₆ += wprod
            ab, abc, abcd, abcde = a + b, a + b + c, a + b + c + d, a + b + c + d + e
            push!(slow_paths_6, (a, ab, abc, abcd, abcde, wprod))
            push!(fast_offsets_6, (_linear_offset(a, r_strides), _linear_offset(ab, r_strides), _linear_offset(abc, r_strides), _linear_offset(abcd, r_strides), _linear_offset(abcde, r_strides)))
            push!(fast_w_6, wprod)
        end
    end

    # Erosion chain: mask_int_k[m] = erode^{m+1}(mask), safe for traces with max displacement m
    n_erosions = k_max <= 3 ? 1 : k_max - 2
    mask_int_k = AbstractArray{Bool, N}[]
    prev = mask_int
    for _ in 1:n_erosions
        prev = _erode_mask(prev, offsets)
        push!(mask_int_k, prev)
    end

    mask_deepest = mask_int_k[end]

    # Partition boundary nodes by safety margin (eliminates branching in hot loops)
    max_margin = k_max >= 6 ? 4 : k_max >= 5 ? 3 : k_max >= 4 ? 2 : 1
    boundary_by_margin = [CartesianIndex{N}[] for _ in 0:max_margin]
    for I in CartesianIndices(r)
        mask[I] && !mask_deepest[I] || continue
        margin = _safety_margin(I, sz, max_margin)
        push!(boundary_by_margin[margin+1], I)
    end

    n_traces = k_max - 1
    n_boundary = sum(length, boundary_by_margin)
    boundary_traces = zeros(T, n_traces, n_boundary)

    # Accumulate boundary traces by safety margin
    col_offset = 0
    for (margin_idx, nodes) in enumerate(reverse(boundary_by_margin))
        margin = max_margin - margin_idx + 1
        isempty(nodes) && (col_offset += length(nodes); continue)
        off = col_offset

        Threads.@threads for i in eachindex(nodes)
            x = nodes[i]
            x_lin = lin_inds[x]
            @inbounds rx = r[x_lin]

            # tr(W²): margin 1 needed
            t2_acc = zero(T)
            if !mask_int_k[1][x]
                if margin >= 1
                    @inbounds @simd ivdep for j in eachindex(fast_w_2)
                        t2_acc += fast_w_2[j] * r[x_lin+fast_la_2[j]]
                    end
                else
                    for j in eachindex(slow_paths_2)
                        a, wprod = slow_paths_2[j]
                        if checkbounds(Bool, r, x + a)
                            @inbounds t2_acc += wprod * r[x_lin+fast_la_2[j]]
                        end
                    end
                end
            end
            boundary_traces[1, off+i] = rx * t2_acc

            # tr(W³): margin 1 needed
            if k_max >= 3
                t3_acc = zero(T)
                if !mask_int_k[1][x]
                    if margin >= 1
                        @inbounds @simd ivdep for j in eachindex(fast_w_3)
                            t3_acc += fast_w_3[j] * r[x_lin+fast_la_3[j]] * r[x_lin+fast_lb_3[j]]
                        end
                    else
                        for j in eachindex(slow_paths_3)
                            a, b, wprod = slow_paths_3[j]
                            if checkbounds(Bool, r, x + a) && checkbounds(Bool, r, x + b)
                                @inbounds t3_acc += wprod * r[x_lin+fast_la_3[j]] * r[x_lin+fast_lb_3[j]]
                            end
                        end
                    end
                end
                boundary_traces[2, off+i] = rx * t3_acc
            end

            # tr(W⁴): margin 2 needed
            if k_max >= 4
                t4_acc = zero(T)
                if length(mask_int_k) < 2 || !mask_int_k[2][x]
                    if margin >= 2
                        @inbounds @simd ivdep for j in eachindex(fast_w_4)
                            t4_acc += fast_w_4[j] * r[x_lin+fast_la_4[j]] * r[x_lin+fast_lab_4[j]] * r[x_lin+fast_labc_4[j]]
                        end
                    else
                        for j in eachindex(slow_paths_4)
                            a, ab, abc, wprod = slow_paths_4[j]
                            if checkbounds(Bool, r, x + a) && checkbounds(Bool, r, x + ab) && checkbounds(Bool, r, x + abc)
                                @inbounds t4_acc += wprod * r[x_lin+fast_la_4[j]] * r[x_lin+fast_lab_4[j]] * r[x_lin+fast_labc_4[j]]
                            end
                        end
                    end
                end
                boundary_traces[3, off+i] = rx * t4_acc
            end

            # tr(W⁵): margin 3 needed
            if k_max >= 5
                t5_acc = zero(T)
                if length(mask_int_k) < 3 || !mask_int_k[3][x]
                    if margin >= 3
                        @inbounds for j in eachindex(fast_w_5)
                            la, lab, labc, labcd = fast_offsets_5[j]
                            t5_acc += fast_w_5[j] * r[x_lin+la] * r[x_lin+lab] * r[x_lin+labc] * r[x_lin+labcd]
                        end
                    else
                        for j in eachindex(slow_paths_5)
                            a, ab, abc, abcd, wprod = slow_paths_5[j]
                            if checkbounds(Bool, r, x + a) && checkbounds(Bool, r, x + ab) && checkbounds(Bool, r, x + abc) && checkbounds(Bool, r, x + abcd)
                                la, lab, labc, labcd = fast_offsets_5[j]
                                @inbounds t5_acc += wprod * r[x_lin+la] * r[x_lin+lab] * r[x_lin+labc] * r[x_lin+labcd]
                            end
                        end
                    end
                end
                boundary_traces[4, off+i] = rx * t5_acc
            end

            # tr(W⁶): margin 4 needed
            if k_max >= 6
                t6_acc = zero(T)
                if length(mask_int_k) < 4 || !mask_int_k[4][x]
                    if margin >= 4
                        @inbounds for j in eachindex(fast_w_6)
                            la, lab, labc, labcd, labcde = fast_offsets_6[j]
                            t6_acc += fast_w_6[j] * r[x_lin+la] * r[x_lin+lab] * r[x_lin+labc] * r[x_lin+labcd] * r[x_lin+labcde]
                        end
                    else
                        for j in eachindex(slow_paths_6)
                            a, ab, abc, abcd, abcde, wprod = slow_paths_6[j]
                            if checkbounds(Bool, r, x + a) && checkbounds(Bool, r, x + ab) && checkbounds(Bool, r, x + abc) && checkbounds(Bool, r, x + abcd) && checkbounds(Bool, r, x + abcde)
                                la, lab, labc, labcd, labcde = fast_offsets_6[j]
                                @inbounds t6_acc += wprod * r[x_lin+la] * r[x_lin+lab] * r[x_lin+labc] * r[x_lin+labcd] * r[x_lin+labcde]
                            end
                        end
                    end
                end
                boundary_traces[5, off+i] = rx * t6_acc
            end
        end

        col_offset += length(nodes)
    end

    # Combine interior (constant) and boundary contributions
    Σboundary = sum(boundary_traces; dims = 2)
    Ck = (C₂, C₃, C₄, C₅, C₆)
    traces = zeros(T, k_max)
    for k in 2:k_max
        int_idx = k <= 3 ? 1 : k - 2
        n_int = int_idx <= length(mask_int_k) ? count(mask_int_k[int_idx]) : 0
        traces[k] = n_int * Ck[k-1] * r_min^k + Σboundary[k-1]
    end

    inv_sqrt_degree = r .= sqrt.(r)

    return total_degree, inv_sqrt_degree, traces
end

# Convert CartesianIndex offset to linear offset
@inline function _linear_offset(a::CartesianIndex{N}, strides::NTuple{N, Int}) where {N}
    return sum(Tuple(a) .* strides)
end

# Compute the safety margin of a CartesianIndex within array bounds, clamped to max_margin.
# Margin m means I[d] ∈ [m+1, sz[d]-m] for all d.
@inline function _safety_margin(I::CartesianIndex{N}, sz::NTuple{N, Int}, max_margin::Int) where {N}
    margin = max_margin
    for d in 1:N
        margin = min(margin, I[d] - 1, sz[d] - I[d])
        margin <= 0 && return 0
    end
    return margin
end

function _erode_mask(mask::AbstractArray{Bool, N}, offsets::AbstractVector{<:CartesianIndex{N}}) where {N}
    out = copy(mask)
    sz = size(mask)
    Threads.@threads for I in CartesianIndices(mask)
        if mask[I]
            is_int = true
            for d in 1:N
                if I[d] == 1 || I[d] == sz[d]
                    is_int = false
                    break
                end
            end
            if !is_int
                out[I] = false
                continue
            end
            for a in offsets
                @inbounds if !mask[I+a]
                    out[I] = false
                    break
                end
            end
        end
    end
    return out
end

"""
    _infer_slice_dim(idx) -> (varying_dims, fixed_dim)

Infer which of the three spatial dimensions is constant across all `idx` tuples.
"""
function _infer_slice_dim(idx::AbstractArray{NTuple{3, I}}) where {I <: Integer}
    return @ignore_derivatives begin
        i1 = @allowscalar first(idx)
        all(i -> i[3] == i1[3], idx) && return ((1, 2), 3)
        all(i -> i[2] == i1[2], idx) && return ((1, 3), 2)
        all(i -> i[1] == i1[1], idx) && return ((2, 3), 1)
        error("No common slice dimension")
    end
end

function unflatten_image(x, idx::AbstractVector{NTuple{3, I}}) where {I <: Integer}
    @assert size(x)[end] == length(idx) "Mismatched dimensions"
    (ix, iy), _ = _infer_slice_dim(idx)
    ixlo, ixhi = extrema(i -> i[ix], idx)
    iylo, iyhi = extrema(i -> i[iy], idx)
    ixwidth, iywidth = ixhi - ixlo + 1, iyhi - iylo + 1
    out = fill_similar(x, NaN, size(x)[1:(end-1)]..., ixwidth, iywidth)
    out[.., (i->CartesianIndex(i[ix] - ixlo + 1, i[iy] - iylo + 1)).(idx)] .= x
    return out
end
unflatten_image(x, idx::AbstractArray) = unflatten_image(x, vec(idx))

"""
    _resolve_spatial_axes(Val(N), idx; slice_dim) -> (varying_dims, fixed_dim, fixed_index)

Map 3D index tuples to `N`-dimensional spatial axes.
`N = 3`: full volume. `N = 2`: infers/validates the fixed slice dimension.
"""
function _resolve_spatial_axes(
    ::Val{N},
    idx::AbstractArray{<:NTuple{3, <:Integer}};
    slice_dim::Union{Int, Nothing},
) where {N}
    if N == 3
        return (1, 2, 3), 0, 0
    end

    fixed_dim, varying_dims = if slice_dim === nothing
        varying_dims, fixed_dim = _infer_slice_dim(idx)
        fixed_dim, varying_dims
    else
        @assert slice_dim in 1:3 "slice_dim must be 1, 2, or 3"
        slice_dim, (slice_dim == 1 ? (2, 3) : slice_dim == 2 ? (1, 3) : (1, 2))
    end

    fixed_index = Int(first(idx)[fixed_dim])
    @assert all(i -> i[fixed_dim] == fixed_index, idx) "Indices are not confined to a single slice along dimension $fixed_dim"

    varying_dims, fixed_dim, fixed_index
end

# Weighted neighbour sum over (ncoeff, spatial...), fused over the nonzero stencil offsets
# `NNlib.conv` would need cuDNN to run on the GPU; its CPU fallback scalar-indexes device arrays.
function _neighbour_sum_features(x::AbstractArray{T}, kernel::AbstractArray{T, N}) where {T, N}
    spatial = ntuple(d -> size(x, d + 1), N)
    centre = CartesianIndex(ntuple(Returns(2), N))
    xp = NNlib.pad_zeros(x, 1; dims = ntuple(d -> d + 1, N))

    # Each nonzero weight of the 3^N stencil adds `x` shifted by its offset, zero-padded at the border
    # It is read on the host, since `op.kernel` is a device array once the operator is moved to the GPU.
    stencil = @ignore_derivatives Array(kernel)
    shifted(a) = @views xp[:, ntuple(d->(2+a[d]) .+ (0:(spatial[d]-1)), N)...]
    offsets = @ignore_derivatives [Tuple(I - centre) for I in CartesianIndices(stencil) if !iszero(stencil[I])]

    isempty(offsets) && return zero(x)
    return Base.materialize(foldl(offsets; init = nothing) do acc, a
        term = Broadcast.broadcasted(*, stencil[centre+CartesianIndex(a)], shifted(a))
        return acc === nothing ? term : Broadcast.broadcasted(+, acc, term)
    end)
end

function spatial_scatter(op::SpatialAdjacency{T}, u::AbstractArray{T}) where {T}
    @assert size(u, ndims(u)) == length(op) "Expected last dim = $(length(op)), got $(size(u, ndims(u)))"
    u_flat = Flux.flatten(u)
    y = fill_similar(u, zero(T), size(u_flat, 1), prod(op.shape))
    y[:, op.linear_idx] .= u_flat
    return reshape(y, size(u_flat, 1), op.shape...)
end

function spatial_gather(op::SpatialAdjacency{T}, x_dense::AbstractArray{T}, outsize::NTuple) where {T}
    return reshape(reshape(x_dense, size(x_dense, 1), :)[:, op.linear_idx], outsize)
end

"""
    apply_normalized_adjacency(op, u)

Symmetric normalized adjacency: ``Wu = D^{-1/2} A D^{-1/2} u``.
"""
function apply_normalized_adjacency(op::SpatialAdjacency{T, N}, u::AbstractMatrix{T}) where {T, N}
    D⁻ʰ = reshape(op.inv_sqrt_degree_dense, 1, op.shape...) # D^{-1/2} reshaped for broadcasting
    u_dense = spatial_scatter(op, u)
    return spatial_gather(op, _neighbour_sum_features(u_dense .* D⁻ʰ, op.kernel) .* D⁻ʰ, size(u))
end
apply_normalized_adjacency(op::SpatialAdjacency{T, N}, u::AbstractArray{T}) where {T, N} = reshape(apply_normalized_adjacency(op, reshape(u, 1, :)), size(u))

function ChainRules.rrule(::typeof(apply_normalized_adjacency), op::SpatialAdjacency{T}, u::AbstractMatrix{T}) where {T}
    Wu = apply_normalized_adjacency(op, u)
    function pullback(ΔWu)
        Δu = apply_normalized_adjacency(op, ΔWu)
        return NoTangent(), NoTangent(), Δu
    end
    return Wu, pullback
end

"""
    precision_energy(op, u, τ = 1, ρ = 0)

Half-quadratic ``E_Q(u) = \\tfrac{1}{2} u^\\top Q u = \\tau(\\tfrac{1}{2}\\|u\\|^2 - \\rho E_W(u))``.
Factored through ``W`` for direct AD of ``\\tau, \\rho``.
"""
function precision_energy(op::SpatialAdjacency{T}, u::AbstractArray{T}, τ::Union{T, AbstractVector{T}} = one(T), ρ::Union{T, AbstractVector{T}} = zero(T)) where {T}
    Wu = apply_normalized_adjacency(op, u)
    return sum(@. τ * (abs2(u) / 2 - ρ * u * Wu / 2))
end

"""
    trace_laplacian_pseudoinverse_deflated(op; k_max=4, m=30, nprobes=0, n_zero=1, kwargs...)

Compute ``s_k = \\operatorname{tr}((L^+)^k)`` for ``k = 0,\\ldots,k_{\\max}-1`` via spectral deflation.

Returns a `traces` vector of length `k_max` where `traces[k]` corresponds to ``\\operatorname{tr}((L^+)^{k-1})``.

Extracts the ``m`` smallest eigenvalues of ``L = I - W`` via `KrylovKit.eigsolve`,
then ``s_k \\approx \\sum_{i=2}^{m} \\lambda_i^{-k}``. When `nprobes > 0`, the bulk
spectrum residual is estimated via projected Hutchinson CG on ``(\\operatorname{span}\\{v_1,\\ldots,v_m\\})^\\perp``.

Returns `(; traces, eigenvalues, eigenvectors, n)`.
"""
function trace_laplacian_pseudoinverse_deflated(
    op::SpatialAdjacency{T};
    k_max::Int = 4,
    m::Int = 3,
    n_zero::Int = 1,
    eigsolve_krylovdim::Int = max(2m, 20),
    eigsolve_tol::Real = sqrt(eps(T)),
    cg_tol::Real = sqrt(eps(T)),
    cg_nprobes::Int = 8,
    cg_naverages::Int = 5,
    verbose::Bool = false,
) where {T}
    @assert k_max >= 1 "k_max must be >= 1 (at least s₀)"
    @assert m >= 2 "m must be >= 2 (at least 1 zero eigenvalue + 1 nonzero)"
    @assert n_zero >= 1
    n = length(op)

    verbose && @info "trace_laplacian_pseudoinverse_deflated: n=$n, m=$m, k_max=$k_max, nprobes=$cg_nprobes"

    # Known zero eigenvector of L = I - W: v₀ = D^{-1/2} 1 (normalized)
    v0 = inv.(sqrt(T(op.total_degree)) .* op.inv_sqrt_degree_dense[op.linear_idx])
    v0 ./= sqrt(sum(abs2, v0))
    verbose && @info "Known zero eigenvector norm: $(sqrt(sum(abs2, v0))), Lv0 norm: $(sqrt(sum(abs2, v0 .- vec(apply_normalized_adjacency(op, reshape(v0, 1, :))))))"

    # Shift v0's eigenvalue 0 → 2 so :SR avoids the zero mode; eigenvalues for all
    # eigenvectors ⊥ v0 are unchanged.
    L_deflated(v::AbstractVector) = (v .- apply_normalized_adjacency(op, v)) .+ T(2) .* T(dot(v, v0)) .* v0

    x0 = randn_similar(v0)
    x0 .-= dot(x0, v0) .* v0
    x0 ./= sqrt(sum(abs2, x0))

    verbose && @info "Starting KrylovKit.eigsolve for m-1=$(m-1) smallest eigenvalues (shift σ=2)"
    vals, vecs, info = KrylovKit.eigsolve(
        L_deflated, x0, m - 1, :SR;
        ishermitian = true, krylovdim = eigsolve_krylovdim, tol = T(eigsolve_tol),
    )
    verbose && @info "KrylovKit.eigsolve converged: $(info.converged)/$(m-1), numops=$(info.numops), numiter=$(info.numiter)"
    verbose && @info "Eigenvalues found: $(vals)"

    # Take at most m-1 eigenvalues; KrylovKit can return more when krylovdim >> n
    # e.g. including the deflated eigenvalue 2 for v0 at the tail of the :SR-sorted list.
    n_take = min(length(vals), m - 1)
    λ_deflated = [T(vals[i]) for i in n_zero:n_take]
    Q_deflated = vecs[n_zero:n_take]
    m_deflated = length(λ_deflated)

    verbose && @info "Deflated eigenvalues: $(λ_deflated)"

    traces = zeros(T, k_max)
    traces[1] = T(n - n_zero)
    for k in 1:(k_max-1)
        traces[k+1] = sum(λ -> λ^(-k), λ_deflated)
        verbose && @info "traces[$k+1] (deflated sum) = $(traces[k+1])"
    end
    if cg_nprobes > 0 && m_deflated < n - n_zero && k_max >= 2
        verbose && @info "Running Hutchinson bulk correction with nprobes=$cg_nprobes"
        Q = zeros_similar(v0, n, m_deflated + 1)
        @views copyto!(Q[:, 1], v0)
        for j in 1:m_deflated
            @views copyto!(Q[:, j+1], vecs[j])
        end
        _deflated_hutchinson_bulk!(traces, op, Q; nprobes = cg_nprobes, naverages = cg_naverages, rtol = T(cg_tol))
        verbose && @info "After Hutchinson correction: traces = $traces"
    end

    return (; traces, eigenvalues = λ_deflated, eigenvectors = Q_deflated, n)
end

# Hutchinson CG bulk correction: estimates residual tr((L⁺)^k) for k=1..k_max-1
# and adds the average over `naverages` independent Rademacher probe rounds to `traces`.
function _deflated_hutchinson_bulk!(traces::Vector{T}, op::SpatialAdjacency{T}, Q::AbstractArray; nprobes::Int, naverages::Int = 1, kwargs...) where {T}
    k_max = length(traces)
    n = length(op)
    m = size(Q, 2)

    # Preallocate all buffers outside the averaging loop.
    trace_accum = zeros(T, k_max)
    z = zeros_similar(op.kernel, nprobes, n)
    x = zeros_similar(op.kernel, nprobes, n)
    r = zeros_similar(op.kernel, nprobes, n)
    p = zeros_similar(op.kernel, nprobes, n)
    Ap = zeros_similar(op.kernel, nprobes, n)
    tmp = zeros_similar(op.kernel, nprobes, n)
    coeffs = zeros_similar(op.kernel, nprobes, m)
    y = zeros_similar(op.kernel, nprobes, k_max >= 4 ? n : 0)

    for _ in 1:naverages
        # Draw Rademacher probes and project onto span(Q)⊥: z ← P_⊥z
        randn!(z)
        @. z = ifelse(z > zero(T), one(T), -one(T))
        mul!(coeffs, z, Q)
        mul!(z, coeffs, Q', -one(T), one(T))

        # Solve L x = z via projected CG; on entry r = p = z.
        fill!(x, zero(T))
        copyto!(r, z)
        copyto!(p, r)
        _projected_cg!(x, r, p, Ap, tmp, op, Q, coeffs; kwargs...)

        trace_accum[2] += mapreduce(*, +, z, x) / T(nprobes) # tr(L⁺) = z⊤L⁺z / S
        k_max >= 3 && (trace_accum[3] += sum(abs2, x) / T(nprobes)) # tr((L⁺)²) = ‖L⁺z‖² / S

        # tr((L⁺)^k) for k ≥ 3: solve L y = P_⊥(prev), accumulate x⊤y / S
        if k_max >= 4
            prev = x
            for k in 3:(k_max-1)
                copyto!(r, prev)
                mul!(coeffs, r, Q)
                mul!(r, coeffs, Q', -one(T), one(T)) # r ← P_⊥(prev)
                fill!(y, zero(T))
                copyto!(p, r)
                _projected_cg!(y, r, p, Ap, tmp, op, Q, coeffs; kwargs...)
                trace_accum[k+1] += mapreduce(*, +, x, y) / T(nprobes)
                prev = y
            end
        end
    end

    for k in 2:k_max
        traces[k] += trace_accum[k] / T(naverages)
    end

    return nothing
end

# Projected CG: solve Lx = b in-place, with r = p = b on entry.
# Periodically reprojects r and p onto span(Q)⊥ to suppress drift.
function _projected_cg!(
    x, r, p, Ap, tmp,
    op::SpatialAdjacency{T}, Q, coeffs;
    maxiter::Int = length(op), rtol::Real = sqrt(eps(T)), reproject_every::Int = 5,
) where {T}
    nprobes = size(r, 1)
    rms_bnorm = sqrt(sum(abs2, r) / nprobes)
    rsold = sum(abs2, r; dims = 2) # nprobes×1
    rsnew = similar(rsold)
    pAp = similar(rsold)
    α = similar(rsold)
    β = similar(rsold)
    for iter in 1:maxiter
        Ap .= p .- apply_normalized_adjacency(op, p)
        @. tmp = p * Ap
        sum!(pAp, tmp)
        @. α = rsold / max(pAp, eps(T))
        @. x += α * p
        @. r -= α * Ap
        @. tmp = abs2(r)
        sum!(rsnew, tmp)
        sqrt(sum(rsnew) / nprobes) <= rtol * max(rms_bnorm, eps(T)) && break
        @. β = rsnew / max(rsold, eps(T))
        @. p = r + β * p
        rsold .= rsnew
        if reproject_every > 0 && iter % reproject_every == 0
            mul!(coeffs, r, Q)
            mul!(r, coeffs, Q', -one(T), one(T))
            mul!(coeffs, p, Q)
            mul!(p, coeffs, Q', -one(T), one(T))
            @. tmp = abs2(r)
            sum!(rsold, tmp)
        end
    end
    return nothing
end

#### Hyperparameter sufficient statistics

"""
    spatial_precision_stats(op, ξ::AbstractMatrix)

Per-dimension sufficient statistics from ``\\xi \\in \\mathbb{R}^{K \\times n}``:
``E_I^{(k)} = \\xi_k^\\top \\xi_k``, ``E_W^{(k)} = \\xi_k^\\top W \\xi_k``.
Returns `(; E_I, E_W, K, nsamples, n)`.
"""
function spatial_precision_stats(op::SpatialAdjacency{T}, ξ::AbstractMatrix{T}) where {T}
    Wξ = apply_normalized_adjacency(op, ξ)
    E_I = vec(sum(ξ .* ξ; dims = 2))
    E_W = vec(sum(ξ .* Wξ; dims = 2))
    K = size(ξ, 1)
    n = length(op)
    return (; E_I, E_W, K, nsamples = 1, n)
end

"""
    spatial_precision_stats(op, ξ::AbstractArray{T, 3})

Monte Carlo averaged sufficient statistics from ``\\xi \\in \\mathbb{R}^{K \\times n \\times S}``.
"""
function spatial_precision_stats(op::SpatialAdjacency{T}, ξ::AbstractArray{T, 3}) where {T}
    K, _, nsamp = size(ξ)
    E_I, E_W = zeros(T, K), zeros(T, K)
    for m in 1:nsamp
        s = spatial_precision_stats(op, view(ξ, :, :, m))
        E_I .+= s.E_I
        E_W .+= s.E_W
    end
    return (; E_I = E_I ./ nsamp, E_W = E_W ./ nsamp, K, nsamples = nsamp, n = length(op))
end

#### (α, β) parameterization: Q = αI + βL, L = I - W, κ = β/α

# log|Q| = N log(α) + log|I + κL|, h(κ) = d/dκ log|I + κL| = Σᵢ ℓᵢ/(1 + κℓᵢ)
# Forward moments tₖ = tr(Lᵏ) → Taylor at κ=0; inverse moments sₖ = tr₊(L⁻ᵏ) → asymptotics at κ=∞
# J-pole Stieltjes approximant ĥⱼ(κ) = Σⱼ wⱼpⱼ/(1 + κpⱼ) matches 2J moments exactly

"""
    laplacian_forward_moments(op, Val(J))

Forward spectral moments ``t_k = \\operatorname{tr}(L^k)`` for ``k = 1,\\ldots,J`` from cached ``\\operatorname{tr}(W^k)``.
Uses ``L = I - W`` with ``\\operatorname{tr}(W) = 0``.

Computed via the binomial expansion:
``t_k = \\operatorname{tr}((I-W)^k) = \\sum_{j=0}^{k} \\binom{k}{j} (-1)^j \\operatorname{tr}(W^j)``
where ``\\operatorname{tr}(W^0) = n`` and ``\\operatorname{tr}(W^1) = 0``.
"""
function laplacian_forward_moments(op::SpatialAdjacency{T}, ::Val{J}) where {T, J}
    @assert J <= length(op.tr_adj_powers) "J=$J exceeds the $(length(op.tr_adj_powers)) available cached forward moments"
    n = length(op)
    # tr(W^0) = n, tr(W^1) = 0, tr(W^k) = op.tr_adj_powers[k] for k >= 1
    trW = ntuple(Val(J + 1)) do j
        j == 1 ? T(n) : (j == 2 ? T(0) : op.tr_adj_powers[j-1])
    end
    # t_k = tr((I-W)^k) = Σⱼ₌₀ᵏ binom(k,j) (-1)^j tr(W^j)
    return ntuple(Val(J)) do k
        sum(j -> binomial(k, j) * (-1)^j * trW[j+1], 0:k)
    end
end

"""
    laplacian_inverse_moments(op)

Inverse spectral moments ``s_k = \\operatorname{tr}_+(L^{-k})`` for ``k = 0,1,2,...,J-1`` from cached ``\\operatorname{tr}_+(L^{-k})``.
"""
function laplacian_inverse_moments(op::SpatialAdjacency{T}, ::Val{J}) where {T, J}
    @assert J <= length(op.tr_lap_inv_powers) "J=$J exceeds the $(length(op.tr_lap_inv_powers)) available cached inverse moments"
    return ntuple(i -> op.tr_lap_inv_powers[i], Val(J))
end

"""
    StieltjesApproximant{J, T}

``J``-pole rational Stieltjes approximant: ``\\hat{h}_J(\\kappa) = \\sum_{j=1}^J w_j p_j / (1 + \\kappa p_j)``
with ``w_j, p_j > 0``, matching ``2J`` spectral moments ``t_1,\\ldots,t_J`` and ``s_0,\\ldots,s_{J-1}``.

Evaluation: [`h_approx`](@ref) (positive-pole sum), [`logdet_approx`](@ref)
(``\\sum_j w_j \\log(1 + \\kappa p_j)``).

# Fields

  - `poles`: effective Laplacian eigenvalues ``p_j``
  - `weights`: effective multiplicities ``w_j``
  - `num`, `den_tail`: numerator/denominator polynomial coefficients for `evalpoly`
  - `n`: number of active voxels
  - `valid`: whether all poles and weights are positive
"""
struct StieltjesApproximant{J, T}
    poles::SVector{J, T}
    weights::SVector{J, T}
    num::NTuple{J, T}
    den_tail::NTuple{J, T}
    n::Int
    valid::Bool
end

@inline npoles(::StieltjesApproximant{J}) where {J} = J

function Base.show(io::IO, sa::StieltjesApproximant{J}) where {J}
    print(io, "StieltjesApproximant{$J}(n=$(sa.n), valid=$(sa.valid), poles=$(round.(sa.poles; digits=4)), weights=$(round.(sa.weights; digits=4)))")
end

"""
    StieltjesApproximant(op; J = length(op.tr_lap_inv_powers))

Construct ``J``-pole approximant using the cached inverse Laplacian traces from `op.tr_lap_inv_powers`.
`J` defaults to the number of stored inverse moments (= `k_max` from construction).
"""
function StieltjesApproximant(op::SpatialAdjacency{T}, ::Val{J}; kwargs...) where {T, J}
    return StieltjesApproximant(length(op), laplacian_forward_moments(op, Val(J)), laplacian_inverse_moments(op, Val(J)); kwargs...)
end

function StieltjesApproximant(n::Int, tr_L_powers::NTuple{J, T}, tr_L_inv_powers::NTuple{J, T}) where {J, T}
    return convert_all_float(T, stieltjes_approximant_rutishauser(n, BigFloat.(tr_L_powers), BigFloat.(tr_L_inv_powers)))
end

"""
    stieltjes_approximant_rutishauser(n, tr_L_powers::NTuple{J}, tr_L_inv_powers::NTuple{J})

``J``-pole Stieltjes approximant via the Rutishauser quotient-difference (QD) algorithm.
Maps the bi-infinite moment sequence ``\\mu_k`` (``t_m`` for ``k \\geq 1``, ``s_{-k}`` for ``k \\leq 0``)
directly to continued-fraction coefficients via QD rhombus rules, avoiding the
ill-conditioned Hankel matrix. The resulting tridiagonal Jacobi matrix is eigendecomposed
to obtain poles ``p_j`` and weights ``w_j``.
"""
function stieltjes_approximant_rutishauser(n::Int, tr_L_powers::NTuple{J, T}, tr_L_inv_powers::NTuple{J, T}) where {J, T}
    @assert J >= 1
    t, s = tr_L_powers, tr_L_inv_powers # t[m] = tₘ, s[k] = s_{k-1} (1-indexed)

    # Bi-infinite moment sequence μₖ = tₖ (k≥1), s_{-k} (k≤0); stored with offset J
    μ = T[(i - J >= 1 ? t[i-J] : s[-(i-J)+1]) for i in 1:2J]

    # QD rhombus: initialize q₁^(m) = μ_{m+1}/μ_m
    q = T[μ[i+1] / μ[i] for i in 1:(2J-1)]
    a = Vector{T}(undef, J)
    b = Vector{T}(undef, J - 1)
    a[1] = q[1]

    if J >= 2
        e = T[q[i+1] - q[i] for i in 1:(2J-2)] # e₁^(m)
        b[1] = q[1] * e[1]
        for k in 2:J
            q_new = T[q[i+1] * e[i+1] / e[i] for i in 1:(2J-2k+1)]
            a[k] = q_new[1] + e[1]
            if k <= J - 1
                e_new = T[e[i+1] + q_new[i+1] - q_new[i] for i in 1:(2J-2k)]
                b[k] = q_new[1] * e_new[1]
                e = e_new
            end
            q = q_new
        end
    end

    # Eigendecompose Jacobi tridiagonal → poles and weights
    valid = all(x -> isfinite(x) && x > 0, b) && all(isfinite, a)
    poles_out, weights_out = zeros(T, J), zeros(T, J)

    if valid
        F = LinearAlgebra.eigen(LinearAlgebra.SymTridiagonal(a, sqrt.(b)))
        weights_raw = T[μ[1] * F.vectors[1, j]^2 * F.values[j]^(J - 1) for j in 1:J]
        valid = all(x -> isfinite(x) && x > 0, F.values) && all(x -> isfinite(x) && x > 0, weights_raw)
        if valid
            perm = sortperm(F.values; rev = true)
            poles_out, weights_out = F.values[perm], weights_raw[perm]
        end
    end

    # Denominator: ∏ⱼ(1 + κpⱼ) = 1 + B₁κ + ⋯ + B_Jκᴶ
    den_coeffs = zeros(T, J + 1)
    den_coeffs[1] = one(T)
    for j in 1:J, k in j:-1:1
        den_coeffs[k+1] += den_coeffs[k] * poles_out[j]
    end

    # Numerator via synthetic division: N(κ) = Σⱼ wⱼpⱼ ∏_{i≠j}(1 + κpᵢ)
    num_coeffs = zeros(T, J)
    for j in 1:J
        c = Vector{T}(undef, J)
        c[J] = den_coeffs[J+1] / poles_out[j]
        for k in (J-1):-1:1
            c[k] = (den_coeffs[k+1] - c[k+1]) / poles_out[j]
        end
        num_coeffs .+= (weights_out[j] * poles_out[j]) .* c
    end

    return StieltjesApproximant{J, T}(
        SVector{J, T}(poles_out), SVector{J, T}(weights_out),
        NTuple{J, T}(num_coeffs), NTuple{J, T}(den_coeffs[2:(J+1)]),
        n, valid,
    )
end

"""
    h_approx(sa, κ)

Evaluate ``\\hat{h}_J(\\kappa)`` from its positive poles and weights.
"""
@inline function h_approx(sa::StieltjesApproximant{J, T}, κ::T) where {J, T}
    sa.valid || return T(NaN)
    return sum(sa.weights .* sa.poles ./ (one(T) .+ κ .* sa.poles))
end

"""
    logdet_approx(sa, κ)

``\\hat{F}_J(\\kappa) = \\sum_j w_j \\log(1 + \\kappa p_j) \\approx \\log|I + \\kappa L|``.
Returns `NaN` if invalid.
"""
@inline function logdet_approx(sa::StieltjesApproximant{J, T}, κ::T) where {J, T}
    sa.valid || return T(NaN)
    return sum(sa.weights .* log1p.(κ .* sa.poles))
end
