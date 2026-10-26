module DynamicEPG

using DECAES
using UnsafeArrays

using DECAES: EPGConstantFlipAngleOptions, EPGOptions, epg_decay_basis!, split_indices

const BasisWork = DECAES.EPGWork_ReIm_Batched_Split_Dynamic # whole decay basis

struct BasisWorkspace{T, W}
    work::W
    T2_times::Vector{T}
end
BasisWorkspace(::Type{T}, ETL::Int, nT2::Int) where {T} = BasisWorkspace(BasisWork(T, ETL), zeros(T, nT2))

function epg_basis(ETL::Int, nT2::Int, α::T, TE::T, T2min::T, T2max::T, T1::T) where {T}
    A = zeros(T, ETL, nT2)
    epg_basis!(BasisWorkspace(T, ETL, nT2), A, ETL, α, TE, T2min, T2max, T1)
    return A
end

function epg_basis!(
    work::BasisWorkspace{T},
    A::AbstractMatrix{T},
    ETL::Int,
    α::T, # degrees
    TE::T,
    T2min::T,
    T2max::T,
    T1::T,
) where {T}
    nTEmax, nT2 = size(A)
    @assert length(work.T2_times) == nT2 "T2 buffer length mismatch"
    @assert ETL <= nTEmax "Echo train exceeds the basis buffer"
    work.T2_times .= exp.(range(log(T2min), log(T2max); length = nT2))
    θ = EPGConstantFlipAngleOptions{T, Int}(ETL, deg2rad(α), TE, T(NaN), T1) # `EPGOptions` stores radians; T2 comes from `T2_times`
    GC.@preserve A begin
        epg_decay_basis!(uview(A, 1:ETL, :), work.work, θ, work.T2_times)
        (ETL < nTEmax) && (uview(A, (ETL+1):nTEmax, :) .= 0)
    end
    return nothing
end

function epg_basis!(
    b::AbstractMatrix{T},
    A::AbstractArray{T, 3},
    nTE::AbstractMatrix{<:Union{Int32, Int}},
    α::AbstractMatrix{T},
    TE::AbstractMatrix{T},
    T2Range::AbstractMatrix{T},
    T1::AbstractMatrix{T};
    ntasks = Threads.nthreads(),
) where {T}
    @assert size(b, 2) == size(A, 3) == size(α, 2) == size(TE, 2) == size(T2Range, 2) "Batch size mismatch"
    @assert size(b, 1) == size(A, 1) "Echo dimension size mismatch"

    nTEmax, nT2, batchsize = size(A)
    blocks = split_indices(; length = batchsize, minchunksize = 1, maxpartitions = ntasks)
    GC.@preserve b A @sync for block in blocks
        Threads.@spawn begin
            work = BasisWorkspace(T, size(A, 1), nT2)
            @inbounds for k in block
                ETL = Int(nTE[1, k])
                epg_basis!(work, uview(A, :, :, k), ETL, α[1, k], TE[1, k], T2Range[1, k], T2Range[2, k], T1[1, k])
            end
        end
    end

    return A
end

end # module DynamicEPG
