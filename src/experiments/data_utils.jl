#### Data partition enum

@enum Partition TRAIN VAL TEST

partition(part::String) =
    part == "train" ? TRAIN :
    part == "test" ? TEST :
    part == "val" ? VAL :
    error("Unknown partition: " * part)

#### Multi spin-echo data

lib.@with_kw struct MSEData
    path::String
    data::Array{Float64, 4}
    data_scale::Array{Float64, 3}
    data_bin_width::Array{Float64, 3}
    mask_inds::Array{CartesianIndex{3}, 3}
    echotime::Float64
    numechoes::Int
    voxelsize::NTuple{3, Float64}
    image_size::NTuple{4, Int}
    adc_bin_width::Float64
end
Base.size(img::MSEData) = size(img.data)[1:3]
nsignals(img::MSEData) = prod(size(img))

function MSEData(img_cfg::Config)
    img_path = lib.clean_realpath(img_cfg.path::String)

    if any(endswith.(img_path, [".par", ".rec", ".PAR", ".REC"]))
        rec = with_rec(img_path) do path
            rec = lib.DECAES.ParXRec.load(path; load_data = true, scale_data = true)
            check_rec_filesize(path, rec.hdr)
            return rec
        end
        data = convert(Array{Float64}, rec.data)
        adc_bin_width = inv(map(i -> iszero(i.Scale_Slope) ? 1.0 : Float64(i.Scale_Slope), rec.hdr.images) |> unique |> only)
        data_discrete = round.(Int, data ./ adc_bin_width)
        echotimes = lib.DECAES.ParXRec.echotimes(rec.hdr)
        voxelsize = lib.DECAES.ParXRec.voxelsize(rec.hdr)::NTuple{3, Float64}
        @assert all(isapprox.(echotimes, echotimes[1] .* (1:length(echotimes)); rtol = 1e-2)) "Echo times are not uniformly spaced: $(echotimes)"
        @assert isapprox(img_cfg.echotime::Float64, echotimes[1]; rtol = 1e-2) "Configured echo time $(img_cfg.echotime) does not match header echo time $(echotimes[1])"
        echotime = echotimes[1]
    elseif any(endswith.(img_path, [".nii", ".nii.gz"]))
        nii = lib.DECAES.NIfTI.niread(img_path)
        # See: https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/datatype.html
        if nii.header.datatype == 4 && nii.header.bitpix == 16
            # Int16 values with linear scaling; the bin width is the scale slope (one value if unscaled)
            slope, inter = nii.header.scl_slope, nii.header.scl_inter
            @assert isnan(inter) || iszero(inter) "Int16 NIfTI with nonzero intercept is not supported: $(img_path)"
            adc_bin_width = isnan(slope) || iszero(slope) ? 1.0 : Float64(slope)
            data_discrete = convert(Array{Int}, nii.raw[:, :, :, :])
            data = adc_bin_width .* data_discrete
        else
            @assert nii.header.datatype == 16 && nii.header.bitpix == 32 # Float32 values
            data = convert(Array{Float64}, nii[:, :, :, :])
            # The two smallest distinct positive values differ by one bin width, up to Float32 rounding
            tol = maximum(data) / 2^16
            x1 = minimum(x -> x > 0 ? x : Inf, data)
            x2 = minimum(x -> x > x1 + tol ? x : Inf, data)
            adc_bin_width = x2 - x1
            data_discrete = round.(Int, data ./ adc_bin_width)
            adc_bin_width = lib.mean(data[data_discrete .>= 1] ./ data_discrete[data_discrete .>= 1])
            data_discrete .= round.(Int, data ./ adc_bin_width)
        end
        voxelsize = Float64.(Tuple(nii.header.pixdim[2:4]))
        echotime = img_cfg.echotime::Float64 # NIfTI headers do not store echo times
    else
        error("Unknown file type: $(img_path)")
    end
    @assert size(data, 4) == img_cfg.numechoes::Int "Configured number of echoes $(img_cfg.numechoes) does not match data size $(size(data))"

    # Values are Float32-rounded bin-width multiples; allow a few ulps relative to max(|x|, bin width)
    @assert all(abs.(adc_bin_width .* data_discrete .- data) .<= 4 * eps(Float32) .* max.(abs.(data), adc_bin_width)) "ADC bin width mismatch"

    data_scale = dropdims(maximum(data; dims = 4); dims = 4)
    data_bin_width = fill(adc_bin_width, size(data_scale))

    return MSEData(;
        path = img_path,
        data = data,
        data_scale = data_scale,
        data_bin_width = data_bin_width,
        mask_inds = collect(CartesianIndices(size(data)[1:3])),
        echotime = echotime,
        numechoes = size(data, 4),
        voxelsize = voxelsize,
        image_size = size(data),
        adc_bin_width = adc_bin_width,
    )
end

# ParXRec reads RECs by seeking, so a gzipped REC is decompressed into a temporary directory
function with_rec(f, img_path::String)
    hdrpath = first(lib.DECAES.ParXRec.parxrec_paths(img_path, false))
    base = first(splitext(hdrpath))
    any(isfile, base .* (".REC", ".rec")) && return f(hdrpath)
    gzpath = base * ".REC.gz"
    @assert isfile(gzpath) "No REC or REC.gz file for $(hdrpath)"
    return mktempdir() do dir
        tmphdr = joinpath(dir, basename(hdrpath))
        cp(hdrpath, tmphdr)
        run(pipeline(`gzip -dc $(gzpath)`; stdout = first(splitext(tmphdr)) * ".REC"))
        return f(tmphdr)
    end
end

# ParXRec zero-fills images missing from a truncated REC file with only a warning
function check_rec_filesize(img_path::String, hdr)
    _, recpath = lib.DECAES.ParXRec.parxrec_paths(img_path, true)
    expected = sum(i -> i.Resolution_X * i.Resolution_Y * (i.Pixel_Size ÷ 8), hdr.images)
    @assert filesize(recpath) == expected "REC file size $(filesize(recpath)) does not match header size $(expected): $(recpath)"
end

# Indexing with a vector results in "flat" arrays of size `(length(i), 1, 1, ...)`
function Base.getindex(img::MSEData, i::AbstractVector)
    I = CartesianIndices(size(img))[i]
    return MSEData(
        img;
        data = reshape(img.data[I, :], length(I), 1, 1, :),
        data_scale = reshape(img.data_scale[I], length(I), 1, 1),
        data_bin_width = reshape(img.data_bin_width[I], length(I), 1, 1),
        mask_inds = reshape(img.mask_inds[I], length(I), 1, 1),
    )
end

function preprocess_signals(cfg::Config, img::MSEData, mask)
    @assert size(mask) == size(img) "Mask does not match image size"
    img = img[findall(==(1), mask)]
    nonzero = @views vec(img.data[:, 1, 1, 1] .> 0)
    img = img[findall(nonzero .& vec(img.data_scale .> 0))]
    img = img[lib.randperm(lib.Xoshiro(cfg.prep.seed::Int), nsignals(img))]
    @assert all(isfinite.(img.data) .&& img.data .>= 0)
    @assert all(isfinite.(img.data_bin_width) .&& img.data_bin_width .> 0)
    return img
end

#### DECAES results

lib.@with_kw struct DECAESConfig
    t2mapopts::lib.DECAES.T2mapOptions{Float64}
    t2partopts::lib.DECAES.T2partOptions{Float64}
end

function DECAESConfig(cfg::Config, data::Union{AbstractArray{<:Any, 4}, Nothing} = nothing; TE, SPCutoff = 25e-3, kwargs...)
    t2mapopts = lib.DECAES.T2mapOptions{Float64}(;
        MatrixSize       = data === nothing ? kwargs[:MatrixSize] : size(data)[1:3],
        nTE              = data === nothing ? kwargs[:nTE] : size(data)[4],
        TE               = TE,
        T2Range          = (TE, 1.0),
        nT2              = cfg.prep.decaes.nT2::Int,
        Reg              = default_reg(cfg::Config),
        Chi2Factor       = cfg.prep.decaes.Chi2Factor::Float64,
        T1               = cfg.prep.decaes.T1::Float64,
        Threshold        = 0.0,
        nRefAnglesMin    = cfg.prep.decaes.nRefAnglesMin::Int,
        nRefAngles       = cfg.prep.decaes.nRefAngles::Int,
        SaveNNLSBasis    = false,
        SaveDecayCurve   = false,
        SaveResidualNorm = true,
        SaveRegParam     = true,
        Silent           = true,
        kwargs...,
    )

    t2partopts = lib.DECAES.T2partOptions{Float64}(;
        MatrixSize = t2mapopts.MatrixSize,
        nT2        = t2mapopts.nT2,
        T2Range    = t2mapopts.T2Range,
        SPWin      = (prevfloat(t2mapopts.T2Range[1]), SPCutoff),
        MPWin      = (nextfloat(SPCutoff), nextfloat(t2mapopts.T2Range[2])),
        Sigmoid    = nothing,
        Silent     = true,
    )

    return DECAESConfig(t2mapopts, t2partopts)
end

function DECAESConfig(cfg::Config, img::MSEData; kwargs...)
    return DECAESConfig(cfg, img.data; TE = img.echotime, kwargs...)
end

lib.@with_kw struct DECAESResults
    t2mapopts::lib.DECAES.T2mapOptions{Float64}
    t2partopts::lib.DECAES.T2partOptions{Float64}
    t2dist::Array{Float64, 4}
    t2maps::Dict{String, Any}
    t2parts::Dict{String, Any}
end

function compute_decaes(data::AbstractArray{<:Any, 4}, decaes_cfg::DECAESConfig)
    (; t2mapopts, t2partopts) = decaes_cfg
    t2maps, t2dist = lib.DECAES.T2mapSEcorr(data, t2mapopts)
    t2parts = lib.DECAES.T2partSEcorr(t2dist, t2partopts)
    return DECAESResults(; t2mapopts, t2partopts, t2dist, t2maps, t2parts)
end
compute_decaes(img::MSEData, decaes_cfg::DECAESConfig) = compute_decaes(img.data, decaes_cfg)
compute_decaes(cfg::Config, img::MSEData; kwargs...) = compute_decaes(img, DECAESConfig(cfg, img; kwargs...))

#### Arrow column layout

# Coalesce data columns into an array
function load_array_from_cols!(x::AbstractMatrix{T}, df, key, I) where {T}
    nrow = lib.Tables.rowcount(df)
    len, n = size(x)
    @assert (I === Colon() && n == nrow) || (n == length(I) && n <= nrow) "Invalid index range"
    Threads.@threads for i in 1:len # columns are independent gathers from the mmap-ed `df`
        xi = df[key_name(key, len, i)]::AbstractVector{T}
        x[i, :] .= xi[I]::Vector{T}
    end
    return x
end

function load_array_from_cols(::Type{T}, df, key, I) where {T}
    nrow = lib.Tables.rowcount(df)
    len = count(startswith(string(key, "_")) ∘ string, keys(df))
    n = I isa Colon ? nrow : length(I)
    x = zeros(T, len, n)
    return load_array_from_cols!(x, df, key, I)
end
key_name(key, len, i) = Symbol(key, :_, lpad(i, ndigits(len), '0'))

#### Offline voxel-table fitting

function create_dataset(cfg::Config, img_cfg::Config, img::MSEData; part::String)
    # Each image is split into `prep.decaes.chunks` random voxel chunks per regularization method,
    # each fitted single-threaded so that results do not depend on thread scheduling.
    regs = reg_methods(cfg)
    decaes_augs = vec(collect(Iterators.product(eachindex(regs), 1:(cfg.prep.decaes.chunks::Int))))
    nrows = nsignals(img)
    indices_bounds = round.(Int, range(1, nrows; length = length(decaes_augs) + 1))
    indices = [indices_bounds[i]:(indices_bounds[i+1]-(i<length(decaes_augs))) for i in 1:length(decaes_augs)]
    @assert !any(isempty, indices) "More augmentations ($(length(decaes_augs))) than data points ($(nrows))"
    @assert reduce(vcat, indices) == 1:nrows "Invalid row partitioning"

    # Voxel table; per-image quantities (TE, nTE, bin width, size, T2 range) are in the image table
    df             = lib.OrderedDict{Symbol, AbstractVector}()
    df[:Dataset]   = zeros(UInt8, nrows) # Partition enum
    df[:ImageID]   = zeros(Int32, nrows) # Key into the image table
    df[:Index]     = fill(UInt16.((0, 0, 0)), nrows) # Cartesian index corresponding to voxel
    df[:FlipAngle] = zeros(Float32, nrows) # Flip angle found from automatic B1-correction
    df[:RegParam]  = zeros(Float32, nrows) # Regularization parameter μ from NNLS fit
    df[:RegMethod] = zeros(UInt8, nrows) # Index into `prep.decaes.Reg` of the regularization method
    df[:ResNorm]   = zeros(Float32, nrows) # L2 norm of residuals from NNLS fit
    df[:GDN]       = zeros(Float32, nrows) # General density: GDN = Σᵢ distᵢ
    df[:GGM]       = zeros(Float32, nrows) # General geometric mean: GGM = exp(log_ggm)
    df[:GVA]       = zeros(Float32, nrows) # General variance: GVA = expm1(log1p_gva)
    df[:FNR]       = zeros(Float32, nrows) # Fit to noise ratio: FNR = Σᵢ distᵢ / √(Σᵢ resᵢ² / (nTE - 1))
    df[:SNR]       = zeros(Float32, nrows) # Signal-to-noise ratio: SNR = max_signal / σ_res
    nTEmax         = cfg.prep.maxechoes::Int
    for j in 1:nTEmax
        df[key_name(:Signal, nTEmax, j)] = zeros(Int16, nrows) # Measured signals in ADC counts
    end
    # BLAS stays single-threaded: concurrent DECAES calls reset its thread count, and threading changes rounding
    @info "Running DECAES using $(Threads.nthreads()) thread(s) ($(nrows) signals)..."
    @time lib.DECAES.with_singlethreaded_blas() do
        Threads.@sync for (inds, (r, _)) in Iterators.zip(indices, decaes_augs)
            Threads.@spawn let
                img_chunk = img[inds]

                decaes_chunk = compute_decaes(
                    cfg, img_chunk;
                    Reg = regs[r],
                    Threaded = false,
                )
                (; nTE, T2Range) = decaes_chunk.t2mapopts

                @assert all(isfinite, decaes_chunk.t2maps["alpha"])

                df[:Dataset][inds]   .= UInt8(partition(part))
                df[:ImageID][inds]   .= Int32(img_cfg.id::Int)
                df[:Index][inds]     .= map(I -> UInt16.(Tuple(I)), vec(img_chunk.mask_inds)) # Cartesian index corresponding to voxel
                df[:FlipAngle][inds] .= vec(decaes_chunk.t2maps["alpha"]) # Flip angle found from automatic B1-correction
                df[:RegParam][inds]  .= vec(decaes_chunk.t2maps["mu"]) # Regularization parameter from NNLS fit
                df[:RegMethod][inds] .= UInt8(r) # Regularization method
                df[:ResNorm][inds]   .= vec(decaes_chunk.t2maps["resnorm"]) # L2 norm of residuals from NNLS fit
                df[:GDN][inds]       .= vec(decaes_chunk.t2maps["gdn"]) # General density: GDN = Σᵢ distᵢ
                df[:GGM][inds]       .= vec(decaes_chunk.t2maps["ggm"]) # General geometric mean: GGM = exp(log_ggm)
                df[:GVA][inds]       .= vec(decaes_chunk.t2maps["gva"]) # General variance: GVA = expm1(log1p_gva)
                df[:FNR][inds]       .= vec(decaes_chunk.t2maps["fnr"]) # Fit to noise ratio: FNR = Σᵢ distᵢ / √(Σᵢ resᵢ² / (nTE - 1))
                df[:SNR][inds]       .= vec(decaes_chunk.t2maps["snr"]) # Signal-to-noise ratio: SNR = max_signal / σ_res
                for j in 1:nTE
                    df[key_name(:Signal, nTEmax, j)][inds] .= round.(Int16, img_chunk.data[:, 1, 1, j] ./ img_chunk.data_bin_width) # Quantized measured signals
                end
            end
        end
    end

    return img, df
end

function default_reg(cfg::Config)
    reg = cfg.prep.decaes.Reg
    return reg isa String ? reg : first(reg)::String
end

function reg_methods(cfg::Config)
    reg = cfg.prep.decaes.Reg
    return reg isa String ? [reg] : Vector{String}(reg)
end

#### Local dataset build

image_list(cfg::Config) = [(part, name, img_cfg) for part in ["train", "val", "test"] for (name, img_cfg) in sort!(collect(pairs(cfg[part].images)); by = first)]
read_mask(file::String) = lib.DECAES.NIfTI.niread(file).raw .> 0

function image_record(img::MSEData, nrows, seconds)
    record = Dict{String, Any}(
        "rows" => nrows,
        "TE" => img.echotime,
        "nTE" => img.numechoes,
        "voxelsize" => collect(img.voxelsize),
        "image_size" => collect(img.image_size),
        "adc_bin_width" => img.adc_bin_width,
        "seconds" => round(seconds; digits = 1),
    )
    record["T2Range"] = [img.echotime, 1.0]
    return record
end

function prepare_data(cfg::Config)
    cfg = deepcopy(cfg)
    filename = joinpath(mkpath(lib.clean_normpath(cfg.output::String)), "preprocessed_dataset.arrow")
    sidecar = replace(filename, ".arrow" => ".toml")
    @assert !isfile(filename) && !isfile(sidecar) "Use a new output directory for each dataset build"
    images = image_list(cfg)
    @assert !isempty(images)
    sources = [lib.clean_realpath(img_cfg.path::String) for (_, _, img_cfg) in images]
    @assert allunique(sources) "Each source image must belong to exactly one partition"
    mktempdir() do tmp
        files = String[]
        for (id, (part, name, img_cfg)) in enumerate(images)
            img_cfg.id = id
            img_cfg.path = sources[id]
            img_cfg.mask = lib.clean_realpath(img_cfg.mask::String)
            elapsed = @elapsed begin
                @info "Loading data" image = name
                img = preprocess_signals(cfg, MSEData(img_cfg), read_mask(lib.clean_realpath(img_cfg.mask::String)))
                img, columns = create_dataset(cfg, img_cfg, img; part)
            end
            merge!(img_cfg, Config(image_record(img, length(columns[:ImageID]), elapsed)))
            imagefile = joinpath(tmp, "$(id).arrow")
            lib.Arrow.write(imagefile, (; columns...))
            push!(files, imagefile)
        end
        metadata = Dict("config" => sprint(io -> lib.TOML.print(io, cfg)))
        tables = [lib.Arrow.Table(file) for file in files]
        lib.Arrow.write(filename, lib.Tables.partitioner(tables); metadata)
        lib.TOMLConfig.save_config(cfg; filename = sidecar)
    end
    return filename
end

#### Generated demonstration dataset

function synthetic_image(rng, cfg::Config, dims, nTE, α)
    TE, δ = 0.008f0, 1 / 4096.0f0
    nT2, nTEmax = cfg.data.nT2::Int, cfg.data.nTE::Int
    T2s = exp.(range(log(TE), log(1.0f0); length = nT2))
    basis = lib.DynamicEPG.epg_basis(nTE, nT2, α, TE, TE, 1.0f0, 1.0f0)
    truth = zeros(Float32, nT2, prod(dims))
    counts = zeros(Int16, nTEmax, prod(dims))
    σ = 10.0f0^(-40.0f0 / 20)

    for j in axes(truth, 2)
        μ1, μ2 = 0.01f0 + 0.015f0 * rand(rng, Float32), 0.05f0 + 0.05f0 * rand(rng, Float32)
        s1, s2 = 0.0025f0 + 0.0025f0 * rand(rng, Float32), 0.005f0 + 0.005f0 * rand(rng, Float32)
        f = 0.25f0 * rand(rng, Float32)
        p1, p2 = exp.(-((T2s .- μ1) ./ s1) .^ 2 ./ 2), exp.(-((T2s .- μ2) ./ s2) .^ 2 ./ 2)
        truth[:, j] .= f .* p1 ./ sum(p1) .+ (1 - f) .* p2 ./ sum(p2)
        ν = basis * view(truth, :, j)
        for k in 1:nTE
            b = hypot(ν[k] + σ * randn(rng, Float32), σ * randn(rng, Float32))
            counts[k, j] = floor(Int16, b / δ)
        end
    end
    return (; counts, truth, T2s, TE, δ)
end

"""
Write disjoint generated train/validation/test images using the training Arrow schema.
"""
function synthetic_dataset(cfg::Config, output; seed = 42)
    mkpath(output)
    rng = lib.Xoshiro(seed)
    specs = (
        ("train", (32, 32, 1), 56, 165.0f0),
        ("val", (16, 16, 1), 48, 170.0f0),
        ("test", (16, 16, 1), 56, 165.0f0),
    )
    data_cfg = Dict{String, Any}(
        "prep" => Dict(
            "maxechoes" => cfg.data.nTE::Int,
            "decaes" => Dict(
                "nT2" => cfg.data.nT2::Int, "T1" => 1.0,
                "Reg" => ["gcv"], "Chi2Factor" => 1.02,
                "nRefAnglesMin" => 16, "nRefAngles" => 64,
            ),
        ),
        "synthetic" => Dict("seed" => seed, "SNR" => 40.0, "scale" => 1.0, "normalization" => "known generating scale"),
    )
    tables = NamedTuple[]
    truths = Dict{String, Any}()

    for (id, (part, dims, nTE, α)) in enumerate(specs)
        image = synthetic_image(rng, cfg, dims, nTE, α)
        n = prod(dims)
        image_cfg = Dict(
            "id" => id, "TE" => Float64(image.TE), "nTE" => nTE,
            "echotime" => Float64(image.TE), "numechoes" => nTE,
            "adc_bin_width" => Float64(image.δ), "image_size" => [dims..., nTE],
            "T2Range" => [Float64(image.TE), 1.0], "path" => "generated:synthetic_$(part)",
            "subset" => (part == "test" ? "test_id" : ""),
        )
        data_cfg[part] = Dict("images" => Dict("synthetic_$(part)" => image_cfg))
        columns = OrderedDict{Symbol, AbstractVector}(
            :Dataset => fill(UInt8(partition(part)), n),
            :ImageID => fill(Int32(id), n),
            :Index => vec([UInt16.(Tuple(I)) for I in CartesianIndices(dims)]),
            :FlipAngle => fill(α, n),
            :GDN => ones(Float32, n),
        )
        for j in 1:(cfg.data.nTE::Int)
            columns[key_name(:Signal, cfg.data.nTE::Int, j)] = image.counts[j, :]
        end
        push!(tables, (; columns...))
        truths[part] = (; xdist = image.truth, T2s = image.T2s, dims, image_id = id)
    end

    filename = joinpath(output, "preprocessed_dataset.arrow")
    metadata = Dict("config" => sprint(io -> lib.TOML.print(io, data_cfg)))
    lib.Arrow.write(filename, lib.Tables.partitioner(tables); metadata)
    open(io -> lib.TOML.print(io, data_cfg), replace(filename, ".arrow" => ".toml"), "w")
    lib.JLD2.jldsave(joinpath(output, "truth.jld2"); truths)
    return filename
end

#### Dataset tables and image metadata

const DictTable = AbstractDict{<:Union{Symbol, String}, <:AbstractVector}

@with_kw_noshow struct PartitionedDataset{Tdata, Ttrain, Tval, Ttest}
    df::Tdata
    train::Ttrain
    val::Tval
    test::Ttest
end
Base.show(io::IO, data::PartitionedDataset) = print(io, "PartitionedDataset($(lib.Tables.rowcount(data.df)) rows: $(lib.Tables.rowcount(data.train)) train / $(lib.Tables.rowcount(data.val)) val / $(lib.Tables.rowcount(data.test)) test)")

# Per-row view of a per-image quantity, gathered through each row's `ImageID`
struct ImageColumn{T, V <: AbstractVector{T}, I <: AbstractVector{<:Integer}} <: AbstractVector{T}
    values::V
    ids::I
end
Base.size(c::ImageColumn) = size(c.ids)
Base.@propagate_inbounds Base.getindex(c::ImageColumn, i::Int) = c.values[c.ids[i]]
Base.getindex(c::ImageColumn, I::AbstractVector{<:Integer}) = c.values[c.ids[I]]

# Per-image columns come from the image table in the Arrow metadata, or per row if absent
function image_columns(table)
    meta = lib.Arrow.getmetadata(table)
    (meta === nothing || !haskey(meta, "config")) && return Dict{Symbol, AbstractVector}()
    ids = lib.Tables.getcolumn(table, :ImageID)
    data_cfg = lib.TOML.parse(meta["config"])
    images = [img for part in ["train", "val", "test"] for img in values(data_cfg[part]["images"])]
    n = maximum(img -> img["id"]::Int, images)
    TE, nTE, bw = zeros(Float32, n), zeros(Int32, n), zeros(Float32, n)
    sz, t2 = fill(Int32.((0, 0, 0, 0)), n), fill((0.0f0, 0.0f0), n)
    for img in images
        id = img["id"]::Int
        TE[id], nTE[id], bw[id] = img["TE"], img["nTE"], img["adc_bin_width"]
        sz[id] = Tuple(Int32.(img["image_size"]))
        haskey(img, "T2Range") && (t2[id] = Tuple(Float32.(img["T2Range"])))
    end
    nT2 = fill(Int32(data_cfg["prep"]["decaes"]["nT2"]), n)
    cols = Dict{Symbol, AbstractVector}(
        :TE => ImageColumn(TE, ids),
        :nTE => ImageColumn(nTE, ids),
        :nT2 => ImageColumn(nT2, ids),
        :ADCBinWidth => ImageColumn(bw, ids),
        :SignalBinWidth => ImageColumn(bw, ids), # signals are stored in ADC counts
        :ImageSize => ImageColumn(sz, ids),
    )
    !(:T2Range in lib.Tables.columnnames(table)) && all(img -> haskey(img, "T2Range"), images) && (cols[:T2Range] = ImageColumn(t2, ids))
    return cols
end

function dataset(filename::AbstractString)
    record_info = map(lib.Arrow.Stream(filename)) do r
        return (; nrow = lib.Tables.rowcount(r), part = Int(r.Dataset[1]))
    end

    function partition(part::Partition)
        istart = findfirst(r -> r.part == Int(part), record_info)
        iend = findlast(r -> r.part == Int(part), record_info)
        istart === nothing && iend === nothing && return 1:0
        @assert all(r -> r.part == Int(part), record_info[istart:iend]) "Partition \"$(part)\" not contiguous"
        @assert !any(r -> r.part == Int(part), record_info[[1:(istart-1); (iend+1):end]]) "Partition \"$(part)\" not unique"
        rows = sum(r -> r.nrow, record_info[1:(istart-1)]; init = 0) .+ (1:sum(r->r.nrow, record_info[istart:iend]))
        return rows
    end

    # Load a memory-mapped Arrow dataset as a `NamedTuple` of vectors, adding per-image columns
    table = lib.Arrow.Table(filename)
    df = OrderedDict{Symbol, AbstractVector}(pairs(table))
    for (k, v) in image_columns(table)
        @assert !haskey(df, k) "Column $(k) is stored both per row and per image"
        df[k] = v
    end
    df = sort(df)

    train_rows, val_rows, test_rows = partition(TRAIN), partition(VAL), partition(TEST)

    return PartitionedDataset(;
        df    = df,
        train = sort(OrderedDict{Symbol, AbstractVector}(k => @views(v[train_rows]) for (k, v) in df)),
        val   = sort(OrderedDict{Symbol, AbstractVector}(k => @views(v[val_rows]) for (k, v) in df)),
        test  = sort(OrderedDict{Symbol, AbstractVector}(k => @views(v[test_rows]) for (k, v) in df)),
    )
end

#### Row filtering and image slices

const ImageSlice = Tuple{Vararg{Union{Colon, Int}}}

isslice(I, slice::Tuple{Int, Colon, Colon}) = I[1] == slice[1]
isslice(I, slice::Tuple{Colon, Int, Colon}) = I[2] == slice[2]
isslice(I, slice::Tuple{Colon, Colon, Int}) = I[3] == slice[3]
isslice(I, slice::ImageSlice) = error("Invalid slice: $(slice)")
isslice(slice::ImageSlice) = Base.Fix2(isslice, slice)

"""
    slice_row_indices(partition::DictTable, slice::ImageSlice) -> Vector{Int}

Return row indices into `partition` for the given `slice`. Errors if the partition is empty,
has more than one image, or contains no rows matching the slice.
"""
function slice_row_indices(partition::DictTable, slice::ImageSlice)
    nrow = lib.Tables.rowcount(partition)
    nrow == 0 && error("Partition is empty")
    image_ids = partition[:ImageID]
    n_unique = length(unique!(copy(image_ids)))
    n_unique > 1 && error("Partition has more than one image ($n_unique images); slice selection requires a single image")
    indices_bool = isslice(slice).(copy(partition[:Index]))
    I = findall(indices_bool)
    isempty(I) && error("No rows found for slice $slice")
    return I
end

function image_partition(df::DictTable, image_id = nothing)
    ids = df[:ImageID]
    @assert !isempty(ids) "Partition is empty"
    if image_id === nothing
        @assert all(==(first(ids)), ids) "Multi-image partition requires an explicit image_id"
        return df
    end
    I = findall(==(image_id), ids)
    @assert !isempty(I) "ImageID $(image_id) is absent from this partition"
    return lib.map_dict(x -> view(x, I), df)
end

function image_slice(cfg::Config, df::DictTable, slice::ImageSlice)
    I = slice_row_indices(df, slice)
    postprocess_batch! = batch_postprocesser(cfg; augment = false)
    return Batch(df, I, postprocess_batch!; nT2 = cfg.data.nT2::Int) |> NamedTuple
end

#### Batch storage

# Raw pools remain read-only while queued batches or forks refer to them.
struct RawBatch
    b::Matrix{Float32}
    δb::Matrix{Float32}
    nTE::Matrix{Int32}
    α::Matrix{Float32}
    TE::Matrix{Float32}
    T2Range::Matrix{Float32}
    idx::Matrix{NTuple{3, Int32}}
end
Base.Tuple(raw::RawBatch) = (raw.b, raw.δb, raw.nTE, raw.α, raw.TE, raw.T2Range, raw.idx)
Base.length(raw::RawBatch) = size(raw.b, 2)
Base.copy(raw::RawBatch) = RawBatch(map(copy, Tuple(raw))...)
Base.show(io::IO, raw::RawBatch) = print(io, "RawBatch(n = $(length(raw)))")
raw_columns(raw::RawBatch, I) = RawBatch(map(x -> x[:, I], Tuple(raw))...)

struct Batch
    b::Matrix{Float32} # [nTE x batchsize] signals
    A::Array{Float32, 3} # [nTE x nT2 x batchsize] EPG basis matrices
    mask::Matrix{Float32} # [nTE x batchsize] binary mask
    δb::Matrix{Float32} # [1 x batchsize] signal quantization bin width
    nTE::Matrix{Int32} # [1 x batchsize] number of echoes
    α::Matrix{Float32} # [1 x batchsize] flip angle
    TE::Matrix{Float32} # [1 x batchsize] echo time
    T2Range::Matrix{Float32} # [2 x batchsize] T2 range
    T1::Matrix{Float32} # [1 x batchsize] T1 time
    idx::Matrix{NTuple{3, Int32}} # [1 x batchsize] voxel indices (`CartesianIndex` hardcodes Int64 indices)
    θ::Matrix{Float32} # [10 x batchsize] array of derived scalar features
end
Base.show(io::IO, batch::Batch) = print(io, "Batch$(map(summary, NamedTuple(batch)))")

Base.NamedTuple(batch::Batch) = lib.Functors.children(batch)
Base.Tuple(batch::Batch) = Tuple(NamedTuple(batch))
Base.pairs(batch::Batch) = pairs(NamedTuple(batch))
Base.iterate(batch::Batch, state...) = iterate(NamedTuple(batch), state...)
Base.length(batch::Batch) = length(NamedTuple(batch))
Base.copy(batch::Batch) = Batch(map(copy, Tuple(batch))...)
Base.isequal(batch1::Batch, batch2::Batch) = all(map(isequal, Tuple(batch1), Tuple(batch2)))
Base.isapprox(batch1::Batch, batch2::Batch; kwargs...) = all(map((x, y) -> isapprox(x, y; kwargs...), Tuple(batch1), Tuple(batch2)))
Base.similar(batch::Batch, n::Int = size(batch.b)[end]) = Batch(map(x -> similar(x, size(x)[1:(end-1)]..., n), Tuple(batch))...)

#### Batch assembly and image statistics

function load_raw_batch(df::DictTable, I::Union{Colon, <:AbstractVector})
    b₀ = load_array_from_cols(Int16, df, :Signal, I)
    δb = permutedims(df[:SignalBinWidth][I])
    normalization = permutedims(df[:GDN][I])
    b = b₀ .* δb ./ normalization
    δb ./= normalization
    batchsize = size(b, 2)
    nTE = permutedims(df[:nTE][I])
    α = permutedims(df[:FlipAngle][I])
    TE = permutedims(df[:TE][I])
    T2Range = copy(reinterpret(reshape, Float32, df[:T2Range][I]))
    idx = NTuple{3, Int32}.(reshape(df[:Index][I], 1, batchsize))
    return RawBatch(b, δb, nTE, α, TE, T2Range, idx)
end

# Scatter each source column directly into the mixed pool; no concatenation-sized intermediate.
function copy_mixed_columns!(dest::AbstractMatrix, src::AbstractMatrix, destinations)
    n = size(src, 1)
    for (j, k) in enumerate(destinations)
        copyto!(dest, (k-1)*n+1, src, (j-1)*n+1, n)
    end
    return dest
end

function mix_raw_batches(raws::AbstractVector{RawBatch}, rng)
    @assert !isempty(raws)
    length(raws) == 1 && return only(raws)
    n = sum(length, raws)
    destinations = invperm(lib.randperm(rng, n))
    pooled = map(x -> similar(x, size(x, 1), n), Tuple(first(raws)))
    offset = 0
    for raw in raws
        cols = @view destinations[(offset+1):(offset+length(raw))]
        foreach((dest, src) -> copy_mixed_columns!(dest, src, cols), pooled, Tuple(raw))
        offset += length(raw)
    end
    return RawBatch(pooled...)
end

# Augmentation uses a per-batch RNG and consumes `raw`.
function finalize_batch(raw::RawBatch, postprocess_batch!; nT2::Int, seed::Union{Nothing, UInt64} = nothing)
    rng = seed === nothing ? lib.Random.default_rng() : lib.Xoshiro(seed)
    (; b, δb, nTE, α, TE, T2Range, idx) = raw
    nTEmax, batchsize = size(b, 1), size(b, 2)

    # EPG postprocessing overwrites every entry; the unprocessed path returns initialized storage
    A = postprocess_batch! === nothing ? zeros(Float32, nTEmax, nT2, batchsize) : Array{Float32}(undef, nTEmax, nT2, batchsize)
    mask = Float32.((1:nTEmax) .<= nTE)
    T1 = ones(Float32, 1, batchsize) # default T1 = 1.0s
    batch = (; b, A, mask, δb, nTE, α, TE, T2Range, T1, idx)

    if postprocess_batch! !== nothing
        postprocess_batch!(batch, rng)
    end

    batch = merge(batch, (; θ = scalar_features(batch)))

    return Batch(batch...)
end

function Batch(df::DictTable, I::Union{Colon, <:AbstractVector}, postprocess_batch!; nT2::Int)
    return finalize_batch(load_raw_batch(df, I), postprocess_batch!; nT2)
end

function scalar_features(batch::NamedTuple)
    (; b, A, mask, δb, nTE, α, TE, T2Range, T1, idx) = batch
    θ = vcat(
        δb, log.(δb),
        cosd.(α), sind.(α), # convert from degrees to unit vector components; more natural for neural networks
        T2Range ./ TE, log.(T2Range ./ TE), # only ratio T2 / TE is physically meaningful
        T1 ./ TE, log.(T1 ./ TE), # only ratio T1 / TE is physically meaningful
    )
    return θ
end
num_scalar_features() = 10 # number of scalar features, i.e. size(θ, 1) where θ = scalar_features(batch)

# log(log(T2max / T2min)); divided by nT2 - 1 it is the log-T2 grid spacing
log_relative_t2_range(θ) = @views log.(θ[8:8, ..] .- θ[7:7, ..])

#### Batch augmentation and EPG/DECAES

@with_kw struct RandomRequantizer
    scale::Float32 = 0.10
end
RandomRequantizer(cfg::Config) = RandomRequantizer(cfg.train.augment.requantize::Float64)

function (f::RandomRequantizer)(batch::NamedTuple, rng = lib.Random.default_rng())
    (; b, mask, δb) = batch
    f.scale == 0 && return batch
    u = rand(rng, eltype(δb), size(δb))
    @. b = lib.apply_mask(mask, b + δb * u) # dequantize signal by adding uniform random noise ϵ ∼ 𝒰(0, δb) to each signal
    random_log_normal_scaling!(δb, f.scale, rng) # rescale the bin width by a log-normal factor with mean 1 and std ≈ scale
    @. b = lib.apply_mask(mask, δb * floor(b / δb)) # requantize signal to new bin width δb
    return batch
end

function random_log_normal_scaling_similar(x::AbstractArray{T, N}, scale::Real, rng) where {T, N}
    # Scale each signal by a log-normal randomly sampled value with mean(z) = 1, std(z) = scale.
    # In particular, we let logz ~ 𝒩(μ, σ²) where σ² = log(1 + scale²), μ = -σ²/2.
    σ² = log1p(T(scale^2)) # log-normal parameters with mean 1 and variance scale²: σ² = log(1 + scale²), μ = -σ²/2
    μ, σ = -σ² / 2, √σ² # mean correction to log-normal distribution, E[z] = 1 = exp(μ + σ²/2) <=> μ = -σ²/2
    u = randn(rng, T, ntuple(i -> i == N ? size(x, N) : 1, N)) # random noise along batch dimension
    z = @. exp(μ + σ * u) # random scaling logz ~ 𝒩(μ, σ²)
    return z
end

# Apply a random scaling to the signals `b` and to the quantization width `δb`
random_log_normal_scaling!(x::AbstractArray, scale::Real, rng) = random_log_normal_scaling!(x, random_log_normal_scaling_similar(x, scale, rng))
random_log_normal_scaling!(x::AbstractArray, z::AbstractArray) = x .*= z

# Retain a uniform ordered subset, with P(nTE-nTE′ = d) ∝ exp(-λd).
struct SignalMasker
    nTEmin::Int
    nTEfrac::Float64
end
SignalMasker(cfg::Config) = SignalMasker(cfg.train.augment.nTEmin::Int, cfg.train.augment.nTEfrac::Float64)

function (f::SignalMasker)(batch::NamedTuple, rng = lib.Random.default_rng())
    (; b, A, mask, nTE) = batch
    nTEmax, nT2, batchsize = size(A)
    (; nTEmin, nTEfrac) = f
    @assert 1 / 2 <= nTEfrac <= 1 "Exponential masking requires nTEfrac ∈ [1/2, 1]; got $nTEfrac"
    nTEoffsets = 0:max(nTEmax-nTEmin, 0)
    λ = lib.truncgeom_rate.((1 - nTEfrac) .* nTEoffsets, nTEoffsets)
    uniforms = rand(rng, batchsize)
    for k in axes(b, 2)
        @inbounds begin
            nTEₖ = nTE[1, k]
            nTEoffset = max(nTEₖ - nTEmin, 0)
            nTE′ = nTEₖ - lib.truncgeom_quantile(λ[nTEoffset+1], nTEoffset, uniforms[k])
            selected = sort!(lib.Random.randperm(rng, Int(nTEₖ))[1:nTE′])
            for (i, source) in enumerate(selected)
                b[i, k] = b[source, k]
                for j in axes(A, 2)
                    A[i, j, k] = A[source, j, k]
                end
            end
            nTE[1, k] = nTE′
            @simd for i in 1:nTE′
                mask[i, k] = 1
            end
            @simd for i in (nTE′+1):nTEmax
                mask[i, k] = 0
                b[i, k] = 0
            end
            for i in (nTE′+1):nTEmax
                @simd for j in axes(A, 2)
                    A[i, j, k] = 0
                end
            end
        end
    end
    return batch
end

@with_kw struct EPGBasisParamJiggler
    Δα::Float32 = 1.0f0 # degrees
    ΔT2minbounds::NTuple{2, Float32} = (-1.0f-3, 3.0f-3) # seconds
    T2maxbounds::NTuple{2, Float32} = (1.0f0, 2.0f0) # seconds
    T1bounds::NTuple{2, Float32} = (1.0f0, 2.0f0) # seconds
end
EPGBasisParamJiggler(cfg::Config) = EPGBasisParamJiggler(;
    Δα = cfg.train.augment.flipdelta::Float64,
    ΔT2minbounds = (cfg.train.augment.t2mindelta.min::Float64, cfg.train.augment.t2mindelta.max::Float64),
    T2maxbounds = (cfg.train.augment.t2maxbounds.min::Float64, cfg.train.augment.t2maxbounds.max::Float64),
    T1bounds = (cfg.train.augment.t1bounds.min::Float64, cfg.train.augment.t1bounds.max::Float64),
)

function (f::EPGBasisParamJiggler)(batch::NamedTuple, rng = lib.Random.default_rng())
    (; Δα, ΔT2minbounds, T2maxbounds, T1bounds) = f
    (; α, TE, T2Range, T1) = batch
    draws = rand(rng, Float32, 4, size(α, 2))
    normals = randn(rng, Float32, size(α, 2))
    Threads.@threads for k in axes(α, 2)
        @inbounds begin
            angle = α[1, k] + Δα * normals[k]
            α[1, k] = draws[1, k] < 0.5f0 ? 360.0f0 - angle : angle
            T2Range[1, k] = TE[1, k] + ΔT2minbounds[1] + (ΔT2minbounds[2] - ΔT2minbounds[1]) * draws[2, k]
            T2Range[2, k] = T2maxbounds[1] + (T2maxbounds[2] - T2maxbounds[1]) * draws[3, k]
            T1[1, k] = T1bounds[1] + (T1bounds[2] - T1bounds[1]) * draws[4, k]
        end
    end
    return batch
end

struct EPGBasisComputer end

function (f::EPGBasisComputer)(batch::NamedTuple)
    (; b, A, nTE, α, TE, T2Range, T1) = batch
    lib.DynamicEPG.epg_basis!(b, A, nTE, α, TE, T2Range, T1)
    return batch
end

struct BatchVerifier end

function (f::BatchVerifier)(batch::NamedTuple)
    (; b, A, mask, δb, nTE) = batch
    @assert size(b, 2) == size(A, 3) == size(δb, 2) "b, A, δb must have the same batch dimension"
    @assert size(b, 1) == size(A, 1) "b, A must have the same number of rows"
    @assert all_masked_fast(==(1), mask, nTE) "mask must be binary and match nTE"
    @assert all_masked_fast(>=(0), b, nTE) "b must be nonnegative and masked"
    @assert all_masked_fast(>=(0), A, nTE) "A must be nonnegative and masked"
    @assert all_fast(>(0), δb) "δb must be positive"
    @assert is_approx_discretized(b, δb) "b is not a multiple of δb"
    return batch
end

function is_approx_discretized(b::AbstractMatrix, δb::AbstractMatrix)
    @assert size(b, 2) == size(δb, 2) "b, δb must have the same batch dimension"
    @assert size(δb, 1) == 1 "δb must be a single row matrix"
    pass = true
    @inbounds for j in axes(b, 2)
        δbⱼ = δb[1, j]
        for i in axes(b, 1)
            pass &= is_approx_multiple_of(b[i, j], δbⱼ)
        end
        !pass && break
    end
    return pass
end
@inline is_approx_multiple_of(x::Real, Δ::Real) = x ≈ Δ * round(x / Δ)

function all_fast(f, x::AbstractArray)
    pass = true
    @inbounds for i in eachindex(x)
        pass &= f(x[i])
    end
    return pass
end

function all_masked_fast(f, x::AbstractMatrix, nTE::AbstractMatrix)
    @assert size(x, 2) == size(nTE, 2)
    pass = true
    @inbounds for k in axes(x, 2)
        nTEₖ = nTE[1, k]
        for i in 1:nTEₖ
            pass &= f(x[i, k])
        end
        for i in (nTEₖ+1):size(x, 1)
            pass &= x[i, k] == 0
        end
    end
    return pass
end

function all_masked_fast(f, x::AbstractTensor3D, nTE::AbstractMatrix)
    batchsize = size(x, 3)
    @assert size(nTE, 2) == batchsize
    pass = fill(true, batchsize)
    @sync for batch in lib.split_indices(; length = batchsize, minchunksize = 128, maxpartitions = Threads.nthreads())
        Threads.@spawn @inbounds for k in batch
            nTEₖ = nTE[1, k]
            for j in axes(x, 2)
                for i in 1:nTEₖ
                    pass[k] &= f(x[i, j, k])
                end
                for i in (nTEₖ+1):size(x, 1)
                    pass[k] &= x[i, j, k] == 0
                end
            end
        end
    end
    return all(pass)
end

#### CPU and GPU streaming loaders

struct LoaderState{I}
    indices::I
    seeds::lib.Xoshiro
    pool::Union{Nothing, RawBatch}
    offset::Int
end
LoaderState(opts) = LoaderState(batch_index_stream(opts), lib.Xoshiro(opts.seed), nothing, 0)

struct DataLoader{O, D, P, I, B}
    opts::O
    df::D
    postprocess_batch!::P
    state::I
    cache::B
end

function data_loader(cfg::Config, df::DictTable; seed::Int, batchsize::Int, augment::Bool = false, sampler::Symbol = :blockmix, single_batch::Bool = false)
    nrow = lib.Tables.rowcount(df)
    @assert 0 < batchsize <= nrow "Loader requires 0 < batchsize <= nrow"
    @assert sampler ∈ (:uniform, :contiguous, :blockmix)
    opts = (; nrow, batchsize, seed, augment, sampler, single_batch, nT2 = cfg.data.nT2::Int)
    return DataLoader(opts, df, batch_postprocesser(cfg; augment), Ref(LoaderState(opts)), Ref{Union{Nothing, Batch}}(nothing))
end

function batch_index_stream(opts)
    (; nrow, seed, sampler) = opts
    rng = lib.Xoshiro(seed)
    return sampler === :contiguous ? lib.ContiguousIndices(rng, nrow) :
           sampler === :uniform ? lib.UniformIndices(rng, nrow) :
           sampler === :blockmix ? lib.BlockMixIndices(rng, nrow, opts.batchsize, 8) :
           error("Invalid sampler: $(sampler)")
end

next_raw_pool!(s, df, batchsize::Int) = load_raw_batch(df, lib.next_indices!(s, batchsize))

function next_raw_pool!(s::lib.BlockMixIndices, df, batchsize::Int)
    blocks = lib.next_blocks!(s)
    raws = map(I -> load_raw_batch(df, I), blocks)
    pool = mix_raw_batches(raws, s.rng)
    return pool
end

# Preserve the raw pool until all columns have been consumed.
function next_loader_raw(state::LoaderState, opts, df)
    (; batchsize, single_batch) = opts
    (; indices, seeds, pool, offset) = state
    seeds = copy(seeds)
    seed = rand(seeds, UInt64)
    if single_batch && pool !== nothing
        return copy(pool), seed, LoaderState(indices, seeds, pool, 0)
    end

    if pool === nothing
        indices = copy(indices)
        pool = next_raw_pool!(indices, df, batchsize)
        offset = 0
    end
    while length(pool) - offset < batchsize
        tail = raw_columns(pool, (offset+1):length(pool))
        indices = copy(indices)
        next = next_raw_pool!(indices, df, batchsize)
        pool = RawBatch(map(hcat, Tuple(tail), Tuple(next))...)
        offset = 0
    end

    if single_batch
        if offset != 0 || length(pool) != batchsize
            pool = raw_columns(pool, (offset+1):(offset+batchsize))
        end
        return copy(pool), seed, LoaderState(indices, seeds, pool, 0)
    elseif offset == 0 && length(pool) == batchsize
        return pool, seed, LoaderState(indices, seeds, nothing, 0)
    else
        raw = raw_columns(pool, (offset+1):(offset+batchsize))
        offset += batchsize
        next_pool = offset == length(pool) ? nothing : pool
        return raw, seed, LoaderState(indices, seeds, next_pool, next_pool === nothing ? 0 : offset)
    end
end

function batch_postprocesser(cfg::Config; augment::Bool = false)
    quantizer, masker, jiggler = augment ? (RandomRequantizer(cfg), SignalMasker(cfg), EPGBasisParamJiggler(cfg)) : (nothing, nothing, nothing)
    function postprocess!(batch, rng)
        if augment
            quantizer(batch, rng)
            jiggler(batch, rng)
        end
        EPGBasisComputer()(batch)
        augment && masker(batch, rng)
        return BatchVerifier()(batch)
    end
    return postprocess!
end

function batch!(dl::DataLoader)
    cached = dl.cache[]
    cached === nothing || return NamedTuple(copy(cached))
    raw, seed, state = next_loader_raw(dl.state[], dl.opts, dl.df)
    batch = finalize_batch(raw, dl.postprocess_batch!; nT2 = dl.opts.nT2, seed)
    dl.state[] = state
    dl.opts.single_batch && !dl.opts.augment && (dl.cache[] = copy(batch))
    return NamedTuple(batch)
end
Base.iterate(dl::DataLoader, state = 0) = batch!(dl), state + 1
Base.IteratorSize(::Type{<:DataLoader}) = Base.IsInfinite()

# Initialization starts from the same seed without advancing training data.
function with_initialization_data(f, dl::DataLoader)
    initial = DataLoader(dl.opts, dl.df, dl.postprocess_batch!, Ref(LoaderState(dl.opts)), Ref{Union{Nothing, Batch}}(nothing))
    try
        return f(initial)
    finally
        empty!(initial)
    end
end

device_batches(dl) = lib.USE_GPU[] ? lib.CuIterator(dl) : dl
lib.with_gpu(f, dl::DataLoader) = f(device_batches(dl))

function Base.empty!(dl::DataLoader)
    dl.cache[] = nothing
    dl.state[] = LoaderState(dl.opts)
    return dl
end

#### Dataset integrity checks

function evaluation_partition(data, data_cfg; select = img -> lowercase(img.subset::String) == "test_id")
    ids = Set(img.id::Int for (_, img) in data_cfg.test.images if select(img))
    @assert !isempty(ids) "No selected test images in dataset metadata"
    selected = findall(id -> id ∈ ids, data.test[:ImageID])
    @assert !isempty(selected) "No rows for the selected test images"
    return OrderedDict{Symbol, AbstractVector}(k => view(v, selected) for (k, v) in data.test)
end
