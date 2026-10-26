using Library
using Library: (..)
using LinearAlgebra

using LaTeXStrings
using PrettyTables
using Printf
import AdvancedHMC as AHMC
import MCMCDiagnosticTools as MCDT
import AdvancedHMC.LogDensityProblems as LDP
using NLopt: NLopt
using Optim: Optim
using SHA: SHA

const lib = Library
const ex = Experiments

#### Output and reproducibility

const FIGURE_DIR = Ref(lib.projectdir("docs", "figures"))

function figuresdir(args...)
    path = joinpath(FIGURE_DIR[], args...)
    if endswith(path, ".jld2")
        path = lib.cachepath(path)
        mkpath(dirname(path))
        return path
    end
    return abspath(path)
end

function savefig(fig, filename; kwargs...)
    filename = abspath(filename)
    mkpath(dirname(filename))
    save(filename, fig; pt_per_unit = 1, kwargs...)
    @info "Saved: $filename"
    return fig
end

macro device_elapsed(expr)
    return quote
        if lib.USE_GPU[]
            CUDA.@elapsed $(esc(expr))
        else
            @elapsed $(esc(expr))
        end
    end
end

#### Interpolation geometry

function lerp_square(rng, reference)
    d, T = size(reference, 1), eltype(reference)
    factor = qr(randn(rng, T, d, 2))
    basis = Matrix(factor.Q[:, 1:2]) .* sign.(diag(factor.R))'
    return basis * T[-1 -1 1; 1 -1 -1] .* sqrt(T(d / 2)) # A, B, C; circumradius √d
end

function lerp_endpoints(rng, reference, n; method = :normal)
    d, T = size(reference, 1), eltype(reference)
    a = randn(rng, T, d, n)
    if method === :normal
        b = randn(rng, T, d, n)
    elseif method === :antipodal
        a .*= sqrt(T(d)) ./ sqrt.(sum(abs2, a; dims = 1))
        b = -a
    else
        throw(ArgumentError("Unknown LERP endpoint method: $method"))
    end
    return (; a, b)
end

function lerp_profile(energies, ts; detrend = false, reference = 0)
    values = if detrend
        i0, i1 = findfirst(==(0), ts), findfirst(==(1), ts)
        t = reshape(ts, :, 1)
        energies .- ((1 .- t) .* energies[i0:i0, :] .+ t .* energies[i1:i1, :])
    else
        energies .- reference
    end
    bounds = [quantile(row, [0.025, 0.975]) for row in eachrow(values)]
    return (; values, mean = vec(mean(values; dims = 2)), lower = first.(bounds), upper = last.(bounds))
end

# Normalize both orthogonal coordinates by one scale, preserving lengths up to that scale.
function lerp_plane(points; center = :bounds, padding = 0, orientation = :edges)
    origin = points[:, 2]
    edges = hcat(points[:, 3] - origin, points[:, 1] - origin)
    factor = qr(edges)
    basis = Matrix(factor.Q[:, 1:2]) .* sign.(diag(factor.R))'

    if orientation === :play
        triples = ((1, 2, 3), (1, 3, 2), (2, 3, 1))
        i, j, k = triples[argmax(map(t -> sum(abs2, points[:, t[2]] - points[:, t[1]]), triples))]
        y = normalize(basis' * (points[:, j] - points[:, i]))
        x = [y[2], -y[1]]
        rotation = hcat(x, y)
        dot(x, basis' * (points[:, k] - points[:, i])) < 0 && (rotation .*= -1)
        basis = basis * rotation
    elseif orientation !== :edges
        throw(ArgumentError("Unknown LERP plane orientation: $orientation"))
    end
    coordinates = basis' * (points .- origin)

    if center === :centroid
        midpoint = mean(coordinates; dims = 2)
        scale = 2maximum(abs, coordinates .- midpoint) / (1 - 2padding)
    elseif center === :bounds
        lo, hi = minimum(coordinates; dims = 2), maximum(coordinates; dims = 2)
        midpoint = (lo + hi) ./ 2
        scale = maximum(hi - lo) / (1 - 2padding)
    else
        throw(ArgumentError("Unknown LERP plane centre: $center"))
    end
    offset = midpoint .- scale / 2

    return (; origin = origin + basis * vec(offset), basis = basis .* scale, points = (coordinates .- offset) ./ scale)
end

function lerp_grid(plane, xs, ys)
    coordinates = permutedims(hcat(repeat(collect(xs), length(ys)), repeat(collect(ys); inner = length(xs))))
    return plane.origin .+ plane.basis * coordinates
end

#### PrettyTables

function default_pretty_table(io, args...; kwargs...)
    pretty_table(
        io, args...;
        merge_column_label_cells = :auto,
        alignment = :c,
        style = LatexTableStyle(; column_label = ["textbf"]),
        backend = :latex,
        table_format = latex_table_format__booktabs,
        kwargs...,
    )
end

function save_table(table_builder, filename)
    filename = abspath(filename)
    mkpath(dirname(filename))
    table = table_builder(String)
    write(filename, table)
    @info "Saved: $filename"
    return table
end

#### Makie

function default_theme(; kwargs...)
    theme = merge(
        theme_latexfonts(),
        Theme(;
            font = "CMU Serif",
            fontsize = 16,
            size = (800, 600),
            linestyles = [nothing, :dash, :dash],
            ishollowmarkers = [false, true, false],
            markers = [:circle, :diamond, :rtriangle],
            linecycle = Cycle([:color, :linestyle]; covary = true),
            scattercycle = Cycle([:color => :markercolor, :strokecolor => :color, :marker]; covary = true),
            markerstrokewidth = 1.5,
            colormap = :viridis,
            Axis = (
                titlesize = 16,
                xticklabelsize = 16,
                yticklabelsize = 16,
            ),
            Axis3 = (
                titlesize = 18,
                xlabelsize = 18,
                ylabelsize = 18,
                zlabelsize = 18,
                xticklabelsize = 16,
                yticklabelsize = 16,
                zticklabelsize = 16,
            ),
            Colorbar = (labelsize = 16, ticklabelsize = 14),
            Legend = (labelsize = 16, titlesize = 16),
            kwargs...,
        ),
    )
    return theme
end

#### Posterior observables and numerical plotting helpers

function seed_all!(seed)
    Random.seed!(seed)
    lib.USE_GPU[] && CUDA.functional() && CUDA.seed!(seed)
    return nothing
end

relerr(x̂, x) = @. abs(x̂ - x) / abs(x)

cummean(x) = (cumsum(x) ./= (1:length(x)))

# Target EPG basis at the same physical acquisition parameters as the native batch.
function epg_basis_at(batch, n_target)
    nTEmax, _, batchsize = size(batch.A)
    A = zeros(Float64, nTEmax, n_target, batchsize)
    lib.DynamicEPG.epg_basis!(lib.cpu64(batch.b), A, lib.cpu(Int32.(batch.nTE)), lib.cpu64(batch.α), lib.cpu64(batch.TE), lib.cpu64(batch.T2Range), lib.cpu64(batch.T1))
    return lib.arr_similar(batch.A, A)
end

# Scatter a masked voxel vector v into a full spatial image, filling background voxels with NaN.
function to_slice(op, v)
    img = fill(NaN, op.shape...)
    img[CartesianIndices(op.shape)[cpu(op.linear_idx)]] = v
    return img
end

# Invert A₁(κ) = I₁(κ)/I₀(κ) via Newton; initial κ from the Mardia & Jupp guess.
function inv_besseli1i0(R::Real)
    R ≤ 0 && return zero(R)
    R ≥ 1 && return oftype(float(R), Inf)
    κ = inv_besseli1i0_approx(R)
    for _ in 1:4
        r, rx, rm1 = FastRicianLikelihoods._besseli1i0_parts(κ)
        κ = max(κ + (r - R) / (rm1 * (rm1 + 2) + rx), zero(R))
    end
    return κ
end
inv_besseli1i0_approx(R::Real) = R < 0.53 ? R * evalpoly(R^2, (2.0, 1.0, 5 / 6)) : R < 0.85 ? -0.4 + R * 1.39 + 0.43 / (1 - R) : inv(R * evalpoly(R, (3.0, -4.0, 1.0)))

# Forward difference with a shared seed couples stochastic gradient evaluations.
function withgradient_and_hvp(f, xs::AbstractArray...; step_size = sqrt(eps(Float32)), seed = nothing)
    function withgradients(xs′)
        seed === nothing || seed_all!(seed)
        local ℓ, 𝒥 = lib.Zygote.pullback(xs′...) do x...
            f(x...)
        end
        return (ℓ, 𝒥(pullback_seed(ℓ))...)
    end
    vs = map(randn_similar, xs)
    δ(x) = eltype(x)(step_size)
    ℓ, ∇xs... = withgradients(xs)
    xs⁺ = map((x, v) -> x .+ δ(x) .* v, xs, vs)
    _, ∇xs⁺... = withgradients(xs⁺)
    hs = map((g, g⁺, x) -> g⁺ .= (g⁺ .- g) ./ δ(x), ∇xs, ∇xs⁺, xs)
    return ℓ, ∇xs, hs, vs
end
pullback_seed(ℓ::Number) = one(ℓ)
pullback_seed(ℓ::AbstractArray) = ones_similar(ℓ)

function optimize_with_optim(f_∇f, x0)
    function fg!(F, G, x)
        f, df = f_∇f(x)
        if G !== nothing
            G .= df
        end
        if F !== nothing
            return only(f)
        end
        return nothing
    end
    obj = Optim.NLSolversBase.only_fg!(fg!)
    alg = Optim.BFGS()
    return Optim.optimize(obj, x0, alg, Optim.Options(; x_abstol = 1e-6, g_abstol = 1e-3))
end

function find_bounding_box_widths(f, x0::NTuple{D}; f_max, init_widths) where {D}
    objective(logw, grad) = -sum(logw)

    function constraint_builder(rel_coord::Vector{Float64})
        function constraint(logw::Vector{Float64}, grad::Vector{Float64})
            @assert length(grad) == 0 "Gradient not supported for constraints"
            x_j = x0 .+ rel_coord .* exp.(logw)
            return f(x_j) - f_max # g_j(w) = f(x_j) - f_max <= 0
        end
    end

    opt = NLopt.Opt(:LN_COBYLA, D)
    NLopt.min_objective!(opt, objective)
    NLopt.xtol_rel!(opt, 1e-3)
    NLopt.ftol_abs!(opt, 1e-3)
    NLopt.maxeval!(opt, 100)

    for rel_coord in Iterators.product(ntuple(_ -> LinRange(-0.5, 0.5, 2), D)...)
        constraint_func_j = constraint_builder(collect(rel_coord))
        NLopt.inequality_constraint!(opt, constraint_func_j, 1e-3)
    end

    _, logw, _ = NLopt.optimize(opt, [log.(init_widths)...])
    return exp.(logw)
end

function refine_mode_and_hessian(f_∇f, ∇²f, x0)
    optim_res = optimize_with_optim(f_∇f, x0)
    x = Optim.minimizer(optim_res)
    refined_loss = first(f_∇f(x))

    hessian = ∇²f(x)
    λ, Q = eigen(hessian)
    ivecs = (argmax(abs.(λ)), argmin(abs.(λ)))
    evecs = (Q[:, ivecs[1]], Q[:, ivecs[2]])
    λvals = (λ[ivecs[1]], λ[ivecs[2]])

    return (; x, refined_loss, hessian, λ, evecs, λvals, ivecs)
end

# Fraction of [logT2ᵢ − Δ/2, logT2ᵢ + Δ/2] below logWin; a boundary at the bin centre gives ½.
function soft_gate(logT2s::AbstractVector{T}, logWin::Real) where {T <: Real}
    Δ = @allowscalar logT2s[2] - logT2s[1]
    return clamp.((logWin .- logT2s) ./ Δ .+ T(0.5), zero(T), one(T))
end

function setup_t2_grid(T2Range, nT2)
    T2min, T2max = @allowscalar (T2Range[1, 1], T2Range[2, 1])
    logT2s = range(log(T2min), log(T2max); length = nT2)
    T2s = exp.(logT2s)
    return (; T2s, logT2s)
end

# Pool fractions are gᵀx̄; T₂ values are in seconds. Gates partition the short/middle/long pools.
function compute_observable_maps(xdist, logT2s; SPWin = 32.0f-3, MPWin = 200.0f-3)
    spgate = lib.arr_similar(xdist, soft_gate(logT2s, log(SPWin)))
    mpgate = lib.arr_similar(xdist, soft_gate(logT2s, log(MPWin)))
    midgate, lpgate = mpgate .- spgate, 1 .- mpgate
    Σxdist = sum(xdist; dims = 1)
    Σxdist .= ifelse.(Σxdist .== 0, one(eltype(Σxdist)), Σxdist)
    f_short = dropdims(sum(spgate .* xdist ./ Σxdist; dims = 1); dims = 1)
    f_mid = dropdims(sum(midgate .* xdist ./ Σxdist; dims = 1); dims = 1)
    f_long = dropdims(sum(lpgate .* xdist ./ Σxdist; dims = 1); dims = 1)
    T2_short = dropdims(exp.(sum(lib.arr_similar(xdist, logT2s) .* spgate .* xdist; dims = 1) ./ sum(spgate .* xdist; dims = 1)); dims = 1)
    T2_mid = dropdims(exp.(sum(lib.arr_similar(xdist, logT2s) .* midgate .* xdist; dims = 1) ./ sum(midgate .* xdist; dims = 1)); dims = 1)
    gmT2 = dropdims(exp.(sum(lib.arr_similar(xdist, logT2s) .* xdist ./ Σxdist; dims = 1)); dims = 1)
    return (; mwf = f_short, iewf = f_mid, cwf = f_long, mwT2 = T2_short, iewT2 = T2_mid, gmT2)
end


# Samples are (draws, chains, parameters...); diagnostics are computed before pooling or thinning.
function reference_chain_diagnostics(samples)
    @assert size(samples, 1) >= 6 && all(isfinite, samples)
    kwargs = (; autocov_method = MCDT.FFTAutocovMethod(), maxlag = size(samples, 1) - 1)
    d = MCDT.ess_rhat(samples; kwargs...)
    return (; rhat = d.rhat, ess_bulk = d.ess, ess_tail = MCDT.ess(samples; kind = :tail, kwargs...), mcse_mean = MCDT.mcse(samples; kwargs...))
end

function reference_posterior_diagnostics(u, xdist, logT2s)
    @assert size(u)[2:3] == size(xdist)[2:3]
    biomarkers = compute_observable_maps(xdist, logT2s)
    return (;
        coordinates = reference_chain_diagnostics(permutedims(u, (2, 3, 1))),
        spectra = reference_chain_diagnostics(permutedims(xdist, (2, 3, 1))),
        biomarkers = map(reference_chain_diagnostics, biomarkers),
    )
end

# Per-site MCMC autocorrelation time τ for each T2-distribution functional from an xdist chain.
# Input: xdist (nsamps, nT2, nvox), logT2s, SPWin, MPWin.
# Same fields as compute_observable_maps; each value is a (1, nvox) matrix of τ estimates
function chain_observable_autocorr_times(xdist::AbstractArray{<:Real, 3}, logT2s; SPWin, MPWin)
    nsamps, _, nvox = size(xdist)
    observables = compute_observable_maps(permutedims(xdist, (2, 1, 3)), logT2s; SPWin, MPWin)
    return map(observables) do obs
        return dropdims(lib.autocorr_time(lib.autocorr_fft(reshape(obs, nsamps, 1, nvox); dims = 1); dims = 1); dims = 1)
    end
end

# Recompute each deterministic posterior chunk during reverse AD to bound activation memory.
concatenate_batches(xs::AbstractArray...) = cat(xs...; dims = ndims(first(xs)))
concatenate_batches(xs::NamedTuple...) = map(concatenate_batches, xs...)

function posterior_chunks(𝒢, 𝒫, u, z, b, A, mask, δb, θ; chunksize = INFERENCE_CHUNKSIZE)
    function evaluate(_u, _z, _b, _A, _mask, _δb, _θ)
        out = ex.ForwardPhysicsTransformedPosterior(𝒢, 𝒫, _z, _b, _A, _mask, _δb, _θ)(_u)
        return (; out.summary, out.negelbo, out.neglogpostu)
    end
    return checkpointed_batches(evaluate, u, z, b, A, mask, δb, θ; chunksize)
end

function checkpointed_batches(f, args...; chunksize)
    n = size(first(args), ndims(first(args)))
    outputs = map(1:chunksize:n) do start
        inds = start:min(start+chunksize-1, n)
        chunk = map(x -> x[.., inds], args)
        return lib.Zygote.checkpointed(f, chunk...)
    end
    return concatenate_batches(outputs...)
end
