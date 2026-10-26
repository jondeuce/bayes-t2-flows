#### Setup

using CairoMakie
include(joinpath(@__DIR__, "plots_utils.jl"))

CairoMakie.activate!()
set_theme!(default_theme())

function load_run(;
    dir,
    seed = 0, slice = (:, :, 1), cohort_size = 128,
    image_id = nothing,
)
    global cfg, data_cfg, models
    (; cfg, data_cfg, models) = ex.load_checkpoint(dir)
    models = merge(models, (; genatr = lib.fix_affine_matrices(models.genatr)))
    cfg.device.seed = seed
    seed_all!(seed)
    path = lib.clean_realpath(cfg.data.path::String)
    data = ex.dataset(path)
    partition = :test
    rows = ex.evaluation_partition(data, data_cfg)
    image = ex.image_partition(data.test, image_id)
    val_sampler = ex.data_loader(cfg, rows; batchsize = cohort_size, seed, sampler = :uniform)
    val_batch, val_cohort = ex.image_slice(cfg, image, slice), first(val_sampler)

    global run_env = (; dir, seed, slice, partition, image_id = first(image[:ImageID]), cfg, data_cfg, models, val_batch, val_cohort)
    return run_env
end

#### Shared inference

const INFERENCE_CHUNKSIZE = 1024

function run_decaes()
    batch = run_env.val_batch
    return map((; lcurve = "lcurve", gcv = "gcv", chi2 = "chi2")) do Reg
        return ex.decaes_inference_variables(run_env.data_cfg, batch; Reg)
    end
end

function run_elbo_metrics(; seed = 0, samples = 1000)
    seed_all!(seed)
    batch = run_env.val_batch |> gpu
    (; b, A, mask, δb, θ) = batch
    ℰ, 𝒢, 𝒫 = run_env.models.enc, run_env.models.genatr, run_env.models.prior

    return @time ex.run_elbo_estimate(ℰ, 𝒢, 𝒫, b, A, mask, δb, θ; samples, encoder_chunksize = INFERENCE_CHUNKSIZE, generator_chunksize = INFERENCE_CHUNKSIZE, verbose = true)
end

function run_mcmc_sampling(; seed = 0, iter = 1000)
    seed_all!(seed)
    batch = run_env.val_batch |> gpu
    (; b, A, mask, δb, θ) = batch
    ℰ, 𝒢, 𝒫 = run_env.models.enc, run_env.models.genatr, run_env.models.prior
    samp_prob = ex.SamplingProblem(ex.MALA(), ℰ, 𝒢, 𝒫)
    mala_results = @time ex.run_sampling(samp_prob, b, A, mask, δb, θ; iter, warmup_iter = 0, save_all_samples = true, encoder_chunksize = INFERENCE_CHUNKSIZE, generator_chunksize = INFERENCE_CHUNKSIZE, verbose = true)
    return mala_results |> cpu
end

function run_map_inference(; seed = 0, iter = 50)
    seed_all!(seed)
    batch = run_env.val_batch |> gpu
    (; b, A, mask, δb, θ) = batch
    ℰ, 𝒢, 𝒫 = run_env.models.enc, run_env.models.genatr, run_env.models.prior

    inf_prob = ex.MAPInferenceProblem(ℰ, 𝒢, 𝒫)
    map_results = @time ex.run_inference(inf_prob, b, A, mask, δb, θ; verbose = true, lr = 0.1, iter, schedule = :linear, max_rank = 10, newton = false, schedule_free = true, whiten_grad = true, encoder_chunksize = INFERENCE_CHUNKSIZE, generator_chunksize = INFERENCE_CHUNKSIZE)
    map_results = merge(map_results, ex.inference_variables(map_results.summary.x) |> NamedTuple)

    return merge(map_results, (; θ)) |> cpu
end

# Evaluate the conditional posterior flow at its Gaussian base origin.
function sample_forward_flow(; seed = 0)
    seed_all!(seed)
    batch = run_env.val_batch |> gpu
    (; b, A, mask, δb, θ) = batch
    ℰ, 𝒢 = run_env.models.enc, run_env.models.genatr
    z = lib.chunked_apply(ℰ, b, A, mask, θ; chunksize = INFERENCE_CHUNKSIZE)
    u = zeros_similar(z, 𝒢.opts.udim, size(z, 2))
    result = lib.chunked_apply(u, z, b, A, mask, δb, θ; chunksize = INFERENCE_CHUNKSIZE) do _u, _z, _b, _A, _mask, _δb, _θ
        𝒢(_u, (_z, _b, _A, _mask, _δb, _θ); deterministic = true)
    end
    return result |> cpu
end

function run_map_inference_id(; seed = 0, iter = 100)
    seed_all!(seed)
    (; u, z, b, A, mask, δb, θ) = map_results |> gpu
    ℰ, 𝒢, 𝒫 = run_env.models.enc, run_env.models.genatr, run_env.models.prior

    # Identity backbone: keep only the terminal SimplexFlow layer
    𝒢₀ = ex.identity_backbone(𝒢)

    # Warm-start u₀ for 𝒢₀ by mapping the learned-flow MAP solution back through 𝒢₀
    x = lib.chunked_apply(u, z, b, A, mask, δb, θ; chunksize = INFERENCE_CHUNKSIZE) do _u, cond...
        ex.flow_forward(ex.recondition(𝒢.model, cond), _u).x
    end
    u₀ = lib.chunked_apply(x, z, b, A, mask, δb, θ; chunksize = INFERENCE_CHUNKSIZE) do _x, cond...
        ex.flow_backward(ex.recondition(𝒢₀.model, cond), _x).u
    end

    inf_prob = ex.MAPInferenceProblem(ℰ, 𝒢₀, 𝒫)
    map_results_id = @time ex.run_inference(inf_prob, b, A, mask, δb, θ, z, u₀; verbose = true, lr = 0.5, iter, schedule = :linear, max_rank = 40, newton = true, whiten_grad = false, encoder_chunksize = INFERENCE_CHUNKSIZE, generator_chunksize = INFERENCE_CHUNKSIZE)

    return merge(map_results_id, (; θ)) |> cpu
end

#### Posterior geometry

function compute_posterior_surface_results(; voxel = 1, npts = (251, 251))
    ℰ, 𝒢, 𝒫 = run_env.models.enc, run_env.models.genatr, run_env.models.prior
    𝒢₀ = ex.identity_backbone(𝒢)
    (; udim, xdim) = 𝒢.opts
    (; u, z, b, A, mask, δb, θ) = map_results
    u_id = map_results_id.u

    jplot = voxel
    @assert 1 <= jplot <= size(b, 2)

    @info "Computing posterior surface for column #$jplot"
    u_jplot = u[begin:(end-2), jplot:jplot]
    u_id_jplot = u_id[begin:(end-2), jplot:jplot]
    data_jplot = map(d -> d[.., jplot:jplot], (; z, b, A, mask, δb, θ)) |> gpu

    panels = []
    for (_u, _𝒢, _title) in zip(
        [u_id_jplot, u_jplot],
        [𝒢₀, 𝒢],
        ["Original Posterior", "Preconditioned Posterior"],
    )
        @info "Processing: $_title"
        inf_prob32 = ex.MAPInferenceProblem(ℰ, _𝒢, 𝒫)
        inf_prob64 = inf_prob32 |> gpu64
        local data32 = map(gpu32, data_jplot)
        local data64 = map(gpu64, data32)

        function f_∇f(x)
            (losses, du) = ex.∇inference_loss(inf_prob64, reshape(x, :, 1) |> gpu64, data64...) |> cpu64
            return (only(losses.neglogpostu), vec(du))
        end

        function ∇²f(x)
            hess = ex.∇²inference_loss(inf_prob64, reshape(x, :, 1) |> gpu64, data64...)
            return reshape(hess, udim - 2, udim - 2) |> cpu64
        end
        res = refine_mode_and_hessian(f_∇f, ∇²f, vec(cpu64(_u)))
        _u = reshape(res.x, :, 1) |> gpu32
        hess = reshape(res.hessian, udim - 2, udim - 2)

        local _u_plot_center = _u
        widths = find_bounding_box_widths((0.0, 0.0); init_widths = 3.0 ./ sqrt.(res.λvals), f_max = res.refined_loss + 5.0) do e⃗
            losses = ex.inference_losses(inf_prob64, _u_plot_center + gpu64(e⃗[1] .* res.evecs[1] .+ e⃗[2] .* res.evecs[2]), data64...) |> cpu64
            return only(losses.neglogpostu)
        end

        scales = widths ./ 2
        offset_ranges = [range(-1.0f0, 1.0f0; length = n) for n in npts]
        plot_offsets = [s .* r for (s, r) in zip(scales, offset_ranges)]
        Δu_grid = res.evecs[1] .* reshape(plot_offsets[1], 1, :, 1) .+ res.evecs[2] .* reshape(plot_offsets[2], 1, 1, :)

        # Center grid around the refined mode and compute losses
        u_grid_plot = vec(_u_plot_center) .+ reshape(Δu_grid |> gpu32, udim - 2, :)
        energy_on_grid = lib.chunked_apply(u_grid_plot; chunksize = INFERENCE_CHUNKSIZE) do _grid
            data_chunk = map(d -> repeat(d, ntuple(i -> i < ndims(d) ? 1 : size(_grid, 2), ndims(d))...), data_jplot)
            return ex.inference_losses(inf_prob32, _grid, data_chunk...).neglogpostu
        end
        neglogpostu_on_grid = reshape(energy_on_grid |> cpu, npts...)

        x_plot = repeat(plot_offsets[1], 1, npts[2])
        y_plot = repeat(plot_offsets[2]', npts[1], 1)
        z_plot = neglogpostu_on_grid .- res.refined_loss

        z_min, z_max = extrema(z_plot)
        z_surf_plot_thresh = min(z_min + 10.0, z_max)
        z_levels = range(z_min, z_surf_plot_thresh; length = 15)

        hess_mat = reshape(hess, udim - 2, udim - 2) |> cpu64
        push!(panels, (; title = _title, x_plot, y_plot, z_plot, z_levels, res.ivecs, res.λ, udim, z_min, hess_mat))
    end

    return (; panels, udim, npts, jplot)
end

function plot_posterior_geometry(; figpath = figuresdir("preconditioning_geometry.pdf"))
    (; panels, udim, npts) = posterior_surface_results
    zaspect = 1.4
    wratio = 1 / (zaspect - 1)

    fig = Figure(; size = (1200, 700), backgroundcolor = :white)

    for (iter, panel) in enumerate(panels)
        (; title, x_plot, y_plot, z_plot, z_levels, ivecs, λ, z_min, hess_mat) = panel
        joffset = 2 * (iter - 1)

        ax1 = Axis3(
            fig[1:3, 1+joffset]; # spans 3 rows, takes 1st or 3rd column
            title = LaTeXString(title),
            xlabel = L"\Delta_{%$(ivecs[1])}", ylabel = L"\Delta_{%$(ivecs[2])}", zlabel = "",
            aspect = (1, 1, zaspect),
            elevation = deg2rad(15.0), azimuth = deg2rad(-110.0),
            xticks = LinearTicks(3), yticks = LinearTicks(3),
        )

        # Level-set contour lines at natural z-heights (not projected)
        contour!(ax1, x_plot, y_plot, z_plot;
            levels = z_levels,
            linewidth = 2.0,
            colormap = :deep,
        )

        # Plot surface
        surface!(
            ax1, x_plot, y_plot, z_plot;
            shading = NoShading, colormap = :deep, # colormap = :viridis,
            alpha = 0.3,
            rasterize = 4, # force rasterization for large meshes to avoid large PDF size
        )

        # Plot contour lines on surface
        contour3d!(ax1, x_plot, y_plot, z_plot;
            levels = z_levels,
            linewidth = 0.5,
            color = :black,
        )
        zlims!(ax1, z_min - 0.01, nothing)

        # Axis 2: Slice along Delta_{i[1]}
        ax2 = Axis(
            fig[1, 2+joffset]; # Row 1, 2nd or 4th column
            title = L"$-\!\log p$ vs. $\Delta_{%$(ivecs[1])}$",
            xticks = LinearTicks(3), yticks = LinearTicks(3),
        )
        lines!(ax2, x_plot[:, round(Int, end / 2)], z_plot[:, round(Int, end / 2)]) # Plot middle slice

        # Axis 3: Slice along Delta_{i[2]}
        ax3 = Axis(
            fig[2, 2+joffset]; # Row 2, 2nd or 4th column
            title = L"$-\!\log p$ vs. $\Delta_{%$(ivecs[2])}$",
            xticks = LinearTicks(3), yticks = LinearTicks(3),
        )
        lines!(ax3, y_plot[round(Int, end / 2), :], z_plot[round(Int, end / 2), :]) # Plot middle slice

        # Axis 4: Eigenvalue Spectrum
        ax4 = Axis(
            fig[3, 2+joffset]; # Row 3, 2nd or 4th column
            title = L"$\log_{10}\lambda_i$ vs. Index $i$",
            xticks = LinearTicks(3), yticks = LinearTicks(3),
        )
        scatter!(ax4, 1:(udim-2), log10.(sort(abs.(λ); rev = true)); marker = :circle, markersize = 3)

        # Hessian heatmap below surface (log-scale to handle orders-of-magnitude range)
        gl = fig[4, (1+joffset):(2+joffset)] # span full half-width (cols 1-2 or 3-4)
        ax5 = Axis(
            gl[1, 1];
            title = L"Hessian $\nabla^2(-\!\log p)$ at mode",
            xticksvisible = false,
            yticksvisible = false,
            xticklabelsvisible = false,
            yticklabelsvisible = false,
        )
        ε = max(maximum(abs.(hess_mat)) * eps(), floatmin())
        hess_log = log10.(abs.(hess_mat) .+ ε)
        hm = heatmap!(ax5, hess_log; colormap = :plasma, colorrange = extrema(hess_log), label = L"$\log_{10}|\nabla^2|$")
        Colorbar(gl[1, 2], hm; width = 15)
        axislegend(ax5; position = :rb)
    end

    # 4 columns total. Ratios wratio : 1 : wratio : 1
    total_ratio = 2 * wratio + 2
    colsize!(fig.layout, 1, Relative(wratio / total_ratio))
    colsize!(fig.layout, 2, Relative(1.0 / total_ratio))
    colsize!(fig.layout, 3, Relative(wratio / total_ratio))
    colsize!(fig.layout, 4, Relative(1.0 / total_ratio))

    colgap!(fig.layout, 1, 0) # no gap between 3D surface and its 1D slice panels
    colgap!(fig.layout, 3, 0) # same for the second panel pair
    colgap!(fig.layout, 2, 0) # no gap between the two main panel blocks
    rowgap!(fig.layout, 1, 10)
    rowgap!(fig.layout, 2, 10)
    rowgap!(fig.layout, 3, 10) # gap between surface rows and Hessian heatmap row

    resize_to_layout!(fig)
    savefig(fig, figpath)
end

#### T2 distributions and component clusters

# Compute T2 distributions and observable maps (MWF, IEWF, gmT2) for all inference methods.
# SPWin / MPWin are nominal short-pool / medium-pool boundaries.
function compute_xdist_results(; SPWin = 32.0f-3, MPWin = 200.0f-3)
    (; T2Range, TE, idx, α) = run_env.val_batch
    nT2, T2min, T2max = run_env.cfg.data.nT2, T2Range[1, 1], T2Range[2, 1]
    (; logT2s, T2s) = setup_t2_grid(T2Range, nT2)
    @info "Pool windows (nominal): SP = $(round(1000 * SPWin; sigdigits = 3)) ms, MP = $(round(1000 * MPWin; sigdigits = 3)) ms"

    xdists = (;
        PseudoMAP = pseudomap_xdist.x.xdist,
        MAP = map_results.xdist,
        Mean = elbo_results["XDist"],
        IW = elbo_results["XDist_IS"],
        MH = elbo_results["XDist_MH"],
        MALA = mala_results.mean_outputs.xdist,
        Chi2 = decaes_results.chi2.x.xdist,
        LCurve = decaes_results.lcurve.x.xdist,
        GCV = decaes_results.gcv.x.xdist,
    )
    xdists = map(_x -> lib.unitsum(_x; dims = 1), xdists)

    observables_per_method = map(xdist -> compute_observable_maps(xdist, logT2s; SPWin, MPWin), xdists)
    mwfs = map(bm -> lib.unflatten_image(bm.mwf, idx), observables_per_method)
    iewfs = map(bm -> lib.unflatten_image(bm.iewf, idx), observables_per_method)
    csfs = map(bm -> lib.unflatten_image(bm.cwf, idx), observables_per_method)
    gmT2s = map(xdist -> lib.unflatten_image(vec(exp.(sum(logT2s .* xdist; dims = 1) ./ sum(xdist; dims = 1))), idx), xdists)

    base_results = (; xdists, mwfs, iewfs, csfs, gmT2s, logT2s, T2s, nT2, idx, SPWin, MPWin)
    clusters = cluster_heuristic(base_results)

    return (; base_results..., clusters)
end

# Publication-facing comparison figure for Results:
#   left block  = mean spectra for spectral component clusters (ordered by GM T2)
#   right block = observable maps for Direct / IW / MH / MALA / DECAES
function plot_xdist_comparison_combined(;
    figpath = figuresdir("xdist_comparison.pdf"),
    method_keys = (:MH, :GCV, :LCurve, :Chi2),
    slice_aspect = 0.85,
)
    (; xdists, T2s, idx, SPWin, MPWin, clusters) = xdist_results
    method_keys = [k for k in method_keys if hasproperty(xdists, k)]
    method_labels = Dict(
        :Mean => "Direct", :IW => "IW", :MH => "MH", :MALA => "MALA",
        :GCV => "GCV", :LCurve => "L-Curve", :Chi2 => L"$\chi^2$",
    )
    method_styles = Dict(
        :MH => :solid, :MALA => :dot, :Mean => :dashdot, :IW => :dashdotdot,
        :GCV => :dash, :LCurve => :dashdot, :Chi2 => :dot,
    )
    cluster_labels = (L"High MWF Voxels$$", L"High IEWF Voxels$$", L"High CWF Voxels$$") # clusters by dominant water fraction (ascending GM T2)
    region_colors = (Makie.RGBf(215 / 255, 145 / 255, 0 / 255), Makie.RGBf(0 / 255, 140 / 255, 210 / 255), Makie.RGBf(128 / 255, 0 / 255, 32 / 255)) # slightly dark orange, light blue, maroon

    # MH posterior 95% spectrum band per cluster: average of per-voxel posterior credible intervals.

    fig = Figure(; size = (1120, 360))

    # Per-cluster mean spectra of DECAES and MH, with the MH 95% posterior band
    for (i, (cluster_label, color)) in enumerate(zip(cluster_labels, region_colors))
        region_idx = findall(clusters.assignments .== i)
        ax = Axis(
            fig[1, i];
            xscale = log10, yscale = log10, title = cluster_label,
            xlabel = L"$T_2$ (ms)", ylabel = i == 1 ? L"$T_2$ amplitude" : "",
            xgridvisible = false, ygridvisible = false,
        )
        for key in method_keys
            xdist = xdists[key]
            x̄ = isempty(region_idx) ? vec(mean(xdist; dims = 2)) : vec(mean(xdist[:, region_idx]; dims = 2))
            lines!(ax, 1000 .* T2s, x̄; color = color, linewidth = 2.0, linestyle = method_styles[key], label = i == 1 ? get(method_labels, key, string(key)) : "")
        end
        vlines!(ax, 1000 .* [SPWin, MPWin]; color = (:gray50, 0.7f0), linestyle = :dash, linewidth = 1.0)
        xlims!(ax, 1000 .* minimum(T2s), 1000 .* maximum(T2s))
        i == 1 && axislegend(ax; position = :rt, framevisible = false, labelsize = 10)
    end

    # Cluster assignment map (categorical colours matching the spectra panels).
    ax_map = Axis(fig[1, 4]; aspect = slice_aspect, yreversed = true, spinewidth = 0)
    heatmap!(ax_map, lib.unflatten_image(Float32.(clusters.assignments), idx)'; colormap = cgrad(collect(region_colors); categorical = true), colorrange = (0.5, 3.5))
    hidedecorations!(ax_map)

    resize_to_layout!(fig)
    savefig(fig, figpath)
    return fig
end

# Order three spectral-fraction clusters by increasing geometric mean T₂.
function cluster_heuristic(results)
    (; xdists, logT2s, SPWin, MPWin) = results
    xdist = xdists.IW
    bm = compute_observable_maps(xdist, logT2s; SPWin, MPWin)
    fracs = hcat(bm.mwf, bm.iewf, bm.cwf)
    fracs_rel = (fracs .- mean(fracs; dims = 1)) ./ std(fracs; dims = 1)
    assignments = vec(map(ci -> ci[2], argmax(fracs_rel; dims = 2)))
    spectra = hcat([vec(mean(xdist[:, assignments .== k]; dims = 2)) for k in 1:3]...)
    order = invperm(sortperm(vec(exp.(spectra' * logT2s))))
    return (; assignments = map(a -> order[a], assignments))
end

#### Flip angle optimization

# Build a SymmetricRealNDFT interpolator for the EPG basis A(α) on a uniform α ∈ [0°, 180°] grid.
function build_epg_interpolator(batch)
    (; b, T2Range, TE, T1) = batch
    T2min, T2max = @allowscalar (T2Range[1, 1], T2Range[2, 1])
    nT2 = size(batch.A, 2)
    ETL = size(b, 1)
    nα_grid = nextpow(2, ETL) + 1
    αs_grid = range(0.0, 180.0; length = nα_grid)
    T2s = exp.(range(log(Float64(T2min)), log(Float64(T2max)); length = nT2))
    _, A_grid = lib.∇EPGBasis_with_primal(collect(αs_grid), collect(T2s); ETL, TE = Float64(@allowscalar TE[1]), T1 = Float64(@allowscalar T1[1]))
    A_interp = lib.SymmetricRealNDFT(A_grid, Val(:even)) |> lib.chop
    return lib.device_similar(b, eltype(b) === Float64 ? A_interp : Flux.f32(A_interp))
end

# Evaluate interpolated EPG basis A(α) at per-voxel flip angles α, masked and scaled by sin(α/2).
# Returns A of size [ETL x nT2 x batchsize].
function compute_epg_matrix(A_interp, mask, α)
    A₀ = A_interp(lib.deg2rad.(vec(α)))
    A = lib.apply_mask.(Flux.unsqueeze(mask; dims = 2), abs.(sind.(reshape(α, 1, 1, :) ./ 2) .* A₀))
    return A
end

# Cached MAP loss: -𝔼[log p(b|x)] where x = φ(u=0|z) is precomputed and only A(α) varies.
function compute_likelihood_loss(x, b, A, mask, δb)
    like = ex.forward_physics_model(x, A, mask, δb)
    return mean(ex.neglogpdf(like, b))
end

# Fit a degree-`degree` polynomial to the flip angle field α.
# Polynomial is fit to the deviation α - 180° (nominally zero), then re-centered.
function fit_flip_angle_polynomial(batch; degree::Int = 4)
    (; idx, α) = batch
    idx_2d = map(x -> x[1:2], idx)
    pf = lib.PolynomialField(vec(idx_2d), vec(α .- 180); degree)
    return lib.CachedPolynomialField(pf, vec(idx_2d))
end

# Recompute the EPG basis A and scalar features θ for a new per-voxel flip angle field α_new.
function recompute_batch(batch, α_new, A_interp)
    A_new = compute_epg_matrix(A_interp, batch.mask, α_new)
    batch_new = merge(batch, (; α = α_new, A = A_new))
    θ_new = ex.scalar_features(batch_new)
    batch_new = merge(batch_new, (; θ = θ_new))
    return batch_new
end

# Flip-angle objectives with fixed model parameters; gradients do not pass through z(α).
# :pseudo_map profiles η at fixed x̄ = φₓ̄(0; z).
# :em minimizes ℱ(α) = 𝔼q[log q − log p(b, x | α)], refreshing z by forward passes.
# :pmmh uses p̂(b | α) = S⁻¹∑ₛ exp(−negelboₛ), with a fitted von Mises prior.
# Fixed z fixes q(x̄|b), but q(η|x̄,b,α) varies with A(α); weighting both keeps p̂ unbiased.
# PMMH keeps p̂ on rejection, so the α marginal targets p(α|b).
# Angles are folded into [0°, 180°]; report the posterior median and standard deviation.
function optimize_flip_angle_elbo(;
    seed = 0, mode::Symbol = :direct, objective::Symbol = :pseudo_map,
    degree::Int = 6, iter::Int = 100,
    sample_newton_samples::Int = 8, pmmh_iters::Int = 250, pmmh_burnin::Int = 50,
)
    lr, lr_final, lr_warmup, lr_precond = 0.5, 0.0, 10, 0.5
    kron_Q, momentum, schedule_free = mode === :poly, 0.9, false
    newton_delta, verbose = 0.005, true
    em_eval_samples, z_update_interval, z_update_multiplier = 8, 1, 2
    sample_newton_eps, sample_newton_hmin, sample_newton_max_step_deg = newton_delta, 1.0f-3, 1.0
    pmmh_elbo_samples, pmmh_init_sigma_deg, pmmh_sigma_min_deg, pmmh_logp_trials = 4, 2.0, 0.1, 8
    seed_all!(seed)
    @assert mode in (:direct, :poly)
    @assert objective in (:pseudo_map, :em, :pmmh)
    @assert objective !== :pmmh || mode === :direct "PM-MCMC is a per-voxel sampler; use mode = :direct"
    @assert objective !== :pmmh || pmmh_burnin < pmmh_iters

    source = run_env.val_batch
    batch = merge(source, (; A = copy(source.A), θ = copy(source.θ))) |> gpu
    (; b, A, mask, δb, θ, α) = batch
    ℰ, 𝒢, 𝒫 = run_env.models.enc, run_env.models.genatr, run_env.models.prior
    α_decaes, T = copy(α), eltype(b)

    A_interp = build_epg_interpolator(batch)
    cpf = mode === :poly ? fit_flip_angle_polynomial(batch; degree) : nothing
    cpf_flip_angle(weights) = 180 .+ reshape(cpf.Φ * weights, 1, :)

    z = lib.chunked_apply(ℰ, b, A, mask, θ; chunksize = INFERENCE_CHUNKSIZE)
    α_from_params(p) = mode === :poly ? cpf_flip_angle(p) : rad2deg.(p) # params are ~O(1) scale
    params = mode === :poly ? cpf.w : deg2rad.(α) # init: poly fit to decaes (poly) or decaes α per voxel (direct)
    Aα(p) = compute_epg_matrix(A_interp, mask, α_from_params(p))
    u_zero = fill!(ex.sample_base_similar(𝒢.model, b), 0.0f0) # pseudo-mode base point u = 0

    # Posterior summary at flip angle α(p) and latent u with the fixed embedding z and features θ
    post_summary(p, u) = posterior_chunks(𝒢, 𝒫, u, z, b, Aα(p), mask, δb, θ)

    # Likelihood at frozen x = φ(u;z); differentiating only this term gives the EM M-step
    # Exact at fixed z; the omitted α-dependence of the variational posterior is the z-refresh's job.
    likelihood_loss(p, x) = compute_likelihood_loss(x, b, Aα(p), mask, δb)

    rule = lib.KronPSGD(; eta = lr, eta_Q = lr_precond, kron_Q, diag_Q = !kron_Q, beta = momentum, schedule_free, whiten_grad = false)
    lrfun(i) = i < lr_warmup ? T(lr) * T(i) / T(lr_warmup) : T(lr) + T(lr_final - lr) * T(i - lr_warmup) / T(max(1, iter - lr_warmup))
    log_iter(i, ℓ, gnorm) = verbose && (i == 0 || i % max(1, iter ÷ 10) == 0 || i == iter) && (αc = α_from_params(params);
        @info "Iter $i/$iter: loss = $(round(ℓ; digits = 4)), α range = [$(round(minimum(αc); digits = 2)), $(round(maximum(αc); digits = 2))], |∇| = $(round(gnorm; digits = 6))")
    make_result(; α_std = nothing) = (; α_decaes = cpu(α_decaes), α_final = cpu(α_from_params(params)), degree, α_std)

    @info "Optimizing flip angles (mode=:$mode, objective=:$objective)"

    # Pseudo-MAP: minimize M(0,α) with z and x̄₀ fixed; the nuisance mode η₀(α) is differentiated by IFT
    if objective === :pseudo_map
        logx̄₀ = post_summary(params, u_zero).summary.x[1:(end-2), :]
        prior = ex.composite_prior(𝒫, (θ,))
        function pseudo_map_loss(p)
            Ap = Aα(p)
            μ = lib.apply_mask.(mask, ex.resampled_batched_vec(Ap, logx̄₀))
            logs, logσ = ex.neglogpdf_qrician_nuis_mode(mask, b, μ, δb, Val(8); logs_prior = 𝒫.model.logs, logσ_prior = 𝒫.model.logσ)
            x = [logx̄₀; logs; logσ]
            return compute_likelihood_loss(x, b, Ap, mask, δb) + mean(ex.neglogpdf(prior, x))
        end
        opt_p = lib.Optimisers.setup(rule, params)
        for i in 0:iter
            lib.Optimisers.adjust!(opt_p, lrfun(i))
            ℓ, (∇p,), (h,), (v,) = withgradient_and_hvp(pseudo_map_loss, params; step_size = newton_delta)
            lib.Optimisers.update!(opt_p, params, ∇p, (h, v))
            log_iter(i, ℓ, lib.norm(∇p))
        end
        return make_result()
    end

    # :pmmh keeps z fixed; the conditional nuisance proposal still depends on α.
    if objective === :pmmh
        logaddexp(a, c) = max(a, c) + log1p(exp(-abs(a - c)))

        # log p̂(b | α) = logsumexpₛ(−negelboₛ) − log S; α is in degrees.
        function marginal_loglike(αv, S)
            Ap = compute_epg_matrix(A_interp, mask, αv)
            post(u) = posterior_chunks(𝒢, 𝒫, u, z, b, Ap, mask, δb, θ)
            acc = .-post(ex.sample_base_similar(𝒢.model, b)).negelbo
            for _ in 2:S
                acc .= logaddexp.(acc, .-post(ex.sample_base_similar(𝒢.model, b)).negelbo)
            end
            acc .-= T(log(S))
            return acc
        end

        # von Mises prior on α with mean 180°; κ solves A₁(κ) = ⟨cos(α - 180°)⟩
        κ_pr = T(inv_besseli1i0(-mean(cosd, α_decaes)))
        log_prior(αv) = .-κ_pr .* cosd.(αv)

        # Use inverse empirical-CDF quantiles in the physical [0°, 180°] gauge.
        function fold_angle(α)
            a = mod(α, 360)
            return ifelse(a > 180, 360 - a, a)
        end

        function finalize_posterior(samples)
            angles = fold_angle.(reduce(vcat, map(cpu, samples)))
            α_med = median(angles; dims = 1)
            copyto!(params, deg2rad.(α_med))
            return make_result(; α_std = std(angles; dims = 1))
        end

        # PMMH retains the likelihood estimate on rejection; refreshing it would change the target.
        σ_prop = fill!(similar(α_decaes), T(pmmh_init_sigma_deg))

        # Estimator noise check: Var(log p̂) ≲ 1.2–3 at α_decaes (Doucet et al., 2015)
        Ls = [marginal_loglike(α_decaes, pmmh_elbo_samples) for _ in 1:pmmh_logp_trials]
        mL = sum(Ls) ./ T(pmmh_logp_trials)
        vL = zero(α_decaes)
        for L in Ls
            vL .+= (L .- mL) .^ 2
        end
        vL ./= T(pmmh_logp_trials - 1)
        verbose && @info "PM-MCMC: mean Var(log p̂) = $(round(mean(vL); digits = 2)) (target ≲ 1.2-3), mean initial σ = $(round(mean(σ_prop); digits = 3))°"

        α_curr = copy(α_decaes)
        L_curr = log_prior(α_curr) .+ marginal_loglike(α_curr, pmmh_elbo_samples)
        chain = typeof(α_decaes)[]
        for n in 1:pmmh_iters
            # Reflecting random walk on [0°, 180°]: the folded kernel remains symmetric, so no Hastings correction
            α_prop = fold_angle.(α_curr .+ σ_prop .* lib.randn_similar(α_curr))
            L_prop = log_prior(α_prop) .+ marginal_loglike(α_prop, pmmh_elbo_samples)
            accept = log.(lib.rand_similar(α_curr)) .< L_prop .- L_curr
            α_curr .= ifelse.(accept, α_prop, α_curr)
            L_curr .= ifelse.(accept, L_prop, L_curr)
            if n <= pmmh_burnin
                # Robbins–Monro adaptation of log σ toward acceptance 0.44, frozen after burn-in
                σ_prop .= max.(σ_prop .* exp.((accept .- T(0.44)) ./ T(n)), T(pmmh_sigma_min_deg))
            else
                push!(chain, copy(α_curr))
            end
            verbose && n % max(1, pmmh_iters ÷ 10) == 0 && @info "PM-MCMC iter $n/$pmmh_iters: mean acceptance = $(round(mean(accept); digits = 2)), mean σ = $(round(mean(σ_prop); digits = 3))°"
        end
        return finalize_posterior(chain)
    end

    # :em - free energy ℱ(α) = 𝔼_u[negelbo] with z(α) refreshed as the variational posterior
    elbo_loss(p, u) = mean(post_summary(p, u).negelbo)
    function refresh_z!()
        batch_curr = recompute_batch(batch, α_from_params(params), A_interp)
        A .= batch_curr.A
        θ .= batch_curr.θ
        z .= lib.chunked_apply(ℰ, b, A, mask, θ; chunksize = INFERENCE_CHUNKSIZE)
        return nothing
    end
    estimate_elbo_loss(p, K) = mean(elbo_loss(p, ex.sample_base_similar(𝒢.model, b)) for _ in 1:K)
    cum_ℓ = 0.0

    if mode === :direct
        epsα, hmin, maxstep = T(sample_newton_eps), T(sample_newton_hmin), T(deg2rad(sample_newton_max_step_deg))
        function sample_newton_step!()
            p₊, p₋ = params .+ epsα, params .- epsα
            g, h, nelbos = zero(params), zero(params), Float64[]
            for _ in 1:sample_newton_samples
                s = post_summary(params, ex.sample_base_similar(𝒢.model, b))
                push!(nelbos, mean(s.negelbo))
                x = s.summary.x
                _, (g₊,) = lib.Zygote.withgradient(p -> likelihood_loss(p, x), p₊)
                _, (g₋,) = lib.Zygote.withgradient(p -> likelihood_loss(p, x), p₋)
                g .+= (g₊ .+ g₋) ./ 2
                h .+= (g₊ .- g₋) ./ (2epsα)
            end
            g ./= T(sample_newton_samples)
            h ./= T(sample_newton_samples)
            Δ = .-g ./ max.(h, hmin)
            maxabs = maximum(abs, Δ)
            maxabs > maxstep && (Δ .*= maxstep / maxabs)
            params .+= Δ
            return lib.norm(g), mean(nelbos)
        end
        for i in 0:iter
            gnorm, meanℓ = sample_newton_step!()
            z_update_interval > 0 && refresh_z!()
            cum_ℓ += (meanℓ - cum_ℓ) / (i + 1)
            log_iter(i, cum_ℓ, gnorm)
        end
        return make_result()
    end

    opt = lib.Optimisers.setup(rule, params)
    for i in 0:iter
        u = ex.sample_base_similar(𝒢.model, b)
        if z_update_interval > 0 && (i == 0 || i % z_update_interval == 0)
            refresh_z!()
            verbose && @info "  Updated encoder embedding z at iter $i"
            refresh_elbo = estimate_elbo_loss(params, em_eval_samples)
            verbose && @info "  Free energy after embedding refresh" estimate = refresh_elbo
            z_update_interval *= z_update_multiplier
        end
        lib.Optimisers.adjust!(opt, lrfun(i))
        ℓ, gnorm = lib.with_trainmode!(params, opt) do _, _
            s = post_summary(params, u)
            x, nelbo = s.summary.x, mean(s.negelbo)
            lossfn(p) = likelihood_loss(p, x)
            _, (∇p,), (h,), (v,) = withgradient_and_hvp(lossfn, params; step_size = newton_delta)
            lib.Optimisers.update!(opt, params, ∇p, (h, v))
            (nelbo, lib.norm(∇p))
        end
        cum_ℓ += (ℓ - cum_ℓ) / (i + 1)
        log_iter(i, cum_ℓ, gnorm)
    end
    return make_result()
end

# Flip-angle fields (DECAES, PMMH median, pseudo-mode, polynomial), posterior σ_α and differences
function plot_flip_angle_comparison(mcmc_res, pseudo_res, poly_res; figpath = figuresdir("flip_angle_elbo_comparison.pdf"), slice_aspect = 0.85)
    (; idx) = run_env.val_batch
    unflatten(v) = lib.unflatten_image(v, idx)[1, :, :]
    α_decaes = mcmc_res.α_decaes
    α_cr = extrema(filter(isfinite, unflatten(α_decaes))) |> bnds -> (floor(bnds[1]), ceil(bnds[2]))

    fig = Figure(; size = (1000, 400), fontsize = 18) # figure fontsize themes all titles, axis labels, ticks, and colorbar labels
    ca = (; aspect = slice_aspect, yreversed = true, xlabel = "")

    function hm_plot!(pos, img, title, cr, cmap, cblabel)
        ax = Axis(fig[pos...]; title, ca...)
        hm = heatmap!(ax, img'; colormap = cmap, colorrange = cr)
        hidedecorations!(ax)
        hidespines!(ax)
        Colorbar(fig[pos[1], pos[2]+1], hm; label = cblabel)
    end
    α_methods = ((mcmc_res, "PM\\!-\\!MCMC", "Voxelwise"), (pseudo_res, "PSEUDO", "Voxelwise"), (poly_res, "POLY", "Polynomial ELBO (degree $(poly_res.degree))"))
    hm_plot!((1, 1), unflatten(α_decaes), L"Voxelwise $\hat{\alpha}_\mathrm{DECAES}$", α_cr, :viridis, "")
    for (k, (res, tag, label)) in enumerate(α_methods)
        hm_plot!((1, 2k + 1), unflatten(res.α_final), L"%$label $\hat{\alpha}_\text{%$tag}$", α_cr, :viridis, k == length(α_methods) ? L"$\hat{\alpha}$ $(\degree)$" : "")
    end

    img_σα = unflatten(mcmc_res.α_std)
    ax_σ = Axis(fig[2, 1]; title = L"Uncertainty $\sigma(\hat{\alpha}_\text{PM\!-\!MCMC})$", ca...)
    hm_σ = heatmap!(ax_σ, img_σα'; colormap = :inferno, colorrange = (0.0, quantile(filter(isfinite, img_σα), 0.99)))
    hidedecorations!(ax_σ)
    hidespines!(ax_σ)
    Colorbar(fig[2, 2], hm_σ; label = L"$\sigma(\hat{\alpha}_\text{PM\!-\!MCMC})$ $(\degree)$")

    rms_vs_decaes(α̂) = round(sqrt(mean((α̂ .- α_decaes) .^ 2)); digits = 3)
    function Δhm_plot!(pos, img, title, cblabel)
        ax = Axis(fig[pos...]; title, ca...)
        hm = heatmap!(ax, img'; colormap = :RdBu, colorrange = maximum(abs, filter(isfinite, img)) .* (-1, 1))
        hidedecorations!(ax)
        hidespines!(ax)
        Colorbar(fig[pos[1], pos[2]+1], hm; label = cblabel)
    end
    for (k, (res, tag, _)) in enumerate(α_methods)
        Δhm_plot!((2, 2k + 1), unflatten(res.α_final .- α_decaes), L"$\Vert\hat{\alpha}_\mathrm{%$tag} - \hat{\alpha}_\mathrm{DECAES}\Vert_\mathrm{RMS}$ = %$(rms_vs_decaes(res.α_final))°", k == length(α_methods) ? L"$\Delta\alpha$ $(\degree)$" : "")
    end

    savefig(fig, figpath)
    return fig
end

# Pairwise flip-angle differences |α_i - α_j|: voxel median with 2.5/97.5 percentiles
function write_flip_angle_table(mcmc_res, pseudo_res, poly_res; figpath = figuresdir("flip_angle_table.tex"))
    methods = [("DECAES", vec(mcmc_res.α_decaes)), ("PM-MCMC", vec(mcmc_res.α_final)), ("Pseudo-mode", vec(pseudo_res.α_final)), ("Polynomial", vec(poly_res.α_final))]
    names, n = first.(methods), length(methods)

    # Show each unordered pair once: drop the DECAES row and the polynomial column
    row_idxs, col_idxs = 2:n, 1:(n-1)
    column_labels = [Any[L"\text{Method}", [latexstring("\\text{$(names[j])}") for j in col_idxs]...]]

    data = Matrix{Any}(undef, length(row_idxs), length(col_idxs) + 1)
    for (ii, i) in enumerate(row_idxs)
        data[ii, 1] = names[i]
        for (jj, j) in enumerate(col_idxs)
            if j >= i
                data[ii, jj+1] = nothing
            else
                d = filter(isfinite, abs.(methods[i][2] .- methods[j][2]))
                data[ii, jj+1] = (median(d), quantile(d, 0.025), quantile(d, 0.975))
            end
        end
    end

    function cell_formatter(v, i, j)
        v isa String && return v
        isnothing(v) && return LatexCell("---")
        med, lo, hi = v
        return LatexCell(latexstring("$(round(med; sigdigits = 2))^{$(round(hi; sigdigits = 2))}_{$(round(lo; sigdigits = 2))}"))
    end
    save_table(figpath) do io
        default_pretty_table(io, data; column_labels, formatters = [cell_formatter])
    end
end

# Posterior-mean T2 distribution at a fixed flip-angle field α (degrees) by independent MH
function flip_angle_mh_xdist(α_field, batch, A_interp; iter = 1000, warmup_iter = 0)
    ℰ, 𝒢, 𝒫 = run_env.models.enc, run_env.models.genatr, run_env.models.prior
    batch_α = recompute_batch(batch, gpu(reshape(Float32.(vec(α_field)), 1, :)), A_interp)
    (; b, A, mask, δb, θ) = batch_α
    samp_prob = ex.SamplingProblem(ex.Metropolis(), ℰ, 𝒢, 𝒫)
    res = ex.run_sampling(samp_prob, b, A, mask, δb, θ; lr = nothing, iter, warmup_iter, verbose = false)
    return lib.unitsum(cpu(res.mean_outputs.xdist); dims = 1)
end

# Observable maps under the per-voxel and polynomial flip-angle estimates, by independent MH
function run_flip_angle_observables(elbo_res, poly_res; seed = 0, iter = 1000, warmup_iter = 0, SPWin = 32.0f-3, MPWin = 200.0f-3)
    seed_all!(seed)
    batch = run_env.val_batch
    (; T2Range, idx) = batch
    batch = batch |> gpu
    A_interp = build_epg_interpolator(batch)
    nT2 = run_env.cfg.data.nT2
    logT2s = range(log(T2Range[1, 1]), log(T2Range[2, 1]); length = nT2)

    function observables(α_field)
        xdist = flip_angle_mh_xdist(α_field, batch, A_interp; iter, warmup_iter)
        bm = compute_observable_maps(xdist, logT2s; SPWin, MPWin)
        mwf = lib.unflatten_image(bm.mwf, idx)
        iewf = lib.unflatten_image(bm.iewf, idx)
        cwf = lib.unflatten_image(bm.cwf, idx)
        gmT2 = lib.unflatten_image(vec(exp.(sum(logT2s .* xdist; dims = 1))), idx) # xdist is normalized to unit sum
        return (; mwf, iewf, cwf, gmT2)
    end
    bm_elbo = observables(elbo_res.α_final)
    bm_poly = observables(poly_res.α_final)

    return (; bm_elbo, bm_poly, T2Range)
end

function plot_flip_angle_observables(results = fa_observable_results; figpath = figuresdir("flip_angle_observables.pdf"), slice_aspect = 0.85)
    (; bm_elbo, bm_poly, T2Range) = results
    fig = Figure(; size = (1200, 500), fontsize = 18)
    ca = (; aspect = slice_aspect, yreversed = true)

    # Columns: MWF, IEWF, CWF, GM T2; rows: per-voxel and polynomial free energy
    cols = ((:mwf, L"\mathrm{MWF}", (0.0f0, 0.25f0), L"\mathrm{MWF}"),
        (:iewf, L"\mathrm{IEWF}", (0.7f0, 1.0f0), L"\mathrm{IEWF}"),
        (:cwf, L"\mathrm{CWF}", (0.0f0, 0.25f0), L"\mathrm{CWF}"),
        (:gmT2, L"GM $T_2$", (T2Range[1, 1], 0.2f0), L"$T_2$ (s)"))
    row_labels = (L"Voxelwise $\hat{\alpha}_\text{ELBO}$", L"Polynomial $\hat{\alpha}_\text{POLY}$")
    for (c, (key, title, cr, cblabel)) in enumerate(cols)
        hm = nothing
        for (r, bm) in enumerate((bm_elbo, bm_poly))
            ax = Axis(fig[r, 2c-1]; title = r == 1 ? title : "", ylabel = c == 1 ? row_labels[r] : "", ca...)
            hm = heatmap!(ax, getproperty(bm, key)'; colormap = :viridis, colorrange = cr)
            hidedecorations!(ax; label = false)
            hidespines!(ax)
        end
        Colorbar(fig[1:2, 2c], hm; label = cblabel)
    end

    savefig(fig, figpath)
    return fig
end

#### Spatial regularization

# Normalized adjacency W = D⁻¹ᐟ²AD⁻¹ᐟ² and precision Q = τ(I − ρW), so Q√d = τ(1 − ρ)√d.

# Empirical Bayes: fit a global smoothing strength κ for the spatial precision Q = I + κL, L = I - W.
# E-step: posterior-mean quadratic forms; M-step: solve h(κ) = E_L via the Stieltjes approximant.

# Solve h(κ) = α * E_L for κ via bisection; h is monotonically decreasing in κ.
# Returns 0 if target ≥ h(0) (boundary case); E_L ≤ 0 would imply κ → ∞ and fails loudly.
function solve_kappa_fixed_alpha(sa, α::T, E_L::T; tol = T(1e-8), maxiter = 200) where {T}
    target = α * E_L
    @assert target > 0 "Nonpositive spatial statistic E_L = $(E_L)"
    h0 = lib.h_approx(sa, zero(T))
    target >= h0 && return zero(T)
    κ_lo, κ_hi = zero(T), one(T)
    while lib.h_approx(sa, κ_hi) > target
        κ_hi *= T(2)
    end
    for _ in 1:maxiter
        κ_mid = (κ_lo + κ_hi) / 2
        abs(κ_hi - κ_lo) / max(κ_mid, one(T)) < tol && return κ_mid
        if lib.h_approx(sa, κ_mid) > target
            κ_lo = κ_mid
        else
            κ_hi = κ_mid
        end
    end
    return (κ_lo + κ_hi) / 2
end

# Unpack the held-out slice batch, models (ℰ, 𝒢, 𝒫), and encoder output z from run_env.
function unpack_run_env()
    batch = run_env.val_batch |> gpu
    ℰ, 𝒢, 𝒫 = run_env.models.enc, run_env.models.genatr, run_env.models.prior

    (; b, A, mask, δb, θ, idx, T2Range) = batch
    z = lib.chunked_apply(ℰ, b, A, mask, θ; chunksize = INFERENCE_CHUNKSIZE)
    nT2, udim_x = 𝒢.opts.nT2, 𝒢.opts.udim - 2

    return (; b, A, mask, δb, θ, idx, T2Range, 𝒢, 𝒫, z, nT2, udim_x)
end

# Construct the joint posterior (ForwardPhysicsTransformedPosterior), conditioned prior flow,
# and amortized marginal posterior q(x|y) from the current batch and model state.
function setup_posteriors(𝒢, 𝒫, z, b, A, mask, δb, θ)
    post(u) = posterior_chunks(𝒢, 𝒫, u, z, b, A, mask, δb, θ)
    prior_xdist = ex.recondition(ex.xdist_prior(𝒫), (θ,))
    qx = ex.marginal_posterior(𝒢, (z, b, A, mask, δb, θ))
    return (; post, prior_xdist, qx)
end

# Forward map: u_x -> u (padded) -> posterior -> log x̄ -> ξ (ILR coords via prior flow inverse).
function forward_ξ(post, prior_xdist, u_x)
    u = [u_x; zeros_similar(u_x, 2, Base.tail(size(u_x))...)]
    (; summary, neglogpostu) = post(u)
    logx̄ = summary.x[1:(end-2), ..]
    ξ = ex.flow_backward(prior_xdist, logx̄).u
    return (; ξ, logx̄, neglogpostu, summary)
end

# Total spatial energy: E = Σ(-log p(u_i|y_i)) - ½||ξ||² + ½ξᵀQξ,
# where ½ξᵀQξ replaces the i.i.d. base-space prior ½||ξ||² with the spatial precision Q = τ(I - ρW).
function spatial_energy_total(post, prior_xdist, op_gpu, u_x, τ, ρ)
    (; ξ, logx̄, neglogpostu) = forward_ξ(post, prior_xdist, u_x)
    iid_energy = sum(abs2, ξ) / 2
    spatial_energy = lib.precision_energy(op_gpu, ξ, τ, ρ)
    energy = sum(neglogpostu) - iid_energy + spatial_energy
    return energy, (; energy, ξ, logx̄, neglogpostu, iid_energy, spatial_energy)
end

# Compute the total spatial energy E (see spatial_energy_total) and ∇_{u_x} E via Zygote.
function spatial_energy_and_grad(post, prior_xdist, op_gpu, u_x, τ, ρ)
    outputs = Ref{NamedTuple}()
    energy, back = lib.Zygote.pullback(u_x) do _u_x
        E, state = spatial_energy_total(post, prior_xdist, op_gpu, _u_x, τ, ρ)
        @ignore_derivatives outputs[] = state
        return E
    end
    ∇u_x = back(one(energy))[1]::typeof(u_x)
    return outputs[], ∇u_x
end

# Compute roughness ||ξ - Wξ||²_F / N and per-voxel observable maps from a forward_ξ output.
spatial_observables(outputs, logT2s; SPWin = 32.0f-3, MPWin = 200.0f-3) = compute_observable_maps(lib.unitsum(exp.(outputs.logx̄); dims = 1), logT2s; SPWin, MPWin)

# Build the spatial adjacency operator and fit a J_max-pole Stieltjes approximant to logdet/h.
# Forward moments are exact; inverse moments use deflated Hutchinson, avoiding an eigendecomposition.
function build_spatial_operators(; seed = 0, J_max::Int = 4)
    seed_all!(seed)
    (; idx) = run_env.val_batch
    op = lib.SpatialAdjacencyOperator(Float32, Val(2), cpu(idx); k_max = J_max)

    @time op = lib.estimate_stochastic_traces!(op |> gpu) |> cpu32
    stieltjes = @time lib.StieltjesApproximant(op, Val(J_max))
    stieltjes.valid || @warn "Stieltjes approximant invalid"

    return (; op, stieltjes)
end

# Validates the J_max-pole Stieltjes approximant against an exact dense eigendecomposition of W:
# BigFloat spectral moments and accuracy plots for logdet/h/h′. Feasible for a 2D slice.
function validate_spatial_stieltjes(; J_max::Int = 4, figpath = figuresdir("spatial_stieltjes_validation.pdf"))
    (; idx) = run_env.val_batch
    op = lib.SpatialAdjacencyOperator(Float32, Val(2), cpu(idx); k_max = J_max)

    W = lib.apply_normalized_adjacency(op, Matrix{Float32}(LinearAlgebra.I, length(op), length(op)))
    λ_W_true = eigvals(Symmetric(W))
    λ_W_true[end] = 1 # pin the trivial eigenvalue to exactly 1
    λ_L = 1 .- λ_W_true

    # Exact moments tₖ = tr(Lᵏ) and sₖ = tr(L⁻ᵏ) in BigFloat to avoid cancellation
    fwd_mom = ntuple(k -> sum(big(λ)^k for λ in λ_L), J_max)
    inv_mom = ntuple(k -> sum(inv(big(λ)^(k - 1)) for λ in λ_L if λ > 0), J_max)
    stieltjes = lib.StieltjesApproximant(length(op), fwd_mom, inv_mom)
    stieltjes = fmap(x -> x isa Union{Int, Bool} ? x : Float32.(x), stieltjes) # downcast to Float32 after BigFloat fit
    @assert stieltjes.valid "Stieltjes approximant invalid"

    n = length(λ_L)

    # Exact spectral functions for comparison
    exact_logdet(κ) = sum(log1p.(κ .* λ_L))
    exact_h(κ) = sum(λ_L ./ (1 .+ κ .* λ_L))
    exact_hprime(κ) = -sum((λ_L .^ 2) ./ ((1 .+ κ .* λ_L) .^ 2))
    stieltjes_hprime(κ) = -sum((stieltjes.weights .* stieltjes.poles .^ 2) ./ ((1 .+ κ .* stieltjes.poles) .^ 2))

    log10κ_vals = range(-1.0f0, 6.0f0; length = 100)
    κ_vals = exp10.(log10κ_vals)

    exact_logdet_vals = [exact_logdet(κ) for κ in κ_vals]
    exact_h_vals = [exact_h(κ) for κ in κ_vals]
    exact_col4_vals = [exact_hprime(κ) for κ in κ_vals]

    stieltjes_logdet_vals = [lib.logdet_approx(stieltjes, κ) for κ in κ_vals]
    stieltjes_h_vals = [lib.h_approx(stieltjes, κ) for κ in κ_vals]
    stieltjes_col4_vals = [stieltjes_hprime(κ) for κ in κ_vals]

    style_exact = (color = :black, linestyle = :solid, linewidth = 3)
    style_stieltjes = (color = :red, linestyle = :dash, linewidth = 2)

    fig_stieltjes = Figure(; size = (900, 450))

    λ_L_pos = filter(>(0), λ_L)
    pole_bins = exp10.(range(log10(minimum(λ_L_pos)), log10(maximum(λ_L_pos)); length = 20))

    # (a) Sorted eigenvalue spectrum of L = I - W
    ax_eig = Axis(fig_stieltjes[1, 1];
        xlabel = L"Index $i$",
        ylabel = L"Eigenvalues $\lambda_i$",
        xticks = WilkinsonTicks(3),
    )
    scatter!(ax_eig, 1:n, sort(λ_L; rev = true); markersize = 4)
    hlines!(ax_eig, [0.0]; linestyle = :dash, color = :red)

    # (b) log det(I + κL) over log(κ)
    ax_logdet = Axis(fig_stieltjes[1, 2];
        title = L"$f(\kappa) = \log|I + \kappa L|$",
        xlabel = L"\kappa",
        xscale = log10,
        yscale = log10,
    )
    p_exact_logdet = lines!(ax_logdet, κ_vals, exact_logdet_vals; style_exact..., label = L"Exact $f(\kappa)$")
    p_stieltjes_logdet = lines!(ax_logdet, κ_vals, stieltjes_logdet_vals; style_stieltjes..., label = L"Stieltjes $J=%$(lib.npoles(stieltjes))$")

    # (c) h(κ) = d/dκ log|I + κL| over log(κ)
    ax_h = Axis(fig_stieltjes[1, 3];
        title = L"$f\,'(\kappa) = \mathrm{tr}((I + \kappa L)^{-1} L)$",
        xlabel = L"\kappa",
        xscale = log10,
        yscale = log10,
    )
    p_exact_h = lines!(ax_h, κ_vals, exact_h_vals; style_exact..., label = L"Exact $f\,'(\kappa)$")
    lines!(ax_h, κ_vals, stieltjes_h_vals; style_stieltjes..., label = L"Stieltjes $J=%$(lib.npoles(stieltjes))$")

    # (d) h′(κ)
    ax_p = Axis(fig_stieltjes[1, 4];
        title = L"$-\!f\,''(\kappa) = \mathrm{tr}(((I + \kappa L)^{-1}L)^2)$",
        xlabel = L"\kappa",
        xscale = log10,
        yscale = log10,
    )
    p_exact_col4 = lines!(ax_p, κ_vals, -exact_col4_vals; style_exact..., label = L"Exact $f\,''(\kappa)$")
    lines!(ax_p, κ_vals, -stieltjes_col4_vals; style_stieltjes..., label = L"Stieltjes $J=%$(lib.npoles(stieltjes))$")

    # (e) Stieltjes poles and weights vs Laplacian eigenvalue histogram
    ax_poles = Axis(fig_stieltjes[2, 1];
        xlabel = L"$p_j$ and $\lambda_i$",
        ylabel = L"Normalized $w_j$ and $\lambda_i$",
        xscale = log10,
    )
    p_hist = hist!(ax_poles, λ_L_pos; bins = pole_bins, normalization = :pdf, color = (:gray, 0.3), strokecolor = :gray, strokewidth = 1, label = L"\lambda > 0")
    p_poles = scatterlines!(ax_poles, [stieltjes.poles...], [stieltjes.weights...] ./ sum(stieltjes.weights); color = :red, markersize = 12, label = L"Stieltjes poles")

    # (f) Relative error in f(κ) = log|I + κL|
    ax_rel_logdet = Axis(fig_stieltjes[2, 2];
        title = L"$|\hat{f} - f|/|f|$",
        xlabel = L"\kappa",
        xscale = log10, yscale = log10,
    )
    hlines!(ax_rel_logdet, [0.0]; linestyle = :solid, color = :black, linewidth = 1)
    lines!(ax_rel_logdet, κ_vals, max.(relerr.(stieltjes_logdet_vals, exact_logdet_vals), eps(Float32)); style_stieltjes...)

    # (g) Relative error in h(κ)
    ax_rel_h = Axis(fig_stieltjes[2, 3];
        title = L"$|\hat{f}' - f\,'|/|f\,'|$",
        xlabel = L"\kappa",
        xscale = log10, yscale = log10,
    )
    hlines!(ax_rel_h, [0.0]; linestyle = :solid, color = :black, linewidth = 1)
    lines!(ax_rel_h, κ_vals, max.(relerr.(stieltjes_h_vals, exact_h_vals), eps(Float32)); style_stieltjes...)

    # (h) Relative error in h′(κ)
    ax_rel_p = Axis(fig_stieltjes[2, 4];
        title = L"$|\hat{f}'' - f\,''|/|f\,''|$",
        xlabel = L"\kappa",
        xscale = log10, yscale = log10,
    )
    hlines!(ax_rel_p, [0.0]; linestyle = :solid, color = :black, linewidth = 1)
    lines!(ax_rel_p, κ_vals, max.(relerr.(stieltjes_col4_vals, exact_col4_vals), eps(Float32)); style_stieltjes...)

    Legend(fig_stieltjes[1:2, 5], [p_exact_logdet, p_exact_h, p_exact_col4, p_stieltjes_logdet, p_hist, p_poles], [L"Exact $f(\kappa)$", L"Exact $f\,'(\kappa)$", L"Exact $-\!f\,''(\kappa)$", L"Stieltjes $J=%$(lib.npoles(stieltjes))$", L"Eigenvalues$$", L"Stieltjes poles$$"]; framevisible = false)

    savefig(fig_stieltjes, figpath)

    return (; op, stieltjes)
end

# Unbiased squared-mean estimate from the sample-mean statistic and the per-sample average
debias_squared_mean(E_mean, E_avg, m) = @. (m * E_mean - E_avg) / (m - 1)

# Fit the global κ: E-step samples ξ from q(u|y); M-step solves h(κ) = E_L by bisection.
function fit_spatial_hyperparameters_em(op, sa; seed = 0, nsamples = 50)
    seed_all!(seed)
    @assert nsamples > 1

    (; b, A, mask, δb, θ, 𝒢, 𝒫, z, udim_x) = unpack_run_env()
    (; post, prior_xdist, qx) = setup_posteriors(𝒢, 𝒫, z, b, A, mask, δb, θ)

    # Streaming accumulators: avoid allocating full (udim_x, n_voxels, nsamples) array.
    n_voxels = size(b, 2)
    ξ_sum = lib.zeros_similar(b, udim_x, n_voxels)
    E_I_sum, E_W_sum = 0.0, 0.0

    # E-step via direct samples u_x ~ q(u|y) = N(0, I)
    for _ in 1:nsamples
        u_x = ex.sample_base_similar(qx, b)
        (; ξ) = forward_ξ(post, prior_xdist, u_x)
        ξ_sum .+= ξ
        stats_s = lib.spatial_precision_stats(op, ξ) |> cpu
        E_I_sum += sum(stats_s.E_I)
        E_W_sum += sum(stats_s.E_W)
    end

    # Posterior-mean quadratic forms, averaged over `ξ` dimensions and debiased by U-statistics
    S = nsamples
    stats_mean = lib.spatial_precision_stats(op, ξ_sum ./ S) |> cpu
    E_I = Float32(debias_squared_mean(sum(stats_mean.E_I), E_I_sum / S, S) / udim_x)
    E_W = Float32(debias_squared_mean(sum(stats_mean.E_W), E_W_sum / S, S) / udim_x)

    κ = solve_kappa_fixed_alpha(sa, 1.0f0, E_I - E_W)
    τ, ρ = 1 + κ, κ / (1 + κ) # Q = τ(I - ρW) = I + κL
    @info "EM spatial hyperparameters using debiased posterior means: E_I = $(E_I), E_W = $(E_W), E_L = $(E_I - E_W), κ = $(κ)"

    return (; τ, ρ, κ, E_I, E_W)
end

# Replace the iid prior by Q = τ(I − ρW): E(u) = −∑ᵢlog p(uᵢ | bᵢ) + ½ξᵀ(Q − I)ξ.
# ξ is the prior's whitened coordinate; Q stays fixed during this MAP optimization.
function run_spatial_map_optimization(em_results; seed = 0, nsteps = 100)
    SPWin, MPWin = 32.0f-3, 200.0f-3
    newton_delta, lr, lr_warmup, lr_precond, max_rank = 0.001, 0.5, 10, 0.5f0, 5
    seed_all!(seed)
    (; b, A, mask, δb, θ, idx, T2Range, 𝒢, 𝒫, z, nT2) = unpack_run_env()
    (; T2s, logT2s) = setup_t2_grid(T2Range, nT2)

    configs = [
        (; label = "global", em_results.τ, em_results.ρ),
        (; label = "unregularized", τ = one(em_results.τ), ρ = zero(em_results.ρ)),
    ]

    op = lib.SpatialAdjacencyOperator(Val(2), cpu(idx))
    op_gpu = gpu32(op)

    (; post, prior_xdist, qx) = setup_posteriors(𝒢, 𝒫, z, b, A, mask, δb, θ)
    u0 = zeros_similar(ex.sample_base_similar(qx, b))

    evaluate_state_and_grad(u_x, τ, ρ) = spatial_energy_and_grad(post, prior_xdist, op_gpu, u_x, τ, ρ)

    # Run each (τ, ρ) config independently from u = 0
    results = map(configs) do cfg
        (; τ, ρ, label) = cfg
        u = copy(u0)
        outputs, ∇u = evaluate_state_and_grad(u, τ, ρ)
        rule = lib.LRAPSGD(; eta = lr, eta_Q = lr_precond, max_rank, whiten_grad = true)
        opt = lib.Optimisers.setup(rule, u)

        for t in 1:nsteps
            lib.Optimisers.adjust!(opt, eltype(u)(lr * (t >= lr_warmup ? 1.0 : t / lr_warmup)))
            δv = eltype(u)(newton_delta)
            v = randn_similar(u)
            _, ∇u⁺ = evaluate_state_and_grad(u .+ δv .* v, τ, ρ)
            h = @. (∇u⁺ - ∇u) / δv
            lib.Optimisers.update!(opt, u, ∇u, (h, v))
            outputs, ∇u = evaluate_state_and_grad(u, τ, ρ)

            !isfinite(outputs.energy) && error("Energy is not finite")
            @info "[$label] Step $t/$nsteps: E=$(round(outputs.energy; sigdigits=4)), ||∇u||=$(round(sqrt(sum(abs2, ∇u)); sigdigits=4))"
        end

        return (; cfg = (; label), final_observables = spatial_observables(outputs, logT2s; SPWin, MPWin), outputs.ξ)
    end

    return (; op, results, T2s, logT2s, SPWin, MPWin) |> cpu
end

# Compare MWF, IEWF, CWF, and gmT2 maps across the spatially regularized (global κ) MAP,
# the unregularized (voxelwise) MAP, the independent-proposal MH posterior mean, and DECAES-GCV.
function plot_spatial_map_visualizations(; figpath = figuresdir("spatial_map_observables.pdf"), slice_aspect = 0.85)
    (; op, results, T2s, logT2s, SPWin, MPWin) = spatial_map_results
    if slice_aspect isa Real
        shape = size(xdist_results.mwfs.MH)
        slice_aspect *= (op.shape[2] / op.shape[1]) / (shape[2] / shape[1])
    end

    cols_spatial = [only(filter(r -> r.cfg.label == "global", results)), only(filter(r -> r.cfg.label == "unregularized", results))]
    col_titles = [L"Spatially Regularized $\mathrm{MAP}$", L"Unregularized $\mathrm{MAP}$"]

    xdist_mh = lib.unitsum(elbo_results["XDist_MH"]; dims = 1)
    mh_bm = compute_observable_maps(xdist_mh, logT2s; SPWin, MPWin)

    decaes_xdist_gcv = lib.unitsum(decaes_results.gcv.x.xdist; dims = 1)
    decaes_bm = compute_observable_maps(decaes_xdist_gcv, logT2s; SPWin, MPWin)

    ncols = length(cols_spatial) + 2
    fig = Figure(; size = (200 * ncols, 800))

    mwf_cr = (0.0f0, 0.25f0)
    iewf_cr = (0.75f0, 1.0f0)
    cwf_cr = (0.0f0, 0.5f0)
    gmT2_cr = (T2s[1], 200.0f-3)

    for (j, r) in enumerate(cols_spatial)
        ax1 = Axis(fig[1, j]; title = col_titles[j], ylabel = j == 1 ? L"\mathrm{MWF}" : "", aspect = slice_aspect, yreversed = true)
        heatmap!(ax1, to_slice(op, r.final_observables.mwf)'; colormap = :viridis, colorrange = mwf_cr)
        hidedecorations!(ax1; label = false)
        hidespines!(ax1)
        ax2 = Axis(fig[2, j]; ylabel = j == 1 ? L"\mathrm{IEWF}" : "", aspect = slice_aspect, yreversed = true)
        heatmap!(ax2, to_slice(op, r.final_observables.iewf)'; colormap = :viridis, colorrange = iewf_cr)
        hidedecorations!(ax2; label = false)
        hidespines!(ax2)
        ax3 = Axis(fig[3, j]; ylabel = j == 1 ? L"\mathrm{CWF}" : "", aspect = slice_aspect, yreversed = true)
        heatmap!(ax3, to_slice(op, r.final_observables.cwf)'; colormap = :viridis, colorrange = cwf_cr)
        hidedecorations!(ax3; label = false)
        hidespines!(ax3)
        ax4 = Axis(fig[4, j]; ylabel = j == 1 ? L"GM $T_2$" : "", aspect = slice_aspect, yreversed = true)
        heatmap!(ax4, to_slice(op, r.final_observables.gmT2)'; colormap = :magma, colorrange = gmT2_cr)
        hidedecorations!(ax4; label = false)
        hidespines!(ax4)
    end

    for (col_offset, (title, bm)) in enumerate([(L"\mathrm{MH}$$", mh_bm), (L"DECAES (GCV)$$", decaes_bm)])
        col = length(cols_spatial) + col_offset
        ax1 = Axis(fig[1, col]; title, aspect = slice_aspect, yreversed = true)
        heatmap!(ax1, to_slice(op, bm.mwf)'; colormap = :viridis, colorrange = mwf_cr)
        hidedecorations!(ax1)
        hidespines!(ax1)
        ax2 = Axis(fig[2, col]; aspect = slice_aspect, yreversed = true)
        heatmap!(ax2, to_slice(op, bm.iewf)'; colormap = :viridis, colorrange = iewf_cr)
        hidedecorations!(ax2)
        hidespines!(ax2)
        ax3 = Axis(fig[3, col]; aspect = slice_aspect, yreversed = true)
        heatmap!(ax3, to_slice(op, bm.cwf)'; colormap = :viridis, colorrange = cwf_cr)
        hidedecorations!(ax3)
        hidespines!(ax3)
        ax4 = Axis(fig[4, col]; aspect = slice_aspect, yreversed = true)
        heatmap!(ax4, to_slice(op, bm.gmT2)'; colormap = :magma, colorrange = gmT2_cr)
        hidedecorations!(ax4)
        hidespines!(ax4)
    end

    savefig(fig, figpath)
    return fig
end

# ξ fields before (unregularized MAP) and after (global-κ prior) smoothing
function plot_spatial_map_ilr_fields(; ncomponents = 4, figpath = figuresdir("spatial_map_ilr_fields.pdf"), slice_aspect = 0.85)
    (; op, results) = spatial_map_results
    κ = spatial_em_results.κ
    if slice_aspect isa Real
        shape = size(xdist_results.mwfs.MH)
        slice_aspect *= (op.shape[2] / op.shape[1]) / (shape[2] / shape[1])
    end

    ξ_indices = 1:ncomponents
    r_before = only(filter(r -> r.cfg.label == "unregularized", results))
    r_after = only(filter(r -> r.cfg.label == "global", results))

    ξ_before_all = [vec(lib.cpu(r_before.ξ[idx:idx, :])) for idx in ξ_indices]
    ξ_after_all = [vec(lib.cpu(r_after.ξ[idx:idx, :])) for idx in ξ_indices]
    cr = maximum(abs, reduce(vcat, [ξ_before_all; ξ_after_all])) .* (-1.0f0, 1.0f0)

    fig = Figure(; size = (175 * length(ξ_indices) + 100, 400))

    hm = nothing
    for (col, idx) in enumerate(ξ_indices)
        ax1 = Axis(fig[1, col]; title = L"\xi_{%$idx}$$", ylabel = col == 1 ? "Unregularized" : "", aspect = slice_aspect, yreversed = true)
        hm = heatmap!(ax1, to_slice(op, ξ_before_all[col])'; colormap = :RdBu, colorrange = cr)
        hidedecorations!(ax1; label = false)
        hidespines!(ax1)

        ax2 = Axis(fig[2, col]; ylabel = col == 1 ? L"Regularized ($\kappa = %$(round(κ; sigdigits=3))$)" : "", aspect = slice_aspect, yreversed = true)
        heatmap!(ax2, to_slice(op, ξ_after_all[col])'; colormap = :RdBu, colorrange = cr)
        hidedecorations!(ax2; label = false)
        hidespines!(ax2)
    end

    Colorbar(fig[1:2, length(ξ_indices)+1], hm; label = L"\xi$$")

    savefig(fig, figpath)
    return fig
end

#### Documentation figures

reg_label(reg) = reg == "lcurve" ? "L-Curve" : reg == "gcv" ? "GCV" : reg == "chi2" ? "Chi2" : uppercase(reg)

# Posterior mean and half-width of the central 95% interval.
function sample_matrix_mean_band(mat::AbstractMatrix)
    μ = dropdims(mean(mat; dims = 2); dims = 2)
    mat_sorted = sort(mat; dims = 2)
    band = similar(mat, size(mat, 1))
    Threads.@threads for i in axes(mat, 1)
        row = @view mat_sorted[i, :]
        band[i] = (quantile(row, 0.975; sorted = true) - quantile(row, 0.025; sorted = true)) / 2
    end
    return (; mean = μ, band)
end

# X = all_outputs.xdist: (n_samples, nT2, nvox); observable matrices keyed by field name.
function mala_posterior_bands(all_outputs, logT2s; SPWin = 32.0f-3, MPWin = 200.0f-3)
    X = all_outputs.xdist
    bms = [compute_observable_maps(lib.unitsum(X[s, :, :]; dims = 1), logT2s; SPWin, MPWin) for s in axes(X, 1)]
    mats = (; mwf = stack(bm.mwf for bm in bms), gmT2 = stack(bm.gmT2 for bm in bms))
    return map(sample_matrix_mean_band, mats)
end

# 2×8 field-data comparison: {MWF, gmT2} × {DECAES, MALA mean, MH mean, Difference, MALA unc, MH unc}.
# Colorbars: mean (col 1), difference (col 5), uncertainty (col 8)
function compare_inference_method_slices(;
    figpath = figuresdir("compare_inference_method_slices.pdf"),
    SPWin = xdist_results.SPWin, MPWin = xdist_results.MPWin,
    decaes_reg = :GCV,
    slice_aspect = 0.85,
)
    (; mwfs, gmT2s, logT2s, idx, xdists) = xdist_results
    bands_mala = mala_posterior_bands(mala_results.all_outputs, logT2s; SPWin, MPWin)
    bands_mh = mala_posterior_bands((; xdist = elbo_results["XDistSamples"]), logT2s; SPWin, MPWin)
    mwf_unc_lbl = L"$\mathrm{MWF}$ 95% CI Radius"
    gmt2_unc_lbl = L"GM $T_2$ 95% CI Radius (ms)"
    kw = (; aspect = slice_aspect, yreversed = true, spinewidth = 0)

    # MH posterior mean observables from running-mean xdist
    xdist_mh_mean = lib.unitsum(elbo_results["XDist_MH"]; dims = 1)
    bm_mh = compute_observable_maps(xdist_mh_mean, logT2s; SPWin, MPWin)

    # Shared uncertainty colorranges (99th pct on in-mask flat vectors, not NaN-padded slice grids).
    mwf_unc_range = (0.0f0, max(quantile(bands_mala.mwf.band, 0.99), quantile(bands_mh.mwf.band, 0.99)))
    gmt2_unc_range = 1000.0f0 .* (0.0f0, max(quantile(bands_mala.gmT2.band, 0.99), quantile(bands_mh.gmT2.band, 0.99)))

    # Difference color limits from flat observable maps (same in-mask voxels as `idx`).
    bm_decaes = compute_observable_maps(getfield(xdists, decaes_reg), logT2s; SPWin, MPWin)
    mwf_diff_lim = quantile(abs.(bm_decaes.mwf .- bm_mh.mwf), 0.99)
    gmT2_diff_lim = quantile(1000.0f0 .* abs.(bm_decaes.gmT2 .- bm_mh.gmT2), 0.99)

    # DECAES - MH difference maps (2D NaN-padded for display)
    mwf_diff = getfield(mwfs, decaes_reg) .- lib.unflatten_image(bm_mh.mwf, idx)
    gmT2_diff = 1000.0f0 .* (getfield(gmT2s, decaes_reg) .- lib.unflatten_image(bm_mh.gmT2, idx))

    fig = Figure(; size = (1400, 500))

    ax = Axis(fig[1, 2]; kw..., title = L"DECAES (%$(decaes_reg))$$")
    hm_mwf = heatmap!(ax, getfield(mwfs, decaes_reg)'; colormap = :viridis, colorrange = (0.0f0, 0.3f0))
    hidedecorations!(ax; label = false)
    hidespines!(ax)
    ax = Axis(fig[1, 3]; kw..., title = L"MALA$$")
    heatmap!(ax, mwfs.MALA'; colormap = :viridis, colorrange = (0.0f0, 0.3f0))
    hidedecorations!(ax)
    hidespines!(ax)
    ax = Axis(fig[1, 4]; kw..., title = L"MH$$")
    heatmap!(ax, lib.unflatten_image(bm_mh.mwf, idx)'; colormap = :viridis, colorrange = (0.0f0, 0.3f0))
    hidedecorations!(ax)
    hidespines!(ax)
    ax = Axis(fig[1, 5]; kw..., title = L"\text{DECAES} - \text{MH}")
    hm_mwf_diff = heatmap!(ax, mwf_diff'; colormap = :RdBu, colorrange = (-mwf_diff_lim, mwf_diff_lim))
    hidedecorations!(ax)
    hidespines!(ax)
    ax = Axis(fig[1, 7]; kw..., title = L"MH$$")
    hm_mwf_unc = heatmap!(ax, lib.unflatten_image(bands_mh.mwf.band, idx)'; colormap = :plasma, colorrange = mwf_unc_range)
    hidedecorations!(ax)
    hidespines!(ax)
    Colorbar(fig[1, 1], hm_mwf; label = L"\mathrm{MWF}", width = 15, ticklabelsize = 11, flipaxis = false)

    ax = Axis(fig[2, 2]; kw...)
    hm_gmt2 = heatmap!(ax, 1000.0f0 .* getfield(gmT2s, decaes_reg)'; colormap = :magma, colorrange = (0.0f0, 200.0f0))
    hidedecorations!(ax; label = false)
    hidespines!(ax)
    ax = Axis(fig[2, 3]; kw...)
    heatmap!(ax, 1000.0f0 .* gmT2s.MALA'; colormap = :magma, colorrange = (0.0f0, 200.0f0))
    hidedecorations!(ax)
    hidespines!(ax)
    ax = Axis(fig[2, 4]; kw...)
    heatmap!(ax, lib.unflatten_image(1000.0f0 .* bm_mh.gmT2, idx)'; colormap = :magma, colorrange = (0.0f0, 200.0f0))
    hidedecorations!(ax)
    hidespines!(ax)
    ax = Axis(fig[2, 5]; kw...)
    hm_gmt2_diff = heatmap!(ax, gmT2_diff'; colormap = :RdBu, colorrange = (-gmT2_diff_lim, gmT2_diff_lim))
    hidedecorations!(ax)
    hidespines!(ax)
    ax = Axis(fig[2, 7]; kw...)
    hm_gmt2_unc = heatmap!(ax, lib.unflatten_image(1000.0f0 .* bands_mh.gmT2.band, idx)'; colormap = :plasma, colorrange = gmt2_unc_range)
    hidedecorations!(ax)
    hidespines!(ax)
    Colorbar(fig[2, 1], hm_gmt2; label = L"GM $T_2$ (ms)", width = 15, ticklabelsize = 11, flipaxis = false)

    # Difference colorbars (col 6, right beside difference maps in col 5)
    Colorbar(fig[1, 6], hm_mwf_diff; label = L"\Delta \mathrm{MWF}", width = 15, ticklabelsize = 11)
    Colorbar(fig[2, 6], hm_gmt2_diff; label = L"$\Delta$ GM $T_2$ (ms)", width = 15, ticklabelsize = 11)

    # Uncertainty colorbar (col 8, right beside MH 95% CI radius map in col 7)
    Colorbar(fig[1, 8], hm_mwf_unc; label = mwf_unc_lbl, width = 15, ticklabelsize = 11)
    Colorbar(fig[2, 8], hm_gmt2_unc; label = gmt2_unc_lbl, width = 15, ticklabelsize = 11)

    resize_to_layout!(fig)
    savefig(fig, figpath)
    return fig
end

# Observable samples: (S, nT₂, nvox) → one (nvox, S) matrix per biomarker.
function bm_sample_mats(xdist_samps::AbstractArray{<:Real, 3}, logT2s, fields; SPWin, MPWin)
    bm_per_samp = [compute_observable_maps(lib.unitsum(xdist_samps[s, :, :]; dims = 1), logT2s; SPWin, MPWin) for s in axes(xdist_samps, 1)]
    return NamedTuple{Tuple(fields)}(ntuple(k -> stack(getfield(bm, fields[k]) for bm in bm_per_samp), length(fields)))
end

# Empirical coverage: fraction of truths inside their voxel's central α-credible interval.
function ci_coverage(bm_mat::AbstractMatrix, truth::AbstractVector, α::Real; weights = nothing)
    nvox = size(bm_mat, 1)
    lo_q, hi_q = (1 - α) / 2, (1 + α) / 2
    lo = similar(truth, nvox)
    hi = similar(truth, nvox)
    Threads.@threads for i in 1:nvox
        row = @view bm_mat[i, :]
        p = sortperm(row)
        if isnothing(weights)
            srow = row[p]
            lo[i] = quantile(srow, lo_q; sorted = true)
            hi[i] = quantile(srow, hi_q; sorted = true)
        else
            w = @view weights[i, :]
            lo[i] = weighted_quantile_sorted(row, w, lo_q, p)
            hi[i] = weighted_quantile_sorted(row, w, hi_q, p)
        end
    end
    return mean((truth .>= lo) .& (truth .<= hi))
end

# Invert the weighted empirical CDF at q * ∑ᵢwᵢ; weights must be nonnegative.
function weighted_quantile_sorted(vals::AbstractVector, weights::AbstractVector, q::Real, perm::AbstractVector)
    cumw = cumsum(weights[perm])
    idx = searchsortedfirst(cumw, q * cumw[end])
    return vals[perm[clamp(idx, 1, length(vals))]]
end

function synthetic_t2_distributions(batch; SNR, seed = 0)
    SPFrac_params = (; lo = 0.0, hi = 0.25)
    T21_params = (; mean = (; lo = 10e-3, hi = 25e-3), logstd = (; lo = 0.1, hi = 0.2))
    T22_params = (; mean = (; lo = 50e-3, hi = 100e-3), logstd = (; lo = 0.1, hi = 0.2))
    (; A, T2Range) = batch
    nTEmax, nT2, batchsize = size(A)
    T = eltype(A)

    # Draw on the CPU so a seed gives the same values on any device
    rng = seed === nothing ? lib.Random.default_rng() : lib.Xoshiro(seed)
    rand_uniform(p) = T(p.lo) .+ T(p.hi - p.lo) .* lib.arr_similar(A, rand(rng, T, 1, batchsize))

    # Logarithmically spaced T2 values
    logT2lo, logΔT2 = log.(T2Range[1:1, :]), log.(T2Range[2:2, :] ./ T2Range[1:1, :])
    unit_range = lib.arr_similar(A, collect(range(zero(T), one(T); length = nT2)))
    logT2 = @. logT2lo + logΔT2 * unit_range

    # Mixture of two Gaussian lobes in log(T2)
    SPFrac = rand_uniform(SPFrac_params)
    T21_μ, T21_σ = rand_uniform(T21_params.mean), rand_uniform(T21_params.logstd)
    T22_μ, T22_σ = rand_uniform(T22_params.mean), rand_uniform(T22_params.logstd)
    xdist1 = @. exp(-(logT2 - log(T21_μ))^2 / (2 * T21_σ^2))
    xdist1 ./= sum(xdist1; dims = 1)
    xdist2 = @. exp(-(logT2 - log(T22_μ))^2 / (2 * T22_σ^2))
    xdist2 ./= sum(xdist2; dims = 1)
    xdist = @. SPFrac * xdist1 + (1 - SPFrac) * xdist2

    # Nuisance parameters
    logs = lib.zeros_similar(A, 1, batchsize)
    logσ = lib.fill_similar(A, -lib.snr_to_log_noise_level(Float32(SNR)), 1, batchsize)

    return ex.inference_variables((; xdist, logs, logσ))
end

# Run all inference methods on synthetic T2 distributions at multiple SNRs.
# snr_results[SNR].decaes[reg]::{mwf,w1,w2}::{μ,σ}; similarly for direct/iw/mh/mala.
# rep_vox stores normalized forward signals + posterior samples for one representative voxel.
function run_synthetic_eval(; seed = 0, spectrum_source = :prior, SNRs = [30, 40, 50], n_voxels = nothing, SPWin = 32.0f-3, MPWin = 200.0f-3, decaes_regs = ["lcurve", "gcv", "chi2"], n_samples = 1000, warmup = 0)
    seed_all!(seed)
    batch = run_env.val_cohort
    _, nT2, _ = size(batch.A)
    logT2s = range(log(batch.T2Range[1, 1]), log(batch.T2Range[2, 1]); length = nT2)

    nvox = n_voxels === nothing ? size(batch.b, 2) : min(n_voxels, size(batch.b, 2))
    bs = map(x -> (x isa AbstractArray && last(size(x)) != nvox) ? x[.., 1:nvox] : x, batch) |> gpu

    ℰ, 𝒢, 𝒫 = run_env.models.enc, run_env.models.genatr, run_env.models.prior
    xdist_true = if spectrum_source === :bimodal
        synthetic_t2_distributions(bs |> cpu; SNR = maximum(SNRs), seed).xdist
    elseif spectrum_source === :prior
        prior = ex.recondition(cpu32(𝒫), (cpu(bs.θ),))
        ex.sample_similar_xdist(prior, cpu(bs.b)).xdist
    else
        error("Unknown synthetic spectrum source: $spectrum_source")
    end
    gt_bm = compute_observable_maps(xdist_true, logT2s; SPWin, MPWin)

    # Voxelwise W_p (nT2, nvox) → (nvox,); scaled to continuous integral over log-T2
    dlogT2 = step(logT2s)
    w1v(est) = vec(lib.wasserstein1(xdist_true, cpu(est); agg = identity, dims = 1)) .* dlogT2
    ms(v) = (; μ = mean(v), σ = std(v))
    mets(xd, bm) = (; mwf = ms(abs.(bm.mwf .- gt_bm.mwf)), gmT2 = ms(1000.0f0 .* abs.(bm.gmT2 .- gt_bm.gmT2)), w1 = ms(w1v(xd)))

    snr_results = Dict{Int, NamedTuple}()

    for SNR in SNRs
        @info "SNR = $SNR dB"
        seed_all!(seed + SNR) # noise and sampler draws at one SNR do not depend on the other SNRs evaluated
        logs = 0.0f0 # unit scale
        logσ = -lib.snr_to_log_noise_level(Float32(SNR)) # σ = exp10(-SNR/20)

        x_true = ex.InferenceVariables(gpu(xdist_true), lib.fill_similar(bs.b, logs, 1, nvox), lib.fill_similar(bs.b, logσ, 1, nvox))
        b_synth = ex.sample_similar(ex.forward_physics_model(x_true, bs.A, bs.mask, bs.δb), bs.b)

        # DECAES (all specified regularization methods)
        decaes_xdists, decaes_bms = Dict{String, Matrix{Float32}}(), Dict{String, NamedTuple}()
        for reg in decaes_regs
            out = ex.decaes_inference_variables(run_env.data_cfg, merge(bs, (; b = cpu(b_synth))); Reg = reg).x
            xd = isnothing(out) ? fill(1.0f0 / nT2, nT2, nvox) : lib.unitsum(out.xdist; dims = 1)
            decaes_xdists[reg] = xd
            decaes_bms[reg] = compute_observable_maps(xd, logT2s; SPWin, MPWin)
        end

        # Direct / IW / MH (single call)
        elbo_res = ex.run_elbo_estimate(ℰ, 𝒢, 𝒫, b_synth, bs.A, bs.mask, bs.δb, bs.θ; samples = n_samples, verbose = false)
        xdist_direct = lib.unitsum(cpu(elbo_res["XDist"]); dims = 1)
        xdist_iw = lib.unitsum(cpu(elbo_res["XDist_IS"]); dims = 1)
        xdist_mh = lib.unitsum(cpu(elbo_res["XDist_MH"]); dims = 1)

        # MALA (warmup + main run; save all samples for rep. voxel figure)
        samp_prob = ex.SamplingProblem(ex.MALA(), ℰ, 𝒢, 𝒫)
        mala_out = ex.run_sampling(samp_prob, b_synth, bs.A, bs.mask, bs.δb, bs.θ; iter = warmup + n_samples, warmup_iter = warmup, save_all_samples = true, verbose = false)
        xdist_mala = lib.unitsum(cpu(mala_out.mean_outputs.xdist); dims = 1)

        # Raw xdist sample arrays
        mala_xdist_samps = cpu(mala_out.all_outputs.xdist) # (n_mala, nT2, nvox)
        mh_xdist_samps = cpu(elbo_res["XDistSamples"]) # (n_elbo-warmup, nT2, nvox); independent-proposal MH chain
        iw_xdist_samps = cpu(elbo_res["XDistProposalSamples"]) # (n_elbo, nT2, nvox); raw flow proposals

        # IW weights: (n_elbo, nvox) normalized, transposed to (nvox, n_elbo) for per-voxel indexing
        iw_weights = Float32.(cpu(elbo_res["Weights_IS"])') # (nvox, n_elbo)

        # Pre-cache observable sample matrices (nvox, n_samples) for calibration and coverage
        bm_fields = (:mwf, :iewf, :mwT2, :gmT2)
        mala_bm = bm_sample_mats(mala_xdist_samps, logT2s, bm_fields; SPWin, MPWin)
        mh_bm = bm_sample_mats(mh_xdist_samps, logT2s, bm_fields; SPWin, MPWin)
        iw_bm = bm_sample_mats(iw_xdist_samps, logT2s, bm_fields; SPWin, MPWin)

        # 95% CI coverage scalars for table (re-use cached bm matrices; IW uses importance weights)
        cov95(bm_mat, truth) = ci_coverage(bm_mat, truth, 0.95f0)
        cov95w(bm_mat, truth) = ci_coverage(bm_mat, truth, 0.95f0; weights = iw_weights)

        # Direct coverage: raw flow proposals with equal weights (same samples as IW, but unweighted)
        direct_bm = iw_bm  # same proposal samples, just unweighted quantiles

        # Raw per-voxel data; voxel selection and forward signal computation in plot_synthetic_voxel
        A_cpu, mask_cpu, θ_cpu, TE_cpu, b_synth_cpu = (bs.A, bs.mask, bs.θ, bs.TE, b_synth) |> cpu

        snr_results[SNR] = (;
            decaes = Dict(reg => mets(decaes_xdists[reg], decaes_bms[reg]) for reg in decaes_regs),
            direct = mets(xdist_direct, compute_observable_maps(xdist_direct, logT2s; SPWin, MPWin)),
            iw = mets(xdist_iw, compute_observable_maps(xdist_iw, logT2s; SPWin, MPWin)),
            mh = mets(xdist_mh, compute_observable_maps(xdist_mh, logT2s; SPWin, MPWin)),
            mala = mets(xdist_mala, compute_observable_maps(xdist_mala, logT2s; SPWin, MPWin)),
            coverage = (;
                direct_mwf = cov95(direct_bm.mwf, gt_bm.mwf), direct_mwT2 = cov95(direct_bm.mwT2, gt_bm.mwT2), direct_gmT2 = cov95(direct_bm.gmT2, gt_bm.gmT2),
                iw_mwf = cov95w(iw_bm.mwf, gt_bm.mwf), iw_mwT2 = cov95w(iw_bm.mwT2, gt_bm.mwT2), iw_gmT2 = cov95w(iw_bm.gmT2, gt_bm.gmT2),
                mh_mwf = cov95(mh_bm.mwf, gt_bm.mwf), mh_mwT2 = cov95(mh_bm.mwT2, gt_bm.mwT2), mh_gmT2 = cov95(mh_bm.gmT2, gt_bm.gmT2),
                mala_mwf = cov95(mala_bm.mwf, gt_bm.mwf), mala_mwT2 = cov95(mala_bm.mwT2, gt_bm.mwT2), mala_gmT2 = cov95(mala_bm.gmT2, gt_bm.gmT2),
            ),
            all_vox = (;
                logT2s, A = A_cpu, mask = mask_cpu, θ = θ_cpu, TE = TE_cpu, b_synth = b_synth_cpu,
                xdist_true, decaes_xdists,
                samples = (;
                    mala = (; xdist = mala_xdist_samps, observables = mala_bm),
                    mh = (; xdist = mh_xdist_samps, observables = mh_bm),
                    iw = (; observables = iw_bm, weights = iw_weights), # skip xdist to save memory
                ),
            ),
        )
    end

    return (; snr_results, SNRs, logT2s, SPWin, MPWin, decaes_regs, spectrum_source, seed, n_samples, nvox, warmup)
end

# Write the synthetic results table as LaTeX.
# Rows: one group per SNR, each listing DECAES methods then Direct/IW/MH/MALA.
# Columns: MWF, GM T2 and W1 errors (μ ± σ), then MWF and GM T2 95% CI coverage
# Bold marks the lowest displayed mean error within each SNR group.
function write_synthetic_table(results = synth_eval_results;
    figpath = figuresdir("synthetic_table.tex"),
    show_std = true,
)
    (; snr_results, SNRs, decaes_regs) = results
    bayes_keys = [:direct, :iw, :mh, :mala]
    bayes_labels = ["Direct", "IW", "MH", "MALA"]
    nmethods = length(decaes_regs) + length(bayes_keys)

    # Store raw values in data matrix; formatting is done by the formatter.
    # Metric cells: (μ, σ) tuples; coverage cells: scalar fraction or nothing (for N/A).
    data = Matrix{Any}(undef, nmethods * length(SNRs), 6)
    bold_cells = Set{Tuple{Int, Int}}()
    error_digits(j) = j == 4 ? 2 : 1
    for (g, snr) in enumerate(SNRs)
        rows = (g-1)*nmethods .+ (1:nmethods)
        res = snr_results[snr]

        for (i, reg) in enumerate(decaes_regs)
            r = res.decaes[reg]
            data[rows[i], :] = [reg_label(reg), (100 * r.mwf.μ, 100 * r.mwf.σ), (r.gmT2.μ, r.gmT2.σ), (r.w1.μ, r.w1.σ), nothing, nothing]
        end

        for (i, (mk, ml)) in enumerate(zip(bayes_keys, bayes_labels))
            r = res[mk]
            cov(field) = getfield(res.coverage, Symbol(mk, :_, field))
            data[rows[length(decaes_regs)+i], :] = [ml, (100 * r.mwf.μ, 100 * r.mwf.σ), (r.gmT2.μ, r.gmT2.σ), (r.w1.μ, r.w1.σ), cov(:mwf), cov(:gmT2)]
        end

        # Compare displayed values so that rounding ties are bolded together
        for j in 2:4
            score = Dict(i => round(data[i, j][1]; digits = error_digits(j)) for i in rows)
            union!(bold_cells, ((i, j) for i in rows if score[i] == minimum(values(score))))
        end

    end

    # Formatter: convert raw cell values to LatexCell strings
    function cell_formatter(v, i, j)
        v isa String && return v
        isnothing(v) && return LatexCell("---")
        fixed(x) = @sprintf("%.*f", error_digits(j), x) # fixed decimals so 0.10 is not printed as 0.1
        s = v isa Tuple{<:Real, <:Real} ? (show_std ? "$(fixed(v[1])) \\pm $(fixed(v[2]))" : fixed(v[1])) :
            v isa Real ? "$(round(100 * v; digits=1))\\%" : # coverage (stored as fraction)
            error("Unknown cell value type: $(typeof(v))")
        s = (i, j) ∈ bold_cells ? "\\mathbf{$s}" : s
        return LatexCell(latexstring(s))
    end

    function render_table(io)
        println(io, raw"\begin{tabular}{clccccc}")
        println(io, raw"  \toprule")
        println(io, raw"   & Method & \multicolumn{3}{c}{Mean errors} & \multicolumn{2}{c}{95\% coverage} \\\\")
        println(io, raw"  \cmidrule(lr){3-5}\cmidrule(lr){6-7}")
        println(io, raw"   & & MWF (\%) & GM $T_2$ (ms) & $W_1$ & MWF & GM $T_2$ \\\\")
        println(io, raw"  \midrule")
        for (g, snr) in enumerate(SNRs)
            g > 1 && println(io, raw"  \midrule")
            for k in 1:nmethods
                i = (g-1)*nmethods + k
                label = k == 1 ? "\\multirow{$nmethods}{*}{\\rotatebox{90}{SNR = $snr dB}}" : ""
                cells = [j == 1 ? data[i, j] : cell_formatter(data[i, j], i, j).data for j in 1:6]
                println(io, "  ", join(vcat(label, cells), " & "), raw" \\\\")
            end
        end
        println(io, raw"  \bottomrule")
        println(io, raw"\end{tabular}")
    end
    save_table(_ -> sprint(render_table), figpath)
end

# Compare fixed-SNR coverage under two spectrum distributions with the model held fixed.
function plot_synthetic_prior_comparison(bimodal, prior; figpath = figuresdir("synthetic_prior_comparison.pdf"))
    @assert bimodal.SNRs == prior.SNRs
    methods = (:direct, :iw, :mh, :mala)
    labels = ["Direct", "IW", "MH", "MALA"]
    colors = Makie.wong_colors()[1:2]
    fields = (:mwf => L"MWF coverage (\%)", :gmT2 => L"GM $T_2$ coverage (\%)")
    fig = Figure(; size = (1000, 600), fontsize = 18)
    for (row, (field, ylabel)) in enumerate(fields), (col, snr) in enumerate(bimodal.SNRs)
        ax = Axis(fig[row, col]; title = row == 1 ? L"SNR $= %$snr$ dB" : "",
            xlabel = row == 2 ? "Method" : "", ylabel = col == 1 ? ylabel : "",
            xticks = (1:4, labels), xgridvisible = false, ygridvisible = false)
        values = [100 * getproperty(result.snr_results[s].coverage, Symbol(method, :_, field)) for result in (bimodal, prior) for s in bimodal.SNRs for method in methods]
        for (result, color) in zip((bimodal, prior), colors)
            p = [100 * getproperty(result.snr_results[snr].coverage, Symbol(method, :_, field)) for method in methods]
            scatterlines!(ax, 1:4, p; color, markersize = 8, linewidth = 2)
        end
        hlines!(ax, [95]; color = :gray40, linestyle = :dash, linewidth = 1)
        ylims!(ax, max(0, 5floor((minimum(values)-3)/5)), 101)
    end
    Legend(fig[3, 1:length(bimodal.SNRs)], [LineElement(; color, linewidth = 2) for color in colors],
        ["Bimodal spectra", "Learned-prior spectra"]; orientation = :horizontal, framevisible = false)
    savefig(fig, figpath)
    return fig
end

# Visual comparison for one representative synthetic voxel.
# Left column: signal decay (top) + residuals b_obs - b_fit (bottom).
# Center column: T2 distribution (top) + MWF posterior histogram with true value (bottom).
# Right column: legend.
# Shaded bands: 2.5-97.5% quantile over MALA posterior samples.
function plot_synthetic_voxel(;
    figpath = figuresdir("synthetic_voxel.pdf"),
    vox_snr = synth_eval_results.SNRs[end],
    vox = 1,
    decaes_regs = synth_eval_results.decaes_regs,
)
    d = synth_eval_results.snr_results[vox_snr].all_vox
    if isnothing(vox)
        # Representative voxel: true spectrum closest to a two-pool reference (MWF 0.2 at 15 ms, IEWF at 65 ms)
        target = 0.15f0 .* exp.(-((d.logT2s .- log(17.5f-3)) ./ 0.15f0) .^ 2 ./ 2) .+ 0.85f0 .* exp.(-((d.logT2s .- log(65.0f-3)) ./ 0.15f0) .^ 2 ./ 2)
        target ./= sum(target)
        vox = argmin(dropdims(sum(abs, cumsum(target) .- cumsum(d.xdist_true; dims = 1); dims = 1); dims = 1))
    end
    valid_vox = findall(d.mask[:, vox] .> 0)
    A_vox = d.A[valid_vox, :, vox]
    TEs_ms_vox = 1000.0f0 .* valid_vox .* d.TE[1, vox]
    mala_samps = d.samples.mala.xdist[:, :, vox]
    mala_samps ./= sum(mala_samps; dims = 2) # (n_samp, nT2)
    b_mala_samps = (A_vox * mala_samps')'
    b_mala_samps ./= b_mala_samps[:, 1:1] # (n_samp, nTE)
    b_true_noiseless = A_vox * d.xdist_true[:, vox]
    b_true_noiseless ./= b_true_noiseless[1]
    b_obs_norm = d.b_synth[valid_vox, vox]
    b_obs_norm ./= b_obs_norm[1]
    decaes_vox = Dict{String, NamedTuple}()
    for reg in decaes_regs
        bd = A_vox * d.decaes_xdists[reg][:, vox]
        decaes_vox[reg] = (; b = bd ./ bd[1], xdist = d.decaes_xdists[reg][:, vox])
    end
    r = (;
        logT2s = d.logT2s, TEs_ms = TEs_ms_vox,
        b = b_obs_norm, b_true_noiseless, b_mala_samps,
        xdist_mala_samps = mala_samps, xdist_true = d.xdist_true[:, vox], decaes = decaes_vox,
    )
    T2ms = 1000.0f0 .* exp.(r.logT2s)
    nT2 = length(r.logT2s)
    colors = Makie.wong_colors()
    nc = length(decaes_regs) # number of DECAES curves; MALA uses colors[nc+1]

    # MALA posterior quantiles over T2 bins and echo times
    q025(v) = quantile(v, 0.025f0)
    q975(v) = quantile(v, 0.975f0)
    xd_lo = [q025(r.xdist_mala_samps[:, k]) for k in 1:nT2]
    xd_hi = [q975(r.xdist_mala_samps[:, k]) for k in 1:nT2]
    xd_mean = vec(mean(r.xdist_mala_samps; dims = 1))
    b_lo = vec(mapslices(q025, r.b_mala_samps; dims = 1))
    b_hi = vec(mapslices(q975, r.b_mala_samps; dims = 1))
    b_mean = vec(mean(r.b_mala_samps; dims = 1))

    (; SPWin, MPWin) = synth_eval_results
    bm_true = compute_observable_maps(reshape(r.xdist_true, :, 1), r.logT2s; SPWin, MPWin)
    bm_samps = compute_observable_maps(r.xdist_mala_samps', r.logT2s; SPWin, MPWin)

    kw_ax = (; xgridvisible = false, ygridvisible = false)
    fig = Figure(; size = (1200, 500))
    ax1 = Axis(fig[1, 1]; kw_ax..., ylabel = L"Signal ($\mathrm{SNR} = %$vox_snr\ \mathrm{dB}$)")
    ax2 = Axis(fig[1, 2]; kw_ax..., xlabel = L"Echo time (ms)$$", ylabel = L"\text{True} - \text{Fit}")
    ax3 = Axis(fig[1, 3]; kw_ax..., xlabel = L"$T_2$ (ms)", ylabel = L"$T_2$ Amplitude", xscale = log10)
    ax4 = Axis(fig[2, 1:3][1, 1]; kw_ax..., xlabel = L"\mathrm{MWF}", ylabel = L"Density$$")
    ax5 = Axis(fig[2, 1:3][1, 2]; kw_ax..., xlabel = L"\mathrm{IEWF}")
    ax6 = Axis(fig[2, 1:3][1, 3]; kw_ax..., xlabel = L"GM $T_2$")
    ax7 = Axis(fig[2, 1:3][1, 4]; kw_ax..., xlabel = L"MW $T_2$")

    # Signal comparison: observed (dotted), noiseless true (solid black), DECAES, MALA
    lines!(ax1, r.TEs_ms, r.b_true_noiseless; color = :black)
    for (ci, reg) in enumerate(decaes_regs)
        lines!(ax1, r.TEs_ms, r.decaes[reg].b; color = colors[ci])
    end
    band!(ax1, r.TEs_ms, b_lo, b_hi; color = (colors[nc+1], 0.3f0))
    lines!(ax1, r.TEs_ms, b_mean; color = colors[nc+1])
    lines!(ax1, r.TEs_ms, r.b; color = :black, linestyle = :dot)

    # Residuals: b_true_noiseless - b_fit
    hlines!(ax2, [0]; color = :black, linestyle = :dash, linewidth = 0.8)
    for (ci, reg) in enumerate(decaes_regs)
        lines!(ax2, r.TEs_ms, r.b_true_noiseless .- r.decaes[reg].b; color = colors[ci])
    end
    band!(ax2, r.TEs_ms, r.b_true_noiseless .- b_hi, r.b_true_noiseless .- b_lo; color = (colors[nc+1], 0.3f0))
    lines!(ax2, r.TEs_ms, r.b_true_noiseless .- b_mean; color = colors[nc+1])

    # T2 spectrum: stems, lines for reconstructions
    for k in eachindex(r.xdist_true)
        lines!(ax3, [T2ms[k], T2ms[k]], [0.0f0, r.xdist_true[k]]; color = :black, linewidth = 1)
    end
    scatter!(ax3, T2ms, r.xdist_true; color = :black, markersize = 6)
    for (ci, reg) in enumerate(decaes_regs)
        lines!(ax3, T2ms, r.decaes[reg].xdist; color = colors[ci])
    end
    band!(ax3, T2ms, xd_lo, xd_hi; color = (colors[nc+1], 0.3f0))
    lines!(ax3, T2ms, xd_mean; color = colors[nc+1])

    # Legend in its own column (col 5)
    elems = vcat(
        [LineElement(; color = :black, linestyle = :dot)],
        [LineElement(; color = :black)],
        [LineElement(; color = colors[ci]) for ci in 1:nc],
        [[PolyElement(; color = (colors[nc+1], 0.3f0)), LineElement(; color = colors[nc+1])]],
        [[LineElement(; color = :black, linewidth = 2.5), MarkerElement(; color = :black, marker = :circle)]],
    )
    labels = vcat([L"Observed Signal$$", L"True Signal$$"], reg_label.(decaes_regs), [L"MALA$$", L"True $T_2$ Distbn."])
    Legend(fig[1, 4], elems, labels; framevisible = false)

    # DECAES observable point estimates for overlay on histograms
    bm_decaes = Dict(reg => compute_observable_maps(reshape(r.decaes[reg].xdist, :, 1), r.logT2s; SPWin, MPWin) for reg in decaes_regs)
    kw_legend = (; framevisible = true, framecolor = (:white, 0.0), backgroundcolor = (:white, 0.7), labelsize = 9)

    # MWF posterior histogram (ax4)
    cmwf = colors[1]
    hist!(ax4, bm_samps.mwf; bins = 40, color = (cmwf, 0.7), normalization = :pdf)
    density!(ax4, bm_samps.mwf; color = :transparent, strokecolor = cmwf, strokewidth = 2)
    vlines!(ax4, [bm_true.mwf[1]]; color = :black, linewidth = 2, label = L"True = %$(round(bm_true.mwf[1]; digits = 3))$$")
    for (ci, reg) in enumerate(decaes_regs)
        vlines!(ax4, [bm_decaes[reg].mwf[1]]; color = colors[ci], linewidth = 1.5, linestyle = :dash, label = L"%$(reg_label(reg)) = %$(round(bm_decaes[reg].mwf[1]; digits = 3))$$")
    end
    axislegend(ax4; position = :rt, kw_legend...)

    # IEWF posterior histogram (ax5)
    ciewf = colors[2]
    hist!(ax5, bm_samps.iewf; bins = 40, color = (ciewf, 0.7), normalization = :pdf)
    density!(ax5, bm_samps.iewf; color = :transparent, strokecolor = ciewf, strokewidth = 2)
    vlines!(ax5, [bm_true.iewf[1]]; color = :black, linewidth = 2, label = L"True = %$(round(bm_true.iewf[1]; digits = 3))$$")
    for (ci, reg) in enumerate(decaes_regs)
        vlines!(ax5, [bm_decaes[reg].iewf[1]]; color = colors[ci], linewidth = 1.5, linestyle = :dash, label = L"%$(reg_label(reg)) = %$(round(bm_decaes[reg].iewf[1]; digits = 3))$$")
    end
    axislegend(ax5; position = :lt, kw_legend...)

    # gmT2 posterior histogram (ax6)
    cgmT2 = colors[3]
    hist!(ax6, bm_samps.gmT2; bins = 40, color = (cgmT2, 0.7), normalization = :pdf)
    density!(ax6, bm_samps.gmT2; color = :transparent, strokecolor = cgmT2, strokewidth = 2)
    vlines!(ax6, [bm_true.gmT2[1]]; color = :black, linewidth = 2, label = L"True = %$(round(bm_true.gmT2[1]; digits = 3))$$")
    for (ci, reg) in enumerate(decaes_regs)
        vlines!(ax6, [bm_decaes[reg].gmT2[1]]; color = colors[ci], linewidth = 1.5, linestyle = :dash, label = L"%$(reg_label(reg)) = %$(round(bm_decaes[reg].gmT2[1]; digits = 3))$$")
    end
    axislegend(ax6; position = :rt, kw_legend...)

    # MW T2 posterior histogram (ax7)
    cmwT2 = colors[4]
    hist!(ax7, 1000.0f0 .* bm_samps.mwT2; bins = 40, color = (cmwT2, 0.7), normalization = :pdf)
    density!(ax7, 1000.0f0 .* bm_samps.mwT2; color = :transparent, strokecolor = cmwT2, strokewidth = 2)
    vlines!(ax7, [1000.0f0 * bm_true.mwT2[1]]; color = :black, linewidth = 2, label = L"True = %$(round(1000.0f0 * bm_true.mwT2[1]; digits = 1)) ms$$")
    for (ci, reg) in enumerate(decaes_regs)
        vlines!(ax7, [1000.0f0 * bm_decaes[reg].mwT2[1]]; color = colors[ci], linewidth = 1.5, linestyle = :dash, label = L"%$(reg_label(reg)) = %$(round(1000.0f0 * bm_decaes[reg].mwT2[1]; digits = 1)) ms$$")
    end
    axislegend(ax7; position = :rt, kw_legend...)

    resize_to_layout!(fig)
    savefig(fig, figpath)
    return fig
end

# Runs ELBO + MALA on a uniform sample from the held-out test_id cohort.
# Used by plot_psis_and_ess, plot_mcmc_diagnostics, plot_posterior_samples
# so that histograms / traces represent the held-out test_id cohort distribution, not just the slice.
function run_val_diagnostics(; seed = 0, n_samples = 1000, warmup = 0, n_voxels = nothing)
    seed_all!(seed)
    batch = run_env.val_cohort |> gpu
    n_voxels !== nothing && (batch = map(x -> x[.., 1:min(n_voxels, size(batch.b, 2))], batch))
    nvox = size(batch.b, 2)
    nT2 = run_env.cfg.data.nT2
    logT2s = @allowscalar range(log(batch.T2Range[1, 1]), log(batch.T2Range[2, 1]); length = nT2)
    ℰ, 𝒢, 𝒫 = run_env.models.enc, run_env.models.genatr, run_env.models.prior

    @info "Held-out diagnostics: ELBO ($nvox vox, $n_samples samples)..."
    elbo_time = @device_elapsed (elbo_res = ex.run_elbo_estimate(ℰ, 𝒢, 𝒫, batch.b, batch.A, batch.mask, batch.δb, batch.θ; samples = n_samples, verbose = false))

    @info "Held-out diagnostics: MALA ($warmup warmup + $n_samples samples)..."
    samp_prob = ex.SamplingProblem(ex.MALA(), ℰ, 𝒢, 𝒫)
    mala_time = @device_elapsed (mala_out = ex.run_sampling(samp_prob, batch.b, batch.A, batch.mask, batch.δb, batch.θ; iter = warmup + n_samples, warmup_iter = warmup, save_all_samples = true, verbose = false))

    return (; logT2s, elbo_res, mala_out, nvox, batch, elbo_time, mala_time, n_samples) |> cpu
end

# PSIS and IS diagnostics: voxelwise Pareto-k̂ histogram + spatial map.
# Spatial map from the held-out slice; histograms from the held-out test_id cohort
# k̂ < 0.5: reliable; 0.5-0.7: ok; > 0.7: unreliable.
function plot_psis_and_ess(; figpath = figuresdir("psis_diagnostic.pdf"))
    # PSIS weights and normalized ESS/S over the held-out test_id cohort sample.
    weights_diag = val_diag_results.elbo_res["Weights_IS"]
    k̂_diag, _ = lib.psis_diagnostic(sort(weights_diag; dims = 1))
    k̂_global, _ = lib.psis_diagnostic(sort(vec(weights_diag)))
    ess_diag = vec(val_diag_results.elbo_res["ESS_IS"])
    ess_xmin = max(0.0, quantile(ess_diag, 0.01))

    fig = Figure(; size = (800, 300))

    # Histogram of per-voxel k̂
    ax1 = Axis(fig[1, 1]; xlabel = L"\hat{k}", title = L"PSIS tail-index $\hat{k}$")
    hist!(ax1, vec(k̂_diag); bins = 50, color = :steelblue, normalization = :pdf)
    vlines!(ax1, [k̂_global]; color = :black, linewidth = 2, linestyle = :dot, label = L"Pooled $\hat{k} = %$(round(k̂_global; digits=3))$")
    vlines!(ax1, [0.5]; color = :orange, linestyle = :dash, linewidth = 1.5, label = L"Reliable $\hat{k} = 0.5$")
    vlines!(ax1, [0.7]; color = :red, linestyle = :dashdot, linewidth = 1.5, label = L"Threshold $\hat{k} = 0.7$")
    axislegend(ax1; position = :lt, framevisible = true, framecolor = (:white, 0.0), backgroundcolor = (:white, 0.7), labelsize = 10)

    # Normalized ESS histogram
    ax3 = Axis(fig[1, 2]; xlabel = L"\mathrm{ESS}/S", title = L"Normalized effective sample size ($\mathrm{ESS}/S$)")
    hist!(ax3, ess_diag; bins = 50, color = :steelblue, normalization = :pdf)
    vlines!(ax3, [mean(ess_diag)]; color = :black, linewidth = 2, linestyle = :dot, label = L"Mean $\mathrm{ESS}/S = %$(round(mean(ess_diag); digits=2))$")
    xlims!(ax3, ess_xmin, 1.0)
    axislegend(ax3; position = :lt, framevisible = true, framecolor = (:white, 0.0), backgroundcolor = (:white, 0.7), labelsize = 10)

    # Scatter: per-voxel ESS/S vs k̂ (held-out test_id cohort val_diag sample)
    ax2 = Axis(fig[1, 3]; xlabel = L"\mathrm{ESS}/S", ylabel = L"\hat{k}", title = L"$\mathrm{ESS}/S$ vs.\ $\hat{k}$", xgridvisible = false, ygridvisible = false)
    scatter!(ax2, ess_diag, vec(k̂_diag); color = (:steelblue, 0.5), markersize = 5, strokewidth = 0)
    hist2d = StatsBase.fit(StatsBase.Histogram, (ess_diag, vec(k̂_diag)); nbins = 30)
    contour!(ax2, StatsBase.midpoints(hist2d.edges[1]), StatsBase.midpoints(hist2d.edges[2]), Float32.(hist2d.weights); levels = 15)
    xlims!(ax2, ess_xmin, 1.0)
    hlines!(ax2, [0.5]; color = :orange, linestyle = :dash, linewidth = 1.5)
    hlines!(ax2, [0.7]; color = :red, linestyle = :dashdot, linewidth = 1.5)

    resize_to_layout!(fig)
    savefig(fig, figpath)
    return fig
end

# MALA + MH convergence diagnostics.
# 2-row layout (MALA top | MH bottom), 3 columns:
#   Col 1: acceptance rate histogram + density  (held-out test_id cohort val_diag sample)
#   Col 2: observable τ histograms; MH also shows its acceptance-sequence τ
#   Col 3: MWF chain traces with cumulative mean at 3 quantile voxels
function plot_mcmc_diagnostics(; figpath = figuresdir("mcmc_diagnostics.pdf"))
    (; SPWin, MPWin) = xdist_results
    ao_diag = val_diag_results.mala_out.all_outputs
    logT2s_diag = val_diag_results.logT2s
    colors = Makie.wong_colors()
    kw_ax = (; xgridvisible = false, ygridvisible = false)

    # Acceptance rates (held-out test_id cohort)
    mala_accrat = vec(dropdims(mean(ao_diag.α; dims = 1); dims = 1))
    mh_accrat = vec(val_diag_results.elbo_res["AccRat"])

    # Autocorrelation times

    # MH: per-voxel τ from acceptance-sequence autocorrelation
    τ_mh = dropdims(val_diag_results.elbo_res["AutoCorrTime"]; dims = 1) # (nvox,)

    # MWF traces: 3 voxels at 25/50/75th percentile of mean MWF
    xdist_mala = ao_diag.xdist  # (window, nT2, nvox)
    xdist_mh = val_diag_results.elbo_res["XDistSamples"] # (mh_steps, nT2, nvox)
    mean_mwf = compute_observable_maps(dropdims(mean(xdist_mala; dims = 1); dims = 1), logT2s_diag; SPWin, MPWin).mwf
    n = length(mean_mwf)
    trace_vox = sortperm(mean_mwf)[[clamp(round(Int, p * n), 1, n) for p in (0.25, 0.50, 0.75)]]
    window_mala, window_mh = size(xdist_mala, 1), size(xdist_mh, 1)
    mala_traces = [compute_observable_maps(lib.unitsum(view(xdist_mala, :, :, v)'; dims = 1), logT2s_diag; SPWin, MPWin).mwf for v in trace_vox]
    mh_traces = [compute_observable_maps(lib.unitsum(view(xdist_mh, :, :, v)'; dims = 1), logT2s_diag; SPWin, MPWin).mwf for v in trace_vox]

    # Observable τ from xdist chains (uses top-level chain_observable_autocorr_times utility)
    τ_mala_bm = chain_observable_autocorr_times(xdist_mala, logT2s_diag; SPWin, MPWin)
    τ_mh_bm = chain_observable_autocorr_times(xdist_mh, logT2s_diag; SPWin, MPWin)
    bm_colors = (; mwf = :orange, iewf = :green, cwf = :red, gmT2 = :purple)

    fig = Figure(; size = (800, 450))

    # Acceptance rate histograms
    ax1a = Axis(fig[1, 1]; kw_ax..., xlabel = L"Acceptance ratio$$", ylabel = L"Density$$", title = L"MALA Acceptance Ratio$$")
    hist!(ax1a, mala_accrat; bins = 50, color = (:steelblue, 0.5), normalization = :pdf)
    density!(ax1a, mala_accrat; color = :transparent, strokecolor = :steelblue, strokewidth = 1.5)
    vlines!(ax1a, [0.574]; color = :red, linestyle = :dash, linewidth = 1.5, label = L"Theory = 0.574$$")
    vlines!(ax1a, [mean(mala_accrat)]; color = :black, linestyle = :dot, linewidth = 1.5, label = L"Mean = %$(round(mean(mala_accrat); digits = 3))$$")
    axislegend(ax1a; position = :lt, framevisible = true, framecolor = (:white, 0.0), backgroundcolor = (:white, 0.7), labelsize = 12)

    ax1b = Axis(fig[2, 1]; kw_ax..., xlabel = L"Acceptance ratio$$", ylabel = L"Density$$", title = L"MH Acceptance Ratio$$")
    hist!(ax1b, mh_accrat; bins = 50, color = (:orangered, 0.5), normalization = :pdf)
    density!(ax1b, mh_accrat; color = :transparent, strokecolor = :orangered, strokewidth = 1.5)
    vlines!(ax1b, [mean(mh_accrat)]; color = :black, linestyle = :dot, linewidth = 1.5, label = L"Mean = %$(round(mean(mh_accrat); digits = 3))$$")
    axislegend(ax1b; position = :lt, framevisible = true, framecolor = (:white, 0.0), backgroundcolor = (:white, 0.7), labelsize = 12)

    # Autocorrelation time histograms with density overlays

    # MALA: MWF/IEWF/CWF/gmT2 observable τ
    ax2a = Axis(fig[1, 2]; kw_ax..., xlabel = L"$\tau^{\mathrm{int}}$ (steps)", ylabel = L"Density$$", title = L"MALA Autocorrelation Time $\tau^{\mathrm{int}}$")
    mala_τ_entries = [
        (vec(τ_mala_bm.mwf), :mwf, L"\tau^{\mathrm{int}}_{\mathrm{MWF}}"),
        (vec(τ_mala_bm.iewf), :iewf, L"\tau^{\mathrm{int}}_{\mathrm{IEWF}}"), (vec(τ_mala_bm.cwf), :cwf, L"\tau^{\mathrm{int}}_{\mathrm{CWF}}"), (vec(τ_mala_bm.gmT2), :gmT2, L"\tau^{\mathrm{int}}_{\text{GM } T_2}"),
    ]
    for (data, sym, lbl) in mala_τ_entries
        c = bm_colors[sym]
        hist!(ax2a, min.(data, 7.5f0); bins = 50, color = (c, 0.4), normalization = :pdf, label = lbl)
        density!(ax2a, min.(data, 7.5f0); color = :transparent, strokecolor = c, strokewidth = 1.5)
    end
    axislegend(ax2a; position = :rt, framevisible = false, labelsize = 12)

    # MH: acceptance-sequence τ + MWF/IEWF/CWF/gmT2 observable τ from xdist chain
    ax2b = Axis(fig[2, 2]; kw_ax..., xlabel = L"$\tau^{\mathrm{int}}$ (steps)", ylabel = L"Density$$", title = L"MH Autocorrelation Time $\tau^{\mathrm{int}}$")
    hist!(ax2b, vec(τ_mh); bins = 50, color = (:gray, 0.4), normalization = :pdf, label = L"\tau^{\mathrm{int}}_{\text{Acc.}}")
    density!(ax2b, vec(τ_mh); color = :transparent, strokecolor = :gray, strokewidth = 1.5)
    for (data, sym, lbl) in [
        (vec(τ_mh_bm.mwf), :mwf, L"\tau^{\mathrm{int}}_{\mathrm{MWF}}"), (vec(τ_mh_bm.iewf), :iewf, L"\tau^{\mathrm{int}}_{\mathrm{IEWF}}"),
        (vec(τ_mh_bm.cwf), :cwf, L"\tau^{\mathrm{int}}_{\mathrm{CWF}}"), (vec(τ_mh_bm.gmT2), :gmT2, L"\tau^{\mathrm{int}}_{\text{GM } T_2}"),
    ]
        c = bm_colors[sym]
        hist!(ax2b, min.(data, 2.0f0); bins = 50, color = (c, 0.4), normalization = :pdf, label = lbl)
        density!(ax2b, min.(data, 2.0f0); color = :transparent, strokecolor = c, strokewidth = 1.5)
    end
    axislegend(ax2b; position = :rt, framevisible = false, labelsize = 12)

    # MWF traces + cumulative mean (demonstrates ergodic convergence)
    ax3a = Axis(fig[1, 3]; kw_ax..., xlabel = L"Step$$", ylabel = L"\mathrm{MWF}", title = L"MALA MWF Traces$$")
    for (i, tr) in enumerate(mala_traces)
        lines!(ax3a, 1:window_mala, tr; color = (colors[i], 0.25), linewidth = 0.6)
        lines!(ax3a, 1:window_mala, cummean(tr); color = colors[i], linewidth = 1.5)
    end

    ax3b = Axis(fig[2, 3]; kw_ax..., xlabel = L"Step$$", ylabel = L"\mathrm{MWF}", title = L"MH MWF Traces$$")
    for (i, tr) in enumerate(mh_traces)
        lines!(ax3b, 1:window_mh, tr; color = (colors[i], 0.25), linewidth = 0.6)
        lines!(ax3b, 1:window_mh, cummean(tr); color = colors[i], linewidth = 1.5)
    end

    resize_to_layout!(fig)
    savefig(fig, figpath)
    return fig
end

# Each NUTS chain runs as a task; density requests from all chains are pooled into one batched gradient.
struct PooledPosterior
    id::Int
    dim::Int
    requests::Channel{Tuple{Int, Vector{Float64}}}
    response::Channel{Tuple{Float64, Vector{Float64}}}
end
LDP.dimension(ℓ::PooledPosterior) = ℓ.dim
LDP.capabilities(::Type{PooledPosterior}) = LDP.LogDensityOrder{1}()
function LDP.logdensity_and_gradient(ℓ::PooledPosterior, u)
    put!(ℓ.requests, (ℓ.id, Vector{Float64}(u)))
    return take!(ℓ.response) # (log p(u|b), ∇u log p(u|b))
end
LDP.logdensity(ℓ::PooledPosterior, u) = first(LDP.logdensity_and_gradient(ℓ, u))

# Worst-mixing-coordinate integrated autocorrelation time of a (dim, n) chain; matches the MALA ablation's τ_u.
chain_tau(u) = maximum(max.(dropdims(lib.autocorr_time(lib.autocorr_fft(permutedims(u); dims = 1); dims = 1); dims = 1), 1.0))

# Per-voxel and aggregate mixing diagnostics for a `run_nuts_baseline` or `run_hmc_baseline` result.
function print_baseline_summary(label, result)
    (; runs, meta) = result
    failures = hasproperty(meta, :divergences) ? (; meta.divergences) : (; meta.numerical_error_iterations)
    @info "$label baseline: $(length(runs)) voxel(s), $(meta.nchains) chains × $(meta.n_samples) samples" failures mean_accept = mean(r.accept for r in runs) ess_per_gradient = sum(r.ess_u for r in runs) / sum(r.grad_evals for r in runs)
    return nothing
end

# NUTS targets the same physical posterior with either backbone and an adapted diagonal metric.
# `NUTS(δ; metric)` builds the multinomial generalised-NoUTurn kernel with Stan windowed adaptation.
# Chains run as cooperative tasks with pooled gradients; seeds depend only on (voxel, chain).
function nuts_chains(
    batch;
    seed = 0,
    backbone::Symbol = :flow,
    voxels = 1:1,
    nchains::Int = 8,
    n_samples::Int = 1000,
    n_adapts::Int = n_samples,
    max_depth::Int = 10,
)
    @assert backbone in (:flow, :id_window)
    chains = vec(collect(Iterators.product(eachindex(voxels), 1:nchains)))
    metric, target_accept = :diagonal, 0.8
    dev = lib.USE_GPU[] ? gpu64 : cpu64
    ℰ, G, 𝒫 = dev(run_env.models.enc), dev(run_env.models.genatr), dev(run_env.models.prior)
    backbone === :id_window && (G = ex.identity_backbone_with_window_flow(G))
    prob = ex.MAPInferenceProblem(ℰ, G, 𝒫)
    udim = G.opts.udim
    sampler = AHMC.NUTS(Float64(target_accept); metric, max_depth)
    chain_seeds = rand(Xoshiro(seed), UInt64, length(voxels), nchains)

    # Conditioning columns, one per chain.
    columns = [voxels[vi] for (vi, _) in chains]
    cond = map(d -> dev(d[.., columns]), (batch.b, batch.A, batch.mask, batch.δb, batch.θ))
    z = backbone === :id_window ? lib.zeros_similar(cond[1], 0, length(columns)) : lib.chunked_apply(ℰ, cond[1], cond[2], cond[3], cond[5]; chunksize = INFERENCE_CHUNKSIZE)
    latents = (z, cond...)

    requests = Channel{Tuple{Int, Vector{Float64}}}(Inf)
    responses = [Channel{Tuple{Float64, Vector{Float64}}}(1) for _ in chains]
    results = Vector{Any}(undef, length(chains))
    done = Ref(0)
    t_start = time()
    println(stderr, "NUTS: $(length(chains)) chain(s) over $(length(voxels)) voxels, $n_adapts warmup + $n_samples samples")
    flush(stderr)
    tasks = map(enumerate(chains)) do (k, (vi, c))
        @async try
            rng = Xoshiro(chain_seeds[vi, c])
            model = AHMC.LogDensityModel(PooledPosterior(k, udim, requests, responses[k]))
            transitions = AHMC.sample(rng, model, sampler, n_adapts + n_samples; n_adapts, initial_params = randn(rng, Float64, udim), progress = false)
            warmup = @view transitions[1:n_adapts]
            chain = @view transitions[(n_adapts+1):end]
            st = [t.stat for t in chain]
            results[k] = (;
                u = reduce(hcat, t.z.θ for t in chain),
                grad_evals = sum(s.n_steps for s in st),
                accept = mean(s.acceptance_rate for s in st),
                tree_depth = mean(s.tree_depth for s in st),
                max_tree_depth = maximum(s.tree_depth for s in st),
                tree_depth_saturated = count(s -> s.tree_depth == max_depth, st),
                divergences = count(s -> s.numerical_error, st),
                warmup_grad_evals = sum((t.stat.n_steps for t in warmup); init = 0),
                warmup_divergences = count(t -> t.stat.numerical_error, warmup),
                warmup_tree_depth_saturated = count(t -> t.stat.tree_depth == max_depth, warmup),
            )
            done[] += 1
            el = time() - t_start
            @info @sprintf("NUTS baseline: chain %d/%d done | elapsed %.1fs, ETA %.1fs", done[], length(chains), el, el / done[] * (length(chains) - done[]))

        finally
            put!(requests, (k, Float64[])) # an empty request marks a finished chain
        end
    end

    active, pending = length(chains), Tuple{Int, Vector{Float64}}[]
    while active > 0
        k, u = take!(requests)
        isempty(u) ? (active -= 1) : push!(pending, (k, u))
        (isempty(pending) || length(pending) < active) && continue
        ks = first.(pending)
        losses, ∇u = lib.chunked_apply(dev(reduce(hcat, last.(pending))), map(d -> d[.., ks], latents)...; chunksize = INFERENCE_CHUNKSIZE) do _u, c...
            ex.∇inference_loss(prob, _u, c...)
        end
        ℓ, g = cpu(-vec(losses.neglogpostu)), cpu(-∇u)
        foreach(((j, k),) -> put!(responses[k], (ℓ[j], g[:, j])), enumerate(ks))
        empty!(pending)
    end
    foreach(wait, tasks)
    return Dict(zip(chains, results))
end

# Assemble each voxel's chains: map u-draws to x̄ through the marginal posterior, then diagnose
# `grad_evals` sums the leapfrog steps for an ESS-per-gradient comparison against MALA.
function assemble_nuts(batch, chains::AbstractDict; seed = 0, backbone::Symbol = :flow, voxels = 1:1, nchains::Int = 8, n_samples::Int = 1000, n_adapts::Int = n_samples, max_depth::Int = 10)
    @assert keys(chains) == Set(Iterators.product(eachindex(voxels), 1:nchains)) "Incomplete NUTS chains"
    G = cpu64(run_env.models.genatr)
    backbone === :id_window && (G = ex.identity_backbone_with_window_flow(G))
    udim = G.opts.udim
    ℰ = cpu64(run_env.models.enc)
    runs = map(eachindex(voxels)) do vi
        v = voxels[vi]
        cs = [chains[(vi, c)] for c in 1:nchains]
        slice(d) = cpu64(d[.., v:v])
        repn(d) = repeat(slice(d), ntuple(i -> i < ndims(d) ? 1 : n_samples, ndims(d))...)
        z = backbone === :id_window ? zeros(Float64, 0, 1) : ℰ(slice(batch.b), slice(batch.A), slice(batch.mask), slice(batch.θ))
        mp = ex.marginal_posterior(G, (repeat(z, 1, n_samples), repn(batch.b), repn(batch.A), repn(batch.mask), repn(batch.δb), repn(batch.θ)))
        τs = [chain_tau(ch.u) for ch in cs]
        U = stack(ch.u for ch in cs)
        X = stack(exp.(ex.flow_forward(mp, ch.u[1:(udim-2), :]).x) for ch in cs)
        logT2s = range(log(cpu(batch.T2Range)[1, v]), log(cpu(batch.T2Range)[2, v]); length = size(X, 1))
        diagnostics = reference_posterior_diagnostics(U, X, logT2s)
        (;
            v, nchains,
            u = reshape(U, udim, :),
            xdist = reshape(X, size(X, 1), :),
            diagnostics,
            τ_u = mean(τs),
            ess_u = sum(n_samples ./ (2 .* τs)), # ESS^int = S/(2τ^int); see `autocorr_time`'s docstring for the τ^int convention
            grad_evals = sum(ch.grad_evals for ch in cs),
            accept = mean(ch.accept for ch in cs),
            divergences = sum(ch.divergences for ch in cs),
            max_tree_depth = maximum(ch.max_tree_depth for ch in cs),
            tree_depth_saturated = sum(ch.tree_depth_saturated for ch in cs),
            warmup_grad_evals = sum(ch.warmup_grad_evals for ch in cs),
            warmup_divergences = sum(ch.warmup_divergences for ch in cs),
            warmup_tree_depth_saturated = sum(ch.warmup_tree_depth_saturated for ch in cs),
        )
    end
    tree_depth = mean(ch.tree_depth for ch in values(chains))
    divergences = sum(ch.divergences for ch in values(chains))
    result = (; runs, meta = (; seed, backbone, metric = :diagonal, voxels, nchains, n_adapts, n_samples, target_accept = 0.8, max_depth, udim, tree_depth, divergences))
    print_baseline_summary("NUTS", result)
    return result
end

run_nuts_baseline(batch; kwargs...) = assemble_nuts(batch, nuts_chains(batch; kwargs...); kwargs...)

# HMC baseline: all `voxels × nchains` chains as columns of one matrix (AdvancedHMC vectorised mode),
# each with its own adapted step size and diagonal metric; gradients are batched on the GPU in Float32.
# A static multinomial trajectory is required, so NUTS is unavailable here.
function run_hmc_baseline(
    batch;
    seed = 0,
    voxels = 1:1,
    nchains::Int = 500,
    backbone::Symbol = :id_window,
    leapfrog::Int = 16,
    n_samples::Int = 1000,
    n_adapts::Int = n_samples,
)
    @assert backbone in (:flow, :id_window)
    metric, init_ϵ, target_accept = :diagonal, 0.1, 0.8
    ℰ, 𝒢, 𝒫 = run_env.models.enc, run_env.models.genatr, run_env.models.prior
    G = backbone === :flow ? 𝒢 : ex.identity_backbone_with_window_flow(𝒢)
    prob = ex.MAPInferenceProblem(ℰ, G, 𝒫)
    udim = G.opts.udim
    nvox = length(voxels)
    N = nvox * nchains # columns ordered voxel-fastest: column k ↔ voxel voxels[((k-1) % nvox)+1], chain ((k-1) ÷ nvox)+1

    # Tile each voxel `nchains` times across columns; the encoder runs once on the full batch
    repn(d) = repeat(gpu(d[.., voxels]), ntuple(i -> i < ndims(d) ? 1 : nchains, ndims(d))...)
    z = backbone === :id_window ? lib.zeros_similar(batch.b, 0, N) : lib.chunked_apply(ℰ, repn(batch.b), repn(batch.A), repn(batch.mask), repn(batch.θ); chunksize = INFERENCE_CHUNKSIZE)
    latents = (z, repn(batch.b), repn(batch.A), repn(batch.mask), repn(batch.δb), repn(batch.θ))

    # Sampler state stays on the CPU; only the batched gradient runs on the GPU
    function ℓπ(θ)
        losses = lib.chunked_apply(gpu32(θ), latents...; chunksize = INFERENCE_CHUNKSIZE) do _u, cond...
            ex.inference_losses(prob, _u, cond...)
        end
        return cpu64(vec(-losses.neglogpostu))
    end # (N,) +log p(u|b)
    function ∇ℓπ(θ)
        losses, ∇u = lib.chunked_apply(gpu32(θ), latents...; chunksize = INFERENCE_CHUNKSIZE) do _u, cond...
            ex.∇inference_loss(prob, _u, cond...)
        end
        return cpu64(vec(-losses.neglogpostu)), cpu64(-∇u) # ((N,), (udim, N))
    end

    rng = Xoshiro(seed)
    θ₀ = randn(rng, Float64, udim, N)
    m = AHMC.DiagEuclideanMetric(Float64, (udim, N))
    integ = AHMC.Leapfrog(fill(Float64(init_ϵ), N)) # per-column step sizes ⇒ vectorised dual-averaging accepts the per-column acceptance vector
    ham = AHMC.Hamiltonian(m, ℓπ, ∇ℓπ)
    kernel = AHMC.HMCKernel(AHMC.Trajectory{AHMC.MultinomialTS}(integ, AHMC.FixedNSteps(leapfrog)))
    adaptor = AHMC.StanHMCAdaptor(AHMC.MassMatrixAdaptor(m), AHMC.StepSizeAdaptor(Float64(target_accept), integ))
    println(stderr, "HMC: $nvox voxels × $nchains chains × $leapfrog leapfrog, $n_adapts warmup + $n_samples samples")
    flush(stderr)
    θs, stats = @time AHMC.sample(rng, ham, kernel, θ₀, n_adapts + n_samples, adaptor, n_adapts; drop_warmup = true, progress = false, verbose = false)

    # Regroup the N columns into (voxel, chain), voxel-fastest as tiled
    Xiter = map(θs) do θ
        logx = lib.chunked_apply(gpu32(θ[1:(udim-2), :]), latents...; chunksize = INFERENCE_CHUNKSIZE) do _u, cond...
            ex.flow_forward(ex.marginal_posterior(G, cond), _u).x
        end
        return exp.(cpu(logx))
    end # n_samples × (nT2, N)
    nT2 = size(first(Xiter), 1)
    X4 = reshape(stack(Xiter; dims = 3), nT2, nvox, nchains, n_samples)
    U4 = reshape(stack(θs; dims = 3), udim, nvox, nchains, n_samples) # already CPU
    Acc = reshape(reduce(hcat, [s.acceptance_rate for s in stats]), nvox, nchains, n_samples)

    runs = map(1:nvox) do i
        τs = [chain_tau(U4[:, i, c, :]) for c in 1:nchains]
        logT2s = range(log(cpu(batch.T2Range)[1, voxels[i]]), log(cpu(batch.T2Range)[2, voxels[i]]); length = nT2)
        diagnostics = reference_posterior_diagnostics(permutedims(U4[:, i, :, :], (1, 3, 2)), permutedims(X4[:, i, :, :], (1, 3, 2)), logT2s)
        (;
            v = voxels[i], nchains,
            diagnostics,
            u = reshape(U4[:, i, :, :], udim, nchains * n_samples),
            xdist = reshape(X4[:, i, :, :], nT2, nchains * n_samples),
            τ_u = mean(τs),
            ess_u = sum(n_samples ./ (2 .* τs)), # ESS^int = S/(2τ^int)
            grad_evals = nchains * leapfrog * n_samples,
            accept = mean(@view Acc[i, :, :]),
        )
    end
    result = (; runs, meta = (; seed, backbone, metric, voxels = voxels, nchains, leapfrog, n_adapts, n_samples, target_accept, udim, numerical_error_iterations = count(s -> s.numerical_error, stats)))
    print_baseline_summary("HMC", result)
    return result
end

#### Reference sampling cache

# HMC/NUTS references are cached by checkpoint, voxels, inputs and sampler settings.
const REFERENCE_DIR = Ref{Union{Nothing, String}}(nothing)
reference_path(name) = lib.cachepath(joinpath(something(REFERENCE_DIR[], figuresdir("reference-samples")), "$(name).jld2"))

function reference_key(batch, voxels; settings...)
    inputs = Dict(string(k) => Array{Float32}(cpu(getproperty(batch, k)[.., voxels])) for k in (:b, :A, :mask, :δb, :θ))
    return Dict{String, Any}("checkpoint" => bytes2hex(open(SHA.sha256, joinpath(run_env.dir, "models.jld2"))), "voxels" => collect(voxels), "inputs" => inputs, "T2Range" => Array{Float32}(cpu(batch.T2Range[:, voxels])), "settings" => Dict(string(k) => v for (k, v) in settings), "diagnostics_version" => 2)
end

function matching_keys(a, b)
    a["checkpoint"] == b["checkpoint"] && a["voxels"] == b["voxels"] && a["settings"] == b["settings"] && get(a, "diagnostics_version", 0) == b["diagnostics_version"] && a["T2Range"] == b["T2Range"] || return false
    return all(isapprox(a["inputs"][k], b["inputs"][k]; rtol = 1.0f-5) for k in keys(b["inputs"]))
end

# Raw u-draws are not stored; pooled spectra are thinned to at most `maxdraws` per voxel
function strip_draws(result; maxdraws = 16000)
    thin(x) = x[:, 1:cld(size(x, 2), maxdraws):end]
    return (; runs = [merge(Base.structdiff(r, NamedTuple{(:u,)}), (; xdist = thin(r.xdist))) for r in result.runs], result.meta)
end

# NUTS keys omit the chain count, which is checked separately against `nchains`
function cached_reference(compute, name, key; nchains::Int)
    path = reference_path(name)
    if isfile(path)
        saved, result = lib.JLD2.load(path, "key", "result")
        @assert matching_keys(saved, key) "Saved $(name) reference does not match this run: $(path)"
        @assert result.meta.nchains == nchains "Saved $(name) reference has $(result.meta.nchains) chains per voxel, expected $(nchains): $(path)"
        @info "Loaded $(name) reference: $(path) ($(result.meta.nchains) chains per voxel)"
        return result
    end
    result = strip_draws(compute())
    mkpath(dirname(path))
    lib.JLD2.jldsave(lib.cachepath(path); key, result)
    return result
end

# Posterior samples for representative voxels (selection, HMC and NUTS), separate from plotting
# Uses val_diag_results MALA chain (held-out test_id cohort sample).
# 3 voxels: MWF | IEWF | CWF, each the `select_quantile`-th percentile of that pool fraction.
# HMC: many independent flow-backbone HMC chains per voxel, pooled into a gold-standard reference.
# NUTS uses learned coordinates by default; the windowed identity backbone remains available.
function run_posterior_samples_data(;
    seed = 0,
    select_quantile = 0.78,
    nchains_hmc::Int = 500,
    nsamples_hmc::Int = 1000,
    nadapts_hmc::Int = nsamples_hmc,
    nvoxels_nuts::Int = 96,
    nchains_nuts::Int = 32,
    nsamples_nuts::Int = 5000,
    nadapts_nuts::Int = nsamples_nuts,
    nuts_backbone::Symbol = :flow,
    nuts_max_depth::Int = 10,
)
    (; SPWin, MPWin) = xdist_results
    xdist_chain = val_diag_results.mala_out.all_outputs.xdist  # (window, nT2, nvox)
    logT2s = val_diag_results.logT2s

    # Mean observables across the MALA chain for voxel selection
    mean_bm = compute_observable_maps(dropdims(mean(xdist_chain; dims = 1); dims = 1), logT2s; SPWin, MPWin)
    n = size(xdist_chain, 3)
    pctile(v) = sortperm(v)[clamp(round(Int, select_quantile * n), 1, n)]
    mwf_vox, iew_vox, cwf_vox = pctile(mean_bm.mwf), pctile(mean_bm.iewf), pctile(mean_bm.cwf)
    voxels = [mwf_vox, iew_vox, cwf_vox]

    batch = val_diag_results.batch
    hmc_settings = (; seed, backbone = :flow, nchains = nchains_hmc, n_samples = nsamples_hmc, n_adapts = nadapts_hmc)
    nuts_settings = (; seed, backbone = nuts_backbone, nchains = nchains_nuts, n_samples = nsamples_nuts, n_adapts = nadapts_nuts, max_depth = nuts_max_depth)

    # NUTS reference voxels: the three figure voxels, then further held-out cohort voxels
    nuts_voxels = vcat(voxels, setdiff(1:size(batch.b, 2), voxels)[1:(nvoxels_nuts-length(voxels))])
    nuts_key = reference_key(batch, nuts_voxels; sampler = "nuts", Base.structdiff(nuts_settings, NamedTuple{(:nchains,)})...)
    nuts_ref = nsamples_nuts == 0 ? nothing : cached_reference(() -> run_nuts_baseline(batch; voxels = nuts_voxels, nuts_settings...), "nuts", nuts_key; nchains = nchains_nuts)
    hmc_ref = nsamples_hmc == 0 ? nothing : cached_reference("hmc", reference_key(batch, voxels; sampler = "hmc", hmc_settings...); nchains = nchains_hmc) do
        run_hmc_baseline(batch; voxels, hmc_settings...)
    end

    return (; mwf_vox, iew_vox, cwf_vox, xdist_chain, logT2s, SPWin, MPWin, hmc_ref, nuts_ref)
end

function write_reference_diagnostics_table(hmc, nuts; figpath = figuresdir("reference_diagnostics_table.tex"))
    function render_table(io)
        println(io, raw"\begin{tabular}{lrrrrrr}")
        println(io, raw"\toprule")
        println(io, raw"Reference & Voxels & Chains & Max. $\hat R$ & Min. bulk ESS & Min. tail ESS & Errors " * "\\\\")
        println(io, raw"\midrule")
        for (name, ref) in (("HMC", hmc), ("NUTS", nuts))
            rhat = maximum(maximum(r.diagnostics.coordinates.rhat) for r in ref.runs)
            bulk = minimum(minimum(r.diagnostics.spectra.ess_bulk) for r in ref.runs)
            tail = minimum(minimum(r.diagnostics.spectra.ess_tail) for r in ref.runs)
            errors = name == "HMC" ? ref.meta.numerical_error_iterations : ref.meta.divergences
            @printf(io, "%s & %d & %d & %.4f & %.0f & %.0f & %d ", name, length(ref.runs), ref.meta.nchains, rhat, bulk, tail, errors)
            println(io, "\\\\")
        end
        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
    end
    return save_table(_ -> sprint(render_table), figpath)
end

function plot_posterior_samples(results = posterior_samples_results; figpath = figuresdir("posterior_samples.pdf"), fontsize = 24)
    (; mwf_vox, iew_vox, cwf_vox, xdist_chain, logT2s, SPWin, MPWin, hmc_ref, nuts_ref) = results
    T2ms = 1000.0f0 .* exp.(logT2s)
    colors = Makie.wong_colors()
    kw_ax = (; xgridvisible = false, ygridvisible = false, titlesize = fontsize, xlabelsize = fontsize, ylabelsize = fontsize, xticklabelsize = fontsize, yticklabelsize = fontsize)

    col_info = [
        (; vox = mwf_vox, field = :mwf, label = L"\mathrm{MWF}", xlabel = L"\mathrm{MWF}", color = colors[1]),
        (; vox = iew_vox, field = :iewf, label = L"\mathrm{IEWF}", xlabel = L"\mathrm{IEWF}", color = colors[2]),
        (; vox = cwf_vox, field = :cwf, label = L"\mathrm{CWF}", xlabel = L"\mathrm{CWF}", color = colors[3]),
    ]

    fig = Figure(; size = (250 * 3 + 100, 400), fontsize)
    amp_floor = 1.0f-4 # log-scale band floor for near-zero spectral tails
    href = :gray25 # gold-standard HMC reference colour
    nutsref = :purple # independent NUTS reference colour
    p_flow = p_hmc = p_nuts = nothing

    for (col, ci) in enumerate(col_info)
        xd_all = lib.unitsum(xdist_chain[:, :, ci.vox]'; dims = 1) # (nT2, window)
        xd_mean = vec(mean(xd_all; dims = 2))
        xd_lo = vec(mapslices(r -> quantile(r, 0.025f0), xd_all; dims = 2))
        xd_hi = vec(mapslices(r -> quantile(r, 0.975f0), xd_all; dims = 2))
        bm_samps = compute_observable_maps(xd_all, logT2s; SPWin, MPWin)
        frac_samps = getfield(bm_samps, ci.field)

        ax_spec = Axis(fig[1, col]; kw_ax..., xlabel = L"$T_2$ (ms)", ylabel = col == 1 ? L"Amplitude$$" : "", xscale = log10, yscale = log10, xticks = ([10, 100, 1000], [L"10^1", L"10^2", L"10^3"]), yticks = ([1e-4, 1e-2, 1], [L"10^{-4}", L"10^{-2}", L"10^0"]))
        band!(ax_spec, T2ms, max.(xd_lo, amp_floor), max.(xd_hi, amp_floor); color = (ci.color, 0.3))
        lines!(ax_spec, T2ms, max.(xd_lo, amp_floor); color = (ci.color, 0.85), linewidth = 0.75)
        lines!(ax_spec, T2ms, max.(xd_hi, amp_floor); color = (ci.color, 0.85), linewidth = 0.75)
        p_flow = lines!(ax_spec, T2ms, max.(xd_mean, amp_floor); color = ci.color, linewidth = 2, label = L"MALA$$")

        ax_hist = Axis(fig[2, col]; kw_ax..., xlabel = ci.xlabel, ylabel = col == 1 ? L"Density$$" : "", xticks = WilkinsonTicks(3; k_max = 3), yticks = WilkinsonTicks(3; k_max = 3))
        hist!(ax_hist, frac_samps; bins = 40, color = (ci.color, 0.4), normalization = :pdf)
        density!(ax_hist, frac_samps; color = :transparent, strokecolor = ci.color, strokewidth = 2)
        vlines!(ax_hist, [mean(frac_samps)]; color = ci.color, linewidth = 1.25)
        for (method, ref, color, linestyle) in ((:hmc, hmc_ref, href, :dash), (:nuts, nuts_ref, nutsref, :dot))
            ref === nothing && continue
            xd = lib.unitsum(ref.runs[col].xdist; dims = 1)
            μ = vec(mean(xd; dims = 2))
            frac = getfield(compute_observable_maps(xd, logT2s; SPWin, MPWin), ci.field)
            if method === :hmc
                lo = vec(mapslices(r -> quantile(r, 0.025f0), xd; dims = 2))
                hi = vec(mapslices(r -> quantile(r, 0.975f0), xd; dims = 2))
                band!(ax_spec, T2ms, max.(lo, amp_floor), max.(hi, amp_floor); color = (color, 0.18))
                lines!(ax_spec, T2ms, max.(lo, amp_floor); color = (color, 0.8), linewidth = 0.75, linestyle)
                lines!(ax_spec, T2ms, max.(hi, amp_floor); color = (color, 0.8), linewidth = 0.75, linestyle)
            end
            p = lines!(ax_spec, T2ms, max.(μ, amp_floor); color, linewidth = 1.5, linestyle)
            method === :hmc ? (p_hmc = p) : (p_nuts = p)
            density!(ax_hist, frac; color = :transparent, strokecolor = color, strokewidth = 2, linestyle)
            vlines!(ax_hist, [mean(frac)]; color, linewidth = 1.25, linestyle)
        end
    end

    entries = filter(p -> first(p) !== nothing, [p_flow => L"MALA$$", p_hmc => L"HMC$$", p_nuts => L"NUTS$$"])
    Legend(fig[1:2, 4], first.(entries), last.(entries); framevisible = false, labelsize = fontsize)

    resize_to_layout!(fig)
    savefig(fig, figpath)
    return fig
end

"""
Low-pass prior window and its effect on T2 spectra, for prior draws and flow posterior draws.
The left panel shows the learned Gaussian window weights against geometric sweeps that scale the cutoff `Fc`, controlling spectral smoothness, and the amplitude `a`, controlling peak sharpness, by ½, 1/√2, √2, and 2.
The middle panels map a single fixed Gaussian draw `u` to a T2 spectrum through each window via the inverse ILR transform. The right panels show nplot importance-weighted posterior draws from the trained flow at a representative voxel, shaded by importance weight, with their importance-weighted mean over all `elbo_results` draws. Each is shown on linear and log amplitude axes.
"""
function plot_prior_lowpass_filter(; figpath = figuresdir("prior_lowpass_filter.pdf"), seed::Int = 0, voxel = nothing, nplot::Int = 10, fontsize = 22)
    ℰ, 𝒢, 𝒫 = run_env.models.enc, run_env.models.genatr, run_env.models.prior
    bt = run_env.val_batch
    T2Range = bt.T2Range

    # Learned low-pass window at a representative voxel: cutoff Fc, amplitude λ0 and floor λmin
    logτ₀ = ex.log_relative_t2_range(bt.θ[:, 1])
    lowpass = cpu(ex.window_filter(𝒢).flow.model)
    nu = length(lowpass.t) # Aitchison dimension nT2-1; the DC component is dropped
    nT2 = nu + 1
    freqs = (1:nu) ./ Float32(nT2) # fᵢ = i/nT2 as in the documentation; the model grid i/nu differs by a constant absorbed in Fc
    Fc, a, λmin = exp(only(logτ₀) - only(lowpass.logτ)) * Float32(nu) / Float32(nT2), exp(only(lowpass.w.w.logw₀)), exp(only(lowpass.w.logw₀)) # Fc rescaled from the model's i/nu grid into the i/nT2 convention
    window(f, Fc, a) = sqrt((a * exp(-(f / Fc)^2 / 2))^2 + λmin^2) # √((a·gaussian)² + λmin²), matching the scale-then-floor parameterization
    T2ms = 1000.0f0 .* exp.(range(log(T2Range[1, 1]), log(T2Range[2, 1]); length = nT2))
    u = randn(lib.Xoshiro(seed), Float32, nu)

    # Geometric sweeps of the cutoff (blue) and amplitude (orange) about the learned prior (black)
    factors = Float32[1/2, 1/√2, √2, 2]
    fc_labels = [L"$F_c \div 2$", L"$F_c \div \sqrt{2}$", L"$F_c \times \sqrt{2}$", L"$F_c \times 2$"]
    a_labels = [L"$\lambda_0 \div 2$", L"$\lambda_0 \div \sqrt{2}$", L"$\lambda_0 \times \sqrt{2}$", L"$\lambda_0 \times 2$"]

    blues, oranges = Makie.resample_cmap(:Blues, length(factors) + 2)[3:end], Makie.resample_cmap(:Oranges, length(factors) + 2)[3:end]
    fc_specs = [(Fc * r, a, blues[i], :solid) for (i, r) in enumerate(factors)]
    a_specs = [(Fc, a * r, oranges[i], :dash) for (i, r) in enumerate(factors)]

    # Conditioning voxel: explicit index, else the median-MWF voxel of the slice
    nvox = size(bt.b)[end]
    v = if voxel !== nothing
        voxel
    else
        x̄₀ = xdist_results.xdists.PseudoMAP
        mwf = vec(compute_observable_maps(x̄₀, collect(range(log(T2Range[1, 1]), log(T2Range[2, 1]); length = nT2)); SPWin = xdist_results.SPWin, MPWin = xdist_results.MPWin).mwf)
        sortperm(mwf)[cld(nvox, 2)]
    end

    # Reuse aligned proposals and IS weights; opacity and mean reflect posterior reweighting.
    gen_samples = Float32.(elbo_results["XDistProposalSamples"][:, :, v]') # (nT2, n_elbo)
    gen_weights = Float32.(elbo_results["Weights_IS"][:, v]) # (n_elbo,), self-normalized to sum to 1
    n_elbo = length(gen_weights)
    gen_mean = gen_samples * gen_weights # importance-weighted mean in T2 space
    greens = Makie.resample_cmap(:Greens, 6)
    mean_color = greens[6]
    plotted = 1:min(nplot, n_elbo)
    w_plot_max = maximum(gen_weights[plotted])
    function draw_gen!(ax)
        for j in plotted
            α = 0.15f0 + 0.75f0 * (gen_weights[j] / w_plot_max) # higher importance weight -> darker/more opaque
            lines!(ax, T2ms, gen_samples[:, j]; color = (greens[4], α), linewidth = 0.8)
        end
        lines!(ax, T2ms, gen_mean; color = mean_color, linewidth = 2.5)
    end

    # Styled curves for one data source; references thin, learned drawn last and bold.
    curveset(curve) = push!(
        [(curve(Fcᵢ, aᵢ)..., color, ls, 1.4) for (Fcᵢ, aᵢ, color, ls) in vcat(fc_specs, a_specs)],
        (curve(Fc, a)..., Makie.to_color(:black), :solid, 2.5),
    )
    draw!(ax, curves) = foreach(c -> lines!(ax, c[1], c[2]; color = c[3], linestyle = c[4], linewidth = c[5]), curves)

    win_curves = curveset((Fcᵢ, aᵢ) -> (freqs, window.(freqs, Fcᵢ, aᵢ)))
    prior_curves = curveset((Fcᵢ, aᵢ) -> (T2ms, lib.aitchison_inv_transform(window.(freqs, Fcᵢ, aᵢ) .* u)))

    fig = Figure(; size = (800, 420), fontsize)
    kw = (; xgridvisible = false, ygridvisible = false, titlesize = fontsize, xlabelsize = fontsize, ylabelsize = fontsize, xticklabelsize = fontsize, yticklabelsize = fontsize)
    ax_win = Axis(fig[1:2, 1]; kw..., title = L"Low-Pass Prior Filter$$", xlabel = L"Frequency $f$", ylabel = L"Window Weight $\lambda(f)$", yscale = log10, xticks = WilkinsonTicks(3; k_max = 3))
    ax_pl = Axis(fig[1, 2]; kw..., title = L"Prior Samples$$", ylabel = L"Amplitude$$", xscale = log10, xticks = ([10, 100, 1000], [L"10^1", L"10^2", L"10^3"]), xticklabelsvisible = false)
    ax_plog = Axis(fig[2, 2]; kw..., xlabel = L"$T_2$ (ms)", ylabel = L"Amplitude$$", xscale = log10, xticks = ([10, 100, 1000], [L"10^1", L"10^2", L"10^3"]), yscale = log10)
    ax_gl = Axis(fig[1, 3]; kw..., title = L"Flow Samples$$", xscale = log10, xticks = ([10, 100, 1000], [L"10^1", L"10^2", L"10^3"]), xticklabelsvisible = false)
    ax_glog = Axis(fig[2, 3]; kw..., xlabel = L"$T_2$ (ms)", xscale = log10, xticks = ([10, 100, 1000], [L"10^1", L"10^2", L"10^3"]), yscale = log10)

    draw!(ax_win, win_curves)
    draw!(ax_pl, prior_curves)
    draw!(ax_plog, prior_curves)
    draw_gen!(ax_gl)
    draw_gen!(ax_glog)

    # Combined legend: add "Prior" heading (like "Flow") and label multipliers properly
    prior_labels = vcat([L"Learned$$"], fc_labels, a_labels)
    prior_elems = vcat(
        [LineElement(; color = :black, linewidth = 2.5)],
        [LineElement(; color, linestyle = :solid) for (_, _, color, _) in fc_specs],
        [LineElement(; color, linestyle = :dash) for (_, _, color, _) in a_specs],
    )
    post_elems = [LineElement(; color = mean_color, linewidth = 2.5), LineElement(; color = greens[4], linewidth = 1.0)]

    Legend(fig[1:2, 4],
        [prior_elems, post_elems],
        [prior_labels, [L"Mean$$", L"Samples$$"]],
        [L"Prior$$", L"Flow$$"];
        framevisible = true,
        framecolor = (:white, 0.0),
        backgroundcolor = (:white, 0.7),
        labelsize = fontsize,
        titlesize = fontsize,
    )

    colgap!(fig.layout, 12)
    rowgap!(fig.layout, 8)
    resize_to_layout!(fig)
    savefig(fig, figpath)
    return fig
end

# Retain at least two states when n ≥ 2.
function post_burn_start(n::Int, burn_in_frac::Real)
    n ≤ 0 && return 1
    n_drop = clamp(floor(Int, burn_in_frac * n), 0, max(0, n - 2))
    return n_drop + 1
end

# Post-burn-in autocorrelation, ESS and observable means; `all_outputs` keeps the full chain
function summarize_mala_run(out, logT2s; SPWin, MPWin, burn_in_frac::Real = 0.1f0)
    ao = out.all_outputs
    n_u = size(ao.summary.u, 1)
    r = post_burn_start(n_u, burn_in_frac):n_u
    n_eff = length(r)
    u_sl = @views ao.summary.u[r, ..]
    x_sl = @views ao.xdist[r, ..]
    α_sl = @views ao.α[r, ..]

    τ_u_nd = dropdims(lib.autocorr_time(lib.autocorr_fft(u_sl; dims = 1); dims = 1); dims = 1) # (udim, nvox)
    τ_u = vec(mean(τ_u_nd; dims = 1))
    τ_u_bottleneck = vec(maximum(τ_u_nd; dims = 1))
    ess_u_min = n_eff ./ (2 .* τ_u_bottleneck) # ESS^int = S/(2τ^int)

    nvox = size(x_sl, 3)
    bm_samps = compute_observable_maps(permutedims(x_sl, (2, 1, 3)), logT2s; SPWin, MPWin) # each field: (n_eff, nvox)
    τ_bm = map(bm_samps) do marker
        dropdims(lib.autocorr_time(lib.autocorr_fft(reshape(marker, n_eff, 1, nvox); dims = 1); dims = 1); dims = 1)
    end
    ess_bm = map(t -> n_eff ./ (2 .* vec(t)), τ_bm) # ESS^int = S/(2τ^int)

    mean_bm = compute_observable_maps(dropdims(mean(x_sl; dims = 1); dims = 1), logT2s; SPWin, MPWin)
    accrat = vec(dropdims(mean(α_sl; dims = 1); dims = 1))
    return (;
        τ_u, τ_u_nd, τ_u_bottleneck, ess_u_min, τ_bm, ess_bm, bm_samps,
        mean_bm, accrat, n_eff, burn_in_frac,
        all_outputs = ao, u = ao.summary.u, xdist = ao.xdist, neglogpostu = ao.losses.neglogpostu, α = ao.α, η = ao.η,
    )
end

# Compare nominal α to the fraction of voxel truths inside central α-credible intervals.
function run_calibration_data(results = synth_eval_results;
    alphas = 0.05f0:0.05f0:0.95f0,
    snr = results.SNRs[end],
    observables = (; mwf = L"\mathrm{MWF}", iewf = L"\mathrm{IEWF}", mwT2 = L"MW $T_2$", gmT2 = L"GM $T_2$"),
    methods = (; mala = "MALA", mh = "MH", iw = "IW"),
)
    d = results.snr_results[snr].all_vox
    (; SPWin, MPWin) = results
    gt_bm = compute_observable_maps(d.xdist_true, d.logT2s; SPWin, MPWin)
    mk, bk = keys(methods), keys(observables)

    coverage = NamedTuple{Tuple(mk)}(ntuple(length(mk)) do mi
        method = mk[mi]
        samp = d.samples[method]
        w = method === :iw ? samp.weights : nothing
        NamedTuple{Tuple(bk)}(ntuple(length(bk)) do fi
            field = bk[fi]
            bm_mat = samp.observables[field]
            truth = getfield(gt_bm, field)
            Float32[ci_coverage(bm_mat, truth, α; weights = w) for α in alphas]
        end)
    end)

    return (; alphas = collect(alphas), coverage, observables, methods, snr)
end

# Plot observed vs nominal coverage curves.
# Color encodes observable; linestyle + marker encodes method.
function plot_calibration(data = cal_data;
    figpath = figuresdir("calibration.pdf"),
)
    all_methods = data.methods
    method_keys = keys(all_methods)
    field_keys = keys(data.observables)
    (; alphas, coverage, observables) = data

    colors = Makie.wong_colors()
    linestyles = [:solid, :dash, :dot, :dashdot]
    markers = [:circle, :rect, :diamond, :utriangle]
    kw_ax = (; xgridvisible = false, ygridvisible = false)

    fig = Figure(; size = (450, 350))
    ax = Axis(fig[1, 1]; kw_ax...,
        xlabel = L"Nominal coverage level$$",
        ylabel = L"Observed coverage$$",
        title = L"Calibration $(\mathrm{SNR} = %$(data.snr)\ \mathrm{dB})$",
        aspect = 1,
    )
    lines!(ax, [0, 1], [0, 1]; color = :gray50, linestyle = :dash, linewidth = 1.5, label = L"Ideal$$")

    for (mi, method) in enumerate(method_keys)
        for (fi, field) in enumerate(field_keys)
            label = L"%$(observables[field]) (\mathrm{%$(all_methods[method])})"
            lines!(ax, alphas, coverage[method][field]; color = colors[fi], linestyle = linestyles[mi], linewidth = 2, label)
            scatter!(ax, alphas, coverage[method][field]; color = colors[fi], marker = markers[mi], markersize = 6)
        end
    end

    xlims!(ax, 0, 1)
    ylims!(ax, 0, 1)
    Legend(fig[1, 2], ax; framevisible = true, framecolor = (:white, 0.0), backgroundcolor = (:white, 0.7), labelsize = 10, tellheight = false)

    resize_to_layout!(fig)
    savefig(fig, figpath)
    return fig
end

# Flow-geometry sampling comparison; identity-backbone chains adapt per-voxel step sizes in warmup
function run_mcmc_ablation(;
    seed = 0,
    n_voxels = nothing,
    SNR = 50,
    n_samples = 1000, # post-warmup samples kept for every chain; must run ≫ τ to estimate autocorrelation time
    warmup_flow = 0, # learned-backbone comparisons start without warmup
    warmup_id = 4000, # adaptive warmup for the identity backbone
    warmup_hmc = 1000, # HMC mass-matrix and step-size adaptation window
    leapfrog_hmc = 16,
    nchains_hmc = 8,
    target_acceptance = 0.574,
)
    seed_all!(seed)
    batch = run_env.val_cohort |> gpu
    nvox = n_voxels === nothing ? size(batch.b, 2) : min(n_voxels, size(batch.b, 2))
    bs = map(x -> (x isa AbstractArray && last(size(x)) != nvox) ? x[.., 1:nvox] : x, batch)
    nT2 = run_env.cfg.data.nT2
    logT2s = @allowscalar range(log(bs.T2Range[1, 1]), log(bs.T2Range[2, 1]); length = nT2)
    ℰ, 𝒢, 𝒫 = run_env.models.enc, run_env.models.genatr, run_env.models.prior

    # Identity backbone keeps the low-pass window and replaces only the learned conditional backbone
    𝒢₀ = ex.identity_backbone_with_window_flow(𝒢)
    samp_id = ex.SamplingProblem(ex.MALA(), ℰ, 𝒢₀, 𝒫)
    samp_flow = ex.SamplingProblem(ex.MALA(), ℰ, 𝒢, 𝒫)
    samp_mh_flow = ex.SamplingProblem(ex.Metropolis(), ℰ, 𝒢, 𝒫)
    samp_kw = (; save_all_samples = true, verbose = false)

    @info "MALA ablation: synthetic data (SNR=$SNR)..."
    bs_cpu = cpu(bs)
    x_true = synthetic_t2_distributions(bs_cpu; SNR, seed)
    b_synth = gpu(ex.sample_similar(ex.forward_physics_model(x_true, bs_cpu.A, bs_cpu.mask, bs_cpu.δb), bs_cpu.b))

    @info "MALA ablation: synthetic calibration chains (flow: $warmup_flow warmup + $n_samples, adaptive id: $warmup_id warmup + $n_samples)..."
    out_cal_mh_flow, t_cal_mh_flow = @timed ex.run_sampling(samp_mh_flow, b_synth, bs.A, bs.mask, bs.δb, bs.θ; warmup_iter = warmup_flow, iter = warmup_flow + n_samples, samp_kw..., verbose = 10, lr = nothing)
    out_cal_flow, t_cal_flow = @timed ex.run_sampling(samp_flow, b_synth, bs.A, bs.mask, bs.δb, bs.θ; warmup_iter = warmup_flow, iter = warmup_flow + n_samples, samp_kw..., verbose = 10)
    out_cal_id, t_cal_id = @timed ex.run_sampling(samp_id, b_synth, bs.A, bs.mask, bs.δb, bs.θ; adaptive = true, target_acceptance, warmup_iter = warmup_id, iter = warmup_id + n_samples, samp_kw...)

    # Identity-backbone HMC: independent chains per voxel, reused for histograms and trajectories
    @info "MALA ablation: identity HMC comparison ($nchains_hmc chains × $n_samples samples, all $nvox voxels)..."
    out_cal_hmc, t_cal_hmc = @timed run_hmc_baseline((; b = b_synth, A = bs.A, mask = bs.mask, δb = bs.δb, θ = bs.θ, T2Range = bs.T2Range); seed, voxels = 1:nvox, backbone = :id_window, nchains = nchains_hmc, leapfrog = leapfrog_hmc, n_adapts = warmup_hmc, n_samples = n_samples)

    cal = (; xdist_true = x_true.xdist, out_cal_mh_flow, t_cal_mh_flow, out_cal_flow, t_cal_flow, out_cal_id, t_cal_id, out_cal_hmc, t_cal_hmc)
    synth = (; b = b_synth, A = bs.A, mask = bs.mask, δb = bs.δb, θ = bs.θ, T2Range = bs.T2Range) # synthetic batch retained for the ξ-space posterior-contour background
    meta = (; seed, n_samples, warmup_flow, warmup_id, warmup_hmc, nchains_hmc, leapfrog_hmc, nvox, SNR, target_acceptance, logT2s)

    return (; cal, synth, meta) |> cpu
end

# Compare samplers on learned and window-preserving identity backbones.
# Identity step sizes minimize worst-biomarker τᵢₙₜ in a separate pilot sweep;
# learned-flow steps use the asymptotic RWM/MALA scalings. HMC uses the identity backbone.
function run_method_backbone_comparison(;
    seed = 0,
    SNR = 50,
    n_voxels = nothing,
    n_samples = 1000, # post-warmup samples kept per chain; also the IS proposal draws and HMC samples
    warmup_flow = 0, # learned-backbone comparisons start without warmup
    warmup_id = 4000, # adaptive warmup for the identity backbone
    warmup_hmc = 1000,
    nchains_hmc = 1,
    leapfrog_hmc = 16,
    sweep_lrs_mala = Float32.(exp10.(-3.0:0.25:-1.75)),
    sweep_lrs_rwm = Float32.(exp10.(-3.0:0.25:-1.0)),
    sweep_iter = 500,
    sweep_burn = 0.5f0, # discard fraction for the short oracle step-size sweep
    SPWin = xdist_results.SPWin, MPWin = xdist_results.MPWin,
    bm_fields = (:mwf, :iewf, :cwf, :gmT2),
)
    seed_all!(seed)
    batch = run_env.val_cohort |> gpu
    nvox = n_voxels === nothing ? size(batch.b, 2) : min(n_voxels, size(batch.b, 2))
    bs = map(x -> (x isa AbstractArray && last(size(x)) != nvox) ? x[.., 1:nvox] : x, batch)
    nT2 = run_env.cfg.data.nT2
    logT2s = @allowscalar range(log(bs.T2Range[1, 1]), log(bs.T2Range[2, 1]); length = nT2)
    dlogT2 = Float32(step(logT2s))
    ℰ, 𝒢, 𝒫 = run_env.models.enc, run_env.models.genatr, run_env.models.prior
    𝒢₀ = ex.identity_backbone_with_window_flow(𝒢)

    bs_cpu = cpu(bs)
    x_true = synthetic_t2_distributions(bs_cpu; SNR, seed)
    b = gpu(ex.sample_similar(ex.forward_physics_model(x_true, bs_cpu.A, bs_cpu.mask, bs_cpu.δb), bs_cpu.b))
    xdist_true = x_true.xdist
    gt_bm = compute_observable_maps(xdist_true, logT2s; SPWin, MPWin)

    sp(alg, G) = ex.SamplingProblem(alg, ℰ, G, 𝒫)
    samp_kw = (; save_all_samples = true, verbose = false)
    worst_tau(s) = Float32[maximum(vec(getfield(s.τ_bm, f))[v] for f in bm_fields) for v in 1:nvox]
    min_ess(s) = Float32[minimum(vec(getfield(s.ess_bm, f))[v] for f in bm_fields) for v in 1:nvox]

    # Oracle per-voxel step-size sweep on the identity backbone, minimizing worst-biomarker τ^int.
    function oracle_lr(alg, lrs)
        τw = Matrix{Float32}(undef, length(lrs), nvox)
        for (k, lr) in enumerate(lrs)
            @info "  identity sweep $(alg) lr=$(lr) ($k/$(length(lrs)))..."
            o = ex.run_sampling(sp(alg, 𝒢₀), b, bs.A, bs.mask, bs.δb, bs.θ; lr, iter = sweep_iter, samp_kw...)
            τw[k, :] .= worst_tau(summarize_mala_run(o, logT2s; SPWin, MPWin, burn_in_frac = sweep_burn))
        end
        return reshape(Float32[lrs[argmin(@view τw[:, v])] for v in 1:nvox], 1, nvox)
    end

    # Keep per-voxel metric vectors so the table can summarize them without resampling
    function metrics(s, ngrad)
        n_w = size(s.xdist, 1)
        r = post_burn_start(n_w, s.burn_in_frac):n_w
        essgrad = ngrad > 0 ? min_ess(s) ./ Float32(ngrad) : Float32[] # ESS per gradient eval; gradient-free → —
        m = (; α = s.accrat, τ = worst_tau(s), essS = min_ess(s) ./ s.n_eff, essgrad)
        mean_xdist = dropdims(mean(@view(s.xdist[r, :, :]); dims = 1); dims = 1)
        mwf_mae = abs.(vec(s.mean_bm.mwf) .- vec(gt_bm.mwf))
        w1 = vec(lib.wasserstein1(xdist_true, mean_xdist; agg = identity, dims = 1)) .* dlogT2
        return merge(m, (; mwf_mae, w1))
    end

    function run_cell(alg, G, lr, warmup)
        out = ex.run_sampling(sp(alg, G), b, bs.A, bs.mask, bs.δb, bs.θ; lr, warmup_iter = warmup, iter = warmup + n_samples, samp_kw...)
        return metrics(summarize_mala_run(out, logT2s; SPWin, MPWin, burn_in_frac = 0), alg isa ex.MALA ? n_samples : 0) # only MALA evaluates gradients; count sampling-phase gradients
    end

    # Identity HMC comparison, batched on the GPU; ESS/∇ counts retained-sampling gradients.
    function hmc_cell(bk)
        res = run_hmc_baseline((; b, A = bs.A, mask = bs.mask, δb = bs.δb, θ = bs.θ, T2Range = bs.T2Range); seed, voxels = 1:nvox, backbone = bk, nchains = nchains_hmc, leapfrog = leapfrog_hmc, n_adapts = warmup_hmc, n_samples = n_samples)
        nsamp = size(first(res.runs).xdist, 2)
        X = permutedims(cat([r.xdist for r in res.runs]...; dims = 3), (2, 1, 3)) # (nsamp, nT2, nvox)

        # Columns interleave independent chains; estimate ESS along time within each chain.
        min_ess_v = map(res.runs) do r
            chains = permutedims(reshape(r.xdist, nT2, r.nchains, :), (3, 1, 2))
            τ_bm = chain_observable_autocorr_times(chains, logT2s; SPWin, MPWin)
            minimum(nsamp / (2 * mean(getfield(τ_bm, f))) for f in bm_fields) # equal-length chains are pooled equally: ESS = CS/(2 mean꜀ τ꜀).
        end

        worst_τ = Float32(nsamp) ./ (2 .* min_ess_v)
        grad_v = Float32[r.grad_evals for r in res.runs]
        m = (; α = Float32[r.accept for r in res.runs], τ = worst_τ, essS = min_ess_v ./ nsamp, essgrad = min_ess_v ./ grad_v)
        mean_x = dropdims(mean(X; dims = 1); dims = 1)
        bm = compute_observable_maps(mean_x, logT2s; SPWin, MPWin)
        mwf_mae = abs.(vec(bm.mwf) .- vec(gt_bm.mwf))
        w1 = vec(lib.wasserstein1(xdist_true, mean_x; agg = identity, dims = 1)) .* dlogT2
        return merge(m, (; mwf_mae, w1))
    end

    # Importance sampling: the flow proposal reweighted instead of accept/reject.
    function iw_cell(G)
        p = ex.run_elbo_estimate(ℰ, G, 𝒫, b, bs.A, bs.mask, bs.δb, bs.θ; samples = n_samples, save_proposal_samples = true)
        w = p["Weights_IS"] # already normalized over proposal samples
        m = (; α = Float32[], τ = Float32[], essS = vec(p["ESS_IS"]), essgrad = Float32[])
        mean_xdist = dropdims(sum(reshape(w, n_samples, 1, nvox) .* p["XDistProposalSamples"]; dims = 1); dims = 1)
        bm = compute_observable_maps(mean_xdist, logT2s; SPWin, MPWin)
        mwf_mae = abs.(vec(bm.mwf) .- vec(gt_bm.mwf))
        w1 = vec(lib.wasserstein1(xdist_true, mean_xdist; agg = identity, dims = 1)) .* dlogT2
        return merge(m, (; mwf_mae, w1))
    end

    cells = Dict{Tuple{Symbol, Symbol}, Any}()
    for method in (:iw, :mh, :rwm, :mala, :hmc)
        @info "Comparison: $(method) (flow + identity, synthetic)..."
        if method === :iw
            cells[(:iw, :flow)] = iw_cell(𝒢)
            cells[(:iw, :id)] = iw_cell(𝒢₀)
        elseif method === :mh
            cells[(:mh, :flow)] = run_cell(ex.Metropolis(), 𝒢, nothing, warmup_flow)
            cells[(:mh, :id)] = run_cell(ex.Metropolis(), 𝒢₀, nothing, warmup_id)
        elseif method === :rwm
            cells[(:rwm, :flow)] = run_cell(ex.Metropolis(), 𝒢, (2.382f0^2 / 2) / (𝒢.opts.udim - 1), warmup_flow)
            cells[(:rwm, :id)] = run_cell(ex.Metropolis(), 𝒢₀, oracle_lr(ex.Metropolis(), sweep_lrs_rwm), warmup_id)
        elseif method === :mala
            cells[(:mala, :flow)] = run_cell(ex.MALA(), 𝒢, (1.650f0^2 / 2) / cbrt(𝒢.opts.udim), warmup_flow)
            cells[(:mala, :id)] = run_cell(ex.MALA(), 𝒢₀, oracle_lr(ex.MALA(), sweep_lrs_mala), warmup_id)
        elseif method === :hmc
            cells[(:hmc, :id)] = hmc_cell(:id_window)
        end
    end

    return cells |> cpu
end

# Separate identity-backbone NUTS efficiency investigation.
function run_nuts_identity_backbone(; seed = 0, n_voxels = 32, n_samples = 1000, n_adapts = n_samples)
    seed_all!(seed)
    batch = run_env.val_cohort |> gpu

    # CPU generation preserves the HMC comparison's seeded spectra/noise across devices and subset sizes.
    batch_cpu = cpu(batch)
    x_true = synthetic_t2_distributions(batch_cpu; SNR = 50, seed)
    b_full = gpu(ex.sample_similar(ex.forward_physics_model(x_true, batch_cpu.A, batch_cpu.mask, batch_cpu.δb), batch_cpu.b))
    nvox = min(n_voxels, size(batch.b, 2))
    bs = map(x -> x[.., 1:nvox], batch)
    nT2, SPWin, MPWin = run_env.cfg.data.nT2, 0.032f0, 0.2f0
    logT2s = range(log(cpu(bs.T2Range)[1, 1]), log(cpu(bs.T2Range)[2, 1]); length = nT2)
    b = b_full[:, 1:nvox]
    inputs = (; b, A = bs.A, mask = bs.mask, δb = bs.δb, θ = bs.θ, T2Range = bs.T2Range)
    settings = (; seed, backbone = :id_window, nchains = 1, n_adapts, n_samples, max_depth = 10)
    key = reference_key(inputs, 1:nvox; sampler = "nuts", Base.structdiff(settings, NamedTuple{(:nchains,)})...)
    res = cached_reference("identity-nuts", key; nchains = settings.nchains) do
        run_nuts_baseline(inputs; voxels = 1:nvox, settings...)
    end
    X = permutedims(cat([r.xdist for r in res.runs]...; dims = 3), (2, 1, 3))
    τ_bm = chain_observable_autocorr_times(X, logT2s; SPWin, MPWin)
    fields = (:mwf, :iewf, :cwf, :gmT2)
    worst_τ = Float32[maximum(vec(getfield(τ_bm, f))[v] for f in fields) for v in 1:nvox]
    essgrad = Float32(n_samples) ./ (2 .* worst_τ) ./ Float32[r.grad_evals for r in res.runs]
    @info "Identity NUTS efficiency" nvox median = median(essgrad) range95 = quantile(essgrad, [0.025, 0.975]) divergences = res.meta.divergences
    return (; essgrad, divergences = res.meta.divergences, reference = res)
end

function write_identity_efficiency_table(cells, reference, logT2s; figpath = figuresdir("identity_efficiency_table.tex"), SPWin = 0.032f0, MPWin = 0.2f0)
    @assert reference.meta.nchains == 1
    @assert all(r -> size(r.xdist, 2) == reference.meta.n_samples, reference.runs) "Unthinned identity chains required"
    X = permutedims(cat([r.xdist for r in reference.runs]...; dims = 3), (2, 1, 3))
    τ = chain_observable_autocorr_times(X, logT2s; SPWin, MPWin)
    worst = [maximum(vec(getproperty(τ, f))[v] for f in (:mwf, :iewf, :cwf, :gmT2)) for v in eachindex(reference.runs)]
    ess = reference.meta.n_samples ./ (2 .* worst)
    voxels = [r.v for r in reference.runs]
    nuts = (; α = [r.accept for r in reference.runs], essS = ess ./ reference.meta.n_samples, essgrad = ess ./ [r.grad_evals for r in reference.runs])
    function render_table(io)
        println(io, raw"\begin{tabular}{lrrr}")
        println(io, raw"\toprule")
        println(io, raw"Identity sampler & Acceptance & ESS/$S$ & ESS/$\nabla$ " * "\\\\")
        println(io, raw"\midrule")
        for (name, metrics) in (("MALA", map(x -> x[voxels], cells[(:mala, :id)][(:α, :essS, :essgrad)])),
            ("HMC", map(x -> x[voxels], cells[(:hmc, :id)][(:α, :essS, :essgrad)])), ("NUTS", nuts))
            @printf(io, "%s & %.4f & %.5f & %.6f ", name, median(metrics.α), median(metrics.essS), median(metrics.essgrad))
            println(io, "\\\\")
        end
        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
    end
    return save_table(_ -> sprint(render_table), figpath)
end

# Per-voxel medians and 95% intervals; unavailable metrics are shown as —.
function write_method_backbone_table(cells; figpath = figuresdir("method_backbone_comparison_table.tex"))
    method_order = (:iw, :mh, :rwm, :mala, :hmc)
    method_lbl = Dict(:iw => "IW", :mh => "MH", :rwm => "RWM", :mala => "MALA", :hmc => "HMC")
    backbone_order = (:flow => "Learned", :id => "Identity")

    # Metric columns: (field, header, better direction); acceptance has no arrow and is not bolded
    sampling_specs = [(:α, L"\text{Acc. ratio}", :none), (:essS, L"$\mathrm{ESS}/S\,\uparrow$", :max), (:essgrad, L"$\mathrm{ESS}/\nabla\,\uparrow$", :max), (:τ, L"$\tau^{\mathrm{int}}\,\downarrow$", :min)]
    quality_specs = [(:mwf_mae, L"$|\Delta\mathrm{MWF}|\ (\%)\,\downarrow$", :min), (:w1, L"$W_1(\bar{x}, \bar{x}_{\mathrm{true}})\,\downarrow$", :min)]
    metric_specs = vcat(sampling_specs, quality_specs)
    column_labels = [vcat(L"\mathrm{Method}", L"\mathrm{Backbone}", [lbl for (_, lbl, _) in metric_specs])]
    present = [(m, bk, lbl) for m in method_order for (bk, lbl) in backbone_order if haskey(cells, (m, bk))]

    # Collapse numerically-zero tails
    function clean(x; sig = 3)
        y = round(x; sigdigits = sig)
        return abs(y) < 1e-6 ? zero(y) : y
    end

    # Best row per metric column, by median, skipping methods that lack the metric; bolded in the output.
    medmat = [
        begin
            v = getfield(cells[(m, bk)], f)
            isempty(v) ? NaN32 : Float32(median(v))
        end for (m, bk, _) in present, (f, _, _) in metric_specs
    ]
    bestrow = map(enumerate(metric_specs)) do (j, (_, _, dir))
        dir === :none && return 0
        idx = findall(!isnan, @view medmat[:, j])
        isempty(idx) ? 0 : idx[dir === :max ? argmax(medmat[idx, j]) : argmin(medmat[idx, j])]
    end
    function fmt(v; bold = false)
        return isempty(v) ? LatexCell("---") : begin
            s = "$(clean(median(v)))^{$(clean(quantile(v, 0.975); sig = 2))}_{$(clean(quantile(v, 0.025); sig = 2))}"
            LatexCell(latexstring(bold ? "\\mathbf{$s}" : s))
        end
    end

    data = Matrix{Any}(undef, length(present), 2 + length(metric_specs))
    for (i, (m, bk, blbl)) in enumerate(present)
        c = cells[(m, bk)]
        data[i, 1], data[i, 2] = method_lbl[m], blbl
        for (j, (f, _, _)) in enumerate(metric_specs)
            values = getfield(c, f)
            data[i, 2+j] = fmt(f === :mwf_mae ? 100 .* values : values; bold = bestrow[j] == i)
        end
    end
    save_table(figpath) do io
        default_pretty_table(io, data; column_labels)
    end
    return data
end

function flow_geometry_histograms(data; burn_in_frac = 0.0f0)
    function summarize_u(out)
        u, α = out.all_outputs.summary.u, out.all_outputs.α
        keep = post_burn_start(size(u, 1), burn_in_frac):size(u, 1)
        τ = dropdims(lib.autocorr_time(lib.autocorr_fft(u[keep, ..]; dims = 1); dims = 1); dims = 1)
        return (; τ_u = vec(mean(τ; dims = 1)), accrat = vec(mean(α[keep, ..]; dims = 1)))
    end
    flow, id = summarize_u(data.cal.out_cal_flow), summarize_u(data.cal.out_cal_id)
    τ_hmc = map(data.cal.out_cal_hmc.runs) do r
        U = reshape(r.u, size(r.u, 1), r.nchains, :)
        keep = post_burn_start(size(U, 3), burn_in_frac):size(U, 3)
        mean(1:r.nchains) do c
            mean(lib.autocorr_time(lib.autocorr_fft(permutedims(U[:, c, keep]); dims = 1); dims = 1))
        end
    end
    hmc = (; τ_u = Float32.(τ_hmc), accrat = Float32[r.accept for r in data.cal.out_cal_hmc.runs])
    return (; flow, id, hmc)
end

function plot_flow_geometry(;
    seed = 0,
    fontsize = 22,
    histograms = nothing,
    figpath = figuresdir("flow_geometry.pdf"),
    npts::Int = 121,
    n_is_voxels::Int = 1000,
    n_is_samples::Int = 16,
    voxel = 124,
    lerp_1d::Symbol = :normal,
    lerp_2d::Symbol = :normal,
    lerp_seed = 2,
    lerp_detrend::Bool = false,
    lerp_band::Bool = true,
    lerp_samples::Int = 100,
    traj_lim::Real = 3, # trajectory-panel axes are clamped to [-traj_lim, traj_lim]
)
    seed_all!(seed)
    rng = Xoshiro(seed)
    (; cal, synth) = ablation_data
    (; flow, id, hmc) = isnothing(histograms) ? flow_geometry_histograms(ablation_data) : histograms
    colors = Makie.wong_colors()
    hmc_color = Makie.resample_cmap(:Greens, 5)[4] # identity HMC ensemble colour, distinct from learned/identity MALA
    kw_ax = (; xgridvisible = false, ygridvisible = false, titlesize = fontsize, xlabelsize = fontsize, ylabelsize = fontsize, xticklabelsize = fontsize, yticklabelsize = fontsize)
    lblsize = round(Int, 0.8 * fontsize) # shared font size for every in-axis legend and text label
    legbg = (:white, 0.8) # shared translucent background for every legend and text label
    finite_pos(x) = filter(v -> isfinite(v) && v > 0, vec(x))
    τ_flow, τ_id, τ_hmc = finite_pos(flow.τ_u), finite_pos(id.τ_u), finite_pos(hmc.τ_u)
    acc_flow, acc_id, acc_hmc = finite_pos(flow.accrat), finite_pos(id.accrat), finite_pos(hmc.accrat)

    ℰ, 𝒢, 𝒫 = run_env.models.enc, run_env.models.genatr, run_env.models.prior
    𝒢₀ = ex.identity_backbone_with_window_flow(𝒢) # match the backbone the identity chains were sampled in (window flow retained)
    ao_id, ao_flow = cal.out_cal_id.all_outputs, cal.out_cal_flow.all_outputs

    v = isnothing(voxel) ? argmin(map(a -> isfinite(a) ? abs(a - 0.574f0) : Inf32, id.accrat)) : voxel
    @assert v in axes(synth.b, 2)
    @info "fig:flow_geometry selection" voxel = v seed
    datav = map(d -> gpu(d[.., v:v]), (; b = synth.b, A = synth.A, mask = synth.mask, δb = synth.δb, θ = synth.θ))
    last_n(m) = m[:, max(1, size(m, 2)-1000+1):end]

    # Map log-x̄ samples to a backbone's base coordinates through its marginal posterior
    z_v = ℰ(datav.b, datav.A, datav.mask, datav.θ)
    function pull(logxbar, G)
        result = lib.chunked_apply(gpu(logxbar); chunksize = INFERENCE_CHUNKSIZE) do _x
            M = size(_x, 2)
            rep(d) = repeat(d, ntuple(i -> i < ndims(d) ? 1 : M, ndims(d))...)
            cond = map(rep, (z_v, datav.b, datav.A, datav.mask, datav.δb, datav.θ))
            return ex.flow_backward(ex.marginal_posterior(G, cond), _x).u
        end
        return cpu(result)
    end
    chain_pull(ao, G) = pull(log.(permutedims(ao.xdist[:, :, v])), G)

    # ξ-space (identity backbone) and u-space (learned backbone) representations of each chain.
    ξ_flow, ξ_id = chain_pull(ao_flow, 𝒢₀), permutedims(ao_id.summary.u[:, :, v])
    u_flow = permutedims(ao_flow.summary.u[:, :, v])

    # Joint posterior weights include η; discarding η leaves weighted x̄ samples.
    # Pool replicate columns by their mean weight exp(-ELBO_IS), preserving between-column differences
    function iw_proposal()
        rep(d) = repeat(gpu(d[.., v:v]), ntuple(i -> i < ndims(d) ? 1 : n_is_voxels, ndims(d))...)
        ep = ex.run_elbo_estimate(ℰ, 𝒢, 𝒫, rep(synth.b), rep(synth.A), rep(synth.mask), rep(synth.δb), rep(synth.θ); samples = n_is_samples, save_proposal_samples = true, verbose = false)
        Xbar = ep["XDistProposalSamples"]                                          # (S, n_xdist, W)
        logxbar = log.(reshape(permutedims(Xbar, (2, 1, 3)), size(Xbar, 2), :))    # (n_xdist, S*W)

        # Pool in log space: column-normalized log-weights shifted by each column's -ELBO_IS
        logw = vec(log.(ep["Weights_IS"]) .- ep["ELBO_IS"])
        wv = exp.(logw .- maximum(logw))
        wv ./= sum(wv)
        @info "fig:flow_geometry IW background: voxel $v, $(length(wv)) proposal draws, ESS = $(round(1 / sum(abs2, wv); digits = 1))"
        return logxbar, wv
    end
    logxbar_iw, w_iw = iw_proposal()
    ξ_iw, u_iw = pull(logxbar_iw, 𝒢₀), pull(logxbar_iw, 𝒢)

    # Importance-weighted 2D KDE with weighted Silverman bandwidth and Nₑ = 1/∑wᵢ²
    function kde_iw(s12, w, gg1, gg2)
        Neff = 1 / sum(abs2, w)
        m1, m2 = sum(w .* s12[1, :]), sum(w .* s12[2, :])
        h1 = sqrt(sum(w .* (s12[1, :] .- m1) .^ 2)) * Neff^(-1 / 6) + 1.0f-6
        h2 = sqrt(sum(w .* (s12[2, :] .- m2) .^ 2)) * Neff^(-1 / 6) + 1.0f-6
        Z = [-log(sum(@. w * exp(-((x - s12[1, :]) / h1)^2 / 2 - ((y - s12[2, :]) / h2)^2 / 2))) for x in gg1, y in gg2]
        return Z .- minimum(Z)
    end
    function wcov(X, w)
        Xc = X .- sum(w' .* X; dims = 2)
        return Symmetric((Xc .* w') * Xc')
    end

    Σξ = wcov(ξ_iw[1:40, :], w_iw)

    ev = eigen(Σξ)
    ord = sortperm(ev.values; rev = true)
    e1, e2 = ev.vectors[:, ord[1]], ev.vectors[:, ord[2]]
    projξ(M) = permutedims(hcat(e1, e2)) * M[1:40, :]

    # HMC and identity MALA chains share the windowed ξ geometry and projection basis.
    r = cal.out_cal_hmc.runs[v]
    U = reshape(r.u, size(r.u, 1), r.nchains, :)
    ξ_hmc = [projξ(U[:, c, :]) for c in 1:r.nchains]

    xlabξ, ylabξ = L"\hat{\xi}_{1}", L"\hat{\xi}_{2}"
    cblabξ = L"-\Delta\log p(\hat{\xi}_{1}, \hat{\xi}_{2} \mid b)"
    bgξ(gg1, gg2) = kde_iw(projξ(ξ_iw), w_iw, gg1, gg2)

    # Posterior energy −log p(·|b) over arbitrary base-coordinate columns P for voxel v.
    function nlp(G, P)
        energies = lib.chunked_apply(gpu32(P); chunksize = INFERENCE_CHUNKSIZE) do _u
            rep(d) = repeat(d, ntuple(i -> i < ndims(d) ? 1 : size(_u, 2), ndims(d))...)
            datap = map(rep, (z_v, datav.b, datav.A, datav.mask, datav.δb, datav.θ))
            return ex.inference_losses(ex.MAPInferenceProblem(ℰ, G, 𝒫), _u, datap...).neglogpostu
        end
        return vec(cpu(energies))
    end
    function to_ξ(U)
        nx = size(ξ_flow, 1)
        logxbar = lib.chunked_apply(gpu32(U[1:nx, :]); chunksize = INFERENCE_CHUNKSIZE) do _u
            rep(d) = repeat(d, ntuple(i -> i < ndims(d) ? 1 : size(_u, 2), ndims(d))...)
            cond = map(rep, (z_v, datav.b, datav.A, datav.mask, datav.δb, datav.θ))
            return ex.flow_forward(ex.marginal_posterior(𝒢, cond), _u).x
        end
        return vcat(pull(cpu(logxbar), 𝒢₀), U[(nx+1):end, :])
    end

    # Independent RNG keeps the marginal-density backgrounds unaffected by LERP choices.
    lerp_rng = Xoshiro(lerp_seed)
    u_corners = lerp_square(lerp_rng, u_flow)

    if lerp_2d === :normal
        # Independent draws leave the 1D endpoint pairs unchanged.
        u_corners = randn(Xoshiro(lerp_seed), eltype(u_flow), size(u_flow, 1), 3)
    elseif lerp_2d !== :square
        throw(ArgumentError("Unknown 2D LERP endpoint method: $lerp_2d"))
    end
    ξ_corners = to_ξ(u_corners)

    function plane_energy(points, G)
        plane = lerp_2d === :normal ? lerp_plane(points; padding = 0.08f0, orientation = :play) : lerp_plane(points)
        ts = lerp_2d === :normal ? range(0.0f0, 1.0f0; length = npts) : range(-0.08f0, 1.08f0; length = npts)
        Z = reshape(nlp(G, lerp_grid(plane, ts, ts)), npts, npts)
        return ts, Z .- minimum(Z), plane.points
    end

    fig = Figure(; size = (1300, 600), fontsize)
    left = fig[1, 1] = GridLayout()
    mid = fig[1, 2] = GridLayout()
    trajξ = fig[1, 3] = GridLayout()

    # Column 1: autocorrelation-time and acceptance-rate histograms, plus the posterior spectrum.
    logyticks = ([0, 1, 10, 100, 1000], ["0", "1", "10", "100", "1000"])
    ax1 = Axis(left[1, 1]; kw_ax..., xscale = log10, xticks = LogTicks(WilkinsonTicks(3; k_max = 3)), yscale = Makie.pseudolog10, yticks = logyticks, xlabel = L"Autocorr. Time $\tau_u$", ylabel = L"Count$$")
    τall = vcat(τ_flow, τ_id, τ_hmc)
    τbins = 10 .^ range(log10(minimum(τall)), log10(maximum(τall) * 1.4); length = 55)
    for (vals, c) in [(τ_flow, colors[1]), (τ_id, colors[2]), (τ_hmc, hmc_color)]
        hist!(ax1, vals; bins = τbins, color = (c, 0.55))
        vlines!(ax1, [median(vals)]; color = c, linestyle = :dash, linewidth = 1.5)
    end

    ax2 = Axis(left[2, 1]; kw_ax..., yscale = Makie.pseudolog10, yticks = logyticks, xlabel = L"Acceptance Rate$$", ylabel = L"Count$$")
    for (vals, c) in [(acc_flow, colors[1]), (acc_id, colors[2]), (acc_hmc, hmc_color)]
        hist!(ax2, vals; bins = range(0, 1; length = 40), color = (c, 0.55))
        vlines!(ax2, [median(vals)]; color = c, linestyle = :dash, linewidth = 1.5)
    end

    λξ = sort(eigvals(Σξ); rev = true)
    λu = sort(eigvals(wcov(u_iw[1:40, :], w_iw)); rev = true)
    sqrtκ_of(λ) = sqrt(λ[1] / max(λ[end], eps(Float32))) # ellipse aspect ratio of the posterior covariance
    axs = Axis(left[3, 1]; kw_ax..., xlabel = L"Mode $k$", ylabel = L"Variance $\lambda_k$", yscale = log10)
    scatterlines!(axs, 1:length(λu), max.(λu, eps(Float32)); color = colors[1], markercolor = colors[1], markersize = 4, linewidth = 1.5, strokewidth = 0, label = L"\sqrt{\kappa} = %$(round(sqrtκ_of(λu); digits = 1))")
    scatterlines!(axs, 1:length(λξ), max.(λξ, eps(Float32)); color = colors[2], markercolor = colors[2], markersize = 4, linewidth = 1.5, strokewidth = 0, label = L"\sqrt{\kappa} = %$(round(sqrtκ_of(λξ); digits = 1))")

    # Covariance spectrum of pooled HMC ξ-draws versus the IW posterior spectrum λξ
    λ_hmc = sort(eigvals(cov(Float32.(cal.out_cal_hmc.runs[v].u[1:40, :]); dims = 2)); rev = true)
    scatterlines!(axs, 1:length(λ_hmc), max.(λ_hmc, eps(Float32)); color = hmc_color, markercolor = hmc_color, markersize = 4, linewidth = 1.5, strokewidth = 0, label = L"\sqrt{\kappa} = %$(round(sqrtκ_of(λ_hmc); digits = 1))")
    axislegend(axs; position = :lb, framevisible = true, framecolor = (:white, 0.0), backgroundcolor = legbg, labelsize = lblsize)

    # Each plane uses orthogonal axes and a single common scale for both coordinates.
    for (j, corners, G, axlbl) in [(1, ξ_corners, 𝒢₀, L"$\xi$-LERP"), (2, u_corners, 𝒢, L"$u$-LERP")]
        gl = mid[1, j] = GridLayout()
        rowsize!(gl, 1, Fixed(14))
        ts, Z, markers = plane_energy(corners, G)
        clip = quantile(vec(Z), 0.92) # clip the upper bound; energy blows up away from the samples
        ticks = lerp_2d === :normal ? ([0, 0.5, 1], ["0", "0.5", "1"]) : Makie.automatic

        # Inverted grayscale (white at the peak), matching the trajectory panels
        ax = Axis(gl[2, 1]; kw_ax..., aspect = DataAspect(), valign = :top, xlabel = L"t_1", ylabel = j == 1 ? L"t_2" : "", xticks = ticks, yticks = ticks, yticklabelsvisible = j == 1, yticksvisible = j == 1, title = "")
        cf = contourf!(ax, ts, ts, Z; levels = range(0, clip; length = 16), colormap = Reverse(:grays), extendhigh = :auto)
        contour!(ax, ts, ts, clamp.(Z, 0, clip); levels = range(0, clip; length = 8), color = (:black, 0.3), linewidth = 0.6)
        scatter!(ax, markers[1, :], markers[2, :]; color = :red, markersize = 14, strokecolor = :white, strokewidth = 1.5)
        Colorbar(gl[1, 1], cf; vertical = false, size = 12, ticks = WilkinsonTicks(3))
        textlabel!(ax, 0.96, 0.96; text = axlbl, text_align = (:right, :top), space = :relative, fontsize = lblsize, background_color = legbg, draw_on_top = true, strokewidth = 0)
        xlims!(ax, extrema(ts)...)
        ylims!(ax, extrema(ts)...)
    end

    let
        K, nshow, nt = lerp_samples, 4, 61
        ts = collect(range(-0.25, 1.25; length = nt))
        u_pairs = lerp_endpoints(lerp_rng, u_flow, K; method = lerp_1d)
        ξ_pairs = map(to_ξ, u_pairs)
        paths(pair) = hcat([(1 - t) .* pair.a[:, k] .+ t .* pair.b[:, k] for k in 1:K for t in ts]...)
        Eξ, Eu = reshape(nlp(𝒢₀, paths(ξ_pairs)), nt, K), reshape(nlp(𝒢, paths(u_pairs)), nt, K)

        # A common physical reference preserves offsets between chords; no per-chord detrending.
        u0 = zeros(eltype(u_flow), size(u_flow, 1), 1)
        refξ, refu = lerp_detrend ? (0, 0) : (only(nlp(𝒢₀, to_ξ(u0))), only(nlp(𝒢, u0)))
        pξ = lerp_profile(Eξ, ts; detrend = lerp_detrend, reference = refξ)
        pu = lerp_profile(Eu, ts; detrend = lerp_detrend, reference = refu)
        ax = Axis(mid[2, 1:2]; kw_ax..., xlabel = L"LERP Fraction $t$", ylabel = L"-\Delta\log p")
        vspan!(ax, 0, 1; color = (:gray, 0.08))
        shown = rand(lerp_rng, 1:K, nshow)
        for (p, c, label) in [(pξ, colors[2], L"$\xi$-LERP"), (pu, colors[1], L"$u$-LERP")]
            if lerp_band
                band!(ax, ts, p.lower, p.upper; color = (c, 0.18))
            else
                for j in shown
                    lines!(ax, ts, p.values[:, j]; color = (c, 0.22), linewidth = 1)
                end
            end
            lines!(ax, ts, p.mean; color = c, linewidth = 3, label)
        end
        axislegend(ax; position = :rt, framevisible = true, framecolor = (:white, 0.0), backgroundcolor = legbg, labelsize = lblsize)
        ylo = min(minimum(lerp_band ? pξ.lower : pξ.mean), minimum(lerp_band ? pu.lower : pu.mean))
        yhi = max(maximum(lerp_band ? pξ.upper : pξ.mean), maximum(lerp_band ? pu.upper : pu.mean))
        d = max(yhi - ylo, 1.0f0)
        ylims!(ax, ylo - 0.15d, yhi + 0.3d)
    end

    # Boundary of the nσ covariance ellipse of a 2D point cloud `P` (2, n)
    function cov_ellipse(P; nσ = 2.0f0, m = 64)
        μ = vec(mean(P; dims = 2))
        vals, vecs = eigen(Symmetric(cov(P; dims = 2)))
        ϕ = range(0, 2π; length = m)
        return μ .+ vecs * ((nσ .* sqrt.(max.(vals, 0.0f0))) .* permutedims(hcat(cos.(ϕ), sin.(ϕ))))
    end

    # Draw one chain as translucent points joined by lines, outlined by its covariance ellipse
    function draw_chain!(ax, P_full, color, marker)
        P = last_n(P_full)
        scatter!(ax, P[1, :], P[2, :]; color = (color, 0.2), markersize = 5, marker, strokewidth = 0)
        ell = cov_ellipse(P)
        lines!(ax, ell[1, :], ell[2, :]; color = (color, 0.8), linewidth = 2.0)
        return ax
    end

    # Trajectory panels over the negative-log marginal posterior, clamped to ±traj_lim
    panels = Any[(trajξ, projξ(ξ_flow), projξ(ξ_id), bgξ, xlabξ, ylabξ, cblabξ, L"Projected $\xi$")]
    for (gl, tf_full, ti_full, bg, xlab, ylab, cblab, title) in panels
        rowsize!(gl, 1, Fixed(14))
        tf, ti = last_n(tf_full), last_n(ti_full)
        function lim(j)
            lo = min(minimum(tf[j, :]), minimum(ti[j, :]))
            hi = max(maximum(tf[j, :]), maximum(ti[j, :]))
            m = max(hi - lo, 0.1f0)
            return (max(lo - 0.08f0 * m, -traj_lim), min(hi + 0.08f0 * m, traj_lim))
        end

        x1lim, x2lim = lim(1), lim(2)
        gg1, gg2 = range(x1lim...; length = npts), range(x2lim...; length = npts)
        Z = bg(gg1, gg2)
        ax = Axis(gl[2, 1]; kw_ax..., xlabel = xlab, ylabel = ylab, title = "")

        # Inverted grayscale (white at the peak) keeps the coloured chains legible
        cf = contourf!(ax, gg1, gg2, Z; levels = range(0, quantile(vec(Z), 0.97); length = 20), colormap = Reverse(:grays), extendhigh = :auto)
        Colorbar(gl[1, 1], cf; vertical = false, size = 12, ticks = WilkinsonTicks(3))
        textlabel!(ax, 0.98, 0.98; text = cblab, text_align = (:right, :top), space = :relative, fontsize = lblsize, background_color = legbg, strokewidth = 0)

        # Draw order: flow chain, then the HMC ensemble, then the identity chain on top
        draw_chain!(ax, tf_full, colors[1], :circle)
        if gl === trajξ
            for (i, P_ξ_hmc) in enumerate(ξ_hmc)
                i > 4 && break
                draw_chain!(ax, P_ξ_hmc, hmc_color, :diamond)
            end
        end
        draw_chain!(ax, ti_full, colors[2], :rect)
        xlims!(ax, x1lim...)
        ylims!(ax, x2lim...)
    end

    # Single shared legend on the right; marker shapes match the trajectory panels.
    legels = Any[MarkerElement(; color = colors[1], marker = :circle, markersize = 12), MarkerElement(; color = colors[2], marker = :rect, markersize = 12)]
    leglbls = [L"Flow MALA$$", L"Id. MALA$$"]
    push!(legels, MarkerElement(; color = hmc_color, marker = :diamond, markersize = 12))
    push!(leglbls, L"Id. HMC$$")
    Legend(fig[1, 4], legels, leglbls; framevisible = false, tellwidth = false, tellheight = false, labelsize = fontsize)

    colsize!(fig.layout, 1, Auto(0.75))
    colsize!(fig.layout, 2, Auto(1.75))
    colsize!(fig.layout, 3, Auto(1.75))
    colsize!(fig.layout, 4, Fixed(150))
    colgap!(fig.layout, 20)
    rowgap!(fig.layout, 8)
    rowgap!(left, 7)
    rowgap!(mid, 7)
    colgap!(mid, 20)
    resize_to_layout!(fig)
    savefig(fig, figpath)
    return fig
end

#### Cross-resolution inference via the encoder-aware sketch basis `B = A_{n′} * L`

# Uniform log-T₂ grid points and spacing for `n` bins over `T2Range`, matching `epg_basis!`.
cross_res_log_t2_grid(T2Range, n::Int) = collect(Float32, range(log(T2Range[1]), log(T2Range[2]); length = n))
cross_res_dlog_t2(T2Range, n::Int) = Float32((log(T2Range[2]) - log(T2Range[1])) / (n - 1))

# Band-limited resample of a stored chain `(iters, n, voxels)` onto `n_target` via kernel `L`
function resample_chain_xdist(results, n_target::Int)
    size(results.all_outputs.xdist, 2) == n_target && return results
    logx̄ = @view results.all_outputs.summary.x[:, 1:(end-2), :]
    iters, n, vox = size(logx̄)
    x̄′ = lib.lift_resample(reshape(permutedims(logx̄, (2, 1, 3)), n, iters * vox), n_target)
    @reset results.all_outputs.xdist = permutedims(reshape(x̄′, n_target, iters, vox), (2, 1, 3))
    return results
end

# The sweep keeps only the chain fields its diagnostics read, bounding host memory across cells.
compact_chain(results) = (; all_outputs = (; results.all_outputs.xdist, results.all_outputs.α, losses = (; results.all_outputs.losses.negloglike)))

# Encoder-aware sketch basis `B = A_{n_target} * L`, broadcast over the voxel batch.
function sketch_basis(batch::NamedTuple, n_target::Int, L::AbstractMatrix)
    n_target_chk, n_trained = size(L)
    @assert n_target_chk == n_target
    A_target = epg_basis_at(batch, n_target)
    return lib.batched_mul(A_target, reshape(lib.arr_similar(A_target, L), n_target, n_trained, 1))
end

# Product lift: sample in the `n_target` latent space against `A_target`, encoding with `A_target * L`
function lifted_cell(models, batch::NamedTuple, n_target::Int; map_shift::Bool = false, map_iter::Int = 250)
    n_trained = models.genatr.opts.nT2
    A_target = epg_basis_at(batch, n_target)
    L = lib.linear_lift_matrix(n_trained, n_target)
    B_n = lib.batched_mul(A_target, reshape(lib.arr_similar(A_target, L), n_target, n_trained, 1))
    z = models.enc(batch.b, B_n, batch.mask, batch.θ)
    lifted = ex.lift_models(models, n_target)
    if map_shift
        prob = ex.MAPInferenceProblem(lifted.enc, lifted.genatr, lifted.prior)
        lr = 0.5 * n_trained / n_target
        Δu = ex.run_inference(prob, batch.b, A_target, batch.mask, batch.δb, batch.θ, z; iter = map_iter, newton = true, lr).u[1:(end-2), :]
        lifted = @set lifted.genatr = ex.prepend_fwd_backbone(lifted.genatr, lib.ShiftFlow(Δu))
    end
    return (; lifted, A_target, z)
end

# Prepend u ↦ u + u⋆, where u⋆ minimizes the sketch-posterior energy with uη = 0.
function map_shift_generator(models, batch, B; iter::Int = 250)
    prob = ex.MAPInferenceProblem(models.enc, models.genatr, models.prior)
    Δu = ex.run_inference(prob, batch.b, B, batch.mask, batch.δb, batch.θ; iter, newton = true, lr = 0.5).u[1:(end-2), :] # drop the zero-padded nuisance dims; the xdist backbone shift is the first udim-2
    return ex.prepend_fwd_backbone(models.genatr, lib.ShiftFlow(Δu))
end

# Sample against B = Aₙ′L in native coordinates, then report Lx̄ on the target grid.
# The sweep shares one optional MAP shift across all samplers in each cell.
function run_mcmc_sampling_sketch(batch::NamedTuple, models, alg; n_target, iter = 1000, kwargs...)
    L = lib.linear_lift_matrix(models.genatr.opts.nT2, n_target)
    B = sketch_basis(batch, n_target, L)
    z = models.enc(batch.b, B, batch.mask, batch.θ)
    samp_prob = ex.SamplingProblem(alg, models.enc, models.genatr, models.prior)
    results = ex.run_sampling(samp_prob, z, batch.b, B, batch.mask, batch.δb, batch.θ; iter, warmup_iter = 0, save_all_samples = true, verbose = true, kwargs...)
    return (; results = compact_chain(resample_chain_xdist(results, n_target)), n_target, T2Range = batch.T2Range[:, 1], kernel = :triangular) |> cpu
end

# IS diagnostics against B = Aₙ′L using the cell generator.
function run_elbo_estimate_sketch(batch::NamedTuple, models; n_target, samples = 1000)
    L = lib.linear_lift_matrix(models.genatr.opts.nT2, n_target)
    B = sketch_basis(batch, n_target, L)
    z = models.enc(batch.b, B, batch.mask, batch.θ)
    results = ex.run_elbo_estimate(models.genatr, models.prior, z, batch.b, B, batch.mask, batch.δb, batch.θ; samples, save_proposal_samples = false, verbose = true)
    return (; results, n_target, T2Range = batch.T2Range[:, 1], kernel = :triangular) |> cpu
end

# Product lift samples on the target grid, so `resample_chain_xdist` is a no-op
function run_mcmc_sampling_lifted(batch::NamedTuple, lifted, A_target, z, alg; n_target, iter = 1000, kwargs...)
    samp_prob = ex.SamplingProblem(alg, lifted.enc, lifted.genatr, lifted.prior)
    results = ex.run_sampling(samp_prob, z, batch.b, A_target, batch.mask, batch.δb, batch.θ; iter, warmup_iter = 0, save_all_samples = true, verbose = true, kwargs...)
    return (; results = compact_chain(resample_chain_xdist(results, n_target)), n_target, T2Range = batch.T2Range[:, 1], kernel = :product) |> cpu
end

# Full `n_target`-dimensional IS diagnostics for the product lift.
function run_elbo_estimate_lifted(batch::NamedTuple, lifted, A_target, z; n_target, samples = 1000)
    results = ex.run_elbo_estimate(lifted.genatr, lifted.prior, z, batch.b, A_target, batch.mask, batch.δb, batch.θ; samples, save_proposal_samples = false, verbose = true)
    return (; results, n_target, T2Range = batch.T2Range[:, 1], kernel = :product) |> cpu
end

# Sweep `kinds × resolutions` with independent MH, RWM, MALA and an IS ELBO estimate per cell.
# At `n = n_trained` cells are stored under `:none`; a `_map` suffix enables MAP recentring.
cross_resolution_base_kind(kind::Symbol) = endswith(string(kind), "_map") ? Symbol(chop(string(kind); tail = 4)) : kind

function run_cross_resolution_sweep(models = run_env.models; seed = 0, n_voxels = nothing, map_iter = 250, resolutions = (41, 81, 121), kinds = (:triangular, :triangular_map, :product, :product_map), mh_iter = 1000, mala_iter::Int = mh_iter, is_samples = 1000)
    seed_all!(seed)
    batch = run_env.val_cohort |> gpu
    n_voxels !== nothing && (batch = map(x -> x[.., 1:min(n_voxels, size(batch.b, 2))], batch))
    n_trained = models.genatr.opts.nT2
    runs = Dict{Tuple{Symbol, Int}, Any}()
    function record_sketch!(key_kind, n, map_shift)
        # Share one MAP shift across all samplers in this cell.
        _models = map_shift ? (@set models.genatr = map_shift_generator(models, batch, sketch_basis(batch, n, lib.linear_lift_matrix(n_trained, n)); iter = map_iter)) : models
        runs[(key_kind, n)] = (;
            mh   = run_mcmc_sampling_sketch(batch, _models, ex.Metropolis(); n_target = n, iter = mh_iter, lr = nothing),
            rwm  = run_mcmc_sampling_sketch(batch, _models, ex.Metropolis(); n_target = n, iter = mh_iter),
            mala = run_mcmc_sampling_sketch(batch, _models, ex.MALA(); n_target = n, iter = mala_iter),
            is   = run_elbo_estimate_sketch(batch, _models; n_target = n, samples = is_samples),
        )
    end
    function record_lifted!(key_kind, n, map_shift)
        cell = lifted_cell(models, batch, n; map_shift, map_iter)
        runs[(key_kind, n)] = (;
            mh   = run_mcmc_sampling_lifted(batch, cell.lifted, cell.A_target, cell.z, ex.Metropolis(); n_target = n, iter = mh_iter, lr = nothing),
            rwm  = run_mcmc_sampling_lifted(batch, cell.lifted, cell.A_target, cell.z, ex.Metropolis(); n_target = n, iter = mh_iter),
            mala = run_mcmc_sampling_lifted(batch, cell.lifted, cell.A_target, cell.z, ex.MALA(); n_target = n, iter = mala_iter),
            is   = run_elbo_estimate_lifted(batch, cell.lifted, cell.A_target, cell.z; n_target = n, samples = is_samples),
        )
    end
    record_sketch!(:none, n_trained, false)
    @assert all(k -> k in (:triangular, :triangular_map, :product, :product_map), kinds)
    for n in resolutions, kind in kinds
        map_shift = endswith(string(kind), "_map")
        kernel = cross_resolution_base_kind(kind)
        if n == n_trained
            key = map_shift ? :none_map : :none # L = I at n = n_trained
            haskey(runs, (key, n)) || record_sketch!(key, n, map_shift)
        elseif kernel === :product
            record_lifted!(kind, n, map_shift)
        else
            record_sketch!(kind, n, map_shift)
        end
    end
    return runs
end
# Per-iteration mean (over voxels) of `all_outputs.losses.negloglike`, as a Vector{Float64}.
nll_trace(results) = vec(Float64.(mean(results.all_outputs.losses.negloglike; dims = 2:ndims(results.all_outputs.losses.negloglike))))

# Voxelwise observables (MWF / IEWF / CWF and mean-T2 variants) from the posterior-mean xdist
function observables_per_voxel(run; SPWin::Real = 32.0f-3, MPWin::Real = 200.0f-3, burn_in_frac::Real = 0.1f0)
    chain = run.results.all_outputs.xdist
    n_burn = post_burn_start(size(chain, 1), burn_in_frac)
    x̄ = dropdims(mean(@view(chain[n_burn:end, :, :]); dims = 1); dims = 1) # (nT2, nvox)
    logT2s = cross_res_log_t2_grid(run.T2Range, run.n_target)
    return compute_observable_maps(x̄, logT2s; SPWin, MPWin)
end

# Voxelwise integrated autocorrelation time of each observable chain
function observable_autocorr_per_voxel(run; SPWin::Real = 32.0f-3, MPWin::Real = 200.0f-3, burn_in_frac::Real = 0.1f0)
    chain = run.results.all_outputs.xdist
    n_burn = post_burn_start(size(chain, 1), burn_in_frac)
    logT2s = cross_res_log_t2_grid(run.T2Range, run.n_target)
    return chain_observable_autocorr_times(@view(chain[n_burn:end, :, :]), logT2s; SPWin, MPWin)
end

function acceptance_per_voxel(run; burn_in_frac::Real = 0)
    α = run.results.all_outputs.α
    r = post_burn_start(size(α, 1), burn_in_frac):size(α, 1)
    return Float64.(vec(mean(selectdim(α, 1, r); dims = 1:(ndims(α)-1))))
end

# Voxelwise normalized effective sample size from independent proposal draws.
is_ess_per_voxel(run) = Float64.(vec(run.results["ESS_IS"]))

# Per-iteration observable traces for one voxel.
function chain_observable_traces(run, voxel::Int; SPWin::Real = 32.0f-3, MPWin::Real = 200.0f-3)
    chain = run.results.all_outputs.xdist[:, :, voxel]
    logT2s = cross_res_log_t2_grid(run.T2Range, run.n_target)
    return compute_observable_maps(permutedims(chain, (2, 1)), logT2s; SPWin, MPWin)
end

# τᵢₙₜ = ½ + ∑ₖ≥₁ ρₖ; report ESS/S = 1/(2τᵢₙₜ), capped at 1 for these diagnostics.
function inv_tau_observables(run; SPWin::Real = 32.0f-3, MPWin::Real = 200.0f-3, burn_in_frac::Real = 0.1f0)
    τbm = observable_autocorr_per_voxel(run; SPWin, MPWin, burn_in_frac)
    return map(τ -> 1.0 ./ max.(2.0 .* Float64.(τ), 1.0), τbm)
end

function tau_per_voxel(run_bundle, sampler::Symbol; pool::Symbol, kwargs...)
    τ⁻¹ = inv_tau_observables(getfield(run_bundle, sampler); kwargs...)
    return 1 ./ Float64.(vec(getfield(τ⁻¹, pool)))
end

# Per-iter cohort acceptance summary: mean across voxels with q/(1-q) quantile band.
function accept_trace_iqr(results, q = 0.25)
    α = results.all_outputs.α
    αflat = reshape(α, size(α, 1), :)
    if q === nothing
        return dropdims(mean(αflat; dims = 2); dims = 2), nothing, nothing
    end
    n_iter = size(αflat, 1)
    mid = Vector{Float64}(undef, n_iter)
    lo = Vector{Float64}(undef, n_iter)
    hi = Vector{Float64}(undef, n_iter)
    Threads.@threads for i in 1:n_iter
        lo[i], mid[i], hi[i] = quantile(αflat[i, :], [q, 0.5, 1 - q])
    end
    return mid, lo, hi
end

function rolling_mean(x::AbstractVector, window::Int)
    y = similar(Float64.(x))
    for i in eachindex(x)
        lo = max(firstindex(x), i - window + 1)
        y[i] = mean(@view x[lo:i])
    end
    return y
end

function voxel_mean_trace(a, voxel::Int)
    sl = selectdim(a, ndims(a), voxel)
    ndims(sl) == 1 && return Float64.(vec(sl))
    return vec(Float64.(mean(sl; dims = Tuple(2:ndims(sl)))))
end

rolling_acceptance(run, voxel::Int; window::Int = 25) = rolling_mean(voxel_mean_trace(run.results.all_outputs.α, voxel), window)

function cross_resolution_styles(runs::AbstractDict)
    kinds = sort(unique(first.(keys(runs))))
    resolutions = sort(unique(last.(keys(runs))))
    base_ls = Dict(:none => :solid, :triangular => :solid, :product => :dash, :triangular_cc => :dash, :bspline => :dashdot, :bspline_cc => :dashdotdot, :gaussian => :dash, :gaussian_cc => :dot, :gaussian_tuned => :dot, :constant_cc => :dashdot)
    return (;
        kinds,
        resolutions,
        colors = Makie.wong_colors()[1:length(resolutions)],
        adapter_ls = Dict(k => get(base_ls, cross_resolution_base_kind(k), :solid) for k in kinds), # `_map` kinds inherit their base kernel's linestyle
        sampler_ls = Dict(:mh => :solid, :rwm => :dash, :mala => :dashdot, :is => :dot),
        sampler_order = (:mh, :rwm, :mala),
        sampler_markers = Dict(:mh => :circle, :rwm => :rect, :mala => :utriangle, :is => :diamond),
        pool_markers = Dict(:mwf => :circle, :iewf => :diamond, :cwf => :rect, :gmT2 => :utriangle),
        pool_labels = Dict(:mwf => L"\mathrm{MWF}", :iewf => L"\mathrm{IEWF}", :cwf => L"\mathrm{CWF}", :gmT2 => L"GM $T_2$"),
        sampler_labels = Dict(:mh => "MH", :rwm => "RWM", :mala => "MALA", :is => "IS"),
    )
end

function cross_resolution_plot_keys(runs::AbstractDict, kind::Symbol)
    baseline = sort([k for k in keys(runs) if first(k) === :none]; by = last)
    selected = sort([k for k in keys(runs) if first(k) === kind]; by = last)
    kind !== :none && isempty(selected) && error("no runs for kind = $kind")
    ks = kind === :none ? baseline : vcat(baseline, selected)
    isempty(ks) && error("no runs for kind = $kind")
    return ks
end

cross_resolution_color(st, k) = st.colors[findfirst(==(last(k)), st.resolutions)]
cross_resolution_label(k) = first(k) === :none ? "Native n=$(last(k))" : "n=$(last(k))"

even_marker_indices(N::Int; nmark::Int = 16) = unique(round.(Int, range(1, N; length = min(nmark, N))))

function stephist_fill_marker!(ax, vals; bins = 20, normalization = :probability, color, alpha = 0.25, linestyle = :solid, linewidth = 2, marker = :circle, markersize = 10, peakmarker::Bool = true)
    p = stephist!(ax, vals; bins, normalization, color, linestyle, linewidth)
    pts = p.points[]
    length(pts) < 3 && return p
    centers = [(pts[i][1] + pts[i+1][1]) / 2 for i in 1:(length(pts)-2)]
    heights = [pts[i+1][2] for i in 1:(length(pts)-2)]
    widths = [pts[i+1][1] - pts[i][1] for i in 1:(length(pts)-2)]
    bp = barplot!(ax, centers, heights; width = widths, color = (color, alpha), strokewidth = 0, gap = 0, dodge_gap = 0)
    if peakmarker
        im = argmax(heights)
        scatter!(ax, centers[im], heights[im]; color, marker, markersize)
    end
    return p
end

function cross_resolution_summary_rows(runs::AbstractDict; SPWin::Real = 32.0f-3, MPWin::Real = 200.0f-3, burn_in_frac::Real = 0.1f0)
    # Quartiles describe the skewed voxelwise distributions without a Gaussian approximation.
    function mq(v)
        w = filter(isfinite, Float64.(vec(v)))
        return (; med = median(w), q1 = quantile(w, 0.25), q3 = quantile(w, 0.75))
    end

    keys_sorted = sort(collect(keys(runs)))
    rows = map(keys_sorted) do key
        kind, n = key
        run = runs[key]
        nl = nll_trace(run.mh.results)
        τ_mh = observable_autocorr_per_voxel(run.mh; SPWin, MPWin, burn_in_frac)
        τ_rwm = observable_autocorr_per_voxel(run.rwm; SPWin, MPWin, burn_in_frac)
        τ_mala = observable_autocorr_per_voxel(run.mala; SPWin, MPWin, burn_in_frac)
        poolstats(τ) = map(t -> mq(max.(2 .* vec(t), 1.0)), τ)
        ess = is_ess_per_voxel(run.is)
        (; kind, n, init_nll = nl[1], final_nll = mean(@view nl[post_burn_start(length(nl), burn_in_frac):end]),
            indep_α = mq(acceptance_per_voxel(run.mh; burn_in_frac)), rwm_α = mq(acceptance_per_voxel(run.rwm; burn_in_frac)), mala_α = mq(acceptance_per_voxel(run.mala; burn_in_frac)),
            ess = mq(ess), mh_τ = poolstats(τ_mh).mwf, rwm_τ = poolstats(τ_rwm), mala_τ = poolstats(τ_mala))
    end
    return rows
end

# Median and quartiles across voxels. IS ESS/S is separate from MCMC τ = 2τᵢₙₜ (MWF).
function cross_resolution_diagnostics_table(runs::AbstractDict; SPWin::Real = 32.0f-3, MPWin::Real = 200.0f-3, burn_in_frac::Real = 0.1f0, figpath = figuresdir("cross_resolution_diagnostics_table.tex"))
    column_labels = [[L"\mathrm{Method}", L"$n_{T_2}$", L"$\mathrm{Acc.}_{\mathrm{MH}}$", L"$\mathrm{Acc.}_{\mathrm{RWM}}$", L"$\mathrm{Acc.}_{\mathrm{MALA}}$", L"$\mathrm{ESS}_{\mathrm{IS}}/S$", L"$\tau_{\mathrm{MH}}$", L"$\tau_{\mathrm{RWM}}$", L"$\tau_{\mathrm{MALA}}$"]]
    rows = sort!(cross_resolution_summary_rows(runs; SPWin, MPWin, burn_in_frac); by = r -> (r.n, r.kind))
    method_labels = Dict(:none => "Native", :triangular => "Pseudo-lift", :product => "Product", :triangular_cc => "Triangular (CC)", :bspline => "B-spline", :bspline_cc => "B-spline (CC)", :gaussian => "Gaussian", :gaussian_cc => "Gaussian (CC)", :gaussian_tuned => "Gaussian (tuned)", :constant_cc => "Constant (CC)")
    methods = [
        begin
            ismap = endswith(string(r.kind), "_map")
            base = cross_resolution_base_kind(r.kind)
            get(method_labels, base, uppercasefirst(string(base))) * (ismap ? " + MAP" : "")
        end for r in rows
    ]
    data = Matrix{Any}(undef, length(rows), length(only(column_labels)))
    for (i, row) in enumerate(rows)
        data[i, :] .= (methods[i], row.n, row.indep_α, row.rwm_α, row.mala_α, row.ess, row.mh_τ, row.rwm_τ.mwf, row.mala_τ.mwf)
    end
    function cell_formatter(v, i, j)
        v isa NamedTuple || return v
        return LatexCell(latexstring("$(round(v.med; sigdigits = 3))^{$(round(v.q3; sigdigits = 2))}_{$(round(v.q1; sigdigits = 2))}"))
    end
    save_table(figpath) do io
        default_pretty_table(io, data; column_labels, formatters = [cell_formatter])
    end
    return data
end

# Cohort diagnostics for one adapter `kind`, always including the native `:none` baseline.
# Compares the three MCMC samplers (MH, RWM, MALA) across resolutions from an inter-voxel perspective.
function plot_cross_resolution_cohort_diagnostics(runs::AbstractDict; kind::Symbol = :triangular_map, burn_in_frac::Real = 0.1f0, SPWin::Real = 32.0f-3, MPWin::Real = 200.0f-3, figpath = figuresdir("cross_resolution_cohort_diagnostics.pdf"))
    st = cross_resolution_styles(runs)
    plot_keys = cross_resolution_plot_keys(runs, kind)
    fig = Figure(; size = (1000, 400))
    pool_order = (:mwf, :iewf, :cwf, :gmT2)

    # Thin per-iteration cohort mean plus thick cumulative running mean with sampler markers.
    ax_acc = Axis(fig[1, 1]; xlabel = L"Sample $S$", ylabel = L"Mean Acceptance Ratio$$", limits = ((nothing, nothing), (0, 1)))
    target_acceptance_rates = (0.234, 0.574)
    hlines!(ax_acc, collect(target_acceptance_rates); color = (:gray50, 0.55), linestyle = :dash, linewidth = 1.2)
    for key in plot_keys
        for method in st.sampler_order
            mid, _, _ = accept_trace_iqr(getfield(runs[key], method).results, nothing)
            N = length(mid)
            cum = cumsum(mid) ./ (1:N)
            mi = even_marker_indices(N)
            color = cross_resolution_color(st, key)
            lines!(ax_acc, 1:N, mid; color, linestyle = st.sampler_ls[method], linewidth = 0.4)
            lines!(ax_acc, 1:N, cum; color, linestyle = st.sampler_ls[method], linewidth = 2)
            scatter!(ax_acc, mi, cum[mi]; color, marker = st.sampler_markers[method], markersize = 7)
        end
    end

    ax_prop = Axis(fig[1, 2]; xlabel = L"Acceptance$$", ylabel = L"Density$$", limits = ((0, 1), (nothing, nothing)))
    vlines!(ax_prop, collect(target_acceptance_rates); color = (:gray50, 0.55), linestyle = :dash, linewidth = 1.2)
    stat_specs = (
        (:mh, st.sampler_ls[:mh], st.sampler_markers[:mh]),
        (:rwm, st.sampler_ls[:rwm], st.sampler_markers[:rwm]),
        (:mala, st.sampler_ls[:mala], st.sampler_markers[:mala]),
    )
    for key in plot_keys
        run = runs[key]
        for (stat, ls, marker) in stat_specs
            stephist_fill_marker!(ax_prop, clamp.(acceptance_per_voxel(getfield(run, stat); burn_in_frac), 0, 1); color = cross_resolution_color(st, key), linestyle = ls, marker)
        end
    end

    ax_eff = Axis(fig[1, 3]; ylabel = L"Density$$", xlabel = L"\log_{10}\tau_{\mathrm{MWF}}")

    ax_is = Axis(fig[1, 4]; xlabel = L"IS $\mathrm{ESS}/S$", ylabel = L"Density$$", limits = ((0, 1), (nothing, nothing)))

    zoom_ranges = map(pool_order) do pool
        vals = Float64[]
        for key in plot_keys
            bm = observables_per_voxel(runs[key].mh; SPWin, MPWin, burn_in_frac)
            append!(vals, vec(getfield(bm, pool)))
        end
        (quantile(vals, 0.01), quantile(vals, 0.99))
    end |> v -> NamedTuple{pool_order}(v)
    bm_axes = NamedTuple{pool_order}(map(enumerate(pool_order)) do (i, pool)
        Axis(fig[2, i]; ylabel = st.pool_labels[pool], xlabel = L"Posterior mean$$", xticks = WilkinsonTicks(3), limits = (zoom_ranges[pool], (nothing, nothing)))
    end)

    for key in plot_keys
        color = cross_resolution_color(st, key)
        for sampler in st.sampler_order
            vals = tau_per_voxel(runs[key], sampler; pool = :mwf, SPWin, MPWin, burn_in_frac)
            stephist_fill_marker!(ax_eff, log10.(filter(>(0), vals)); color, linestyle = st.sampler_ls[sampler], marker = st.sampler_markers[sampler])
        end
        stephist_fill_marker!(ax_is, clamp.(is_ess_per_voxel(runs[key].is), 0, 1); color, linestyle = st.sampler_ls[:is], marker = st.sampler_markers[:is])
        bm = observables_per_voxel(runs[key].mh; SPWin, MPWin, burn_in_frac)
        for pool in pool_order
            density!(getfield(bm_axes, pool), vec(getfield(bm, pool)); color = :transparent, strokecolor = color, strokewidth = 2)
        end
    end

    res_elems = [LineElement(; color = cross_resolution_color(st, key), linewidth = 3) for key in plot_keys]
    res_labels = cross_resolution_label.(plot_keys)
    sampler_keys = (:mh, :rwm, :mala, :is)
    sampler_elems = [[LineElement(; color = :black, linestyle = st.sampler_ls[s], linewidth = 2), MarkerElement(; color = :black, marker = st.sampler_markers[s], markersize = 8)] for s in sampler_keys]
    sampler_labels = [s === :is ? L"IS $\mathrm{ESS}/S$" : st.sampler_labels[s] for s in sampler_keys]
    target_elems = [LineElement(; color = (:gray50, 0.55), linestyle = :dash, linewidth = 1.2) for _ in target_acceptance_rates]
    target_labels = [L"RWM target $0.234$", L"MALA target $0.574$"]
    Legend(fig[1:2, 5],
        [res_elems, sampler_elems, target_elems],
        [res_labels, sampler_labels, target_labels],
        ["Resolution", "MCMC Sampler", "Target Acceptance"];
        framevisible = false, labelsize = 11, titlesize = 12, patchsize = (14, 12), patchlabelgap = 4,
    )
    rowsize!(fig.layout, 1, Auto(3))
    rowsize!(fig.layout, 2, Auto(1))

    for c in fig.content
        c isa Axis && (c.xgridvisible = false; c.ygridvisible = false)
    end
    savefig(fig, figpath)
    return fig
end

# Voxel-level diagnostics for one adapter `kind`, always including the native `:none` baseline.
function plot_cross_resolution_voxel_diagnostics(runs::AbstractDict; voxel::Int = 1, kind::Symbol = :triangular_map, burn_in_frac::Real = 0.1f0, SPWin::Real = 32.0f-3, MPWin::Real = 200.0f-3, window::Int = 25, figpath = figuresdir("cross_resolution_voxel_diagnostics.pdf"))
    st = cross_resolution_styles(runs)
    plot_keys = cross_resolution_plot_keys(runs, kind)
    fig = Figure(; size = (900, 520))
    pool_order = (:mwf, :iewf, :cwf, :gmT2)
    pool_sampler = (mwf = :mh, iewf = :rwm, cwf = :mala, gmT2 = :mh)

    left = fig[1, 1] = GridLayout()
    ax_t2 = Axis(left[1, 1]; xscale = log10, xlabel = L"T_2\;(\mathrm{s})", ylabel = L"\bar{x} / \Delta\log T_2")
    vlines!(ax_t2, [SPWin, MPWin]; color = (:black, 0.35), linestyle = :dash, linewidth = 1)
    for key in plot_keys
        mh = runs[key].mh
        n = last(key)
        ℓ = cross_res_log_t2_grid(mh.T2Range, n)
        Δℓ = cross_res_dlog_t2(mh.T2Range, n)
        chain = mh.results.all_outputs.xdist[:, :, voxel]
        n_burn = post_burn_start(size(chain, 1), burn_in_frac)
        x̄ = vec(mean(chain[n_burn:end, :]; dims = 1))
        lines!(ax_t2, exp.(ℓ), x̄ ./ Δℓ; color = cross_resolution_color(st, key), linewidth = 2)
    end

    ax_acc = Axis(left[2, 1]; xlabel = L"Sample $S$", ylabel = L"Rolling Acceptance Ratio$$", limits = ((nothing, nothing), (0, 1)))
    for key in plot_keys
        for sampler in st.sampler_order
            run = getfield(runs[key], sampler)
            ar = rolling_acceptance(run, voxel; window)
            color = cross_resolution_color(st, key)
            lines!(ax_acc, 1:length(ar), ar; color, linestyle = st.sampler_ls[sampler], linewidth = 0.4)
            acc = voxel_mean_trace(run.results.all_outputs.α, voxel)
            N = length(acc)
            cum = cumsum(acc) ./ (1:N)
            mi = even_marker_indices(N)
            lines!(ax_acc, 1:N, cum; color, linestyle = st.sampler_ls[sampler], linewidth = 2)
            scatter!(ax_acc, mi, cum[mi]; color, marker = st.sampler_markers[sampler], markersize = 7)
        end
    end

    # Four rows MWF / IEWF / CWF / GM T2; only the trace column carries pool y-labels.
    right = fig[1, 2] = GridLayout()
    trace_axes = NamedTuple{pool_order}(map(enumerate(pool_order)) do (i, pool)
        Axis(right[i, 1]; ylabel = st.pool_labels[pool], xlabel = i == length(pool_order) ? L"Sample $S$" : "")
    end)
    pdf_axes = NamedTuple{pool_order}(map(enumerate(pool_order)) do (i, pool)
        Axis(right[i, 2]; xlabel = i == length(pool_order) ? L"Posterior mean$$" : "", ylabel = L"Density$$", xticks = WilkinsonTicks(3))
    end)
    for key in plot_keys
        color = cross_resolution_color(st, key)
        for pool in pool_order
            sampler = getfield(pool_sampler, pool)
            traces = chain_observable_traces(getfield(runs[key], sampler), voxel; SPWin, MPWin)
            vals = Float64.(vec(getfield(traces, pool)))
            N = length(vals)
            cum = cumsum(vals) ./ (1:N)
            mi = even_marker_indices(N)
            ax = getfield(trace_axes, pool)
            lines!(ax, 1:N, vals; color, linestyle = st.sampler_ls[sampler], linewidth = 0.4)
            lines!(ax, 1:N, cum; color, linestyle = st.sampler_ls[sampler], linewidth = 2)
            scatter!(ax, mi, cum[mi]; color, marker = st.sampler_markers[sampler], markersize = 7)

            v = vals[post_burn_start(N, burn_in_frac):end]
            stephist_fill_marker!(getfield(pdf_axes, pool), v; color, linestyle = st.sampler_ls[sampler], peakmarker = false)
        end
    end

    res_elems = [LineElement(; color = cross_resolution_color(st, key), linewidth = 3) for key in plot_keys]
    res_labels = cross_resolution_label.(plot_keys)
    sampler_elems = [[LineElement(; color = :black, linestyle = st.sampler_ls[s], linewidth = 2), MarkerElement(; color = :black, marker = st.sampler_markers[s], markersize = 8)] for s in st.sampler_order]
    sampler_labels = [st.sampler_labels[s] for s in st.sampler_order]
    Legend(fig[1, 3],
        [res_elems, sampler_elems],
        [res_labels, sampler_labels],
        ["Resolution", "MCMC Sampler"];
        framevisible = false, labelsize = 11, titlesize = 12, patchsize = (14, 12), patchlabelgap = 4,
    )
    colsize!(fig.layout, 1, Auto(1))
    colsize!(fig.layout, 2, Auto(2))

    for c in fig.content
        c isa Axis && (c.xgridvisible = false; c.ygridvisible = false)
    end
    savefig(fig, figpath)
    return fig
end

#### Manuscript figure drivers

function run_shared_inference!(; seed = 0, quick = false)
    global decaes_results, elbo_results, mala_results, map_results, pseudomap_xdist, map_results_id, xdist_results
    decaes_results = run_decaes()
    elbo_results = run_elbo_metrics(; seed, samples = quick ? 128 : 1000)
    mala_results = run_mcmc_sampling(; seed, iter = quick ? 128 : 1000)
    map_results = run_map_inference(; seed, iter = quick ? 5 : 50)
    pseudomap_xdist = sample_forward_flow(; seed)
    map_results_id = run_map_inference_id(; seed, iter = quick ? 5 : 100)
    xdist_results = compute_xdist_results()
    return nothing
end

# HMC and NUTS references, using the held-out diagnostic voxels
function run_reference_samples!(; seed = 0, quick = false, nuts_backbone = :flow, nuts_max_depth = 10)
    global val_diag_results = run_val_diagnostics(; seed, n_voxels = quick ? 64 : nothing, n_samples = quick ? 128 : 1000)
    run_posterior_samples_data(; seed, nsamples_hmc = quick ? 0 : 1000, nsamples_nuts = quick ? 0 : 5000, nvoxels_nuts = quick ? 3 : 96, nuts_backbone, nuts_max_depth)
    return nothing
end

function generate_main_figures!(; seed = 0, quick = false, references = true, nuts_backbone = :flow, nuts_max_depth = 10)
    global posterior_surface_results, synth_eval_results, val_diag_results, posterior_samples_results, cal_data, ablation_data, comparison_data
    plot_xdist_comparison_combined()
    compare_inference_method_slices()
    plot_prior_lowpass_filter(; seed)
    posterior_surface_results = compute_posterior_surface_results(; npts = quick ? (31, 31) : (251, 251))
    plot_posterior_geometry()
    synth_eval_results = run_synthetic_eval(; seed, n_voxels = quick ? 64 : nothing, n_samples = quick ? 128 : 1000)
    write_synthetic_table()
    plot_synthetic_voxel()
    cal_data = run_calibration_data()
    plot_calibration()
    for snr in synth_eval_results.SNRs[1:(end-1)]
        plot_calibration(run_calibration_data(; snr); figpath = figuresdir("calibration_$(snr)dB.pdf"))
    end
    bimodal = run_synthetic_eval(; seed, spectrum_source = :bimodal, n_voxels = quick ? 64 : nothing, n_samples = quick ? 128 : 1000)
    write_synthetic_table(bimodal; figpath = figuresdir("synthetic_bimodal_table.tex"))
    plot_synthetic_prior_comparison(bimodal, synth_eval_results)
    val_diag_results = run_val_diagnostics(; seed, n_voxels = quick ? 64 : nothing, n_samples = quick ? 128 : 1000)
    plot_psis_and_ess()
    plot_mcmc_diagnostics()
    posterior_samples_results = run_posterior_samples_data(; seed, nsamples_hmc = references && !quick ? 1000 : 0, nsamples_nuts = references && !quick ? 5000 : 0, nvoxels_nuts = quick ? 3 : 96, nuts_backbone, nuts_max_depth)
    plot_posterior_samples()
    if posterior_samples_results.hmc_ref !== nothing && posterior_samples_results.nuts_ref !== nothing
        write_reference_diagnostics_table(posterior_samples_results.hmc_ref, posterior_samples_results.nuts_ref)
    end
    ablation_data = run_mcmc_ablation(; seed, n_voxels = quick ? 16 : nothing, n_samples = quick ? 128 : 1000, warmup_id = quick ? 100 : 4000, warmup_hmc = quick ? 100 : 1000, nchains_hmc = quick ? 2 : 8)
    plot_flow_geometry(; seed, voxel = quick ? nothing : 124, npts = quick ? 31 : 121, n_is_voxels = quick ? 64 : 1000, n_is_samples = quick ? 8 : 16)
    comparison_data = run_method_backbone_comparison(; seed, n_voxels = quick ? 16 : nothing, n_samples = quick ? 128 : 1000, warmup_id = quick ? 100 : 4000, warmup_hmc = quick ? 100 : 1000, sweep_iter = quick ? 64 : 500)
    write_method_backbone_table(comparison_data)
    return nothing
end

function run_flip_angle_data!(; seed = 0, quick = false)
    global fa_results_pseudo, fa_results_poly, fa_results_elbo, fa_results_pmmh, fa_observable_results
    fa_results_pseudo = optimize_flip_angle_elbo(; seed, mode = :direct, objective = :pseudo_map, iter = quick ? 5 : 100)
    fa_results_poly = optimize_flip_angle_elbo(; seed, mode = :poly, objective = :em, degree = 6, iter = quick ? 5 : 100)
    fa_results_elbo = optimize_flip_angle_elbo(; seed, mode = :direct, objective = :em, iter = quick ? 5 : 100, sample_newton_samples = 8)
    fa_results_pmmh = optimize_flip_angle_elbo(; seed, mode = :direct, objective = :pmmh, pmmh_iters = quick ? 30 : 250, pmmh_burnin = quick ? 10 : 50)
    fa_observable_results = run_flip_angle_observables(fa_results_elbo, fa_results_poly; seed, iter = quick ? 128 : 1000)
    return nothing
end

function make_flip_angle_figures!()
    plot_flip_angle_comparison(fa_results_pmmh, fa_results_pseudo, fa_results_poly)
    write_flip_angle_table(fa_results_pmmh, fa_results_pseudo, fa_results_poly)
    plot_flip_angle_observables()
    return nothing
end

function run_spatial_data!(; seed = 0, quick = false)
    global spatial_operators, spatial_em_results, spatial_map_results
    spatial_operators = build_spatial_operators(; seed)
    spatial_em_results = fit_spatial_hyperparameters_em(gpu32(spatial_operators.op), spatial_operators.stieltjes; seed, nsamples = quick ? 8 : 50)
    spatial_map_results = run_spatial_map_optimization(spatial_em_results; seed, nsteps = quick ? 5 : 100)
    return nothing
end

function make_spatial_figures!()
    plot_spatial_map_visualizations()
    plot_spatial_map_ilr_fields()
    return nothing
end

function run_cross_resolution_data!(; seed = 0, quick = false)
    global lifted_runs
    lifted_runs = run_cross_resolution_sweep(; seed, n_voxels = quick ? 16 : nothing, map_iter = quick ? 5 : 250, mh_iter = quick ? 128 : 1000, is_samples = quick ? 128 : 1000)
    return nothing
end

function make_cross_resolution_figures!()
    cross_resolution_diagnostics_table(lifted_runs)
    plot_cross_resolution_cohort_diagnostics(lifted_runs)
    plot_cross_resolution_voxel_diagnostics(lifted_runs)
    return nothing
end
