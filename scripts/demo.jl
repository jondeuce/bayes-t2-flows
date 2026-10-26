using Library
using CairoMakie
const lib = Library
const ex = Library.Experiments

function demo(args)
    run_dir = isempty(args) ? lib.projectdir("outputs", "demo-run") : only(args)
    abspath(run_dir)
    (; cfg, data_cfg, models) = ex.load_checkpoint(run_dir)
    data = ex.dataset(cfg.data.path::String)
    test_data = ex.evaluation_partition(data, data_cfg)
    loader = ex.data_loader(cfg, test_data; batchsize = 256, seed = cfg.device.seed::Int, augment = false, sampler = :contiguous, single_batch = true)

    try
        (; b, A, mask, δb, θ) = first(loader) |> lib.gpu
        samples = 128
        result = ex.run_elbo_estimate(models.enc, models.genatr, models.prior, b, A, mask, δb, θ; samples, verbose = false)
        result = lib.cpu(result)
        posterior_mean = result["XDist_MH"]
        ess = result["ESS_IS"]

        @assert all(isfinite, posterior_mean) "Nonfinite posterior mean"
        total_mass = vec(sum(posterior_mean; dims = 1))
        @assert all(isapprox.(total_mass, 1; atol = 1e-4)) "Posterior distributions must have unit mass"
        @assert all(x -> isfinite(x) && 1 / samples - 1e-6 <= x <= 1 + 1e-6, ess) "Invalid importance sampling ESS"

        truth = lib.JLD2.load(joinpath(dirname(cfg.data.path::String), "truth.jld2"), "truths")["test"]
        draws = result["XDistSamples"][:, :, 1]
        lower = [lib.quantile(draws[:, j], 0.025) for j in axes(draws, 2)]
        upper = [lib.quantile(draws[:, j], 0.975) for j in axes(draws, 2)]

        fig = Figure(; size = (900, 400))
        spectrum_axis = Axis(fig[1, 1]; xlabel = "T₂ (s)", ylabel = "Probability mass", xscale = log10)
        band!(spectrum_axis, truth.T2s, lower, upper; color = (:steelblue, 0.25))
        lines!(spectrum_axis, truth.T2s, posterior_mean[:, 1]; label = "Posterior mean")
        lines!(spectrum_axis, truth.T2s, truth.xdist[:, 1]; color = :black, label = "Generating distribution")
        axislegend(spectrum_axis)

        ess_axis = Axis(fig[1, 2]; xlabel = "Importance sampling ESS / S", ylabel = "Voxel count")
        hist!(ess_axis, vec(ess); color = :steelblue)
        figure_path = abspath(joinpath(run_dir, "demo.pdf"))
        save(figure_path, fig)

    finally
        empty!(loader)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && demo(ARGS)
