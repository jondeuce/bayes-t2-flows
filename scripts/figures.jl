include("plots.jl")

function generate_figures(args)
    @assert length(args) >= 2 "Usage: figures.jl RUN_DIR OUTPUT_DIR [TARGET[,TARGET...]] [--quick] [--image-id=ID] [--slice=INDEX] [--references=DIR] [--skip-references] [--nuts-backbone=flow|id_window] [--nuts-max-depth=INT]; targets: all, main, references, flip-angle, spatial, cross-resolution, nuts-efficiency"
    run_dir, output_dir = args[1:2]
    targets = ["all"]
    quick = false
    references = true
    nuts_backbone = :flow
    nuts_max_depth = 10
    image_id = nothing
    slice_index = 1

    for arg in args[3:end]
        if arg == "--quick"
            quick = true
        elseif arg == "--skip-references"
            references = false
        elseif startswith(arg, "--nuts-backbone=")
            nuts_backbone = Symbol(split(arg, '='; limit = 2)[2])
            @assert nuts_backbone in (:flow, :id_window) "Unknown NUTS backbone"
        elseif startswith(arg, "--nuts-max-depth=")
            nuts_max_depth = parse(Int, split(arg, '='; limit = 2)[2])
            @assert nuts_max_depth > 0
        elseif startswith(arg, "--image-id=")
            image_id = parse(Int, split(arg, '='; limit = 2)[2])
        elseif startswith(arg, "--slice=")
            slice_index = parse(Int, split(arg, '='; limit = 2)[2])
        elseif startswith(arg, "--references=")
            REFERENCE_DIR[] = lib.cachepath(split(arg, '='; limit = 2)[2])
        else
            targets = split(arg, ',')
            @assert all(in(("all", "main", "references", "flip-angle", "spatial", "cross-resolution", "nuts-efficiency")), targets) "Unknown argument: $arg"
        end
    end

    @assert references || !("references" in targets) "Explicit references target conflicts with --skip-references"
    println(stderr, "Loading checkpoint and held-out data")
    flush(stderr)
    load_run(; dir = run_dir, image_id, slice = (:, :, slice_index), cohort_size = quick ? 128 : 1000)
    FIGURE_DIR[] = mkpath(abspath(output_dir))
    ex.save_run_info(cfg, args; dir = FIGURE_DIR[], model_path = joinpath(run_dir, "models.jld2"))

    runs(name) = name in targets || ("all" in targets && name != "references")
    if any(runs, ("main", "references", "flip-angle", "spatial"))
        println(stderr, "Computing shared slice inference")
        flush(stderr)
        run_shared_inference!(; quick)
    end

    if runs("references")
        println(stderr, "Computing reference samples")
        flush(stderr)
        run_reference_samples!(; quick, nuts_backbone, nuts_max_depth)
    end

    if runs("main")
        println(stderr, "Generating main figures")
        flush(stderr)
        generate_main_figures!(; quick, references, nuts_backbone, nuts_max_depth)
    end

    if runs("flip-angle")
        println(stderr, "Generating flip-angle figures")
        flush(stderr)
        run_flip_angle_data!(; quick)
        make_flip_angle_figures!()
    end

    if runs("spatial")
        println(stderr, "Generating spatial figures")
        flush(stderr)
        run_spatial_data!(; quick)
        make_spatial_figures!()
        if !quick
            validate_spatial_stieltjes()
        end
    end

    if runs("cross-resolution")
        println(stderr, "Generating cross-resolution figures")
        flush(stderr)
        run_cross_resolution_data!(; quick)
        make_cross_resolution_figures!()
    end

    if references && runs("nuts-efficiency")
        println(stderr, "Computing identity NUTS efficiency")
        flush(stderr)
        identity = run_nuts_identity_backbone(;
            n_voxels = quick ? 4 : 32,
            n_samples = quick ? 128 : 1000, n_adapts = quick ? 64 : 1000)
        cells = if runs("main") && !quick
            comparison_data
        else
            run_method_backbone_comparison(; n_samples = quick ? 128 : 1000, warmup_id = quick ? 100 : 4000, warmup_hmc = quick ? 100 : 1000, sweep_iter = quick ? 64 : 500)
        end
        batch = cpu(run_env.val_cohort)
        logT2s = range(log(batch.T2Range[1, 1]), log(batch.T2Range[2, 1]); length = run_env.cfg.data.nT2)
        write_identity_efficiency_table(cells, identity.reference, logT2s)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && generate_figures(ARGS)
