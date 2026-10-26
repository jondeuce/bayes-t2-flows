using Library
const lib = Library
const ex = Library.Experiments

function train(args; filename = lib.projectdir("config", "demo.toml"))
    cmd_args = args
    if !isempty(args) && !startswith(first(args), "--")
        filename = first(args)
        args = args[2:end]
    end

    cfg, data_cfg = ex.load_config(args; filename)
    ex.save_run_info(cfg, cmd_args)

    data = ex.dataset(lib.clean_realpath(cfg.data.path::String))
    dl_train = ex.data_loader(cfg, data.train; batchsize = cfg.train.batchsize::Int, seed = cfg.device.seed::Int, augment = true, sampler = :blockmix, single_batch = false)
    dl_val = ex.data_loader(cfg, data.val; batchsize = cfg.eval.batchsize::Int, seed = cfg.device.seed::Int, augment = false, sampler = :contiguous, single_batch = true)

    try
        (; models, optimizers) = ex.init_state!(cfg, dl_train)
        schedule = ex.build_learning_rate_schedule(cfg)
        steps, eval_every = cfg.train.steps::Int, cfg.train.eval_every::Int
        completed_steps = cfg.checkpoint.resume_from_step::Int
        lib.TOMLConfig.save_config(data_cfg; filename = ex.logdir(cfg, "data-config.toml"))

        lib.with_trainmode!(models, optimizers) do _, _
            lib.with_gpu(dl_train) do batches
                for (step, batch) in zip((completed_steps+1):steps, batches)
                    η = schedule(step)::Float64
                    ex.adjust_learning_rate!(cfg, models, optimizers, η)
                    _, models, optimizers = ex.train_step(batch, cfg, models, optimizers)
                    cfg.state.step = step
                    (step % eval_every == 0 || step == steps) || continue

                    validation_batch = first(dl_val) |> lib.gpu
                    validation_loss = lib.with_testmode!(models, optimizers) do evaluation_models, _
                        ex.train_loss(cfg, evaluation_models, validation_batch)
                    end

                    @assert isfinite(validation_loss) "Nonfinite validation loss"
                    @info "Validation: step $step/$steps, loss $(round(validation_loss; sigdigits = 5))"
                    flush(stderr)

                    ex.save_training_checkpoint(models, optimizers; savefolder = cfg.log.dir::String)
                    lib.TOMLConfig.save_config(cfg; filename = ex.logdir(cfg, "train-config.toml"))
                end
            end
        end

        return cfg.log.dir::String

    finally
        empty!(dl_train)
        empty!(dl_val)
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && train(ARGS)
