#### Config

logdir(cfg::Config, args...) = joinpath(cfg.log.dir::String, args...)

function load_from_checkpoint(dir; key = "models", verbose = true)
    filename = joinpath(lib.clean_realpath(dir), key * ".jld2")
    verbose && @info "Loading $(key) from checkpoint: $(relpath(filename))"
    return lib.safe_load(filename)[key] |> gpu
end

function parse_toml_from_checkpoint(dir; name = "train-config", verbose = true)
    filename = joinpath(lib.clean_realpath(dir), name * ".toml")
    verbose && @info "Loading config from checkpoint: $(relpath(filename))"
    return lib.TOML.parsefile(Config, filename)
end

function init_state!(cfg::Config, dl_train)
    dir = cfg.checkpoint.dir::String
    if isempty(dir)
        models = build_models(cfg, dl_train) |> gpu
        optimizers = build_optimizers(cfg, models)
    else
        ckpt_cfg = parse_toml_from_checkpoint(dir; name = "train-config")
        cfg.arch = ckpt_cfg.arch
        cfg.opt = ckpt_cfg.opt
        models = load_from_checkpoint(dir; key = "training-models")
        optimizers = load_from_checkpoint(dir; key = "optimizers")
    end
    optimizers = lib.assert_shared_optimizer_state(models, optimizers; error_on_mismatch = true)
    return (; models, optimizers)
end

function save_training_checkpoint(models, optimizers; savefolder)
    lib.safe_save(joinpath(savefolder, "training-models.jld2"), dict("training-models" => models))
    lib.safe_save(joinpath(savefolder, "optimizers.jld2"), dict("optimizers" => optimizers))
    lib.with_testmode!(models, optimizers) do eval_models, _
        lib.safe_save(joinpath(savefolder, "models.jld2"), dict("models" => eval_models))
    end
    return nothing
end

function load_checkpoint(dir; verbose = true)
    cfg = parse_toml_from_checkpoint(dir; name = "train-config", verbose)
    data_cfg = parse_toml_from_checkpoint(dir; name = "data-config", verbose)
    init_env!(cfg; verbose)
    models = load_from_checkpoint(dir; key = "models", verbose)
    lib.Flux.testmode!(models)
    return (; cfg, data_cfg, models)
end

function load_config!(cfg::Config, args...; filename, verbose = true)
    defaults = lib.TOML.parsefile(Config, lib.projectdir("config", "train.toml"))
    merge!(defaults, lib.TOML.parsefile(Config, filename))
    cfg = merge!(defaults, cfg)
    cfg = lib.TOMLConfig.ArgParse.parse_args(args..., cfg)
    if !isempty(cfg.checkpoint.dir::String)
        dir = lib.clean_realpath(cfg.checkpoint.dir::String)
        cfg = parse_toml_from_checkpoint(dir; verbose)
        cfg.checkpoint.dir = dir
        cfg.checkpoint.resume_from_step = cfg.state.step::Int
        cfg.log.dir = dir
        cfg = lib.TOMLConfig.ArgParse.parse_args(args..., cfg)
    end
    @assert !isempty(cfg.data.path::String) "Set data.path to an explicit Arrow dataset"
    isempty(cfg.data.config::String) && (cfg.data.config = replace(cfg.data.path::String, r"(?:\.shuffled)?\.arrow$" => ".toml"))
    cfg.data.path = lib.clean_realpath(cfg.data.path::String)
    cfg.data.config = lib.clean_realpath(cfg.data.config::String)
    data_cfg = lib.TOML.parsefile(Config, cfg.data.config::String)
    init_env!(cfg; verbose)
    init_stateful_config!(cfg)
    return cfg, data_cfg
end

load_config(args...; kwargs...) = load_config!(Config(), args...; kwargs...)

function init_env!(cfg::Config; verbose = true)

    verbose && @info "Multi-threading: $(Threads.nthreads()) threads"

    if cfg.device.gpu::Bool
        @assert lib.CUDA.functional() "CUDA is unavailable; set --device.gpu=false"
        lib.CUDA.allowscalar(false)
        lib.CUDA.device!(0)
        verbose && @info "CUDA enabled: $(lib.CUDA.name(lib.CUDA.device()))"
    else
        verbose && @info "Using CPU"
    end

    lib.USE_GPU[] = cfg.device.gpu::Bool

    global_seed_rng!(cfg)

    cfg.log.dir = mkpath(lib.clean_normpath(cfg.log.dir::String))
    return cfg
end

function global_seed_rng!(cfg::Config, salt::Int = 0)
    lib.Random.seed!(cfg.device.seed::Int + salt)
    if cfg.device.gpu
        lib.CUDA.seed!(cfg.device.seed::Int + salt)
    end
    return nothing
end

#### Run provenance

function save_run_info(cfg, args; dir = cfg.log.dir::String, model_path = nothing)
    root = lib.projectdir()
    info = Dict{String, Any}(
        "command" => [PROGRAM_FILE; args],
        "seed" => cfg.device.seed::Int,
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(),
        "cpu" => Sys.CPU_NAME,
        "julia_heap_size_hint_bytes" => Base.JLOptions().heap_size_hint,
        "blas_vendor" => string(lib.BLAS.vendor()),
        "blas_threads" => lib.BLAS.get_num_threads(),
        "device" => (cfg.device.gpu::Bool ? lib.CUDA.name(lib.CUDA.device()) : "CPU"),
    )
    if cfg.device.gpu::Bool
        info["cuda_runtime_version"] = string(lib.CUDA.runtime_version())
        info["cuda_driver_api_version"] = string(lib.CUDA.driver_version())
    end

    files = String[]
    for folder in ("src", "scripts")
        for (source_dir, _, names) in walkdir(joinpath(root, folder))
            append!(files, (joinpath(source_dir, name) for name in names if endswith(name, ".jl")))
        end
    end
    sort!(files)

    info["source_sha256"] = Dict(relpath(file, root) => bytes2hex(open(lib.SHA.sha256, file)) for file in files)
    if model_path !== nothing
        info["checkpoint_sha256"] = bytes2hex(open(lib.SHA.sha256, model_path))
    end

    if ispath(joinpath(root, ".git"))
        if success(pipeline(`git -C $root rev-parse --verify --quiet HEAD`; stdout = devnull))
            info["git_commit"] = readchomp(`git -C $root rev-parse HEAD`)
        end
        info["git_dirty"] = !isempty(readchomp(`git -C $root status --porcelain`))
    end

    filename = abspath(joinpath(dir, "run-info.toml"))
    open(filename, "w") do io
        lib.TOML.print(io, info)
    end
    cp(lib.projectdir("Manifest.toml"), abspath(joinpath(dir, "Manifest.toml")); force = true)

    return nothing
end

#### Schedulers

function LinearSchedule(; start::Float64, final::Float64, iters::Int)
    tri = lib.ParameterSchedulers.Triangle(; λ0 = start, λ1 = final, period = 2 * (iters - 1))
    return start <= final ? tri : lib.ParameterSchedulers.Shifted(tri, iters)
end

ConstantSchedule(; start::Float64) = lib.ParameterSchedulers.Constant(start)

function build_learning_rate_schedule(cfg)
    start, peak = cfg.opt.lr.start::Float64, cfg.opt.lr.middle::Float64
    warmup_iters = cfg.opt.lr.warmup::Int # optimizer steps
    schedule = warmup_iters <= 1 ? ConstantSchedule(; start = peak) :
               lib.ParameterSchedulers.Sequence(LinearSchedule(; start, final = peak, iters = warmup_iters) => warmup_iters, ConstantSchedule(; start = peak) => Inf)
    return schedule
end

function init_stateful_config!(cfg::Config)
    if cfg.arch.encoder.tokensize::Int == 0
        cfg.arch.encoder.tokensize::Int = cfg.arch.encoder.hdim::Int ÷ 4
    end
    if cfg.arch.encoder.headsize::Int == 0
        cfg.arch.encoder.headsize::Int = cfg.arch.encoder.tokensize::Int ÷ cfg.arch.encoder.nheads::Int
    end
    @assert cfg.train.steps::Int > 0 && cfg.train.eval_every::Int > 0 "train.steps and train.eval_every must be positive"
    cfg.state.step = 0
    return cfg
end

function adjust_learning_rate!(cfg::Config, models, optimizers, η)

    lib.Optimisers.adjust!(optimizers, η)

    # Match the head learning rate to its initialization gain.
    c_head = cfg.arch.flows.initialization.gain::Float64
    if c_head != 1
        heads = Base.IdSet{Any}(flow_head_parameters(models))
        adjusted = Base.IdSet{Any}() # Forward and inverse flows share parameter arrays.
        lib.fmap_trainable_biwalk(models, optimizers) do p, leaf
            if p ∈ heads && p ∉ adjusted
                push!(adjusted, p)
                lib.Optimisers.adjust!(leaf, c_head * leaf.rule.eta)
            end
            return p
        end
    end

    return nothing
end

#### Training

function train_step(batch::NamedTuple, cfg::Config, models::NamedTuple, optimizers::NamedTuple)
    ℓ, ∇models = train_loss_gradient(cfg, models, batch)
    @assert isfinite(ℓ) "Nonfinite training loss"
    grads = IdDict{lib.Optimisers.Leaf, Tuple{Vararg{AbstractArray}}}()
    lib.Optimisers._grads!(grads, optimizers, models, ∇models)
    @assert all(gs -> all(g -> all(isfinite, g), gs), values(grads)) "Nonfinite gradients"
    models = lib.Optimisers._update!(optimizers, models; grads, params = IdDict())
    return ℓ, models, optimizers
end

function train_loss_gradient(cfg::Config, models::NamedTuple, batch)
    Total_Loss, back = lib.Zygote.pullback(models) do _models
        return train_loss(cfg, _models, batch)
    end
    isfinite(Total_Loss) || return Total_Loss, nothing
    return Total_Loss, only(back(one(Total_Loss)))
end

function auxiliary_train_loss(cfg::Config, models::NamedTuple, synth_batch, enc_cond, z)
    ℰ = models.enc
    (; A, mask, δb, θ) = synth_batch

    b̂, x̂ = synth_batch.b, synth_batch.x # mock signals and corresponding labels
    ẑ = encode_signal(ℰ, b̂, A, enc_cond)

    ℓ_synth = supervised_flow_loss(models, x̂, ẑ, b̂, A, mask, δb, θ)
    ℓ_rep = latent_weighted_embedding_loss(cfg, z, ẑ)

    return ℓ_synth + ℓ_rep
end

function train_loss(cfg::Config, models::NamedTuple, batch)
    ℰ = models.enc
    (; b, A, mask, δb, θ) = batch

    enc_cond = encode_conditionals(ℰ, A, mask, θ)
    z = encode_signal(ℰ, b, A, enc_cond)

    ℓ_flow = flow_loss(models, z, b, A, mask, δb, θ)
    synth_batch = @ignore_derivatives synthetic_batch(models, batch)
    ℓ_aux = auxiliary_train_loss(cfg, models, synth_batch, enc_cond, z)

    return ℓ_flow + ℓ_aux
end

function flow_loss(models, z, b, A, mask, δb, θ)
    𝒢, 𝒫 = models.genatr, models.prior
    u = sample_base_similar(𝒢.model, b)
    post = ForwardPhysicsTransformedPosterior(𝒢, 𝒫, z, b, A, mask, δb, θ)
    return lib.mean(post(u; pathgradient = true).negelbo)
end

function supervised_flow_loss(models, x̂, ẑ, b̂, A, mask, δb, θ)
    𝒢 = models.genatr
    ℳ = marginal_posterior(𝒢, (ẑ, b̂, A, mask, δb, θ))

    # Synthetic supervision fits q(x̄|b); shared nuisance priors are learned through the ELBO.
    logx̄ = @views x̂[1:(end-2), ..] # extract logx̄ from x = [logx̄, logs, logσ] for marginal posterior
    summary = flow_backward(ℳ, logx̄) # -log q(x̄ | ẑ); ℳ takes log x̄ for numerical stability
    return lib.mean(summary.neglogpx)
end

function latent_weighted_embedding_loss(cfg::Config, z1::AbstractMatrix, z2::AbstractMatrix)
    λ = @ignore_derivatives convert(eltype(z1), cfg.train.reg.lambda_encwass2::Float64)
    z = hcat(z1, z2)
    μ = dropdims(lib.mean(z; dims = 2); dims = 2)
    centered = z .- μ
    Σ = (centered * centered') ./ size(z, 2)
    d, n = size(z)
    α = (d + 2) / (n + d + 2)
    return λ * lib.wasserstein2_unit_multivariate_gaussian(μ, Σ; α)
end

function synthetic_batch(models, batch)
    (; b, A, mask, δb, θ) = batch
    x = sample_similar(recondition(models.prior, (θ,)), b)
    synth_batch = @insert batch.x = x
    @reset synth_batch.b = sample_similar(forward_physics_model(x, A, mask, δb), b)
    return synth_batch
end

function decaes_inference_variables(data_cfg::Config, batch; kwargs...)
    (; b, A, nTE, TE, T2Range, T1) = batch
    nTEmax, nT2, batchsize = size(A)

    if !@views(all(nTE[1, 1:1] .== nTE) && all(TE[1, 1:1] .== TE) && all(T2Range[1, 1:1] .== T2Range[1, :]) && all(T2Range[2, 1:1] .== T2Range[2, :]) && all(T1[1, 1:1] .== T1))
        # DECAES shares acquisition options within a call; fit matching groups and restore voxel order.
        metadata = lib.cpu((; nTE, TE, T2Range, T1))
        acquisition(j) = (metadata.nTE[j], metadata.TE[j], metadata.T2Range[1, j], metadata.T2Range[2, j], metadata.T1[j])
        groups = OrderedDict{typeof(acquisition(1)), Vector{Int}}()
        for j in 1:batchsize
            push!(get!(() -> Int[], groups, acquisition(j)), j)
        end
        xdist = lib.zeros_similar(b, nT2, batchsize)
        logs, logσ = lib.zeros_similar(b, 1, batchsize), lib.zeros_similar(b, 1, batchsize)
        results = map(collect(values(groups))) do I
            group_batch = (; b = b[:, I], A = view(A, :, :, I), map(v -> v[:, I], metadata)...)
            fit = decaes_inference_variables(data_cfg, group_batch; kwargs...)
            xdist[:, I], logs[:, I], logσ[:, I] = fit.x.xdist, fit.x.logs, fit.x.logσ
            return (; indices = I, results = fit.results)
        end
        return (; x = inference_variables((; xdist, logs, logσ)), results)
    end

    results = @allowscalar let nTE = Int(nTE[1]), TE = Float64(TE[1]), T2Range = Float64.((T2Range[1, 1], T2Range[2, 1])), T1 = Float64(T1[1])
        data = reshape(permutedims(b[1:nTE, :]), batchsize, 1, 1, nTE) |> cpu64
        decaes_cfg = DECAESConfig(data_cfg, data; TE, T2Range, T1, nT2, kwargs...)
        compute_decaes(data, decaes_cfg)
    end

    xdist = lib.arr_similar(b, results.t2dist) |> lib.Flux.flatten |> permutedims
    scale = sum(xdist; dims = 1)
    xdist ./= scale
    logs = log.(scale)
    logσ = .-log.(lib.arr_similar(b, results.t2maps["snr"])) |> lib.Flux.flatten |> permutedims

    return (; x = inference_variables((; xdist, logs, logσ)), results)
end

#### Inference utils

function print_progress(i, ℓ; scalars = nothing, verbose::Bool)
    verbose || return nothing
    pp(x) = round(x; sigdigits = 4)
    pp(x::NamedTuple) = join(["$k => $(pp(v))" for (k, v) in pairs(x)], ", ")
    @info "iter $i: mean: $(pp(ℓ))" * (scalars === nothing ? "" : ", $(pp(scalars))")
    return nothing
end

#### MAP Inference

struct MAPInferenceProblem{Tℰ, T𝒢, T𝒫}
    ℰ::Tℰ # Encoder
    𝒢::T𝒢 # Generator
    𝒫::T𝒫 # Prior
end

function inference_state(prob::MAPInferenceProblem, u::AbstractArray, z::AbstractArray, b::AbstractMatrix, A::AbstractTensor3D, mask::AbstractMatrix, δb::AbstractMatrix, θ::AbstractArray)
    (; udim) = prob.𝒢.opts
    @assert_ad size(u, 1) ∈ (udim, udim - 2) "u must have either $(udim) or $(udim - 2) dimensions; got $(size(u, 1))"
    if size(u, 1) == udim - 2
        u = [u; lib.zeros_similar(u, 2, Base.tail(size(u))...)] # nuisance dimensions fixed to zero
    end
    post = ForwardPhysicsTransformedPosterior(prob.𝒢, prob.𝒫, z, b, A, mask, δb, θ)
    postsummary = post(u) # joint probability stats: p(u,y) = p(y|u)p(u)
    return merge(postsummary, (; u, z, b, A, mask, δb))
end

function inference_losses(prob::MAPInferenceProblem, u::AbstractArray, z::AbstractArray, b::AbstractMatrix, A::AbstractTensor3D, mask::AbstractMatrix, δb::AbstractMatrix, θ::AbstractArray)
    (; negloglike, neglogprior, neglogpostu) = inference_state(prob, u, z, b, A, mask, δb, θ)
    return (;
        negloglike, # data consistency, i.e. negative log-likelihood -logp(y|u) = -logp(y|x=φ(u;z))
        neglogprior, # prior likelihood, i.e. negative log-prior -logp(x) = -logp(u) - log|du/dx|
        neglogpostu, # negative log joint -log p(u,y) = -log p(y|x) - log p(x) - log|dx/du|
    )
end

function ∇inference_loss(prob::MAPInferenceProblem, u::AbstractArray, z::AbstractArray, b::AbstractMatrix, A::AbstractTensor3D, mask::AbstractMatrix, δb::AbstractMatrix, θ::AbstractArray)
    losses, back = lib.Zygote.pullback(u) do _u
        return inference_losses(prob, _u, z, b, A, mask, δb, θ)
    end
    Δ = lib.fmapstructure(_ -> nothing, losses)
    @reset Δ.neglogpostu = lib.ones_similar(losses.neglogpostu) # backpropagate through -logp(u,y)
    ∇u = back(Δ)[1]::typeof(u)
    return losses, ∇u
end

function ∇²inference_loss(prob::MAPInferenceProblem, u, z, b, A, mask, δb, θ)
    conditionals = map(lib.repeatbatch(2 * size(u, 1)), (z, b, A, mask, δb, θ))
    return lib.batched_hessian_from_grad(u, cbrt(eps(eltype(u)))) do _u
        return ∇inference_loss(prob, _u, conditionals...)[2]
    end
end

function run_inference(
    prob::MAPInferenceProblem,
    b::AbstractMatrix{T},
    A::AbstractTensor3D{T},
    mask::AbstractMatrix{T},
    δb::AbstractMatrix{T},
    θ::AbstractMatrix{T},
    z::Union{AbstractMatrix{T}, Nothing} = nothing,
    u₀::Union{AbstractMatrix{T}, Nothing} = nothing;
    lr = 0.1,
    lrmin = 0.0,
    schedule = :constant,
    iter = 250,
    newton = false,
    gtol = cbrt(eps(T))^2, # early-exit tolerance on the mean absolute gradient component
    verbose = false,
    encoder_chunksize = size(b, 2),
    generator_chunksize = size(b, 2),
    kwargs...,
) where {T}
    # Learning rate scheduler; warming up from a very small lr is crucial
    lrfun = schedule === :constant ? ConstantSchedule(; start = lr) :
            schedule === :linear ? LinearSchedule(; start = lr, final = lrmin, iters = iter) :
            error("Unknown learning rate schedule: $(schedule)")

    z = z === nothing ? lib.chunked_apply(prob.ℰ, b, A, mask, θ; chunksize = encoder_chunksize) : z

    # Layers can assume `z` and `θ` are fixed during sampling
    _∇lossfun(args...) = ∇inference_loss(prob, args...)
    ∇lossfun(_u) = lib.chunked_apply(_∇lossfun, _u, z, b, A, mask, δb, θ; chunksize = generator_chunksize)

    (; udim) = prob.𝒢.opts
    u = u₀ === nothing ? lib.zeros_similar(b, udim - 2, size(b, 2)) : u₀[1:(udim-2), ..] # non-nuisance dimensions, initialized to zero or u₀
    rule = lib.LRAPSGD(; eta = lr, eta_Q = 0.5, max_rank = 10, kwargs...)
    opt = lib.Optimisers.setup(rule, u)

    all_ℓ, ∇u = lib.with_trainmode!(u, opt) do _, _
        return ∇lossfun(u)
    end
    print_progress(0, map(lib.mean, all_ℓ); verbose)

    for i in 1:iter
        step_size = lib.ofeltypefloat(u, lrfun(i))
        lib.Optimisers.adjust!(opt, step_size)

        lib.with_trainmode!(u, opt) do _, _
            if newton
                δv = sqrt(eps(eltype(u))) # perturbation for finite-difference
                v = lib.randn_similar(u) # random perturbation direction
                @. u += δv * v # perturb u
                _, ∇u⁺ = ∇lossfun(u)
                h = (@. ∇u⁺ = (∇u⁺ - ∇u) / δv) # finite-difference gradient estimate
                @. u -= δv * v # unperturb u
                lib.Optimisers.update!(opt, u, ∇u, (h, v))
            else
                lib.Optimisers.update!(opt, u, ∇u)
            end
        end

        all_ℓ, ∇u = ∇lossfun(u)
        ℓ = map(lib.mean, all_ℓ)
        print_progress(i, ℓ; scalars = (; step_size), verbose)

        if (ḡ = lib.mean(abs, ∇u)) < gtol
            verbose && @info "run_inference converged at iter $(i): mean |∇| = $(ḡ) < gtol = $(gtol)"
            break
        end
    end

    if verbose
        ℓ = map(lib.mean, all_ℓ)
        print_progress(iter, ℓ; verbose)
    end

    return inference_state(prob, u, z, b, A, mask, δb, θ)
end

#### Sampling utils

abstract type AbstractSamplingAlgorithm end
struct Metropolis <: AbstractSamplingAlgorithm end

abstract type AbstractLangevinSamplingAlgorithm <: AbstractSamplingAlgorithm end
struct MALA <: AbstractLangevinSamplingAlgorithm end

struct SamplingProblem{alg <: AbstractSamplingAlgorithm, Tℰ, T𝒢, T𝒫}
    alg::alg
    ℰ::Tℰ # Encoder
    𝒢::T𝒢 # Generator
    𝒫::T𝒫 # Prior
end

struct SamplingState{Tpos, Tmom}
    pos::Tpos
    mom::Tmom
end

SamplingState(arr::AbstractArray, dim::Int) = SamplingState(lib.randn_similar(arr, dim, size(arr, 2)), lib.randn_similar(arr, dim, size(arr, 2))) # initial position and initial momentum
Base.NamedTuple(state::SamplingState) = lib.Functors.children(state)
Base.iterate(state::SamplingState, iter_state...) = iterate(NamedTuple(state), iter_state...)

function init_state(prob::SamplingProblem, z::AbstractMatrix, b::AbstractMatrix, A::AbstractTensor3D, mask::AbstractMatrix, δb::AbstractArray, θ::AbstractMatrix)
    (; udim) = prob.𝒢.opts
    state = SamplingState(b, udim)
    conditionals = (; z, b, A, mask, δb, θ)
    return (; state, conditionals)
end

function run_sampling(prob::SamplingProblem, b::AbstractMatrix{T}, A::AbstractTensor3D{T}, mask::AbstractMatrix{T}, δb::AbstractMatrix{T}, θ::AbstractMatrix{T}; encoder_chunksize = nothing, generator_chunksize = nothing, kwargs...) where {T}
    z = lib.chunked_apply(prob.ℰ, b, A, mask, θ; chunksize = encoder_chunksize) # initial embedding
    return run_sampling(prob, z, b, A, mask, δb, θ; chunksize = generator_chunksize, kwargs...)
end

struct SamplingFunctor{
    alg <: AbstractSamplingAlgorithm,
    Tprob <: SamplingProblem{alg},
    Tcond <: NamedTuple,
}
    prob::Tprob
    conditionals::Tcond
end

function sampling_losses(f::SamplingFunctor, u::AbstractArray; chunksize = nothing)
    return lib.chunked_apply(f.conditionals, u; chunksize) do _conditionals, _u
        return sampling_losses(f.prob, _conditionals, _u)
    end
end

function ∇sampling_loss(f::SamplingFunctor, u::AbstractArray; chunksize = nothing)
    return lib.chunked_apply(f.conditionals, u; chunksize) do _conditionals, _u
        return ∇sampling_loss(f.prob, _conditionals, _u)
    end
end

function sampling_losses(prob::SamplingProblem, conditionals, u::AbstractArray)
    post = ForwardPhysicsTransformedPosterior(prob.𝒢, prob.𝒫, conditionals...)
    (; summary, negloglike, neglogprior, neglogpostu) = post(u)
    return (; summary, losses = (; negloglike, neglogprior, neglogpostu))
end

function ∇sampling_loss(prob::SamplingProblem, conditionals, u::AbstractArray)
    outputs, back = lib.Zygote.pullback(u) do _u
        return sampling_losses(prob, conditionals, _u)
    end
    Δ = lib.fmapstructure(_ -> nothing, outputs)
    @reset Δ.losses.neglogpostu = lib.ones_similar(outputs.losses.neglogpostu) # backpropagate through -logp(u,y)
    ∇u = back(Δ)[1]::typeof(u)
    return outputs, ∇u
end

# Retain only the chain fields consumed by manuscript diagnostics and resolution conversion.
function sampling_record(outputs)
    return (; summary = (; outputs.summary.u, outputs.summary.x),
        losses = (; outputs.losses.negloglike, outputs.losses.neglogpostu),
        outputs.xdist, outputs.α, outputs.η)
end

#### Langevin sampling

function step_dynamics!(f::SamplingFunctor{MALA}, (u, v)::SamplingState, ∇u::AbstractArray{T}, η::Union{T, AbstractArray{T}}; chunksize = nothing) where {T}
    # Metropolis-adjusted Langevin algorithm (MALA)
    τ = @. √(2η) # time step in Hamiltonian mechanics physics analogy
    lib.randn!(v) # sample new indepedent momentum
    ū = @. u - (τ^2 / 2) * ∇u + τ * v # equivalently: `u - η * ∇u + √(2η) * v`
    ōutputs, ∇ū = ∇sampling_loss(f, ū; chunksize)
    v̄ = @. v - (τ / 2) * (∇u + ∇ū) # equivalently v - τ∇u - τ²∇²u/2 with ∇²u = (∇ū - ∇u)/τ
    return ōutputs, SamplingState(ū, v̄), ∇ū
end

function update_state!(f::SamplingFunctor{MALA}, outputs::NamedTuple, state::SamplingState, ∇u::AbstractArray{T}, η::Union{T, AbstractArray{T}}; chunksize = nothing) where {T}
    (u, v) = state
    ōutputs, (ū, v̄), ∇ū = step_dynamics!(f, state, ∇u, η; chunksize)

    K = lib.neglogpdf_isotropic_gaussian(v) # kinetic energies
    K̄ = lib.neglogpdf_isotropic_gaussian(v̄)
    V, V̄ = outputs.losses.neglogpostu, ōutputs.losses.neglogpostu # potential energies
    ℋ, ℋ̄ = K + V, K̄ + V̄ # Hamiltonians

    α = @. min(1, exp(ℋ - ℋ̄)) # acceptance probability
    ξ = lib.rand_similar(α)
    reject = @. ξ > α

    # Replace rejected proposals with old values and flip corresponding momenta
    @. ū = ifelse(reject, u, ū)
    @. v̄ = ifelse(reject, -v, v̄)
    @. ∇ū = ifelse(reject, ∇u, ∇ū)
    ōutputs = lib.fmap_array((x̄, x) -> (@. x̄ = ifelse(reject, x, x̄)), ōutputs, outputs)
    scalars = (; acceptance_ratio = lib.mean(α), step_size = lib.mean(η))

    return merge(ōutputs, (; η, α, scalars)), SamplingState(ū, v̄), ∇ū
end

function run_sampling(
    prob::SamplingProblem{<:AbstractLangevinSamplingAlgorithm},
    z::AbstractMatrix{T},
    b::AbstractMatrix{T},
    A::AbstractTensor3D{T},
    mask::AbstractMatrix{T},
    δb::AbstractMatrix{T},
    θ::AbstractMatrix{T};
    lr = (1.650^2 / 2) / cbrt(prob.𝒢.opts.udim), # optimal η = τ²/2 for proposal variance τ² = l² / n^(1/3), l ≈ 1.650 (Roberts & Rosenthal, 1998)
    lrmin = 1e-6,
    lrmax = 1.0,
    lrrate = 2.0,
    target_acceptance = 0.574, # optimal MALA acceptance rate (Roberts & Rosenthal, 1998)
    adaptive = false, # opt-in per-voxel step size adaptation during warmup
    iter = 100,
    warmup_iter = 1,
    chunksize = nothing,
    save_all_samples = false,
    verbose = false,
) where {T}

    if adaptive
        # PI control of each voxel's acceptance toward `target_acceptance`, starting from `lr` (or `lrmin`)
        @assert 0 < target_acceptance < 1 "For adaptive learning rate, `target_acceptance` must be between 0.0 and 1.0"
        η = lr === nothing ? lib.fill_similar(b, T(lrmin), 1, size(b, 2)) : lr isa Number ? lib.fill_similar(b, T(lr), 1, size(b, 2)) : lib.arr_similar(b, lr)
        Kp = T(log(lrrate) / (1 - target_acceptance)) # gain such that a fully (un)accepted step rescales lr by ≈ `lrrate`
        ctrl = lib.AcceptanceController(η; setpoint = T(target_acceptance), Kp, Ki = Kp / warmup_iter, lrmin = T(lrmin), lrmax = T(lrmax))
    else
        @assert lr !== nothing "Must specify `lr` for non-adaptive Langevin sampling, or pass `adaptive = true`"
        @assert lr isa Number || (lr isa AbstractArray && size(lr) == (1, size(b, 2))) "Learning rate array must have the same number of elements as the number of voxels"
        η = lr isa Number ? T(lr) : lib.arr_similar(b, lr)
        ctrl = nothing
    end

    (; state, conditionals) = init_state(prob, z, b, A, mask, δb, θ)
    f = SamplingFunctor(prob, conditionals)

    outputs, ∇u = ∇sampling_loss(f, state.pos; chunksize)
    outputs = merge(outputs, inference_variables(outputs.summary.x) |> NamedTuple)

    mean_outputs = (; xdist = lib.zeros_similar(outputs.xdist))
    all_outputs = nothing
    print_progress(0, map(lib.mean, outputs.losses); verbose = verbose > 0)

    for i in 1:iter
        outputs, state, ∇u = update_state!(f, outputs, state, ∇u, η; chunksize)
        outputs = merge(outputs, inference_variables(outputs.summary.x) |> NamedTuple)

        window = iter - warmup_iter
        if i >= iter - window + 1
            mean_outputs.xdist .+= outputs.xdist ./ window
            if save_all_samples
                i′ = i - iter + window
                record = sampling_record(outputs)
                if i′ == 1
                    all_outputs = lib.fmap(x -> zeros(T, window, size(x)...), record; exclude = lib.isnumeric)
                end
                lib.fmapstructure(all_outputs, record; exclude = lib.isnumeric) do x, x_i
                    x[i′, ..] .= x_i |> cpu
                    return nothing
                end
            end
        end

        # Update learning rate based on acceptance ratio; frozen after warmup
        if adaptive && i <= warmup_iter
            lib.update!(ctrl, outputs.α)
        end

        ℓ = map(lib.mean, outputs.losses)
        print_progress(i, ℓ; scalars = outputs.scalars, verbose = verbose > 0 && (mod(i, max(iter ÷ verbose, 1)) == 0 || i == iter))
        any(!isfinite, ℓ) && break
    end

    return (; mean_outputs, all_outputs)
end

#### Metropolis sampling

function update_state!(f::SamplingFunctor{Metropolis}, outputs::NamedTuple, u::AbstractArray{T}, η::Union{T, AbstractArray{T}}; chunksize = nothing) where {T}
    if iszero(η)
        # Independent Metropolis: propose from the conditional flow's base distribution π_U(u) = N(0, I)
        ū = v̄ = lib.randn_similar(u)
    else
        # Random-walk Metropolis: symmetric proposal ū = u + τv where v ~ N(0, I)
        v̄ = lib.randn_similar(u)
        τ = @. √(2η)
        ū = @. u + τ * v̄
    end
    ōutputs = sampling_losses(f, ū; chunksize)

    # MH acceptance α = min(1, [p_U(ū,y) T(u|ū)] / [p_U(u,y) T(ū|u)]), with -log p_U(u,y) = neglogpostu
    if iszero(η)
        # Independent MH: T(ū|u) = π_U(ū), so ℋ = neglogpostu - neglogpu (a negative log importance weight)
        ℋ = outputs.losses.neglogpostu .- outputs.summary.neglogpu
        ℋ̄ = ōutputs.losses.neglogpostu .- ōutputs.summary.neglogpu
    else
        # Random-walk MH: symmetric proposal, so ℋ = neglogpostu
        ℋ = outputs.losses.neglogpostu
        ℋ̄ = ōutputs.losses.neglogpostu
    end

    α = @. min(1, exp(ℋ - ℋ̄)) # acceptance probability
    ξ = lib.rand_similar(α)
    reject = @. ξ > α

    # Replace rejected proposal values with old values
    @. ū = ifelse(reject, u, ū)
    lib.fmap_array((x̄, x) -> (@. x̄ = ifelse(reject, x, x̄)), ōutputs, outputs)
    scalars = (; acceptance_ratio = lib.mean(α), step_size = lib.mean(η))

    return merge(ōutputs, (; η, α, scalars)), ū, v̄
end

function run_sampling(
    prob::SamplingProblem{Metropolis},
    z::AbstractMatrix{T},
    b::AbstractMatrix{T},
    A::AbstractTensor3D{T},
    mask::AbstractMatrix{T},
    δb::AbstractMatrix{T},
    θ::AbstractMatrix{T};
    lr = (2.382^2 / 2) / (prob.𝒢.opts.udim - 1), # optimal η = τ²/2 for proposal variance τ² = l² / (n - 1), l ≈ 2.382 (Roberts et al., 1997)
    lrmin = 1e-2,
    lrmax = 1.0,
    lrrate = 2.0,
    target_acceptance = 0.234, # optimal random-walk acceptance rate (Roberts et al., 1997)
    adaptive = false, # opt-in per-voxel step size adaptation during warmup
    iter = 100,
    warmup_iter = 1,
    chunksize = nothing,
    save_all_samples = false,
    verbose = false,
) where {T}

    if adaptive
        # PI control of each voxel's acceptance toward `target_acceptance`, starting from `lr` (or `lrmin`)
        @assert 0 < target_acceptance < 1 "For adaptive learning rate, `target_acceptance` must be between 0.0 and 1.0"
        η = lr === nothing ? lib.fill_similar(b, T(lrmin), 1, size(b, 2)) : lr isa Number ? lib.fill_similar(b, T(lr), 1, size(b, 2)) : lib.arr_similar(b, lr)
        Kp = T(log(lrrate) / (1 - target_acceptance)) # gain such that a fully (un)accepted step rescales lr by ≈ `lrrate`
        ctrl = lib.AcceptanceController(η; setpoint = T(target_acceptance), Kp, Ki = Kp / warmup_iter, lrmin = T(lrmin), lrmax = T(lrmax))
    else
        # `lr === nothing` gives η = 0, which selects independent Metropolis rather than a random walk
        @assert lr === nothing || lr isa Number || (lr isa AbstractArray && size(lr) == (1, size(b, 2))) "Learning rate array must have the same number of elements as the number of voxels"
        η = lr === nothing ? zero(T) : lr isa Number ? T(lr) : lib.arr_similar(b, lr)
        ctrl = nothing
    end

    (; state, conditionals) = init_state(prob, z, b, A, mask, δb, θ)
    f = SamplingFunctor(prob, conditionals)
    u = state.pos

    outputs = sampling_losses(f, u; chunksize)
    outputs = merge(outputs, inference_variables(outputs.summary.x) |> NamedTuple)

    mean_outputs = (; xdist = lib.zeros_similar(outputs.xdist))
    all_outputs = nothing
    print_progress(0, map(lib.mean, outputs.losses); verbose = verbose > 0)

    for i in 1:iter
        outputs, u = update_state!(f, outputs, u, η; chunksize)
        outputs = merge(outputs, inference_variables(outputs.summary.x) |> NamedTuple)

        window = iter - warmup_iter
        if i >= iter - window + 1
            mean_outputs.xdist .+= outputs.xdist ./ window
            if save_all_samples
                i′ = i - iter + window
                record = sampling_record(outputs)
                if i′ == 1
                    all_outputs = lib.fmap(x -> zeros(T, window, size(x)...), record; exclude = lib.isnumeric)
                end
                lib.fmapstructure(all_outputs, record; exclude = lib.isnumeric) do x, x_i
                    x[i′, ..] .= x_i |> cpu
                    return nothing
                end
            end
        end

        # Update learning rate based on acceptance ratio; frozen after warmup
        if adaptive && i <= warmup_iter
            lib.update!(ctrl, outputs.α)
        end

        ℓ = map(lib.mean, outputs.losses)
        print_progress(i, ℓ; scalars = outputs.scalars, verbose = verbose > 0 && (mod(i, max(iter ÷ verbose, 1)) == 0 || i == iter))
        any(!isfinite, ℓ) && break
    end

    return (; mean_outputs, all_outputs)
end

#### ELBO estimate

function run_elbo_estimate(
    ℰ::Encoder, 𝒢::Generator, 𝒫::Prior,
    b, A, mask, δb, θ;
    encoder_chunksize = size(b, 2),
    kwargs...,
)
    z = lib.chunked_apply(ℰ, b, A, mask, θ; chunksize = encoder_chunksize)
    return run_elbo_estimate(𝒢, 𝒫, z, b, A, mask, δb, θ; kwargs...)
end

function run_elbo_estimate(
    𝒢::Generator, 𝒫::Prior,
    z::AbstractMatrix, b, A, mask, δb, θ;
    samples = 125,
    generator_chunksize = size(b, 2),
    save_proposal_samples = true,
    verbose = false,
)
    @assert samples >= 2
    batchsize = size(b, 2)
    params, mcmc = dict(), dict()
    elbo, elbosq = nothing, nothing
    for i in 1:samples
        result = lib.chunked_apply(z, b, A, mask, δb, θ; chunksize = generator_chunksize) do _z, _b, _A, _mask, _δb, _θ
            post = ForwardPhysicsTransformedPosterior(𝒢, 𝒫, _z, _b, _A, _mask, _δb, _θ)
            return post(sample_base_similar(𝒢.model, _z))
        end
        (; summary, negloglike, neglogprior) = result
        x = inference_variables(summary.x)
        neglogw = negloglike .+ neglogprior .- summary.neglogpx
        elbo = lib.running_neglogmeanexp(i, elbo, cpu64(neglogw))
        elbosq = lib.running_neglogmeanexp(i, elbosq, cpu64(2 .* neglogw))
        params["ELBO_IS"] = elbo
        params["ESS_IS"] = exp.(.-(2 .* elbo .- elbosq))
        if i == 1
            params["Weights_IS"] = zeros(Float64, samples, batchsize)
            save_proposal_samples && (params["XDistProposalSamples"] = zeros(Float32, samples, size(x.xdist)...))
        end
        xdist64 = Float64.(x.xdist)
        params["XDist"] = lib.running_mean(i, i == 1 ? nothing : params["XDist"], xdist64)
        params["XDist_IS"] = lib.running_neglogmeanexp(i, i == 1 ? nothing : params["XDist_IS"], Float64.(neglogw .- log.(x.xdist)))
        save_proposal_samples && (params["XDistProposalSamples"][i, ..] .= cpu32(x.xdist))
        params["Weights_IS"][i:i, :] .= cpu64(neglogw)

        if i == 1
            mcmc["AcceptHist"] = lib.bool_similar(b, false, samples - 1, batchsize)
            mcmc["NegLogIR"], mcmc["XDist"] = neglogw, x.xdist
            mcmc["XDistSamples"] = zeros(Float32, samples - 1, size(x.xdist)...)
        else
            α = exp.(min.(0, mcmc["NegLogIR"] .- neglogw))
            accepted = lib.rand_similar(α) .< α
            mcmc["NegLogIR"] = ifelse.(accepted, neglogw, mcmc["NegLogIR"])
            mcmc["XDist"] = ifelse.(accepted, x.xdist, mcmc["XDist"])
            j = i - 1
            mcmc["AcceptHist"][j:j, :] .= accepted
            mcmc["XDistSamples"][j, ..] .= cpu32(mcmc["XDist"])
            params["XDist_MH"] = lib.running_mean(j, j == 1 ? nothing : params["XDist_MH"], mcmc["XDist"])
            params["AccRat"] = lib.running_mean(j, j == 1 ? nothing : params["AccRat"], α)
        end
        if verbose && (mod(i, max(1, samples ÷ 10)) == 0 || i == samples)
            @info "iter $i" mean_elbo = lib.mean(elbo) mean_ess = lib.mean(params["ESS_IS"])
        end
    end

    params["XDist"], params["XDist_IS"] = cpu64(params["XDist"]), cpu64(params["XDist_IS"])
    params["Weights_IS"] = exp.(elbo .- params["Weights_IS"]) ./ samples
    params["Weights_IS"] ./= sum(params["Weights_IS"]; dims = 1)
    params["XDist_IS"] = exp.(elbo .- params["XDist_IS"])
    for key in ("XDist", "XDist_IS", "XDist_MH")
        params[key] ./= sum(params[key]; dims = 1)
    end
    ρ = lib.autocorr_from_acceptance(cpu64(mcmc["AcceptHist"]))
    params["AutoCorrTime"] = lib.autocorr_time(ρ; dims = 1)
    params["XDistSamples"] = mcmc["XDistSamples"]
    return lib.map_dict(cpu32, params)
end
