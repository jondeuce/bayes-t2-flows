module Library

#### Dependencies

using Reexport: @reexport

@reexport using Adapt: Adapt
@reexport using Accessors: Accessors, @delete, @insert, @set, @reset
@reexport using Arrow: Arrow
@reexport using Base: Base
@reexport using CUDA: CUDA, @allowscalar, CuArray, CuIterator, CuMatrix
@reexport using ChainRules: ChainRules, @ignore_derivatives, @non_differentiable, @not_implemented, @scalar_rule, NoTangent, ZeroTangent, ignore_derivatives
@reexport using DECAES: DECAES, split_indices
@reexport using DataStructures: OrderedDict
@reexport using Distributions: Distributions, Gaussian, log2π
@reexport using EllipsisNotation: Ellipsis, (..)
@reexport using FFTW: rfft, irfft, dct, idct, idct!
@reexport using FastRicianLikelihoods: FastRicianLikelihoods, @define_unary_dual_scalar_rule, @define_binary_dual_scalar_rule, @define_ternary_dual_scalar_rule, withjacobian
@reexport using Flux: Flux, logsoftmax, logsumexp, softmax
@reexport using ForwardDiff: ForwardDiff, DiffResults
@reexport using Functors: Functors, fmap, fmap_with_path, fmapstructure, fmapstructure_with_path, functor
@reexport using JLD2: JLD2
@reexport using KrylovKit: KrylovKit
@reexport using LaTeXStrings: LaTeXStrings, @L_str, LaTeXString, latexstring
@reexport using LinearAlgebra: LinearAlgebra, BLAS, Hermitian, Symmetric, LowerTriangular, UpperTriangular, diag, dot, eigvals, logabsdet, logdet, mul!, norm, normalize, tr, ×, ⋅
@reexport using MLDataDevices: AbstractDevice, CPUDevice, CUDADevice, cpu_device, get_device
@reexport using MacroTools: MacroTools
@reexport using NNlib: NNlib, batched_mul, batched_transpose, bias_act! # `batched_vec` is patched below
@reexport using Optimisers: Optimisers, OptimiserChain, @lazy, AbstractRule, isnumeric, lazy
@reexport using ParameterSchedulers: ParameterSchedulers
@reexport using Parameters: @with_kw, @with_kw_noshow
@reexport using PrettyTables: PrettyTables, pretty_table
@reexport using Random: Random, AbstractRNG, Xoshiro, randexp, randn!, randperm, randperm!
@reexport using SHA: SHA
@reexport using StaticArrays: StaticArrays, @SMatrix, @SVector, MVector, StaticArray, SVector, SMatrix, SDiagonal
@reexport using Statistics: Statistics, cov, mean, median, quantile, std, var
@reexport using StatsBase: StatsBase, Histogram, sample, sample!
@reexport using TOML: TOML
@reexport using TOMLConfig: TOMLConfig, Config
@reexport using Tables: Tables
@reexport using UnsafeArrays: UnsafeArrays, uview
@reexport using Zygote: Zygote

#### Exports

export cpu, gpu, cpu32, gpu32, cpu64, gpu64
export @call, dict
export AbstractTensor3D, AbstractTensor4D, CuTensor3D, CuTensor4D
export arr_similar, fill_similar, ones_similar, rand_similar, randn_similar, zeros_similar

#### Fix

include("fix/misc.jl")
include("fix/DynamicEPG.jl")
include("fix/OptimisersSetup.jl")

@reexport using .DynamicEPG: DynamicEPG
@reexport using .OptimisersSetup: OptimisersSetup

#### Includes

include("common.jl")
include("flux.jl")
include("math_utils.jl")
include("losses.jl")
include("layers.jl")
include("flows.jl")
include("spatial.jl")
include("xformer.jl")

#### Experiments

include("experiments/Experiments.jl")

@reexport using .Experiments: Experiments

end # module Library
