# Bayesian $T_2$ distribution estimation using conditional normalizing flows

Source code accompanying *Bayesian $T_2$ distribution estimation using conditional normalizing flows* (Doucette, Kames, and Rauscher).

## Setup

Install dependencies with Julia 1.13 from the repository root:

```sh
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Documentation

The [technical documentation](docs/docs.pdf) provides extended mathematical descriptions, derivations, and experiments.

## Demo usage

Generate synthetic data, train a tiny model, and plot diagnostics:

```sh
julia --project scripts/prepare_synthetic_data.jl
julia --project scripts/train.jl
julia --project scripts/demo.jl
```

Outputs are stored in `outputs/demo-run/`.

## Data preparation

Paths, acquisition parameters, and train/val/test splits are set in `config/data.toml`.
Then, preprocess images for training and evaluation:

```sh
julia --project scripts/prepare_data.jl YOUR_CONFIG.toml
```

Example images and masks with 56 echoes and 7-ms uniform echo spacing [are available](https://github.com/jondeuce/mwiexamples).

## Training

Train a model on preprocessed data:

```sh
julia --project scripts/train.jl config/train.toml \
    --data.path=outputs/preprocessed/preprocessed_dataset.arrow \
    --data.config=outputs/preprocessed/preprocessed_dataset.toml \
    --log.dir=outputs/full-run
```

## Figures and tables

Generate figures and tables from a trained model:

```sh
julia --project scripts/figures.jl RUN_DIR docs/figures --image-id=ID --slice=INDEX
```
