using Library
const lib = Library
const ex = Library.Experiments

function prepare_synthetic_data(args)
    config_path = lib.projectdir("config", "train.toml")
    cfg = lib.TOMLConfig.parse_config(String[]; filename = config_path)

    output = isempty(args) ? lib.projectdir("outputs", "synthetic") : only(args)
    output = abspath(output)
    filename = ex.synthetic_dataset(cfg, output; seed = 42)
    @info "Generated synthetic dataset" filename
end

abspath(PROGRAM_FILE) == (@__FILE__) && prepare_synthetic_data(ARGS)
