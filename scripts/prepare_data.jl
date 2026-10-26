using Library
const lib = Library
const ex = Library.Experiments

function prepare_data(args)
    filename = isempty(args) ? lib.projectdir("config", "data.toml") : only(args)
    cfg = lib.TOML.parsefile(lib.Config, filename)
    cfg.output = abspath(cfg.output::String)

    return ex.prepare_data(cfg)
end

abspath(PROGRAM_FILE) == (@__FILE__) && prepare_data(ARGS)
