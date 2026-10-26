# Per-layer optimizer rule initialization, adapted from Optimisers.jl, src/interface.jl

module OptimisersSetup

using Optimisers
using Optimisers: AbstractRule, Leaf
using Optimisers: _trainable, init, isnumeric, mapvalue

function setup(replace, rule::AbstractRule, model)
    cache = IdDict()
    tree = _setup(replace, rule, model; cache)
    isempty(cache) && @warn "setup found no trainable parameters in this model"
    return tree
end
setup(rule::AbstractRule, model) = setup(Returns(nothing), rule, model)

# _setup is almost fmapstructure, but needs a _trainable_walk, and a cache which ignores numbers etc.
function _setup(replace, rule, x; cache)
    haskey(cache, x) && return cache[x]
    if isnumeric(x)
        ℓ = Leaf(rule, init(rule, x))
        if isbits(x)
            cache[nothing] = nothing  # just to disable the warning
            ℓ
        else
            cache[x] = ℓ
        end
    else
        tr = _trainable(x)
        new_rules = replace(x)
        rules = mapvalue(_ -> rule, tr)
        rules = new_rules === nothing ? rules : merge(rules, new_rules)
        mapvalue((ruleᵢ, xᵢ) -> _setup(replace, ruleᵢ, xᵢ; cache), rules, tr)
    end
end

end # module OptimisersSetup
