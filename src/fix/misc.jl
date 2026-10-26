# `NNlib.batched_vec` doesn't handle reshaped views or provide a `batched_vec!`
batched_vec(A, x::SubArray) = batched_vec(A, copy(x))
batched_vec(A, x) = NNlib.batched_vec(A, x) # fallback
