using LinearAlgebra
using ITensors
using Random: Random
using PyPlot
matplotlib.use("QtAgg")
include("det_aux.jl")
let
  my_rng = Random.MersenneTwister(42)
  n = 8
  M = zeros(Float64, n, n, 2, 2)
  Mm = Matrix{Matrix}(undef, n, n)
  for i in 1:n
    for j in 1:n
      vals = exp.(rand(my_rng, Float64, 4) .- 0.5)
      M[i, j, :, :] = vals
      Mm[i, j] = reshape(vals, 2, 2)
    end
  end
  #@show 
  @show size(Mm)
  @show size(M)
  parity = false

  @show restricted_determinant(M; parity=parity)
  se = get_stacked_eigvals(Mm)
  @show sort(vec(real.(se)))
  matshow(real.(se))
  matshow(imag.(se))
  show()
  return nothing
  offset = 1
  counter = false

  contr = get_contribution(copy(Mm), offset; counter=counter, parity=parity)
  @show contr
  ref = get_contribution_ref(copy(Mm), offset, counter; parity=parity)
  @show ref
  #return
  result = 0.0
  other_result = 0.0
  for offset in 0:(n - 1)
    for counter in (false, true)
      result = result + get_contribution_ref(copy(Mm), offset, counter; parity=parity)
      other_result =
        other_result + get_contribution(copy(Mm), offset; counter=counter, parity=parity)
      @show offset
      @assert abs(result - other_result) < 1e-3
    end
  end
  @show result, other_result
end
