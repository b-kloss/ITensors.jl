
using LinearAlgebra
using ITensors
using ITensorGaussianMPS
using HDF5
using PyPlot
matplotlib.use("QtAgg")
using F_utilities
using Interpolations
#using GR
const Fu = F_utilities
ITensors.disable_contraction_sequence_optimization()
#@show ITensors.mkl_get_num_threads()

#@show ITensors.mkl_get_num_threads()

include("imp.jl")
include("bath.jl")
let
  B = h5read("/mnt/home/bkloss/projects/IM_solver/propagator_benchmark/B_rand.h5", "B")
  Bexp = exp_bcs_julian(B)
  Bexp_ref = h5read(
    "/mnt/home/bkloss/projects/IM_solver/propagator_benchmark/Bexp_rand.h5", "c"
  )
  matshow(real.(Bexp_ref))
  matshow(log10.(abs.(real.(Bexp - Bexp_ref))))
  show()
end
