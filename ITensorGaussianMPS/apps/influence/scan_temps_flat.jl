
#using Pkg
#Pkg.activate("../../../")
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
include("aux.jl")
parameter_file = "params.jl"
include(parameter_file)
#using Main.params

let
  #U=0.0  #0.23
  U = params.U
  ed = Pair(params.ed_u, params.ed_d)  #0.52,0.32
  beta = params.beta
  dt = params.dt
  maxdim = params.maxdim
  minblocksize = params.minblocksize
  maxblocksize = params.maxblocksize
  eigval_cutoff = params.eigval_cutoff
  cutoff = params.cutoff

  D = 1.0
  V = 1.0
  using MKL
  betas = [2.50, 5.0, 10.0, 20.0, 40.0]
  Nts = [50, 100, 200, 400]
  res = zeros(Float64, length(betas), length(Nts))
  beta_counter = 0
  Nt_counter = 0
  for beta in betas
    beta_counter += 1
    for Nt in Nts
      Nt_counter += 1
      dt = beta / Nt
      #Nt=Int(round(beta/dt))
      taus = Vector((0:(Nt - 1))) * dt

      #@show taus
      Delta_t = get_Delta_t_flatDOS_mat(beta, V, D, 10001, Nt)

      #@show BLAS.set_num_threads(32)
      println("getting IM")
      c = get_IM(
        taus,
        Delta_t,
        "Julian",
        false;
        eigval_cutoff=eigval_cutoff,
        minblocksize=minblocksize,
        maxblocksize=maxblocksize,
        maxdim=maxdim,
        cutoff=cutoff,
      )
      #matshow(real.(c))
      #show()
      SvN_ni = get_noninteracting_bipartite_entropy(c)
      res[beta_counter, Nt_counter] = SvN_ni
      @show SvN_ni
    end
    #yscale("log")
    #xscale("log")
    #show()
    Nt_counter = 0
  end
  colors = ["r", "magenta", "b", "c"]
  for i in 1:length(Nts)
    plot(betas, res[:, i]; color=colors[i], label=string(Nts[i]), marker=".")
  end
  xscale("log")
  legend()
  show()
end
