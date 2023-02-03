using LinearAlgebra
using ITensors
using ITensorGaussianMPS
using HDF5
using PyPlot
matplotlib.use("QtAgg")
#using F_utilities
using Interpolations
#using GR
#const Fu=F_utilities
ITensors.disable_contraction_sequence_optimization()
#@show ITensors.mkl_get_num_threads()

#@show ITensors.mkl_get_num_threads()

#include("imp_refactored.jl")
#include("imp.jl")
include("contract_imp_env.jl")
include("bath.jl")
include("aux.jl")
let 
    c=h5read("/mnt/home/bkloss/projects/IM_solver/propagator_benchmark/generic_env/correlation_matrix_Jx0.27_Jy0.11.hdf5","corr_t=6")
    beta=6.0
    Nt=6
    dt=1.0
    c=Matrix(transpose(c))
    N=size(c,1)
    shift,env=(true,false)
    shuffledinds=sortperm(vcat(Vector(3:N),[1,2]))
    c=c[shuffledinds,:][:,shuffledinds]
    #matshow(real.(c))
    #show()
    ed_up=0.52
    ed_down=0.32
    ed=Pair(ed_up,ed_down)
    U=0.23
    is_ph=false
    eigval_cutoff=1e-12
    minblocksize=6
    maxblocksize=8
    maxdim=256
    cutoff=0.0
    taus=dt*Vector(0:5)
    psi_r,c=get_IM_from_corr(real.(c),shift,is_ph;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
    #@show eltype(psi_r[1])
    #U,dt,ed=imp_parameters
    #beta,Nt=disc_parameters
    Z,res=contract(psi_r,(U,dt,ed),(beta,Nt);shift=shift,ph=is_ph,env=env)
    @show real.(res)
    return

end
