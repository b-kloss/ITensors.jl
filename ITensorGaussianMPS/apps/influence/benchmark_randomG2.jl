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
include("imp.jl")

include("bath.jl")
include("aux.jl")
let 
    c=h5read("/mnt/home/bkloss/projects/IM_solver/propagator_benchmark/generic_env/correlation_matrix_Jx0.27_Jy0.11.hdf5","corr_t=6")
    beta=6
    Nt=5
    dt=1.0
    c=Matrix(transpose(c))
    matshow(real.(c))
    show()
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
    psi_r,c=get_IM_from_corr(c,false,is_ph;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
    @show eltype(psi_r[1])
    psi_l=copy(psi_r)
    combined_sites_r,psi_r_fused,=fuse_indices_pairwise(psi_r)
    combined_sites_l,psi_l_fused,=fuse_indices_pairwise(psi_l)
    @show inner(psi_l_fused,psi_r_fused)
    @show U, dt, ed
    Z_MPO=get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_Z_MPO_fun)
    #Z_MPO=get_Z_MPO(U,dt,ed,combiners_l,combiners_r,get_Z_MPO_fun)

    Z=logdot(dag(psi_l_fused),(Z_MPO*dag(prime(psi_r_fused))))
    @show Z
    #return
    phase=exp(-1im*imag(Z)/2.0)
    @show phase
    #psi_r_fused=psi_r_fused*phase
    #psi_l_fused=psi_l_fused*phase
    #Z=logdot(dag(psi_l_fused),Z_MPO*dag(prime(psi_r_fused)))
    #@show Z
    centers=get_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_1PGreens_MPO;spin0="up",spin1="up")
    #centers=get_MPO(U,dt,ed,combiners_l,combiners_r,get_1PGreens_MPO;spin0="up",spin1="up")

    results=ComplexF64[]
    counter=0
    BLAS.set_num_threads(1)
    #@show exp(logdot(dag(prime(psi_l_fused)),centers[length(centers)]*psi_r_fused)-Z)
    results=zeros(ComplexF64,length(taus))
    sitefactor=real(-Z)/float(length(centers[1]))
    Threads.@threads for i = 1:length(taus)
        for site in 1:length(centers[i])
            centers[i][site]*=exp(sitefactor)
        end
        #M=normalize(centers[i]; (lognorm!)=[-real(Z)])
        results[i] = exp(logdot(dag(psi_l_fused),centers[i]*dag(prime(psi_r_fused))))
    end
    @show results
    #fout=h5open("results_beta"*string(beta)*"_Nt"*string(Nt)*"_chi"*string(maxdim)*".h5","r+")
    #fout["G"] = results
    #fout["t"] = taus[2:end,1:1]
    #close(fout)
    #plot(real.(results[1:10]),"b")
    #show()
    return

end
