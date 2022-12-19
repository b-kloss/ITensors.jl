
using LinearAlgebra
using ITensors
using ITensorGaussianMPS
using HDF5
#using F_utilities
#using GR
#const Fu=F_utilities
ITensors.disable_contraction_sequence_optimization()

include("imp.jl")
let
U=0.23  #0.23
ed=Pair(0.52,0.32)  #0.52,0.32
dt=1.0

#f = h5open("/mnt/home/bkloss/projects/IM_solver/propagator_benchmark/vacuum_env/correlation_matrix_vacuum.hdf5", "r")
#fout=
#f = h5open("/mnt/home/bkloss/projects/IM_solver/propagator_benchmark/generic_env/correlation_matrix_Jx0.27_Jy0.11.hdf5", "r")
G=rand(12,12)
G=G-G'



c = f["corr_t=6"][:,:]
N = size(c,1)
sites_l = siteinds("Fermion", div(N,2); conserve_qns=false)
sites_r = siteinds("Fermion", div(N,2); conserve_qns=false)
psi_r=ITensorGaussianMPS.correlation_matrix_to_mps(sites_r,copy(c);eigval_cutoff=1e-14,maxblocksize=14,cutoff=1e-14)
psi_l=ITensorGaussianMPS.correlation_matrix_to_mps(sites_l,copy(c);eigval_cutoff=1e-14,maxblocksize=12,cutoff=1e-14)
psi_l=copy(psi_r)
#return

@show norm(psi_r)
@show inner(psi_l,psi_r)
#@show psi_r
combined_sites_r,psi_r_fused=fuse_indices_pairwise(psi_r)
combined_sites_l,psi_l_fused=fuse_indices_pairwise(psi_l)
@show inner(psi_l_fused,psi_r_fused)
Z_MPO=get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_Z_MPO_fun)
Z=dot(dag((prime(psi_l_fused))),Z_MPO*psi_r_fused)
@show Z
psi_l_fused=1/sqrt(Z)*psi_l_fused
psi_r_fused=1/sqrt(Z)*psi_r_fused

centers=get_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_1PGreens_MPO;spin0="up",spin1="up")
results=[]
for center in centers
    push!(results,dot(dag(prime(psi_l_fused)),center*psi_r_fused))
end
@show reverse(results)

return
end
