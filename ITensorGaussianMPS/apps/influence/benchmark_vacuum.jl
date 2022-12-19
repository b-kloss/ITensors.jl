
using LinearAlgebra
using ITensors
using ITensorGaussianMPS
using HDF5
#using F_utilities
#using GR
#const Fu=F_utilities
using PyPlot
matplotlib.use("QtAgg")
ITensors.disable_contraction_sequence_optimization()

include("imp.jl")
include("bath.jl")

let
U=0.0 #0.23
ed=Pair(0.0,0.0)  #0.52,0.32
dt=1.0

#f = h5open("/mnt/home/bkloss/projects/IM_solver/propagator_benchmark/vacuum_env/correlation_matrix_vacuum.hdf5", "r")

M=get_vaccuum_correlation_matrix(7)#
#shuffled_inds=vcat(Vector(3:size(M,1)),[1,2])
#M=M[shuffled_inds,shuffled_inds]
#matshow(real.(M))
#show()
#matshow(real.(ITensorGaussianMPS.reverse_interleave(M)))
#show()
#return
#fout=
#f = h5open("/mnt/home/bkloss/projects/IM_solver/propagator_benchmark/generic_env/correlation_matrix_Jx0.27_Jy0.11.hdf5", "r")
##c = Matrix(transpose(f["corr_t=6"][:,:]))
#for i in 1:size(c,1)
#    for j in 1:i
#        c[i,j]=c[j,i]
#    end
#end
#c += f["corr_t=6"][:,:]
#c = 0.5 .* c
#matshow(real.(c))
#show()
#return
c= M
cb=ITensorGaussianMPS.reverse_interleave(c)
ncb=zeros(eltype(cb),size(cb))

N = size(c,1)
K = div(N,2)
ncb[1:K,1:K]=cb[K+1:end,K+1:end]
ncb[K+1:end,K+1:end]=cb[1:K,1:K]
ncb[1:K,K+1:end]=cb[K+1:end,1:K]
ncb[K+1:end,1:K]=cb[1:K,K+1:end]
nc=zeros(eltype(cb),size(cb))
for i in 1:K
    for j in 1:K
       nc[2*i-1,2*j-1]=c[2*i,2*j]
       nc[2*i,2*j]=c[2*i-1,2*j-1]
       nc[2*i-1,2*j]=c[2*i,2*j-1]
       nc[2*i,2*j-1]=c[2*i-1,2*j]
       
    end
end

#c=c
#c = c[reverse(1:N),reversec(1:N)]
#c[:,N] = c[:,N]*-1im
#c[N,:] = c[N,:]*-1im
#c[:,N-1] = c[:,N-1]*-1im
#c[N-1,:] = c[N-1,:]*-1im

sites_l = siteinds("Fermion", div(N,2); conserve_qns=false)
sites_r = siteinds("Fermion", div(N,2); conserve_qns=false)
psi_r=ITensorGaussianMPS.correlation_matrix_to_mps(sites_r,copy(c);eigval_cutoff=1e-14,maxblocksize=14,cutoff=1e-14)
psi_l=ITensorGaussianMPS.correlation_matrix_to_mps(sites_l,copy(c);eigval_cutoff=1e-14,maxblocksize=14,cutoff=1e-14)

#psi_l=copy(psi_r)
#return

@show norm(psi_r)
@show inner(psi_l,psi_r)
#@show psi_r
combined_sites_r,psi_r_fused=fuse_indices_pairwise(psi_r)
combined_sites_l,psi_l_fused=fuse_indices_pairwise(psi_l)
@show inner(psi_l_fused,psi_r_fused)
Z_MPO=get_Z_MPO(U,dt,ed,combined_sites_r,combined_sites_l,get_Z_MPO_fun)
@show dot(dag(psi_l_fused),psi_r_fused)
Z=logdot((prime(psi_l_fused)),Z_MPO*(psi_r_fused))
altZ=inner((prime(psi_l_fused)),Z_MPO,psi_r_fused)
@show exp(Z), altZ
#@show Z_MPO[1]
#@show Z_MPO[2]

@show Z
#@show 1.0/Z
@show length(Z_MPO)
#return
#psi_l_fused=1/sqrt(Z)*psi_l_fused
#psi_r_fused=1/sqrt(Z)*psi_r_fused

centers=get_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_1PGreens_MPO;spin0="up",spin1="up")
results=[]
for center in centers
    push!(results,logdot((psi_l_fused),center*(prime(psi_r_fused))))
end
@show exp.(results .- Z)

return
end