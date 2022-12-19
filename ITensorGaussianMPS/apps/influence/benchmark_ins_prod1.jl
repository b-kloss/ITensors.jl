
#using Pkg
#Pkg.activate("../../../")
using LinearAlgebra
using ITensors
using ITensorGaussianMPS
using HDF5
#using PyPlot
#matplotlib.use("QtAgg")
using F_utilities
using Interpolations
#using GR
const Fu=F_utilities
ITensors.disable_contraction_sequence_optimization()
#@show ITensors.mkl_get_num_threads()

#@show ITensors.mkl_get_num_threads()

include("imp.jl")
include("bath.jl")
include("aux.jl")
parameter_file=pwd()*"/params.jl"
include(parameter_file)
#using Main.params



let
#U=0.0  #0.23
U=params.U
ed=Pair(params.ed_u,params.ed_d)  #0.52,0.32
beta=params.beta
#dt=params.dt
Nt=params.Nt
dt=beta/Nt

taus=Vector((0:Nt-1))*dt
maxdim=params.maxdim
minblocksize=params.minblocksize
maxblocksize=params.maxblocksize
eigval_cutoff=params.eigval_cutoff
cutoff=params.cutoff
D=params.D
V=params.D
DOSfunc=params.DOS

#Deltafn="/mnt/home/bkloss/projects/IM_solver/triqs_benchmark/Delta_beta10.h5"
#reftaus=h5read(Deltafn,"taus")
#@show last(reftaus)
#refvals=h5read(Deltafn,"data")
#Delta_inter=linear_interpolation(reftaus[:], refvals[:])
#taus=Vector(LinRange(0,beta,Nt))
#taus=taus[1:Nt-1]
#taus_all=Vector(LinRange(-beta,beta,Nt*2 -1))
#@show taus
#Delta_t= V^2 * D*Base.Math.sinc.(D*taus/2.0)
#Delta_t_ref=Delta_inter.(taus)#*0.001
#println("going")
Nw=10001
omegas=Vector(LinRange(-D,D,Nw))
Delta_t=get_Delta_t_genDOS_mat(beta,DOSfunc,omegas,Nt;boundary=params.boundary)
#Delta_t=Delta_t[1:(Nt-1),1:(Nt-1)]
#println("leaving")
#plot(taus[1:end],real.(Delta_t_g)[:,1])
#show()
#plot(taus,real.(Delta_t)[:,1],"b")
#plot(-taus,real.(Delta_t)[1,:],"r")
#plot(taus,real.(Delta_t_mat)[:,1],"b--")
#plot(-taus,real.(Delta_t_mat)[1,:],"r--")

#show()

using MKL
@show BLAS.set_num_threads(32)
println("getting IM")
psi_r,c=get_IM(taus,Delta_t,mode="Julian";eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
#matshow(real.(c))
#show()

SvN_ni=get_noninteracting_bipartite_entropy(c)
SvN_mps, SvN_spectrum=get_interacting_bipartite_entropy(psi_r)
@show SvN_ni
#return
fout=h5open("results.h5","w")
fout["svals"] = SvN_spectrum

@show SvN_mps

@show maxlinkdim(psi_r)
flush(stdout)
close(fout)

##for identical bath for both spin species reuse both 
psi_l=copy(psi_r)
combined_sites_r,psi_r_fused=fuse_indices_pairwise(psi_r)
combined_sites_l,psi_l_fused=fuse_indices_pairwise(psi_l)
@show inner(psi_l_fused,psi_r_fused)
@show U, dt, ed
Z_MPO=get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_Z_MPO_fun)
Z=logdot(dag(psi_l_fused),Z_MPO*prime(psi_r_fused))
@show Z
centers=get_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_1PGreens_MPO;spin0="up",spin1="up")
results=ComplexF64[]
counter=0
BLAS.set_num_threads(1)
#@show exp(logdot(dag(prime(psi_l_fused)),centers[length(centers)]*psi_r_fused)-Z)
results=zeros(ComplexF64,length(taus))
Threads.@threads for i = 1:length(taus)
    results[i] = exp(logdot(dag(psi_l_fused),centers[i]*prime(psi_r_fused))-Z)
end
@show results
fout=h5open("results.h5","r+")
fout["G"] = results
fout["t"] = taus[2:end,1:1]
close(fout)
#plot(real.(results[1:10]),"b")
#show()
return
end
