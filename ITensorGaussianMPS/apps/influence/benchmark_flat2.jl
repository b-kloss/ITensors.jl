
#using Pkg
#Pkg.activate("../../../")
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
parameter_file="params.jl"
include(parameter_file)
#using Main.params



let
#U=0.0  #0.23
U=params.U
ed=Pair(params.ed_u,params.ed_d)  #0.52,0.32
beta=params.beta
Nt=params.Nt
dt=beta/Nt
taus=Vector((0:Nt-1))*dt
maxdim=params.maxdim
minblocksize=Int(log2(maxdim))
maxblocksize=Int(log2(maxdim))
@show maxdim,minblocksize

#minblocksize=params.minblocksize
#maxblocksize=params.maxblocksize
eigval_cutoff=params.eigval_cutoff
cutoff=params.cutoff


is_ph=false
D=1.0
V=1.0

spec_dens(omega)=V^2

function g_lesser_beta(omega,tau,tau_p)
    return g_lesser(omega,beta,tau,tau_p) * spec_dens(omega)
end
function g_greater_beta(omega,tau,tau_p)
    return g_greater(omega,beta,tau,tau_p) * spec_dens(omega)
end
println(typeof(g_lesser))
@show typeof(g_greater)
G_new=get_G(g_lesser_beta,g_greater_beta,dt,Nt,-D,D;alpha=1.0)#,convention="b")

matshow(real.(G_new))
show()

fout=h5open("/mnt/home/bkloss/projects/IM_solver/triqs_benchmark/G_julia_beta2.h5","w")
fout["G"] = G_new
close(fout)
return
println("getting IM")
using MKL
@show BLAS.set_num_threads(32)
psi_r,c=get_IM(G_new,true,is_ph;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
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
#Delta_t=get_Delta_t_flatDOS_mat(beta,V,D,10001,Nt)
#println("getting IM")
#psi_r,c=get_IM(taus,Delta_t,"Julian",true;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
#@show BLAS.set_num_threads(32)

#matshow(real.(c))
#show()
#return
#plot(abs.((real.(c[:,1]))))
#plot(Vector(1:size(c,1)),1e-2 * 1.0 ./ Vector(1:size(c,1)))

#xscale("log")
#yscale("log")
#show()
#SvN_ni=get_noninteracting_bipartite_entropy(c)
SvN_mps, SvN_spectrum=get_interacting_bipartite_entropy(psi_r)
#@show SvN_ni
#return
fout=h5open("results_beta"*string(beta)*"_Nt"*string(Nt)*"_chi"*string(maxdim)*".h5","w")
fout["svals"] = SvN_spectrum

@show SvN_mps

@show maxlinkdim(psi_r)
flush(stdout)
close(fout)

##for identical bath for both spin species reuse both 
psi_l=copy(psi_r)
#combiners_r,combined_sites_r,psi_r_fused,=fuse_indices_pairwise(psi_r)
#combiners_l,combined_sites_l,psi_l_fused=fuse_indices_pairwise(psi_l)
combined_sites_r,psi_r_fused,=fuse_indices_pairwise(psi_r)
combined_sites_l,psi_l_fused,=fuse_indices_pairwise(psi_l)
#@show linkinds(psi_l_fused)
#@show linkinds(psi_r_fused)
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
fout=h5open("results_beta"*string(beta)*"_Nt"*string(Nt)*"_chi"*string(maxdim)*".h5","r+")
fout["G"] = results
fout["t"] = taus[2:end,1:1]
close(fout)
#plot(real.(results[1:10]),"b")
#show()
return
end
