
#using Pkg
#Pkg.activate("../../../")
using LinearAlgebra
using ITensors
using ITensorGaussianMPS
using HDF5
using PyPlot
using Random
using MKL
using SkewLinearAlgebra
Random.seed!(1234)
matplotlib.use("QtAgg")
#using F_utilities
using Interpolations
#using GR
#const Fu=F_utilities
@show Threads.nthreads()
ITensors.disable_contraction_sequence_optimization()
ITensors.enable_threaded_blocksparse()
ITensors.NDTensors.Strided.disable_threads()
#@show ITensors.mkl_get_num_threads()

#@show ITensors.mkl_get_num_threads()

#include("imp_noshift_refactored.jl")
#include("imp_noshift.jl")

include("bath.jl")
include("aux.jl")
include("contract_imp_env.jl")
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
@assert iszero(Nt%4)
dt=beta/(Nt)
is_ph=true
taus=Vector((0:Nt-1))*dt
maxdim=params.maxdim
minblocksize=params.minblocksize
maxblocksize=params.maxblocksize
eigval_cutoff=params.eigval_cutoff
cutoff=params.cutoff

D=1.0
V=1.0#/sqrt(D)

spec_dens(omega)=V^2

function g_lesser_beta(omega,tau,tau_p)
    return g_lesser(omega,beta,tau,tau_p) * spec_dens(omega)
end
function g_greater_beta(omega,tau,tau_p)
    return g_greater(omega,beta,tau,tau_p) * spec_dens(omega)
end

function g_lesser_beta(omega,tdiff)
    return g_lesser(omega,beta,tdiff) * spec_dens(omega)
end
function g_greater_beta(omega,tdiff)
    return g_greater(omega,beta,tdiff) * spec_dens(omega)
end

println(typeof(g_lesser))
@show typeof(g_greater)



println("getting G")
@time begin
G_new=get_G(g_lesser_beta,g_greater_beta,dt,Nt,-D,D;alpha=1.0)#,convention="b")
#matshow(real.(G_new))
#show()
#return
G2=get_G(g_lesser_beta,g_greater_beta,dt/8.,8*Nt,-D,D;alpha=1.0)#,convention="b")
#res=evaluate_ni_aim(G2,ed;convention='a')
#res2=evaluate_ni_aim(integrate_out_timesteps(G2,8),ed;convention='a')
res2=evaluate_ni_aim(G_new,ed;convention='a')

#plot(real.(res[1,1:4:end]))
#plot(real.(res2[1,:]),".")
#show()
end
#return

println("getting IM")
psi_r,c=get_IM(G_new,false,is_ph;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)

@show linkdims(psi_r)
#plot(linkdims(psi_r))
#show()
#matshow(real.(c))
#show()

#SvN_ni=get_noninteracting_bipartite_entropy(c)
SvN_mps, SvN_spectrum=get_interacting_bipartite_entropy(psi_r;b=div(length(psi_r),2))
@show SvN_mps
SvN_mps, SvN_spectrum=get_interacting_bipartite_entropy(psi_r;b=div(length(psi_r),2)+1)
@show SvN_mps


fout=h5open("results.h5","w")
fout["svals"] = SvN_spectrum

@show SvN_mps

@show maxlinkdim(psi_r)
flush(stdout)
close(fout)
Z,res=contract(psi_r,(U,dt,ed),(beta,Nt);shift=false,ph=true,env=true)
@show real.(res)
#@show real.(reverse(res2[2,:]))

#plot(abs.(real.(res2[1,:])-real.(reverse(res[1,:])))./abs.(res2[1,:]),"k")
#plot(abs.(real.(res2[1,:])-real.(reverse(res[1,:]))),"r")
#yscale("log")
#plot(,"r")
show()
return
end