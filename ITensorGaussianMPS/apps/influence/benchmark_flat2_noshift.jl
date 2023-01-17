
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
ITensors.enable_combine_contract()
include("imp_noshift.jl")
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

D=1.0
V=1.0
#Deltafn="/mnt/home/bkloss/projects/IM_solver/triqs_benchmark/Delta_beta10.h5"
#reftaus=h5read(Deltafn,"taus")
#@show last(reftaus)
#refvals=h5read(Deltafn,"data")
#Delta_inter=linear_interpolation(reftaus[:], refvals[:])
#taus=Vector(LinRange(0,beta,Nt))
#taus=taus[1:Nt-1]
#taus_all=Vector(LinRange(-beta,beta,Nt*2 -1))
@show taus
#Delta_t= V^2 * D*Base.Math.sinc.(D*taus/2.0)
#Delta_t_ref=Delta_inter.(taus)#*0.001
println("going")
Delta_t=get_Delta_t_flatDOS_mat(beta,V,D,10001,Nt)
#Delta_t[1,Nt]*=-1
#Delta_t[Nt,1]*=-1

matshow(real.(Delta_t))
show()
#Delta_t=Delta_t[1:(Nt-1),1:(Nt-1)]
println("leaving")
#plot(taus[1:end],real.(Delta_t_g)[:,1])
#show()
plot(taus,real.(Delta_t)[:,1],"b")
plot(-taus,real.(Delta_t)[1,:],"r")
#plot(taus,real.(Delta_t_mat)[:,1],"b--")
#plot(-taus,real.(Delta_t_mat)[1,:],"r--")

show()

using MKL
@show BLAS.set_num_threads(32)
println("getting IM")
psi_r,c=get_IM(taus,Delta_t,"Julian",true,false;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
@show length(psi_r)
matshow(real.(c))
show()
#plot(abs.((real.(c[:,1]))))
#plot(Vector(1:size(c,1)),1e-2 * 1.0 ./ Vector(1:size(c,1)))

#xscale("log")
#yscale("log")
#show()

#SvN_ni=get_noninteracting_bipartite_entropy(c)
SvN_mps, SvN_spectrum=get_interacting_bipartite_entropy(psi_r)
#@show SvN_ni
#return
fout=h5open("mew_results_beta"*string(beta)*"_Nt"*string(Nt)*"_chi"*string(maxdim)*".h5","w")
fout["svals"] = SvN_spectrum

@show SvN_mps

@show maxlinkdim(psi_r)
flush(stdout)
close(fout)

##for identical bath for both spin species reuse both 
psi_l=copy(psi_r)
for i in 1:length(psi_l)
    @show inds(psi_l[i])
end
#@return
combined_sites_r,psi_r_fused=fuse_indices_pairwise(psi_r)
combined_sites_l,psi_l_fused=fuse_indices_pairwise(psi_l)
@show linkinds(psi_r_fused)
@show linkinds(psi_l_fused)

@show inner(psi_l_fused,psi_r_fused)
@show U, dt, ed
projs=get_state_projections(combined_sites_r[1],combined_sites_l[1])
Z_MPOs=[]
for proj in projs
    push!(Z_MPOs,get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_Z_MPO_fun,proj))
end
Z=0.0

@show Z_MPOs[1][1]
#@show Z_MPOs[1][2]
weights=[1.0,-1.0,-1.0,1.0]
for (i,Z_MPO) in enumerate(Z_MPOs)
    contr=logdot(dag(psi_l_fused),Z_MPO*prime(psi_r_fused))
    Z=Z+weights[i]*exp(contr)
end
Z=log(Z)
phase=exp(-1im*imag(Z)/2.0)
@show phase
psi_r_fused=psi_r_fused*phase
psi_l_fused=psi_l_fused*phase
Z=0
for (i,Z_MPO) in enumerate(Z_MPOs)
    contr=logdot(dag(psi_l_fused),Z_MPO*prime(psi_r_fused))
    Z=Z+weights[i]*exp(contr)
end
Z=log(Z)
@show Z
centerss=[]
for proj in projs
    push!(centerss,get_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_1PGreens_MPO,proj;spin0="up",spin1="up"))
end
results=ComplexF64[]
counter=0
BLAS.set_num_threads(1)
#@show exp(logdot(dag(prime(psi_l_fused)),centers[length(centers)]*psi_r_fused)-Z)
results=zeros(ComplexF64,(length(taus),length(projs)))
sitefactor=real(-Z)/float(length(centerss[1][1]))
for i in 1:length(taus)
    for j in 1:length(projs)
        for site in 1:length(centerss[j][i])
            centerss[j][i][site]*=exp(sitefactor)
        end
    end
end
nZ=0.0
for (i,Z_MPO) in enumerate(Z_MPOs)
    for site in 1:length(Z_MPO)
        Z_MPO[site]*=exp(sitefactor)
    end
    contr=dot(dag(psi_l_fused),Z_MPO*prime(psi_r_fused))
    nZ=nZ+weights[i]*contr
end
@show exp(Z), nZ
Threads.@threads for i = 1:length(taus)
    for j in 1:length(projs)
        
        results[i,j] = weights[j]*inner(dag(psi_l_fused),centerss[j][i]*prime(psi_r_fused))
    end
end
#@show results
results[isnan.(results)] .= 0.0
results=dropdims(sum(results,dims=2),dims=2)
@show results
#fout=h5open("results_beta"*string(beta)*"_Nt"*string(Nt)*"_chi"*string(maxdim)*".h5","r+")
#fout["G"] = results
#fout["t"] = taus[2:end,1:1]
#close(fout)
#plot(real.(results[1:10]),"b")
#show()
return
end
