
#using Pkg
#Pkg.activate("../../../")
using LinearAlgebra
using ITensors
using ITensorGaussianMPS
using HDF5
using PyPlot
using Random
Random.seed!(1234)
#matplotlib.use("QtAgg")
#using F_utilities
using Interpolations
#using GR
#const Fu=F_utilities
ITensors.disable_contraction_sequence_optimization()
#@show ITensors.mkl_get_num_threads()

#@show ITensors.mkl_get_num_threads()

include("imp_noshift.jl")
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
dt=beta/(Nt)

taus=Vector((0:Nt-1))*dt
maxdim=params.maxdim
minblocksize=params.minblocksize
maxblocksize=params.maxblocksize
eigval_cutoff=params.eigval_cutoff
cutoff=params.cutoff

D=10.0
V=10.0#/sqrt(D)
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
spec_dens(omega)=V^2

function g_lesser_beta(omega,tau,tau_p)
    return g_lesser(omega,beta,tau,tau_p) * spec_dens(omega)
end
function g_greater_beta(omega,tau,tau_p)
    return g_greater(omega,beta,tau,tau_p) * spec_dens(omega)
end
println(typeof(g_lesser))
@show typeof(g_greater)


Delta_t=get_Delta_t_flatDOS_mat(beta,V,D,10001,Nt;boundary=params.boundary)
Delta_t_alt=get_Delta_t_functional_mat(Nt,beta,-D,D,g_lesser_beta,g_greater_beta)
#matshow(real.(Delta_t))

#colorbar()
#matshow(real.(Delta_t_alt))
#colorbar()
#matshow(real.(Delta_t_alt)-real.(Delta_t))
#colorbar()
#show()
G_new=get_G(g_lesser_beta,g_greater_beta,dt,Nt,-D,D;alpha=1.0)#,convention="b")
G_old=get_G(Delta_t,dt)
#matshow(real.(G_new))
#matshow(real.(G_old))
#matshow(real.(G_new-G_old))


#show()
#return
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
@show BLAS.set_num_threads(Threads.nthreads())
@show BLAS.get_num_threads()

println("getting IM")
#psi_r,c=get_IM(taus,Delta_t,"Julian",true,false;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
psi_r,c=get_IM(G_new,false;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)

matshow(real.(c))
show()

#SvN_ni=get_noninteracting_bipartite_entropy(c)
SvN_mps, SvN_spectrum=get_interacting_bipartite_entropy(psi_r)
@show SvN_mps
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
projs=get_state_projections(combined_sites_r[1],combined_sites_l[1])
Z_MPOs=[]
env_MPOs=[]
env_boundary_MPOs=[]

for proj in projs
    push!(Z_MPOs,get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_Z_MPO_fun,proj))
    push!(env_MPOs,get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_env_MPO_fun,proj;spin="up")) ### do equivalent for spin down if spin's not equivalent
    push!(env_boundary_MPOs,get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_env_boundary_MPO_fun,proj;spin="up")) ### do equivalent for spin down if spin's not equivalent

end

Z=0.0

#@show Z_MPOs[1][1]
#@show Z_MPOs[1][2]
weights=[1.0,-1.0,-1.0,1.0]
@time begin
for (i,Z_MPO) in enumerate(Z_MPOs)
    contr=logdot(dag(psi_l_fused),product(Z_MPO,prime(psi_r_fused);cutoff=1e-16))
    Z=Z+weights[i]*exp(contr)
end
end
Z=log(Z)
phase=exp(-1im*imag(Z)/2.0)
@show phase
if abs(imag(phase)/real(phase))<1e-10
    phase=real(phase)
end
psi_r_fused=psi_r_fused*phase
psi_l_fused=psi_l_fused*phase
Z=0
###recompute Z with applied phase, should be real now if there's no bug. Can be removed, just a sanity checl
@time begin
for (i,Z_MPO) in enumerate(Z_MPOs)
    contr=logdot(dag(psi_l_fused),product(Z_MPO,prime(psi_r_fused);cutoff=1e-16))
    @show contr
    Z=Z+weights[i]*exp(contr)
end
end
Z=log(Z)
@show Z
centerss=[]
for proj in projs
    push!(centerss,get_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_1PGreens_MPO,proj;spin0="up",spin1="up"))
end
sitefactor=real(-Z)/float(length(env_MPOs[1]))
@show sitefactor
#sitefactor=0.0
#apply normalization (distributed over sites) by partition function to avoid over/underflow errors
#res=Vector{ComplexF64}[]
res=zeros(ComplexF64,(length(taus),length(projs)))

###Decide on whether to multithread here, probably better to just put all the threads into BLAS
#Threads.@threads for i in 1:length(projs)
#    anmpo=env_MPOs[i]
@time begin
for (i,anmpo) in enumerate(env_MPOs)
    for site in 1:length(anmpo)
        anmpo[site]*=exp(sitefactor)
    end
    @show storage(anmpo[length(anmpo)])
    #if all(iszero.(storage(anmpo[length(anmpo)])))
    #    continue
    #end
    res[:,i]=exp(sitefactor)*weights[i]*get_corr_from_env(U,dt,ed,anmpo,env_boundary_MPOs[i],psi_l_fused,dag(psi_r_fused);spin="up")
end
end
#res=Matrix(hcat(res...))
#res[isnan.(res)] .= 0.0
#@show res
ress=sum(res,dims=2)
plot(vcat(ress[end,1],ress[2:end-1,1]))
show()
@show ress

return


counter=0
BLAS.set_num_threads(1)
#@show exp(logdot(dag(prime(psi_l_fused)),centers[length(centers)]*psi_r_fused)-Z)
results=zeros(ComplexF64,(length(taus),length(projs)))
counter=0
BLAS.set_num_threads(1)
#@show exp(logdot(dag(prime(psi_l_fused)),centers[length(centers)]*psi_r_fused)-Z)
results=zeros(ComplexF64,(length(taus),length(projs)))
Threads.@threads for i = 1:length(taus)
    for j in 1:length(projs)
        results[i,j] = weights[j]*exp(logdot(dag(psi_l_fused),centerss[j][i]*prime(psi_r_fused))-Z)
    end
end
#@show results
results[isnan.(results)] .= 0.0
results=dropdims(sum(results,dims=2),dims=2)
@show results
fout=h5open("results.h5","r+")
fout["G"] = results
fout["t"] = taus[2:end,1:1]
close(fout)
#plot(real.(results[1:10]),"b")
#show()
return
end
