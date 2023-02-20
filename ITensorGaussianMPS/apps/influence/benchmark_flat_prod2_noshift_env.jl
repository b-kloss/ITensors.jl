
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
is_ph=false
taus=Vector((0:Nt-1))*dt
maxdim=params.maxdim
minblocksize=params.minblocksize
maxblocksize=params.maxblocksize
eigval_cutoff=params.eigval_cutoff
cutoff=params.cutoff

D=1.0
V=1.0#/sqrt(D)
V=1.0
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

function g_lesser_beta(omega,tdiff)
    return g_lesser(omega,beta,tdiff) * spec_dens(omega)
end
function g_greater_beta(omega,tdiff)
    return g_greater(omega,beta,tdiff) * spec_dens(omega)
end

println(typeof(g_lesser))
@show typeof(g_greater)


#Delta_t=get_Delta_t_flatDOS_mat(beta,V,D,10001,Nt;boundary=params.boundary)
#Delta_t_alt=get_Delta_t_functional_mat(Nt,beta,-D,D,g_lesser_beta,g_greater_beta)
#matshow(real.(Delta_t))

#colorbar()
#matshow(real.(Delta_t_alt))
#colorbar()
#matshow(real.(Delta_t_alt)-real.(Delta_t))
#colorbar()
#show()
println("getting G")
@time begin
G_new=get_G(g_lesser_beta,g_greater_beta,dt,Nt,-D,D;alpha=1.0)#,convention="b")
G2=get_G(g_lesser_beta,g_greater_beta,dt/8.,8*Nt,-D,D;alpha=1.0)#,convention="b")
#res=evaluate_ni_aim(G2,ed;convention='a')
res2=evaluate_ni_aim(G_new,ed;convention='a')
@show res2[1,:]
#plot(real.(res[1,1:4:end]))

end
#return
#@show res[1,:]
#G_old=get_G(Delta_t,dt)
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

#using MKL
#@show BLAS.set_num_threads(Threads.nthreads())
#@show BLAS.get_num_threads()
#matshow(ITensorGaussianMPS.reverse_interleave(real.(G_new)))
#show()
println("getting IM")
#psi_r,c=get_IM(taus,Delta_t,"Julian",true,false;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
#psi_r,c=get_IM(G_new,false;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
#psi_r,c=get_IM(integrate_out_timesteps(G2,8),false,is_ph;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
psi_r,c=get_IM(G_new,false,is_ph;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)

#if is_ph
#    c2=zeros(eltype(c),size(c,1)*2,size(c,2)*2)
#    c2[1:size(c,1),1:size(c,1)]=-c
#    c2[size(c,1)+1:end,size(c,1)+1:end]=c
#    #matshow(real.(c2))
#    #show()
#    SvN_ni=Fu.VN_entropy(Fu.Reduce_gamma(c2,div(size(c,1),2),1))
#    @show (SvN_ni)
#    SvN_ni=Fu.VN_entropy(Fu.Reduce_gamma(c2,div(size(c,1),2)+1,1))
#    @show (SvN_ni,SvN_ni-log(2))
#end
#SvN_ni=Fu.VN_entropy(Fu.Reduce_gamma(c2,div(size(c,1),2),2))
#@show (SvN_ni)
#show()
#return

@show linkdims(psi_r)
plot(linkdims(psi_r))
show()
#matshow(real.(c))
#show()

#SvN_ni=get_noninteracting_bipartite_entropy(c)
SvN_mps, SvN_spectrum=get_interacting_bipartite_entropy(psi_r;b=div(length(psi_r),2))
@show SvN_mps
SvN_mps, SvN_spectrum=get_interacting_bipartite_entropy(psi_r;b=div(length(psi_r),2)+1)
@show SvN_mps

#return
#@show SvN_ni
#return
fout=h5open("results.h5","w")
fout["svals"] = SvN_spectrum

@show SvN_mps

@show maxlinkdim(psi_r)
flush(stdout)
close(fout)

##for identical bath for both spin species reuse both 
psi_l=copy(psi_r)
combiners_r,combined_sites_r,psi_r_fused=fuse_indices_pairwise(psi_r)
combiners_l,combined_sites_l,psi_l_fused=fuse_indices_pairwise(psi_l)
@show inner(psi_l_fused,psi_r_fused)
@show U, dt, ed
projs=get_state_projections(combined_sites_r[1],combined_sites_l[1])

#return
Z_MPOs=[]
env_MPOs=[]
env_boundary_MPOs=[]

for proj in projs
    push!(Z_MPOs,get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,combiners_l,combiners_r,get_Z_MPO_fun,proj,is_ph))
    push!(env_MPOs,get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,combiners_l,combiners_r,get_env_MPO_fun,proj,is_ph;spin="up")) ### do equivalent for spin down if spin's not equivalent
    push!(env_boundary_MPOs,get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,combiners_l,combiners_r,get_env_boundary_MPO_fun,proj,is_ph;spin="up")) ### do equivalent for spin down if spin's not equivalent
    #push!(Z_MPOs,get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,combiners_l,combiners_r,get_Z_MPO_fun,proj))
    #push!(env_MPOs,get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,combiners_l,combiners_r,get_env_MPO_fun,proj;spin="up")) ### do equivalent for spin down if spin's not equivalent
    #push!(env_boundary_MPOs,get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,combiners_l,combiners_r,get_env_boundary_MPO_fun,proj;spin="up")) ### do equivalent for spin down if spin's not equivalent

end
println("Done with creating MPOs...")
#return
#@show inds.(Z_MPOs[1][1:3])
#@show inds.(psi_r_fused[1:3])

#return
Z=0.0

#@show Z_MPOs[1][1]
#@show Z_MPOs[1][2]
if is_ph
    weights=[1.0,1.0,1.0,1.0]   ###appropriate for PH transformed
    #weights=[1.0,-1.0,-1.0,1.0] ###appropriate for standard
else
    weights=[-1.0,1.0,1.0,-1.0] ###appropraite for non-PH transformed with refactored code
    #weights=[1.0,-1.0,-1.0,1.0]   ###appropriate for standard

end
    

@time begin
println("In logdots")
for (i,Z_MPO) in enumerate(Z_MPOs)
    #@show siteinds(dag(prime(psi_r_fused)))
   # @show siteinds(Z_MPO)
    contr=logdot(conj(dag(psi_l_fused)),product(Z_MPO,dag(prime(psi_r_fused));cutoff=1e-16))
    Z=Z+weights[i]*exp(contr)
    @show contr
end
println("Out logdots")

end
Z=log(Z)
@show Z

phase=exp(-1im*imag(Z)/2.0)
if abs(real(phase)-1.0) >1e-8

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
        contr=logdot(dag(psi_l_fused),product(Z_MPO,dag(prime(psi_r_fused));cutoff=1e-16))
        @show contr
        Z=Z+weights[i]*exp(contr)
    end
    Z=log(Z)
    end
end
println("Done with calculating Z...")
@show Z
centerss=[]
#for proj in projs
#    push!(centerss,get_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_1PGreens_MPO,proj;spin0="up",spin1="up"))
#end
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
    #@show storage(anmpo[length(anmpo)])
    #if all(iszero.(storage(anmpo[length(anmpo)])))
    #    continue
    #end
    res[:,i]=exp(sitefactor)*weights[i]*get_corr_from_env(U,dt,ed,anmpo,env_boundary_MPOs[i],psi_l_fused,psi_r_fused,combiners_l,combiners_r,is_ph;spin="up")
    #res[:,i]=exp(sitefactor)*weights[i]*get_corr_from_env(U,dt,ed,anmpo,env_boundary_MPOs[i],psi_l_fused,psi_r_fused,combiners_l,combiners_r;spin="up")

end
end
#res=Matrix(hcat(res...))
res[isnan.(res)] .= 0.0
#@show res
ress=sum(res,dims=2)
#plot(vcat(ress[end,1],ress[2:end-1,1]))
#show()
@show real.(ress)
plot(abs.(real.(res2[1,:])-real.(reverse(ress)))./abs.(res2[1,:]),"k")
plot(abs.(real.(res2[1,:])-real.(reverse(ress))),"r")
yscale("log")
#plot(,"r")
show()
return


counter=0

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
