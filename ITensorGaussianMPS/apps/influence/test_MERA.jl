using LinearAlgebra
using ITensors
using ITensorGaussianMPS
using HDF5
using ITensors.HDF5
using PyPlot
using Random
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
include("renormalize.jl")
parameter_file=pwd()*"/params.jl"
include(parameter_file)
let 
    #c=h5read("/mnt/home/bkloss/projects/IM_solver/propagator_benchmark/generic_env/correlation_matrix_Jx0.27_Jy0.11.hdf5","corr_t=6")
    U=params.U
    ed=Pair(params.ed_u,params.ed_d)  #0.52,0.32
    @assert ed[1] == ed[2]
    ed=Pair(0.0,0.0)
    @show U,ed
    beta=params.beta
    Nt=params.Nt
    dt=beta/Nt
    taus=Vector((0:Nt-1))*dt
    minblocksize=Int(params.minblocksize)
    maxblocksize=Int(params.maxblocksize)
    maxdim=params.maxdim
    eigval_cutoff=params.eigval_cutoff
    cutoff=params.cutoff
    is_ph=Bool(params.ph)
    shift=Bool(params.shift)
    alpha=params.alpha
    D=params.D
    V=params.V
    gap=params.gap
    Random.seed!(1234)
    Tren=params.T_ren
    save_psi=Bool(params.save_psi)
    save_B=Bool(params.save_B)
    save_c=Bool(params.save_c)
    shift=true
    ph=false
    is_ph=ph
    if shift
        include("imp_refactored.jl")
    else
        include("imp_noshift_refactored.jl")
    end
    Tren=1


    spec_dens(omega)= abs(omega-ed[1]) < gap ? 0.0 : V^2
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
    #Tren=8
    #siteinds("Fermion", div(N,2);conserve_nfparity=false,conserve_nf=false)
    if is_ph
        sites_r=siteinds("Fermion",Nt*Tren*2;conserve_nf=true,conserve_nfparity=true)
        sites_l=siteinds("Fermion",Nt*Tren*2;conserve_nf=true,conserve_nfparity=true)
    else
        sites_r=siteinds("Fermion",Nt*Tren*2;conserve_nf=false,conserve_nfparity=false)
        sites_l=siteinds("Fermion",Nt*Tren*2;conserve_nf=false,conserve_nfparity=false)
    end

    if is_ph
        space_ii=[QN() => 1]
        links = [Index(space_ii,"Link,l=$ii") for ii in 1:Nt*Tren -1]
    else
        space_ii=1
        links = [Index(space_ii,"Link,l=$ii") for ii in 1:Nt*Tren -1]
        #error("trying only with symmetry now")
    end
    println(U,dt,ed,beta,Nt)
    Zmpo=get_Z_MPO(U,dt,ed,sites_l,sites_r,links,get_Z_MPO_fun,is_ph)

    G=get_G(g_lesser_beta,g_greater_beta,dt/Tren,Nt*Tren,-D,D;alpha=alpha)
    c=apply_ph_everyother(exp_bcs_julian(G))
    cr=copy(c)
    cl=copy(c)
    reshuffle=true
    N=size(c,1)
    @show N, length(sites_r)
    if reshuffle==true
        shuffledinds=vcat(Vector(3:N),[1,2])
        c=c[shuffledinds,:][:,shuffledinds]
    end
    show()
    matshow(real.(c))
    colorbar()
    show()
    #Λ, V, indsnext, relinds=ITensorGaussianMPS.correlation_matrix_to_gmps_brickwall_tailed(c,collect(1:N);eigval_cutoff=1e-12,maxblocksize=4)
    niter=1
    z=deepcopy(Zmpo)
    
    @show length(z)
    
    psil,_=get_IM_from_corr(c,dag(sites_l);eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
    psir,_=get_IM_from_corr(c,dag(sites_r);eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
    #@show inds()?
    contract(psil,psir,(U,dt,ed),(beta,Nt);shift=shift,ph=is_ph,env=!shift)

    println("outside of contract")
    #@show siteinds(z*psir)
    #@show siteinds(psil)
    
    Z=inner(dag(psil),z,psir)
    @show log(Complex(Z))
    #return
    #psir2=*(z,psir;cutoff=-eps(Float64))
    Z=logdot(dag(psil),z*psir)
    @show Z
    return
    for i in 1:niter
        cl,cr,sites_l,sites_r,z,Ul,Ur,Uldag,Urdag=coarsen_once(cl,cr,sites_l,sites_r,z)
        @show i, maxlinkdim(z),length(z) , length(sites_l),size(cl)
        psil=apply(Uldag,psil;cutoff=1e-10)
        psir=apply(Urdag,psir;cutoff=1e-10)
        #psil=apply((reverse(Ul)),psil;cutoff=1e-10)
        #psir=apply((reverse(Ur)),psir;cutoff=1e-10)
        
        #psil,_=get_IM_from_corr(cl,dag(sites_l);eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
        #psir,_=get_IM_from_corr(cr,dag(sites_r);eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
        #@show inds(z[4])
        #@show inds(psil[4])
        #@show inds(psir[4])
        Z=inner(dag(psil),z,psir)
        @show log(Z)
        #Z=logdot(dag(psil),*(z,psir;cutoff=0.0,alg="naive"))
        #@show Z
    end
    #Ur,projr,remaining_sites_r,Λcr=get_coarsening_by_layer(cr,sites_r)
    #Ul,projl,remaining_sites_l,Λcl=get_coarsening_by_layer(cl,sites_l)        ###this is due to my stupid handling of sites ... gonna be a headache
    #zmpo1=apply_coarsening_by_layer(Ur,Zmpo)
    #zmpo2=apply_coarsening_by_layer(Ul,zmpo1)
    #zmpof=project_out_sites(zmpo2,[projr,projl])
    #@show maxlinkdim(zmpof)
    #@show length(zmpof), length(Zmpo)
    #@show siteinds(zmpof)
    #@show projr[1]*zmpo1
    #@show zmpo1
    #@show isodd.(relinds)
    ##matshow(log10.(abs.(real.(Λ)));vmin=-10,vmax=0)
    ##colorbar()
    #show()

    #return Zmpo
    end