using LinearAlgebra
using ITensors
using ITensorGaussianMPS
using HDF5
using ITensors.HDF5
#using PyPlot
#matplotlib.use("QtAgg")
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
parameter_file=pwd()*"/params.jl"
include(parameter_file)
let 
    #c=h5read("/mnt/home/bkloss/projects/IM_solver/propagator_benchmark/generic_env/correlation_matrix_Jx0.27_Jy0.11.hdf5","corr_t=6")
    U=params.U
    ed=Pair(params.ed_u,params.ed_d)  #0.52,0.32
    @assert ed[1] == ed[2]
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
    Tren=params.T_ren
    save_psi=Bool(params.save_psi)
    save_B=Bool(params.save_B)
    save_c=Bool(params.save_c)
    if shift
        include("imp_refactored.jl")
    else
        include("imp_noshift_refactored.jl")
    end
    fout=h5open("results.h5","w")
    close(fout)
    #spec_dens(omega)= abs(omega-ed[1]) < gap ? 0.0 : V^2
    spec_dens=params.DOS
    lower,upper=params.domain()
    #plot(LinRange(lower,upper,Nt),spec_dens.(LinRange(lower,upper,Nt)))
    #show()
    #nu=params.nu
    #e_c=params.e_c
    #tanh()
    #spec_dens(omega) = (1.0 ./ (((1.0 .+ exp.(nu*(abs(omega-e_c))))) * (1.0 .+ exp(-nu*(abs(omega+e_c))))))
    #plot(LinRange(-5,5,1001),tanh.(e_c*(LinRange(-5,5,1001) .+ nu)) .* -tanh.(e_c*(LinRange(-5,5,1001) .- nu)) .+1)
    #spec_dens = x -> 0.5 * (tanh(e_c*(x+ nu)) * (-tanh(e_c*(x - nu))) +1)
    ##return  2 * gamma /((1+np.exp(nu*(energy - e_c))) * (1+np.exp(-nu*(energy + e_c)))) #this gives a flat band with smooth edges
    #return  2 * gamma /((1+np.exp(nu*(energy - e_c))) * (1+np.exp(-nu*(energy + e_c)))) #this gives a flat band with smooth edges
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
    #lower=-(D+gap)
    #upper=D+gap
    
    
    #Tren=4
    
    @show Tren
    Delta_tau=zeros(ComplexF64,(Nt*Tren)+1)
    
    for i in 0:(Nt*Tren)
        Delta_tau[i+1]=quadgk(omega -> 1*g_greater_beta(omega,i*dt/Tren),lower,upper)[1]
    end
    taus=Vector((0:Tren*Nt-1))*dt/Tren
    
    println("Getting G")
    @time begin
    #G=get_G(g_lesser_beta,g_greater_beta,dt/Tren,Nt*Tren,lower,upper;alpha=alpha)#,convention="b")
    G=get_G(Delta_tau,dt/Tren,Nt*Tren;alpha=alpha)#,convention="b")    
    end
    
    Delta_tau_test=zeros(ComplexF64,(Nt*Tren))
    alpha=1.0
    for i in 0:1#(Nt*Tren)
        Delta_tau_test[i+1]= quadgk(omega -> 1*g_greater_beta(omega,(Nt*Tren + i-1-Int(alpha))*dt/Tren),lower,upper)[1]
    end
    for i in 2:(Nt*Tren)-1
        Delta_tau_test[i+1]= - quadgk(omega -> 1*g_greater_beta(omega,(i-1-Int(alpha))*dt/Tren),lower,upper)[1]
    end
    
    
    println("Getting G")
    #@time begin
    G2=get_G_cont(Delta_tau_test,dt/Tren,Nt*Tren;alpha=alpha)
    matshow(real.(G))
    matshow(real.(G2))
    matshow(real.(G-G2))
    show()
    

    #@show G
    #end
    #G=get_G(Delta_tau,dt/Tren,Nt*Tren;alpha=alpha)#,convention="b")
    
    #matshow(real.(G))
    #matshow(real.(G-Gc))
    
    #show()
    #@show size(G)
    #end

    println("Integrating out auxiliary timesteps")
    @time begin
    G=integrate_out_timesteps(G,Tren)
    end
    if save_B
        fout=h5open("results.h5","w")
        fout["B"] = G
        close(fout)
    end

    
    #get_G(g_lesser_beta,g_greater_beta,dt,Nt,-D,D;alpha=1.0)#,convention="b")
    if iszero(U)
        println("Evaluating exact noninteracting solution")
        res2=evaluate_ni_aim(G,ed;convention='a')
        plot(real.(res2[1,1:end]))
        show()
        fout=h5open("results.h5","r+")
        fout["G_ex"] = real.(res2)
        close(fout)
    end
    println("Computing the IM and converting to MPS")
    ###getting correlation matrix and transforming it
    c=exp_bcs_julian(G)
    N=size(c,1)
    if shift
        shuffledinds=vcat(Vector(3:N),[1,2])
        c=c[shuffledinds,:][:,shuffledinds]
    end
    if !is_ph
        sites = siteinds("Fermion", div(N,2);conserve_nfparity=false,conserve_nf=false)
        sites_l = siteinds("Fermion", div(N,2);conserve_nfparity=false,conserve_nf=false)
    else
        sites = siteinds("Fermion", div(N,2);conserve_nfparity=true,conserve_nf=true)
        sites_l = siteinds("Fermion", div(N,2);conserve_nfparity=true,conserve_nf=true)
        c=apply_ph_everyother(c)
    end
    psi_r,_ = get_IM_from_corr(c,sites;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
    #psi_l,_ = get_IM_from_corr(c,sites_l;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
    
    psi_l = deepcopy(psi_r)
    #function swap_all_inds(T::ITensor)
    #    indpairs=[[ind,sim(ind)] for ind in inds(T)]
    #    oinds=collect(indpairs[i][1] for i in 1:length(indpairs))
    #    sinds=collect(indpairs[i][2] for i in 1:length(indpairs))
    #    
    #    T=swapinds(T,oinds,sinds)
    #    return T
    #end

    ####Ideally we want to swap inds instead of redoing the MPS construction for the identical environment
    #for i in 1:length(sites)
    #    #@show inds(psi_l[i])
    #    psi_l[i]=swap_all_inds(psi_l[i])#
##
 #       #@show inds(psi_l[i])
 #   end
    
    #psi_r,c=get_IM(G,shift,is_ph;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
    #psi_l,c=get_IM(G,shift,is_ph;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
    
    #psi_r2,c=get_IM_from_corr(c,siteinds(psi_r);eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
    #psi_l2,c=get_IM_from_corr(c,siteinds(psi_l);eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
    
    ###alternatively replaceinds on psi instead of redoing the whole calculation
    if save_B
        fout=h5open("results.h5","w")
        fout["c"] = c
        close(fout)
    end
    if save_psi
        fout=h5open("psi_IM.h5","w")
        fout["psi"]=psi_r
        close(fout)
    end
    #@show eltype(psi_r[1])
    #U,dt,ed=imp_parameters
    #beta,Nt=disc_parameters
    #
    #Z2,res2=contract(psi_l2,psi_r2,(U,dt,ed),(beta,Nt);shift=shift,ph=is_ph,env=!shift)
    #@show Z2
    println("Contracting")
    Z,res=contract(prime(psi_l),psi_r,(U,dt,ed),(beta,Nt);shift=shift,ph=is_ph,env=!shift)
    @show Z
        
   # Z,res=contract(psi_l,psi_r,(U,dt,ed),(beta,Nt);shift=shift,ph=is_ph,env=!shift)
    #@show Z
    ##log
    #@show size(res), size(res2)
    #@show (real.(res))
    #@show (real.(res2))
    #@show (real.(res[1:(end-1)]-res2[1,2:end]))
    #plot(abs.((real.(res[1,1:(end-1),1]-res2[2,2:end]))),"r")
    
    #plot(abs.((real.(res[1,1:(end-1),1]-res2[1,2:end]) ./ real.(res[1,1:(end-1),1]))))
    @show res
    @show size(res)
    plot(real.(res2[2,2:end]),"r-")
    plot(real.(res2[1,2:end]),"b-")
    
    plot(real.(res[1,1:(end-1)])./real.(res2[1,2:end]),"k--")
    #show()
    yscale("log")
    show()
    
    fout=h5open("results.h5","r+")
    fout["G"] = res
    fout["t"] = taus[2:end,1:1]
    fout["Z"] = Z
    close(fout)
    return

end
