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
    is_ph=true
    shift=false
    D=1.0
    V=1.0

    spec_dens(omega)= abs(omega) <0.3 ? 0.0 : V^2
    #spec_dens(omega)= V^2
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

    G_new=get_G(g_lesser_beta,g_greater_beta,dt,Nt,-D,D;alpha=1.0)#,convention="b")
    #get_G(g_lesser_beta,g_greater_beta,dt,Nt,-D,D;alpha=1.0)#,convention="b")
    if iszero(U)
        res2=evaluate_ni_aim(G_new,ed;convention='a')
        @show res2[1,:]
    end
    psi_r,c=get_IM(G_new,shift,is_ph;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)
    #@show eltype(psi_r[1])
    #U,dt,ed=imp_parameters
    #beta,Nt=disc_parameters
    Z,res=contract(psi_r,(U,dt,ed),(beta,Nt);shift=shift,ph=is_ph,env=!shift)
    @show real.(res)
    if iszero(U)
        @show size(res), size(res2)
        plot(abs.(real.(res[1:end-1])-real.(res2[1,2:end])) ./ abs.(real.(res2[1,2:end])),"k")
        plot(abs.(real.(res[1:end-1])-real.(res2[1,2:end])),"r")
        
        yscale("log")
        show()
    end
    return

end
