
contract(psi_l::MPS,imp_parameters, disc_parameters;shift=false,ph=true)=contract(psi_l,psi_l,imp_parameters, disc_parameters;shift=false,ph=true)

function contract(psi_l::MPS,psi_r::MPS, imp_params,disc_params;shift=false,ph=true)
    contract(psi_l,psi_r, imp_params,disc_params,Val(shift),Val(ph))
end

function contract(psi_l::MPS,psi_r::MPS, imp_parameters, disc_parameters,shift::Val{false},ph::Val{true})
    is_ph=true
    U,dt,ed=imp_parameters
    beta,Nt=disc_parameters#potentially more in the future, like Trotter order
    @assert beta/Nt==dt
    taus=Vector((0:Nt-1))*dt
    # merge pairs of sites (1,),(2,3),(4,5)...,(N,) except for the boundary sites
    combiners_r,combined_sites_r,psi_r_fused=fuse_indices_pairwise(psi_l)
    combiners_l,combined_sites_l,psi_l_fused=fuse_indices_pairwise(psi_r)
    # MPS on a ring implemented by projecting onto each element of the trace
    projs=get_state_projections(combined_sites_r[1],combined_sites_l[1])
    Z_MPOs=[]
    env_MPOs_up=[]
    env_boundary_MPOs_up=[]
    env_MPOs_dn=[]
    env_boundary_MPOs_dn=[]    
    for proj in projs
        push!(Z_MPOs,get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,combiners_l,combiners_r,get_Z_MPO_fun,proj,is_ph))
        push!(env_MPOs_up,get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,combiners_l,combiners_r,get_env_MPO_fun,proj,is_ph;spin="up")) 
        push!(env_boundary_MPOs_up,get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,combiners_l,combiners_r,get_env_boundary_MPO_fun,proj,is_ph;spin="up")) 
        #push!(env_MPOs_dn,get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,combiners_l,combiners_r,get_env_MPO_fun,proj,is_ph;spin="down"))
        #push!(env_boundary_MPOs_dn,get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,combiners_l,combiners_r,get_env_boundary_MPO_fun,proj,is_ph;spin="down"))
    end
    Z=0.0
    weights=[1.0,1.0,1.0,1.0]   ###appropriate for PH transformed
    ##compute partition function sequentially
    for (i,Z_MPO) in enumerate(Z_MPOs)
        contr=logdot(dag(psi_l_fused),product(Z_MPO,dag(prime(psi_r_fused));cutoff=1e-16))
        Z=Z+weights[i]*exp(contr)
        @show contr
    end
    Z=log(Z)
    @show Z
    phase=exp(-1im*imag(Z)/2.0)
    ###Handle complex phase
    if abs(real(phase)-1.0) >1e-8

        @show phase
        if abs(imag(phase)/real(phase))<1e-10
            phase=real(phase)
        end
        psi_r_fused=psi_r_fused*phase
        psi_l_fused=psi_l_fused*phase
        Z=0
        ###recompute Z with applied phase, should be real now if there's no bug. Can be removed, just a sanity checl
        for (i,Z_MPO) in enumerate(Z_MPOs)
            contr=logdot(dag(psi_l_fused),product(Z_MPO,dag(prime(psi_r_fused));cutoff=1e-16))
            @show contr
            Z=Z+weights[i]*exp(contr)
        end
        Z=log(Z)
    end
    res=zeros(ComplexF64,(2,length(taus),length(projs)))
    println("Done with calculating Z...")
    sitefactor=real(-Z)/float(length(env_MPOs_up[1]))
    @show sitefactor

    @time begin
        for (i,anmpo) in enumerate(env_MPOs_up)
            for site in 1:length(anmpo)
                anmpo[site]*=exp(sitefactor)
            end
            res[1,:,i]=exp(sitefactor)*weights[i]*get_corr_from_env(U,dt,ed,anmpo,env_boundary_MPOs_up[i],psi_l_fused,psi_r_fused,combiners_l,combiners_r,is_ph;spin="up")
        end
    end
    #@time begin
    #    for (i,anmpo) in enumerate(env_MPOs_dn)
    #        for site in 1:length(anmpo)
    #            anmpo[site]*=exp(sitefactor)
    #        end
    #        res[2,:,i]=exp(sitefactor)*weights[i]*get_corr_from_env(U,dt,ed,anmpo,env_boundary_MPOs_dn[i],psi_l_fused,psi_r_fused,combiners_l,combiners_r,is_ph;spin="up")
    #    end
    #end
    res[isnan.(res)] .= 0.0
    #@show res
    ress=sum(res,dims=3)
    return Z,ress
end