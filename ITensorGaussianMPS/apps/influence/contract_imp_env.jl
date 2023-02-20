
get_val(::Val{x}) where x = x
#contract(psi_l::MPS,imp_parameters, disc_parameters;shift=false,ph=true,env=true)=contract(psi_l,psi_l,imp_parameters, disc_parameters;shift=shift,ph=ph,env=env)

function contract(psi_l::MPS,psi_r::MPS, imp_params,disc_params;shift=false,ph=true,env=true)
    contract(psi_l,psi_r, imp_params,disc_params,Val(shift),Val(ph),Val(env))
end

function contract(psi_l::MPS,psi_r::MPS,imp_parameters,disc_parameters,shift::Val{true},ph::Union{Val{true},Val{false}},env::Val{false})
    println("in contract with ph and shift")
    include("imp_refactored.jl")
    is_ph=get_val(ph)
    @show is_ph
    U,dt,ed=imp_parameters
    beta,Nt=disc_parameters#potentially more in the future, like Trotter order
    @assert beta/Nt==dt
    taus=Vector((0:Nt-1))*dt
    # merge pairs of sites (1,2),(3,4),(N-1,N)
    println("fuse inds")
    combiners_r,combined_sites_r,psi_r_fused=fuse_indices_pairwise(psi_l)
    combiners_l,combined_sites_l,psi_l_fused=fuse_indices_pairwise(psi_r)
    #combined_sites_r,psi_r_fused,=fuse_indices_pairwise(psi_r)
    #combined_sites_l,psi_l_fused,=fuse_indices_pairwise(psi_l)
    @show inner(psi_l_fused,psi_r_fused)
    @show U, dt, ed
    #Z_MPO=get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_Z_MPO_fun)
    Z_MPO=get_Z_MPO(U,dt,ed,combiners_l,combiners_r,get_Z_MPO_fun,is_ph)
    println("got ZMPO")
    Z=logdot(dag(psi_l_fused),(Z_MPO*psi_r_fused))
    @show Z
    #return
    phase=exp(-1im*imag(Z)/2.0)
    @show phase
    if abs(real(phase)-1.0) >1e-8
        psi_r_fused=psi_r_fused*phase
        psi_l_fused=psi_l_fused*phase
        Z=logdot(dag(psi_l_fused),Z_MPO*(psi_r_fused))
        @show Z
    
    end
    #centers=get_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_1PGreens_MPO;spin0="up",spin1="up")
    centers=get_MPO(U,dt,ed,combiners_l,combiners_r,get_1PGreens_MPO,is_ph;spin0="up",spin1="up")

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
        results[i] = exp(logdot(dag(psi_l_fused),centers[i]*psi_r_fused))
    end
    return Z, results
end



function contract(psi_l::MPS,psi_r::MPS, imp_parameters, disc_parameters,shift::Val{false},ph::Union{Val{true},Val{false}},env::Val{true})
    print("in contract false,true,true")
    include("imp_noshift_refactored.jl")
    #is_ph=true
    is_ph=get_val(ph)
    #@show is_ph
    #return
    U,dt,ed=imp_parameters
    beta,Nt=disc_parameters#potentially more in the future, like Trotter order
    @assert beta/Nt==dt
    taus=Vector((0:Nt-1))*dt
    # merge pairs of sites (1,),(2,3),(4,5)...,(N,) except for the boundary sites
    #@show "combining right"
    combiners_r,combined_sites_r,psi_r_fused=fuse_indices_pairwise(psi_r)
    #@show "combining left"
    combiners_l,combined_sites_l,psi_l_fused=fuse_indices_pairwise(psi_l)
    # MPS on a ring implemented by projecting onto each element of the trace
    projs=get_state_projections(combined_sites_r[1],combined_sites_l[1])    ###FIXME
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
    if is_ph
        weights=[1.0,1.0,1.0,1.0]   ###appropriate for PH transformed
        #weights=[-1.0,1.0,1.0,-1.0] 
    else
        #weights=[1.0,-1.0,-1.0,1.0]    #legacy
        weights=[-1.0,1.0,1.0,-1.0]  ###appropriate for non-PH transformed
        
    end
    @show weights, is_ph
        ##compute partition function sequentially
        
    
    for (i,Z_MPO) in enumerate(Z_MPOs)
        #@show siteinds(psi_r_fused)
        #@show siteinds(product(Z_MPO,psi_r_fused;cutoff=1e-16))
        #@show eltype.(Z_MPO)
        #contr=logdot(dag(psi_l_fused),product(Z_MPO,psi_r_fused;cutoff=1e-16))  #the bra gets daggered inside dot
        contr=logdot(dag(psi_l_fused),*(Z_MPO,psi_r_fused;cutoff=1e-16))  #the bra gets daggered inside dot
        
        Z=Z+weights[i]*exp(contr)
        @show contr
    end
    #@show Z
    Z=log(Z)
    @show Z
    #return
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
            contr=logdot(dag(psi_l_fused),product(Z_MPO,psi_r_fused;cutoff=1e-16))
            @show contr
            Z=Z+weights[i]*exp(contr)
        end
        Z=log(Z)
    end
    res=zeros(ComplexF64,(2,length(taus),length(projs)))
    println("Done with calculating Z...")
    #return
    sitefactor=real(-Z)/float(length(env_MPOs_up[1]))
    @show sitefactor

    @time begin
        for (i,anmpo) in enumerate(env_MPOs_up)
            for site in 1:length(anmpo)
                anmpo[site]*=exp(sitefactor)
            end
            res[1,:,i]=exp(sitefactor)*weights[i]*get_corr_from_env(U,dt,ed,anmpo,env_boundary_MPOs_up[i],psi_l_fused,dag(psi_r_fused),combiners_l,combiners_r,is_ph;spin="up")
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

