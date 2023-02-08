#using ITensors
include("impurity_tensors.jl")

function get_any_T(which_T, U,dt,ed,which_trafo::Int,prefactor, args)
    T=which_T(U,dt,ed,Val(2))
    #T=which_T(U,dt,ed)
    
    return convert_to_itensor(transform_particle_hole(T,which_trafo;prefactor=prefactor),args...)
end

###template from imp_noshift
function get_MPO(U,dt,ed::Pair,combined_sites_l,combined_sites_r,combiners_l,combiners_r,states::Function,state_projection::ITensor;spin0="up",spin1="up",)
    #assumes merged pairs of sites
    #evaluates G(t[tind],0)=<cdag(0)c(t[tind])> when contracted with IM-MPSs
    M=length(combined_sites_l)
    states_vec=MPO[]
    for i in 1:M-1
        thefun = states(i,M-1;spin0=spin0,spin1=spin1)
        preMPO=ITensor[]
        push!(preMPO,state_projection)
        #@show state_projection
        #return
        for n in 2:M-1
            push!(preMPO, get_any_T(thefun(n-1),U,dt,ed,0,prefactor_fun(n),(combiners_l[n],combiners_r[n])))
        end
        push!(preMPO, get_any_T(thefun(M-1),U,dt,ed,0,prefactor_fun(m-1),(combined_sites_l[1],combined_sites_l[M],combined_sites_r[1],combined_sites_r[M]))*dag(state_projection))
        #@show preMPO
        push!(states_vec,MPO(preMPO))

    end
    return states_vec
end

function get_state_projections(site_r,site_l)
    possible_states=["Emp","Occ"]
    projections=ITensor[]
    for astate in possible_states
        for bstate in reverse(possible_states)
            push!(projections,(state(site_r,astate))*(state(site_l,bstate)))
        end
    end
    return projections
end


get_Z_MPO(U,dt,ed::Pair,combined_sites_l,combined_sites_r,combiners_l,combiners_r,states::Function,state_projection::ITensor,;states_kwargs...)=get_Z_MPO(U,dt,ed::Pair,combined_sites_l,combined_sites_r,combiners_l,combiners_r,states,state_projection, false;states_kwargs...)
function get_Z_MPO(U,dt,ed::Pair,combined_sites_l,combined_sites_r,combiners_l,combiners_r,states::Function,state_projection::ITensor, is_ph::Bool;states_kwargs...)
    #assumes merged pairs of sites
    #evaluates G(t[tind],0)=<cdag(0)c(t[tind])> when contracted with IM-MPSs
    
    mode = is_ph ? 1 : 0
    M=length(combined_sites_l)
    #i=1 ##irrelevant here
    thefun=states(M-1;states_kwargs...)
    preMPO=ITensor[]
    push!(preMPO,dag(state_projection))
    #@show typeof(combiners_l)
    #@show typeof(combiners_r)
    #return MPO([get_any_T(thefun(n-1),U,dt,ed,1,prefactor_fun(n),(combiners_l[n],combiners_r[n])) for n in 1:M])
    prefactor_fun = i -> isodd(i) ? -1 : 1
    for n in 2:M-1
        #@show n,M
        push!(preMPO, get_any_T(thefun(n-1),U,dt,ed,mode,prefactor_fun(n),(combiners_l[n],combiners_r[n])))
    end
    #println("done with bulk")
    #@show inds(thefun(M-1)(U,dt,ed,combined_sites_l[1],combined_sites_l[M],combined_sites_r[1],combined_sites_r[M]))
    #@show inds(state_projection)
    #@show inds(thefun(M-1)(U,dt,ed,combined_sites_l[1],combined_sites_l[M],combined_sites_r[1],combined_sites_r[M])*dag(state_projection))
    println("Is it problem with dag stateproj?")
    @show inds(last(preMPO))
    push!(preMPO, (dag(get_any_T(thefun(M-1),U,dt,ed,mode,prefactor_fun(M-1),(combined_sites_l[1],combined_sites_l[M],combined_sites_r[1],combined_sites_r[M]))*dag(state_projection))))
    #println("done with all")
    
    @show inds(last(preMPO))
    @show inds(state_projection)
    println("Apparently not.")
    return MPO(preMPO)
end



###CHANGE
function get_corr_from_env(U,dt,ed::Pair,envMPO::MPO,boundaryMPO::MPO, psil::MPS,psir::MPS,combiners_l::Vector{ITensor},combiners_r::Vector{ITensor},is_ph::Bool;spin="up")
    ###take care of G(0) separately?
    #0, length(H) + 1, 2, H, Vector{ITensor}(undef, length(H))
    @show is_ph
    P=ITensors.ProjMPO(0, length(envMPO) + 1, 1, envMPO, Vector{ITensor}(undef, length(envMPO)))
    @show "getcorr"
    #set position to beginning of chain
    P=position!(P,psil,psir,1)
    res=ComplexF64[]
    #@show "before loop"
    #return
    mode = is_ph ? 1 : 0
    for pos in 2:length(envMPO)
        #@show pos
        P=position!(P,psil,psir,pos)
        L=lproj(P)
        R=rproj(P)
        combined_site_l=siteind(psil,pos)
        combined_site_r=siteind(psir,pos)
        combl=nothing
        combr=nothing
        for apos in 1:length(combiners_l)
            if hascommoninds(combiners_l[apos],combined_site_l)
                #println(pos, apos)
                combl=combiners_l[apos]
            end
        end
        for apos in 1:length(combiners_r)
            if hascommoninds(combiners_r[apos],combined_site_r)
                #println(pos," ",apos)
                combr=combiners_r[apos]
            end
        end
        if isnothing(combl) || isnothing(combr)
            error("no matching pos found")
        end
        if spin=="up"
            if pos==length(envMPO)
                localterm=boundaryMPO[length(envMPO)]
            else
                prefactor_fun = i -> isodd(i) ? -1 : 1
                ##the prefactor in front of get_any_T for is_ph takes into account that
                ##Tcr has an additional sign from the creation operator that the definition didn't take care of yet
                localterm=(is_ph ? prefactor_fun(pos) : 1) *get_any_T(get_Tcr,U,dt,ed,mode,prefactor_fun(pos),(combl,combr))
            end
        else
            if pos==length(envMPO)
                localterm=boundaryMPO[length(envMPO)]
            else
                localterm=get_any_T(get_Tcl,U,dt,ed,mode,prefactor_fun(pos),(combl,combr))
                #localterm=get_Tcl(U,dt,ed,combined_site_l,combined_site_r,combl,combr)
            end
        end
        println("after obtaining local term")
        val=(L*psil[pos])*localterm*((psir[pos])*R)
        println("after contracting local term")
        @assert order(val)==0
        @show scalar(val)
        push!(res,Complex(scalar(val)))
    end
    
    ###FIXME:take care of boundary G(0) later, once the rest works  
    return res
end

###no change below here
function get_1PGreens_MPO(tind::Int,Nt::Int; spin0="up",spin1="up")
    if spin0=="up"
        #right
        if tind==Nt
            T0=get_TBnr
        else
            T0=get_TBcr
        end
    else
        T0=get_TBcl
    end
    if spin1=="up"
        if tind==Nt
            T1=get_TBnr
        else
            T1=get_Tcr
        end
    else
        T1=get_Tcl
    end
    T=get_T
    #@assert tind!=Nt
    #return n -> (n==(Nt-tind+1) ? T1 : (n==1 ? T0 : T) )::Function
    return n -> (n==tind ? T1 : (n==Nt ? T0 : T) )::Function

end

function get_Z_MPO_fun(Nt::Int;states_kwargs...)
    #dummy kwargs not used here
    T=get_T
    T0=get_TB
    #Tedge=state
    #tind=6
    #println("tind=",tind)
    #return n -> (n==Nt ? T0 : T)::Function
    return n -> (n==Nt ? T0 : T)::Function

end

function get_env_MPO_fun(Nt::Int;spin="up")
    T=get_T
    if spin=="up"
        T0=get_TBcr
    elseif spin=="down"
        T0=get_TBcl
    end
    return (n -> (n==Nt ? T0 : T))::Function
end

function get_env_boundary_MPO_fun(Nt::Int;spin="up")
    T=get_T
    if spin=="up"
        T0=get_TBnr
    elseif spin=="down"
        @assert false   #not implemented
    end
    return (n -> (n==Nt ? T0 : T))::Function
end



function fuse_indices_pairwise(Ψ::MPS)
    oinds=siteinds(Ψ)
    M=length(oinds)
    #println("Fusing")
    #@show oinds
    #@show inds(Ψ[1])
    T=eltype(Ψ[1])
    cinds=Index[]
    cinds2=Index[]
    push!(cinds2,oinds[1])
    combiners=ITensor[]
    ###first is the corner case
    c=combiner(oinds[1],oinds[M])
    push!(cinds,combinedind(c))
    push!(combiners,c)
    
    for i in 1:div(length(oinds),2)-1
        c=combiner(oinds[2*i+1],oinds[2*i])
        #c=combiner(oinds[2*i-1],oinds[2*i])
        
        push!(cinds,combinedind(c))
        #if i<div(length(oinds),2)-1
        push!(cinds2,combinedind(c))
        #end
        push!(combiners, c)
    end
    push!(cinds2,oinds[M])
    #@show cinds2
    #@show length(cinds2)
    #@show combiners[div(length(oinds),2)-1]
    Φ=MPS(T,cinds2)
    Φ[1]=Ψ[1]
    for i in 2:div(length(oinds),2)
        #@show combiners[i]
        #@show inds(Ψ[2*(i-1)])
        #@show inds(Ψ[2*i-1])
        
        newT=Ψ[2*i-1]*Ψ[2*(i-1)]
        #@show inds(newT)
        #Φ[i]=newT*combiners[i]
        #if i==div(length(oinds),2)-1
        newT=newT*combiners[i]
        rl,ll,phys=inds(newT)
        Φ[i]=permute(newT,phys,ll,rl)
        #else
        #    Φ[i]=newT*combiners[i]
        #end
        #@show inds(Φ[i])
    end
    #@show inds(Φ[div(M,2)-1])
    Φ[div(M,2)+1]=Ψ[M]
    #@show inds(div(M,2))
    
    return combiners,cinds2,Φ
end


#dtau=0.1im
#U=0.0
#ed=0.5





