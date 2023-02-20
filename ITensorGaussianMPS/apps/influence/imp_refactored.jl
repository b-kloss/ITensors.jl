#using ITensors


include("impurity_tensors.jl")

function get_any_T(which_T, U,dt,ed,which_trafo::Int,prefactor, args)
    T=which_T(U,dt,ed,Val(2))
    #T=which_T(U,dt,ed)
    return convert_to_itensor(transform_particle_hole(T,which_trafo;prefactor=prefactor),args...)
end

function get_any_T_split(which_T, U,dt,ed,which_trafo::Int,prefactor, args)
    T=which_T(U,dt,ed,Val(2))
    #T=which_T(U,dt,ed)
    
    return convert_to_two_itensors(transform_particle_hole(T,which_trafo;prefactor=prefactor),args...)
end


function get_MPO(U,dt,ed::Pair,combined_sites_l,combined_sites_r,states::Function,is_ph::Bool;spin0="up",spin1="up")
    #assumes merged pairs of sites
    #evaluates G(t[tind],0)=<cdag(0)c(t[tind])> when contracted with IM-MPSs
    M=length(combined_sites_l)
    states_vec=MPO[]
    mode = is_ph ? 2 : 0
    for i in 1:M
        thefun = states(i,M;spin0=spin0,spin1=spin1)
        prefactor_fun = j -> isodd(j) ? -1 : 1
        prefactor_for_op = !is_ph ?  (n -> 1) : isodd(i) ? (n -> n==i ? -1 : 1) : (n -> 1)
        push!(states_vec,MPO([prefactor_for_op(n) * convert_to_itensor(transform_particle_hole(thefun(n)(U,dt,ed),mode;prefactor=prefactor_fun(n)),combined_sites_l[n],combined_sites_r[n]) for n in 1:M]))
        
        #return 
    
    end
    return states_vec
end

function get_cdagMPO(U,dt,ed::Pair,combined_sites_l,combined_sites_r,spin0="up",spin1="up")
    """useful for computing ProjMPO to calculate all correlators at O(Nt)"""
    M=length(combined_sites_l)
    preMPO=Vector[ITensor]
    for i in 1:M
        if i==M
            if spin0=="up"
                thefun=get_TBcr
            else
                thefun=get_TBcl
            end
        else
            thefun=get_T
        end
        push!(preMPO,thefun(U,dt,ed,combined_sites_l[i],combined_sites_r[i]))
    end
    return MPO(preMPO)
end

function get_GFels(M::MPO,IMu::MPS,IMd::MPS,MPOs)
    #PM=ProjMPO(M,nsite=1)

    PM=ProjMPO(0, length(M) + 1, 1, M, Vector{ITensor}(undef, length(M)))
    N=length(IMu)
    res=zeros(ComplexF64,N)
           ###maybe a single call to position is enough?
    for i in 1:N
        position!(PM, IMu,IMd, i)
        Lenv=projL(PM)
        Renv=projL(PM)
        res[i]=prime(IMd)*(Renv*(MPOs[i,i]*(Lenv*IMu[i])))
    end
    return res
end

function get_GFel(PM::ProjMPO,IMu::MPS,IMd::MPS,i::Int)
end


function get_Z_MPO(U,dt,ed::Pair,combined_sites_l,combined_sites_r,states::Function,is_ph::Bool;)
    #assumes merged pairs of sites
    #evaluates G(t[tind],0)=<cdag(0)c(t[tind])> when contracted with IM-MPSs
    M=length(combined_sites_l)
    #i=1 ##irrelevant here
    mode= is_ph ? 2 : 0
    thefun=states(M)
    prefactor_fun = i -> isodd(i) ? -1 : 1
    m=ITensor[]
    return MPO([get_any_T(thefun(n),U,dt,ed,mode,prefactor_fun(n),(combined_sites_l[n],combined_sites_r[n])) for n in 1:M])
    #return MPO([convert_to_itensor(transform_particle_hole(thefun(n)(U,dt,ed),1;prefactor=prefactor_fun(n)),combined_sites_l[n],combined_sites_r[n]) for n in 1:M])
    #return MPO([thefun(n)(U,dt,ed,combined_sites_l[n],combined_sites_r[n]) for n in 1:M])
end

function get_Z_MPO(U,dt,ed::Pair,sites_l::Vector{<:Index},sites_r::Vector{<:Index},links::Vector{<:Index},states::Function,is_ph::Bool;)
    M=div(length(sites_l),2)
    #i=1 ##irrelevant here
    mode= is_ph ? 2 : 0
    thefun=states(M)
    prefactor_fun = i -> isodd(i) ? -1 : 1
    sites= i -> i==1 ? (sites_l[2*i-1],sites_l[2*i],sites_r[2*i-1],sites_r[2*i],nothing,links[i]) : i==M ? (sites_l[2*i-1],sites_l[2*i],sites_r[2*i-1],sites_r[2*i],links[M-1],nothing) :  (sites_l[2*i-1],sites_l[2*i],sites_r[2*i-1],sites_r[2*i],links[i-1],links[i]) 
    #sites= i -> i==1 ? (sites_l[2*i],sites_l[2*i-1],sites_r[2*i],sites_r[2*i-1],nothing,links[i]) : i==M ? (sites_l[2*i],sites_l[2*i-1],sites_r[2*i],sites_r[2*i-1],links[M-1],nothing) :  (sites_l[2*i],sites_l[2*i-1],sites_r[2*i],sites_r[2*i-1],links[i-1],links[i]) 
    
    sitetensors=ITensor[]
    for n in 1:M
        A,B=get_any_T_split(thefun(n),U,dt,ed,mode,prefactor_fun(n),sites(n))
        #@show dense(A*B)
        #C=get_any_T(thefun(n),U,dt,ed,mode,prefactor_fun(n),(combined_sites_l[n],combined_sites_r[n]))
        
        push!(sitetensors,A)
        push!(sitetensors,B)
    end
    return MPO(sitetensors)
    #return MPO([get_any_T(thefun(n),U,dt,ed,mode,prefactor_fun(n),sites(n)) for n in 1:M])
    #return MPO([convert_to_itensor(transform_particle_hole(thefun(n)(U,dt,ed),1;prefactor=prefactor_fun(n)),combined_sites_l[n],combined_sites_r[n]) for n in 1:M])
    #return MPO([thefun(n)(U,dt,ed,combined_sites_l[n],combined_sites_r[n]) for n in 1:M])
end


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

function get_Z_MPO_fun(Nt::Int)
    T=get_T
    T0=get_TB
    #tind=6
    #println("tind=",tind)
    #return n -> (n==Nt ? T0 : T)::Function
    return n -> (n==Nt ? T0 : T)::Function

end



function fuse_indices_pairwise(Ψ::MPS)
    oinds=siteinds(Ψ)
    #println("Fusing")
    #@show oinds
    #@show inds(Ψ[1])
    T=eltype(Ψ[1])
    cinds=Index[]
    combiners=ITensor[]
    for i in 1:div(length(oinds),2)
        c=combiner(oinds[2*i],oinds[2*i-1])
        #c=combiner(oinds[2*i-1],oinds[2*i])
        
        push!(cinds,combinedind(c))
        push!(combiners, c)
    end
    Φ=MPS(T,cinds)
    for i in 1:div(length(oinds),2)
        Φ[i]=Ψ[2*i-1]*Ψ[2*i]*combiners[i]
    end
    return combiners,cinds,Φ
end


#dtau=0.1im
#U=0.0
#ed=0.5





