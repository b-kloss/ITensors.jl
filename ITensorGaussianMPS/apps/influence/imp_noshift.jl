using ITensors


###define the individual elements of the product MPO
function get_TB(U,dt,ed::Pair,rs1::Index,rs2::Index,ls1::Index,ls2::Index)
    ed_up,ed_dn=ed
    T=[
        1 0 0 exp(-dt*ed_up)
        0 0 0 0
        0 0 0 0 
        exp(-dt*ed_dn) 0 0 exp(-dt*(ed_up+ed_dn+U))
    ]
    return itensor(T, ls1', ls2', dag(rs1), dag(rs2))
end

function get_TB(U,dt,ed::Pair,rs::Index,ls::Index)
    ed_up,ed_dn=ed
    T=[
        1 0 0 exp(-dt*ed_up)
        0 0 0 0
        0 0 0 0 
        exp(-dt*ed_dn) 0 0 exp(-dt*(ed_up+ed_dn+U))
    ]
    return itensor(T, ls', dag(rs))
end

function get_T(U,dt,ed::Pair,rs1::Index,rs2::Index,ls1::Index,ls2::Index)
    ed_up,ed_dn=ed
    T=[
        1 0 0 -exp(-dt*ed_up)
        0 0 0 0
        0 0 0 0 
        -exp(-dt*ed_dn) 0 0 exp(-dt*(ed_up+ed_dn+U))
    ]
    return itensor(T, ls1', ls2', dag(rs1), dag(rs2))
end

function get_T(U,dt,ed::Pair,rs::Index,ls::Index)
    ed_up,ed_dn=ed
    T=[
        1 0 0 -exp(-dt*ed_up)
        0 0 0 0
        0 0 0 0 
        -exp(-dt*ed_dn) 0 0 exp(-dt*(ed_up+ed_dn+U))
    ]
    return itensor(T, ls', dag(rs))
end

function get_Tcr(U,dt,ed::Pair,rs1::Index,rs2::Index,ls1::Index,ls2::Index)
    ed_up,ed_dn=ed
    T=[
        0 0 exp(-dt*ed_up) 0
        0 0 0 0
        0 0 0 0 
        0 0 -exp(-dt*(ed_up+ed_dn+U)) 0
    ]
    return itensor(T, ls1', ls2', dag(rs1), dag(rs2))
end

function get_Tcr(U,dt,ed::Pair,rs::Index,ls::Index)
    ed_up,ed_dn=ed
    T=[
        0 0 exp(-dt*ed_up) 0
        0 0 0 0
        0 0 0 0 
        0 0 -exp(-dt*(ed_up+ed_dn+U)) 0
    ]
    return itensor(T, ls', dag(rs),)
end

function get_Tcl(U,dt,ed::Pair,rs1::Index,rs2::Index,ls1::Index,ls2::Index)
    ed_up,ed_dn=ed
    T=[
        0 0 0 0
        0 0 0 0
        -exp(-dt*ed_dn) 0 0 exp(-dt*(ed_up+ed_dn+U)) 
        0 0  0
    ]
    return itensor(T, ls1', ls2', dag(rs1), dag(rs2))
end

function get_Tcl(U,dt,ed::Pair,rs::Index,ls::Index)
    ed_up,ed_dn=ed
    T=[
        0 0 0 0
        0 0 0 0
        -exp(-dt*ed_dn) 0 0 exp(-dt*(ed_dn+ed_up+U)) 
        0 0 0 0
    ]
    return itensor(T, ls',dag(rs))
end    

function get_TBcr(U,dt,ed::Pair,rs1::Index,rs2::Index,ls1::Index,ls2::Index)
    ed_up,ed_dn=ed
    T=[
        0 1 0 0
        0 0 0 0
        0 0 0 0 
        0 exp(-dt*ed_dn) 0 0
    ]
    return itensor(T, ls1', ls2', dag(rs1), dag(rs2))
end

function get_TBcr(U,dt,ed::Pair,rs::Index,ls::Index)
    ed_up,ed_dn=ed
    T=[
        0 1 0 0
        0 0 0 0
        0 0 0 0 
        0 exp(-dt*ed_dn) 0 0
    ]
    return itensor(T, ls', dag(rs))
end

function get_TBnr(U,dt,ed::Pair,rs1::Index,rs2::Index,ls1::Index,ls2::Index)
    ed_up,ed_dn=ed
    T=[
        1 0 0 0
        0 0 0 0
        0 0 0 0 
        exp(-dt*ed_dn) 0 0 0
    ]
    return itensor(T,  ls1', ls2', dag(rs1), dag(rs2))
end

function get_TBnr(U,dt,ed::Pair,rs::Index,ls::Index)
    ed_up,ed_dn=ed
    T=[
        1 0 0 0
        0 0 0 0
        0 0 0 0 
        exp(-dt*ed_dn) 0 0 0
    ]
    return itensor(T, ls', dag(rs))
end


function get_TBcl(U,dt,ed::Pair,rs1::Index,rs2::Index,ls1::Index,ls2::Index)
    ed_up,ed_dn=ed
    T=[
        0 0 0 0
        -1 0 0 -exp(-dt*ed_up)
        0 0 0 0 
        0 0 0 0
    ]
    return itensor(T, ls1', ls2', dag(rs1), dag(rs2))
end

function get_TBcl(U,dt,ed::Pair,rs::Index,ls::Index)
    ed_up,ed_dn=ed
    T=[
        0 0 0 0
        -1 0 0 -exp(-dt*ed_up)
        0 0 0 0 
        0 0 0 0
    ]
    return itensor(T, ls', dag(rs))
end


function get_MPO(U,dt,ed::Pair,combined_sites_l,combined_sites_r,states::Function,state_projection::ITensor;spin0="up",spin1="up",)
    #assumes merged pairs of sites
    #evaluates G(t[tind],0)=<cdag(0)c(t[tind])> when contracted with IM-MPSs
    M=length(combined_sites_l)
    states_vec=MPO[]
    for i in 1:M-1
        thefun = states(i,M-1;spin0=spin0,spin1=spin1)
        preMPO=ITensor[]
        push!(preMPO,state_projection)
        for n in 2:M-1
            push!(preMPO,thefun(n-1)(U,dt,ed,combined_sites_l[n],combined_sites_r[n]))
        end
        push!(preMPO, thefun(M-1)(U,dt,ed,combined_sites_l[1],combined_sites_l[M],combined_sites_r[1],combined_sites_r[M])*state_projection)
        #@show preMPO
        push!(states_vec,MPO(preMPO))

    end
    return states_vec
end

function get_state_projections(site_r,site_l)
    possible_states=["Emp","Occ"]
    projections=ITensor[]
    for astate in possible_states
        for bstate in possible_states
            push!(projections,state(site_r,astate)*state(prime(site_l),bstate))
        end
    end
    return projections
end


function get_Z_MPO(U,dt,ed::Pair,combined_sites_l,combined_sites_r,states::Function,state_projection::ITensor)
    #assumes merged pairs of sites
    #evaluates G(t[tind],0)=<cdag(0)c(t[tind])> when contracted with IM-MPSs
    M=length(combined_sites_l)
    #i=1 ##irrelevant here
    thefun=states(M-1)
    preMPO=ITensor[]
    push!(preMPO,state_projection)
    for n in 2:M-1
        push!(preMPO, thefun(n-1)(U,dt,ed,combined_sites_l[n],combined_sites_r[n]))
    end
    push!(preMPO, thefun(M-1)(U,dt,ed,combined_sites_l[M],combined_sites_l[1],combined_sites_r[M],combined_sites_r[1])*state_projection)
    
    return MPO(preMPO)
    
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
    #Tedge=state
    #tind=6
    #println("tind=",tind)
    #return n -> (n==Nt ? T0 : T)::Function
    return n -> (n==Nt ? T0 : T)::Function

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
    
    return cinds2,Φ
end


#dtau=0.1im
#U=0.0
#ed=0.5





