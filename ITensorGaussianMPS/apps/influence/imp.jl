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


function get_MPO(U,dt,ed::Pair,combined_sites_l,combined_sites_r,states::Function;spin0="up",spin1="up")
    #assumes merged pairs of sites
    #evaluates G(t[tind],0)=<cdag(0)c(t[tind])> when contracted with IM-MPSs
    M=length(combined_sites_l)
    states_vec=MPO[]
    for i in 1:M
        thefun = states(i,M;spin0=spin0,spin1=spin1)
        push!(states_vec,MPO([thefun(n)(U,dt,ed,combined_sites_l[n],combined_sites_r[n]) for n in 1:M]))
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


function get_Z_MPO(U,dt,ed::Pair,combined_sites_l,combined_sites_r,states::Function)
    #assumes merged pairs of sites
    #evaluates G(t[tind],0)=<cdag(0)c(t[tind])> when contracted with IM-MPSs
    M=length(combined_sites_l)
    #i=1 ##irrelevant here
    thefun=states(M)
    return MPO([thefun(n)(U,dt,ed,combined_sites_l[n],combined_sites_r[n]) for n in 1:M])
    
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
    return cinds,Φ
end


#dtau=0.1im
#U=0.0
#ed=0.5





