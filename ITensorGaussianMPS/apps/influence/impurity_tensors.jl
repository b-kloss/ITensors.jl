

uncombinedinds(cs::ITensor)=noncommoninds(combinedind(cs),inds(cs))
function get_T(U::Number,dt::Number,ed::Pair)
    ed_up,ed_dn=ed
    T=[
        1 0 0 -exp(-dt*ed_up)
        0 0 0 0
        0 0 0 0 
        -exp(-dt*ed_dn) 0 0 exp(-dt*(ed_up+ed_dn+U))
    
        ]
    return T
    #return itensor(T, ls1', ls2', dag(rs1), dag(rs2))
end

#no change in evolution for second order if no intermittent c,cdag
get_T(U::Number,dt::Number,ed::Pair,order::Val{2})=get_T(U,dt,ed)

get_TB(U::Number,dt::Number,ed::Pair,order::Val{2})=get_TB(U,dt,ed)
function get_TB(U::Number,dt::Number,ed::Pair)
    ed_up,ed_dn=ed
    T=[
    1 0 0 exp(-dt*ed_up)
    0 0 0 0
    0 0 0 0 
    exp(-dt*ed_dn) 0 0 exp(-dt*(ed_up+ed_dn+U))
    ]   
    return T
end

function get_Tcr(U::Number,dt::Number,ed::Pair)
    ed_up,ed_dn=ed
    T=[
        0 0 exp(-dt*ed_up) 0
        0 0 0 0
        0 0 0 0 
        0 0 -exp(-dt*(ed_up+ed_dn+U)) 0
    ]
    return T
end

function get_Tcr(U::Number,dt::Number,ed::Pair,order::Val{2})
    #println("second order!!!")
    ed_up,ed_dn=ed
    T=[
        0 0 exp(-dt/2.0*ed_up) 0
        0 0 0 0
        0 0 0 0 
        0 0 -exp(-dt/2.0*(ed_up+2*ed_dn+U)) 0
    ]
    return T
end

get_Tcl(U::Number,dt::Number,ed::Pair,order::Val{2})=get_Tcl(U,dt,ed)   #FIXME: implement later
function get_Tcl(U::Number,dt::Number,ed::Pair)
    ed_up,ed_dn=ed

    T=[
        0 0 0 0
        0 0 0 0
        -exp(-dt*ed_dn) 0 0 exp(-dt*(ed_up+ed_dn+U)) 
        0 0 0 0
    ]
    return T
end

function get_TBcr(U::Number,dt::Number,ed::Pair,order::Val{2})
    #println("second order!!!")
    ed_up,ed_dn=ed
    T=[
        0 exp(-dt/2.0 * ed_up) 0 0
        0 0 0 0
        0 0 0 0 
        0 exp(-dt/2.0 * (ed_up + 2*ed_dn + U)) 0 0
    ]
    return T
end

function get_TBcr(U::Number,dt::Number,ed::Pair)
    ed_up,ed_dn=ed
    T=[
        0 1 0 0
        0 0 0 0
        0 0 0 0 
        0 exp(-dt*ed_dn) 0 0
    ]
    return T
end

function get_TBcl(U::Number,dt::Number,ed::Pair)
    ed_up,ed_dn=ed
    T=[
        0 0 0 0
        -1 0 0 -exp(-dt*ed_up)
        0 0 0 0 
        0 0 0 0
    ]
    return T
end

get_TBnr(U::Number,dt::Number,ed::Pair,order::Val{2})=get_TBnr(U,dt,ed)
function get_TBnr(U::Number,dt::Number,ed::Pair)
    ed_up,ed_dn=ed
    T=[
        1 0 0 0
        0 0 0 0
        0 0 0 0 
        exp(-dt*ed_dn) 0 0 0
    ]
    return T
end

function transform_particle_hole(T,which::Int;prefactor=1)
    newT=similar(T)
    if which==1 ##c^dag_m -> -1(mod,m2) * d_m
        perm=[3,4,1,2] #from [0,1,2,3] to perm
        rperm=sortperm(perm) #should be [3,4,1,2] as well, so redundant
        prefactors=[1,1,prefactor,prefactor]
        for i in 1:size(T,1)
            for j in 1:size(T,2)
                newT[i,j]=T[rperm[i],rperm[j]]*prefactors[i]*prefactors[j]
            end
        end
    elseif which==2
        perm=[2,1,4,3]
        rperm=sortperm(perm)
        prefactors=[1,prefactor,1,prefactor]
        for i in 1:size(T,1)
            for j in 1:size(T,2)
                newT[i,j]=T[rperm[i],rperm[j]] * prefactors[i] * prefactors[j]
            end
        end
    elseif which==0
        newT=T       
    end
    return newT
end

function convert_to_itensor(T,linds::t,rinds::t) where t<:Index
    #return itensor(T,linds',dag(rinds))
    return itensor(T,linds,rinds)
end

function convert_to_itensor(T,lsc::ITensor,rsc::ITensor)
    r1,r2=uncombinedinds(rsc)
    l1,l2=uncombinedinds(lsc)
    #return itensor(T,dag(l1'),dag(l2'),r1,r2)*dag(rsc)*lsc'
    return itensor(T,l1,l2,r1,r2)*dag(rsc)*dag(lsc) ##dag on combiners should be correct?

end

function convert_to_itensor(T,ls1::t,ls2::t,rs1::t,rs2::t) where t<:Index
    #return itensor(T, ls1', ls2', dag(rs1), dag(rs2))
    return itensor(T, ls1, ls2, rs1, rs2)
end

#space_ii = all(hasqns, sites_a) || all(hasqns,sites_b)  ? [QN() => 1] : 1 #short circuiting or should be fine
#l = [Index(space_ii, "Link,l=$ii") for ii in 1:(N - 1)]

function convert_to_itensor(T,ls1::t,ls2::t,rs1::t,rs2::t,ll::t,lr::t) where t<:Index
    #return itensor(T, dag(ll),ls1', ls2', dag(rs1), dag(rs2),lr)
    return itensor(T, dag(ll),ls1, ls2, rs1, rs2,lr)
end

function convert_to_itensor(T,ls1::t,ls2::t,rs1::t,rs2::t,ll::Nothing,lr::t) where t<:Index
    #return itensor(T,ls1', ls2', dag(rs1), dag(rs2),lr)
    return itensor(T,ls1, ls2, rs1, rs2,lr)
end

function convert_to_itensor(T,ls1::t,ls2::t,rs1::t,rs2::t,ll::t,lr::Nothing) where t<:Index
    #return itensor(T,dag(ll), ls1', ls2', dag(rs1), dag(rs2))
    return itensor(T,dag(ll), ls1, ls2, rs1, rs2)

end

function convert_to_two_itensors(T,ls1::t,ls2::t,rs1::t,rs2::t,ll::t,lr::t) where t<:Index
    iT=convert_to_itensor(T,ls1,ls2,rs1,rs2,ll,lr)
    #u,s,v=svd(iT, dag(ll),ls1',dag(rs1))
    u,s,v=svd(iT, dag(ll),ls1,rs1)
    
    #absorb s in one of the tensors, say V
    return u, s*v
end

function convert_to_two_itensors(T,ls1::t,ls2::t,rs1::t,rs2::t,ll::Nothing,lr::t) where t<:Index
    iT=convert_to_itensor(T, ls1, ls2, rs1, rs2,ll,lr)
    #u,s,v=svd(iT, ls1',dag(rs1))
    u,s,v=svd(iT, ls1,rs1)
    
    #absorb s in one of the tensors, say V
    return u, s*v
end

function convert_to_two_itensors(T,ls1::t,ls2::t,rs1::t,rs2::t,ll::t,lr::Nothing) where t<:Index
    #iT=itensor(T, ls1', ls2', dag(rs1), dag(rs2),lr)
    iT=convert_to_itensor(T, ls1, ls2, rs1, rs2,ll,lr)
    #u,s,v=svd(iT, ll,ls1',dag(rs1))
    u,s,v=svd(iT, ll,ls1,rs1)
    
    #absorb s in one of the tensors, say V
    return u, s*v
end
        
    