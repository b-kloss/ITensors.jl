

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

function get_Tcl(U::Number,dt::Number,ed::Pair)
    ed_up,ed_dn=ed

    T=[
        0 0 0 0
        0 0 0 0
        -exp(-dt*ed_dn) 0 0 exp(-dt*(ed_up+ed_dn+U)) 
        0 0  0
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

function convert_to_itensor(T,rinds::t,linds::t) where t<:Index
    return itensor(T,linds',dag(rinds))
end

function convert_to_itensor(T,rsc::ITensor,lsc::ITensor)
    r1,r2=uncombinedinds(rsc)
    l1,l2=uncombinedinds(lsc)
    return itensor(T,dag(l1'),dag(l2'),r1,r2)*dag(rsc)*lsc'
end

function convert_to_itensor(T,ls1::t,ls2::t,rs1::t,rs2::t) where t<:Index
    return itensor(T, ls1', ls2', dag(rs1), dag(rs2))
end


