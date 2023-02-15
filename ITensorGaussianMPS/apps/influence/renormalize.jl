
function coarsen_once(cl::AbstractMatrix,cr::AbstractMatrix,sites_l::Vector{<:Index},sites_r::Vector{<:Index},Zmpo::MPO)
    Ur,Urdag,projr,pmr,remaining_sites_r,Λcr=get_coarsening_by_layer(cr,dag(sites_r))
    Ul,Uldag,projl,pml,remaining_sites_l,Λcl=get_coarsening_by_layer(cl,dag(sites_l))        ###this is due to my stupid handling of sites ... gonna be a headache
    @show length(siteinds(Zmpo))
    @show length(sites_l)
    Uc=ITensorGaussianMPS.interleave(Ur,Ul)
    zmpo1=apply_coarsening_by_layer(Ur,Zmpo)
    zmpo1=apply_coarsening_by_layer(Ul,Zmpo)
    
    #zmpo1p=apply_coarsening_by_layer([dag(U) for U in reverse(Ur)],zmpo1)
    
    #zmpo1p=apply(zmpo1,pmr)
    #@show lognorm(Zmpo),lognorm(zmpo1),norm(Zmpo-zmpo1p)
    #return
    ###construct MPO out of projectors and Id to check whether MPO is invariant under projecting out sites.
    #zmpo1=project_out_sites(zmpo1,[projr,projl])
    #compare_siteinds(zmpo1,remaining_sites_r,remaining_sites_l)
    #zmpo2=apply_coarsening_by_layer(Ul,zmpo1)
    return Λcl,Λcr,dag(remaining_sites_l),dag(remaining_sites_r),zmpo1,Ul,Ur,Uldag,Urdag
end

function compare_siteinds(m::MPO,sr,sl)
    for s in sr
        @show s, findsite(m,s)
    end
    for s in sl
            @show s, findsite(m,s)
    end       
end


function get_coarsening_by_layer(c,sites)
    #get GMERA circuit
    N=length(sites)
    allinds=collect(1:N)
    Λ, V, indsnext, relinds=ITensorGaussianMPS.correlation_matrix_to_gmps_brickwall_tailed(c,collect(1:N);eigval_cutoff=1e-10,maxblocksize=6)
    removedinds=setdiff(allinds,relinds)
    
    #transform into many body gates
    U=ITensor[]
    Udag=ITensor[]
    for g in reverse(V.rotations)
        @assert abs(imag(g.s))<1e-8
        conjg=LinearAlgebra.Givens(g.i1,g.i2,conj(g.c),-conj(g.s))
        push!(U,dag(ITensor(sites,conjg))) #dag takes complex conjugate elementwise and reverses 
        push!(Udag,ITensor(sites,g))
            
    end
        #U = [ITensor(sites, g') for g in V.rotations]    ###in contrast to application to MPS no reverse here 
    
    #add projection onto states to remove disentangled states from MPO 
    Λc=Λ[removedinds,removedinds]
    Λr=Λ[relinds,relinds]
    
    ##add check for orthogonality of Λc
    #matshow(real.(Λc))
    #show()
    states=round.(Int,real.(diag(Λc))) .+ 1
    stateproj=ITensor[]
    projmpo=ITensor[]
    
    for (ind,st) in zip(removedinds,states)
        push!(stateproj,state((sites[ind]),st))
        
    end
    for ind in allinds
        if ind in removedinds
            st=states[findall(x->x==ind,removedinds)][1]
            push!(projmpo,state(dag(sites[ind]),st)*state(sites[ind]',st))
        else
            push!(projmpo,op("Id",sites[ind]))
        end
    end
    ##not entirely clear whether this needs to be done separately or can just be pushed to gates
    ##(whether apply works for projective 1-site gates)
    #@show projmpo
    #return transformation
    return U,Udag,stateproj,MPO(projmpo),sites[relinds],Λr
end

function apply_coarsening_by_layer(coarsening::Vector{ITensor}, m0::MPO)
    m=copy(m0)
    #origsites=siteinds(m)
    #remsites=siteinds(m)#not gonna work
    ##remove sites that were coarsened out
    for c in coarsening
        #@show inds(c)
        site=findsite(m,noprime(inds(c)))
        #@show site, dims(m[site])
        #println("in apply coarsening", c)
        #@show siteinds(m)[site]
        #m=noprime(c*m)

        m=apply(c,m;apply_dag=false,cutoff=0.0,maxdim=100,mindim=4)
        #@show siteinds(m)[site]
        
    end
    return m
    #    return apply(coarsening,m;apply_dag=false,cutoff=0.0,maxdim=100,mindim=4)     ###for symmetric bath this may actually be preferable
    
end

function apply_coarsenings_by_layer(coarsenings,m0::MPO)
    m=copy(m0)
    for coarsening in coarsenings
        m=apply_coarsening_by_layer(coarsening,m)
    end
    return m
    remsites=siteinds(m)
    ###figure out the signature of remsites, does it return a vector of length(m) with empty indset for sites that were coarsened out?
    ###in that case, simply loop over the empties and contract them to the right or left and extract the tensor to a new list
    ###that after end of iteration is converted to MPO and returned (such that next layer is nearest-neighbour again)

    orig_sites=siteinds(m0)
end

function project_out_sites(m::MPO,stateprojs)
    #sites=[]
    sites2proj=Dict()
    for stateproj in stateprojs
        for p in stateproj
            #push!(sites,findsite(m,ind(p)))
            site=findsite(m,first(inds(p)))

            if haskey(sites2proj,site)
                push!(sites2proj[site],p)
            else
                sites2proj[site]=[p,]
            end
        end
    end
    N=length(m)
    n=N-length(keys(sites2proj))
    nm=ITensor[]
    #apply projectors
    for site in 1:N
        T=copy(m[site])
        if haskey(sites2proj,site)
            #mortho=orthogonalize(m,site)
            #@show site, svd(mortho[site],uniqueinds(mortho[site],mortho[site+1]))
            #return
            for p in sites2proj[site]
                T=T*p
            end
        end
        push!(nm,T)
    end
    #apply 2nd order tensors to neighbours
    nm2=ITensor[]
    i=1
    while true
        if !hascommoninds(inds(nm[i]),siteinds(m)[i])
            if i==N ###handle boundary condition separately
                nm2[end]*=nm[i]
            else
                push!(nm2,nm[i]*nm[i+1])
                i+=2
            end
        else
            push!(nm2,nm[i])
            i+=1
        end
        if i>N
            break
        end
    end    
        
    return MPO(nm2)
end