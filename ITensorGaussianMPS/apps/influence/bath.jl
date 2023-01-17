#using F_utilities
using LinearAlgebra
using PyPlot
using QuadGK

function get_IM(G::Matrix,reshuffle::Bool=true;kwargs...)
    eigval_cutoff=get(kwargs, :eigval_cutoff, 1e-12)
    maxblocksize=get(kwargs, :maxblocksize, 8)
    minblocksize=get(kwargs, :maxblocksize, 1)
    cutoff=get(kwargs,:cutoff,0.0)
    maxdim=get(kwargs,:maxdim,2^maxblocksize)
    c=exp_bcs_julian(G)
    N=size(c,1)
    if reshuffle==true
    #c=c[reverse(Vector(1:N)),reverse(Vector(1:N))]
        #shuffledinds=vcat(Vector(3:N),[1,2])
        shuffledinds=sortperm(vcat(Vector(3:N),[1,2]))
        
        c=c[shuffledinds,:][:,shuffledinds]
    end
    sites_r = siteinds("Fermion", div(N,2);conserve_qns=false)
    psi=ITensorGaussianMPS.correlation_matrix_to_mps(
        sites_r,copy(c);
        eigval_cutoff=eigval_cutoff,maxblocksize=maxblocksize,minblocksize=minblocksize,cutoff=cutoff,maxdim=maxdim
    )    
    return psi, c
end
function get_IM(taus::Vector,Delta::Matrix,mode::String="Julian",make_mps::Bool=true,reshuffle::Bool=true;kwargs...)
    dt=taus[2]-taus[1]
    @assert all(isapprox.(dt,diff(taus)))
    @show dt
    if mode=="Julian"
        N = size(Delta,1)
        #shuffledinds=vcat(Vector(2:N),[1,])
        #Delta=Delta[shuffledinds,shuffledinds]
        #matshow(real.(Delta))
        #show()
        
        c=get_correlation_matrix_Julian(Delta,dt)
        N = size(c,1)
        if reshuffle==true
        #c=c[reverse(Vector(1:N)),reverse(Vector(1:N))]
            shuffledinds=vcat(Vector(3:N),[1,2])
            c=c[shuffledinds,:][:,shuffledinds]
        end
        #matshow(real.(c))
        #show()
    else
        println("BK method")
        c=get_correlation_matrix(Delta,dt)
        N= size(c,1)
    end
    if make_mps==false
        return c
    else
        sites_r = siteinds("Fermion", div(N,2); conserve_qns=false)
        #sitesA = siteinds("Fermion", div(N,4); conserve_qns=false,conserve_nf=true)
        #sitesB = siteinds("AntiFermionIM", div(N,4); conserve_qns=false,conserve_nf=true)
        
        #sites_r=ITensorGaussianMPS.interleave(sitesA,sitesB)
        eigval_cutoff=get(kwargs, :eigval_cutoff, 1e-12)
        maxblocksize=get(kwargs, :maxblocksize, 8)
        minblocksize=get(kwargs, :maxblocksize, 1)
        cutoff=get(kwargs,:cutoff,0.0)
        maxdim=get(kwargs,:maxdim,2^maxblocksize)
        
        @time psi=ITensorGaussianMPS.correlation_matrix_to_mps(
            sites_r,copy(real.(c));
            eigval_cutoff=eigval_cutoff,maxblocksize=maxblocksize,minblocksize=minblocksize,cutoff=cutoff,maxdim=maxdim
        )    
        return psi, c
    end
end


function get_Delta_t_flatDOS(beta,V,D,N_w,N_taus)
    dt=beta/(N_taus-1)
    #dt*=(N_taus-1)/N_taus
    omegas=Vector(LinRange(-D,D,N_w))#linspace(-D,D,N_w)
    taus=Vector(LinRange(-beta,beta,2*N_taus-1))
    res=ComplexF64[]
    resp=ComplexF64[]
    resn=ComplexF64[]
    
    #respp=ComplexF64[]
    
    for tau in taus
        #p=V^2*(1.0/N_w)*sum(-exp.(-omegas .* (tau-beta))./(1 .+ exp.(omegas*beta)))
        #pp=V^2*(1.0/N_w)*sum(exp.(-omegas .* (tau-beta)) .*(1.0 .- (1.0 ./(1 .+ exp.(omegas*beta)))))        
        p=V^2*(2*D/N_w)*sum(-exp.(-omegas(tau)) ./ (1+exp.(-omegas*beta)))
        n=V^2*(2*D/N_w)*sum(exp.(omegas.*tau)./(1 .+ exp.(omegas*beta)))
        push!(res,(tau>=0.0) ? p : n)
        if (tau>=0.0)
            push!(resn,n)
            push!(resp,p)
            #push!(respp,pp)
            
        end
    end
    return res, resp,resn#,respp
end


#function get_integrated_gf(g::Function,beta::Number,tau::Number,tau_p::Number,spec_dens::Function,lower::Number,upper::Number)
#    g_wrap(omega) = 1.0/(2.0*pi)*g(omega,beta,tau,tau_p)*spec_dens(omega)
#    return quadgk(g_wrap, lower,upper;)
#end
        

function get_Delta_t_genDOS_mat(beta::Float64,Gamma::Function,omegas::Vector,N_taus::Int;boundary::Float64=1.0)
    "returns Delta_t,t' for a generic Gamma(omega). Assumes equidistant grid of omegas."
    N_w=length(omegas)
    Γ=(1.0/N_w).*(Gamma.(omegas))
    denom=1 .+ exp.(omegas*beta)
    mdenom=1 .+exp.(-omegas*beta)
    diagonal_el=sum(Γ .*(-boundary .+ (1.0 ./ denom)))
    dt=beta/(N_taus)
    Delta=zeros(ComplexF64,(N_taus,N_taus))
    for i in 1:N_taus
        for j in 1:N_taus

            if i<j
                Delta[i,j]= sum(Γ .*(exp.(-omegas.*(dt*(i-j)))./denom))
            elseif i==j
                #println(V^2*(1.0/N_w)*sum(-1 .+ 1 ./ (1 .+ exp.(omegas*beta))))
                Delta[i,j]=diagonal_el
            elseif i>j
                Delta[i,j]=sum( Γ .* (-exp.(-omegas*(dt*(i-j))) ./ mdenom) )
            end
        end
    end
    return Delta
end


function get_Delta_t_functional_mat(N_taus,beta,lower,upper,g_lesser::Function,g_greater::Function)
    dt=beta/(N_taus+1)
    @show dt
    V=1
    
    Delta=zeros(ComplexF64,(N_taus,N_taus))
    for i in 1:N_taus
        for j in 1:N_taus
            g_l(omega) = 0.5*g_lesser(omega,dt*i,dt*j)
            g_g(omega) = 0.5*g_greater(omega,dt*i,dt*j)
            
            if i<j
                Delta[i,j]=quadgk(g_l, lower,upper)[1]
            elseif i==j
                Delta[i,j]=quadgk(g_g, lower,upper)[1]
            else
                Delta[i,j]=quadgk(g_g, lower,upper)[1]
            end
        end
    end
    return Delta

end



function get_Delta_t_flatDOS_mat(beta,V,D,N_w,N_taus;boundary=1.0)
    #println("in")
    #@show beta,V,D,N_w,N_taus
    dt=beta/(N_taus)
    @show dt
    #@show Vector(LinRange(-D,D,N_w))
    omegas=Vector(LinRange(-D,D,N_w+1))[1:end-1]#linspace(-D,D,N_w)
    Delta=zeros(ComplexF64,(N_taus,N_taus))

    #respp=ComplexF64[]
    for i in 1:N_taus
        for j in 1:N_taus

            if i<j
                Delta[i,j]=V^2*(D/N_w)*sum(exp.(-omegas.*(dt*(i-j)))./(1 .+ exp.(omegas*beta)))
            elseif i==j
                #println(V^2*(1.0/N_w)*sum(-1 .+ 1 ./ (1 .+ exp.(omegas*beta))))
                Delta[i,j]=V^2*(D/N_w)*sum(-1.0 .+ (1.0 ./ (1 .+ exp.(omegas*beta))))
            elseif i>j
                Delta[i,j]=V^2*(D/N_w)*sum(-exp.(-omegas*(dt*(i-j))) ./ (1 .+exp.(-omegas*beta)))
                #Delta[i,j]=V^2*(1.0/N_w)*sum(-1 .+ exp.(-omegas*(dt*(i-j))) ./ (1 .+exp.(omegas*beta)))
                #Delta[i,j]=V^2*(1.0/N_w)*sum(-exp.(-omegas*(dt*(i-j))) .* (1.0 .- (1.0 ./ (1 .+ exp.(omegas*beta)))))
            
            end
        end
    end
    #matshow(real.(Delta))
    #show()
    #println(out)
    return Delta
end

function get_Delta_t_flatDOS_mat2(beta,V,D,N_w,N_taus)
    #println("in")
    #@show beta,V,D,N_w,N_taus
    dt=beta/(N_taus-1)
    #@show Vector(LinRange(-D,D,N_w))
    omegas=Vector(LinRange(-D,D,N_w))#linspace(-D,D,N_w)
    #taus=Vector(LinRange(-beta,beta,2*N_taus-1))
    #res=ComplexF64[]
    #resp=ComplexF64[]
    resn=ComplexF64[]
    Delta_lesser=zeros(ComplexF64,(N_taus,N_taus))
    Delta_greater=zeros(ComplexF64,(N_taus,N_taus))

    #respp=ComplexF64[]
    for i in 1:N_taus
        for j in 1:N_taus
            Delta_lesser[i,j]=V^2*(1.0/N_w)*sum(exp.(-omegas.*(dt*(i-j)))./(1 .+ exp.(omegas*beta)))
            Delta_greater[i,j]=V^2*(1.0/N_w)*sum(-exp.(-omegas*(dt*(i-j))) ./ (1 .+exp.(-omegas*beta)))
            
        end
    end
    #println(out)
    return Delta_lesser,Delta_greater
end


function get_G(Delta1::Vector,Delta2::Vector,dt)
    n=size(Delta1,1)
    G=zeros(ComplexF64,2*n,2*n)
    for i in 1:n
        for j in 1:i
            if i>j
                G[2*i,2*j-1]= Delta[abs(i-j)+1]*(-(dt^2))
                G[2*i-1,2*j]= Delta1[abs(i-j)+1]*(-(dt^2))
             elseif i<j
                #pass
                G[2*i,2*j-1]= Delta2[n-abs(i-j)]*(dt^2)
                G[2*i-1,2*j]= -Delta1[n-abs(i-j)]*(dt^2)
            elseif i==j
                G[2*i,2*i-1]= 1 - (0.5*Delta2[1]*(dt^2))
                G[2*i-1,2*i]= (0.5*Delta2[1]*(dt^2)) - 1       
            end
        end
    end
    G=0.5*(G-transpose(G))
    matshow(real.(G))
    matshow(imag.(G))
    show()
    
    G=G[vcat(Vector(2:2*n),[1]),vcat(Vector(2:2*n),[1])]
    #newG=G[vcat(Vector(2:2*n),[1]),:][:,vcat(Vector(2:2*n),[1])]
    return G    
end

function evaluate_flatDOS_lesser(lower,upper,beta,tau,tau_p)
    t=tau-tau_p
    up=1/t *exp(upper*t) * _₂F₁(1,-t/beta,1-t/beta,-exp(-beta*upper))
    lo=1/t *exp(lower*t) * _₂F₁(1,-t/beta,1-t/beta,-exp(-beta*lower))
    return up-lo
end

function evaluate_flatDOS_greater(lower,upper,beta,tau,tau_p)
    t=tau-tau_p
    up=1/t *exp(upper*t) * _₂F₁(1,-t/beta,1-t/beta,-exp(-beta*upper))
    lo=1/t *exp(lower*t) * _₂F₁(1,-t/beta,1-t/beta,-exp(-beta*lower))
    return up-lo
end
    

function get_G(g_lesser::Function,g_greater::Function,dt::Number,Nt::Number,lower::Number,upper::Number;alpha=1,convention="a")
    """exactly the way Julian does it up to units/factors"""
    G=zeros(ComplexF64,(2*Nt,2*Nt))
    #convention="b"
    factor=0.5/upper
    for m in 0:Nt-1   ##0 2*Nt-1 for python convention match
        tau=m*dt
        for n in m+1:Nt-1
            tau_p=n*dt
            #g_l(omega) = 0.5*g_lesser(omega,dt*i,dt*j)
            #g_g(omega) = 0.5*g_greater(omega,dt*i,dt*j)
            
            G[2*m+1,2*n+1+1]+= dt^2 * quadgk(omega -> factor*g_greater(omega,tau_p,tau+(dt*alpha)),lower,upper)[1]
            G[2*m+1+1,2*n+1]+= -1.0 * dt^2 * quadgk(omega -> factor*g_lesser(omega,tau,tau_p+(dt*alpha)),lower,upper)[1]
        end
        G[2*m+1,2*m+1+1] += dt^2 * quadgk(omega -> factor*g_lesser(omega,tau,tau+(dt*alpha)),lower,upper)[1]
        G[2*m+1,2*m+1+1] += - 1.
    end
    #matshow(real.(G))
    #show()
    G -=transpose(G)
    if convention=="b"
        for m in 0:Nt-1
            G[2*m+1,2*m+1+1] -= - 1.0
            G[2*m+1+1,2*m+1] -= 1.0
        end
        for m in 0:Nt*2-1
            G[m+1,((m+1)%2)+1:2:2*Nt] *= -1.0
        end
        G[end,:] *=-1
        G[:,end] *=-1
        id_meas= zeros(eltype(G),size(G))
        for m in 0:Nt-2
            id_meas[2*m+1+1,2*(m+1)+1] += 1.0
        end
        id_meas[1,end] -= 1.0
        id_meas -= transpose(id_meas)
        G +=id_meas
        G = inv(G)
        #shuffledinds=vcat(Vector(3:N),[1,2])
        #c=c[shuffledinds,:][:,shuffledinds]
    end

            
    return G
end

function get_G(Delta::Matrix,dt;alpha::Int=1)
    n=size(Delta,1)
    G=zeros(ComplexF64,2*n,2*n)
    #Delta[n,1]=-Delta[n,1]
   # matshow(real.(Delta))
    #show()
    #for i in 1:2*n-1
    #    G[i,i+1]=-0.5
    #    G[i+1,i]=+0.5
    #end
    
    for i in 1:n
        for j in 1:n
            if i>j
                G[2*i,2*j-1]= -(dt^2)*Delta[i,j]
                #G[2j-1,2*i]=-G[2*i,2*j-1]
                G[2*i-1,2*j]= (dt^2)*Delta[j,i]
                #G[2j,2*i-1]=-G[2*i-1,2*j]
                
            elseif i==j
                G[2*i,2*j-1]= (-(dt^2)*Delta[i,j])+1.0
                G[2*i-1,2*j]= ((dt^2)*Delta[i,j])-1.0
                #G[2*i,2*j+1]= (-(dt^2)*Delta[i,j])+0.5
                #G[2*i-1,2*j]= ((dt^2)*Delta[i,j])-0.5
                
            elseif i<j
                G[2*i,2*j-1]= -(dt^2)*Delta[i,j]
                G[2*i-1,2*j]= (dt^2)*Delta[j,i]
            end
        end
    end
    ###match Julian's convention and multiply last two rows/columns by -1
    #G[:,2*n-1:2*n]*=-1.0
    #G[2*n-1:2*n,:]*=-1.0
    
    #matshow(log10.(abs.(real.(G))))
    #matshow(sign.(real.(G)))

    #show()
    #G=0.5(G-transpose(G))
    #matshow(real.(G))
    #matshow(log10.(abs.(real.(G+transpose(G)))))
    #show()
    #fout=h5open("D_beta10.h5","r+")
    #fout["Deltat_G"] = G
    #close(fout)
    #matshow(real.(G))
    #show()
    #G=G[vcat(Vector(2:2*n),[1]),vcat(Vector(2:2*n),[1])]
    #newG=G[vcat(Vector(2:2*n),[1]),:][:,vcat(Vector(2:2*n),[1])]
    return G    
end

function get_G(Delta_l::Matrix,Delta_g::Matrix,dt)
    n=size(Delta_l,1)
    G=zeros(ComplexF64,2*n,2*n)
    for i in 1:n
        for j in 1:n
            if i>j
                G[2*i,2*j-1]= -(dt^2)*Delta_g[i,j]
                G[2*i-1,2*j]= (dt^2)*Delta_l[j,i]
            elseif i==j
                G[2*i,2*j-1]= (-(dt^2)*Delta_g[i,j])+1
                G[2*i-1,2*j]= ((dt^2)*Delta_g[i,j])-1
            elseif i<j
                G[2*i,2*j-1]= -(dt^2)*Delta_g[j,i]
                G[2*i-1,2*j]= (dt^2)*Delta_l[i,j]
            end
        end
    end
    #matshow(log10.(abs.(real.(G))))
    #matshow(imag.(G))
    #show()
    
    G=G[vcat(Vector(2:2*n),[1]),vcat(Vector(2:2*n),[1])]
    #newG=G[vcat(Vector(2:2*n),[1]),:][:,vcat(Vector(2:2*n),[1])]
    return G    
end
function get_correlation_matrix_Julian(Delta1,Delta2,dt)
    G=get_G(Delta1,Delta2,dt)
    #B=h5read("/mnt/home/bkloss/Downloads/B_julian.h5","B")
    
    #matshow(real.(G)+real.(B))
    #show()
    n=size(G,1)
    Gh=transpose(G) * conj(G)
    Gdd,R=eigen(Gh)
    #plot(real.(Gdd))
    #plot(imag.(Gdd),"--")
    #show()
    G_schur_complex = conj(transpose(R)) * G * conj(R)
    #matshow(real.(G_schur_complex))
    #matshow(imag.(G_schur_complex))
    eigenvalues_complex=diag(G_schur_complex,1)[1:2:end]
    D_phases=diagm(exp.(0.5im*angle.(ITensorGaussianMPS.interleave(eigenvalues_complex,eigenvalues_complex))))
    R = R * D_phases
    G_schur_real=conj(transpose(R)) * G * conj(R)
    #matshow(real.(G_schur_real))
    #show()
    eigenvalues_real=real.(diag(G_schur_real,1)[1:2:end])
    #plot(eigenvalues_real)
    #show()
    #hist(log10.(abs.(diff(eigenvalues_real))))
    #plot(log10.(abs.(diff(eigenvalues_real))),'.')
    #set_yscale("log")
    #show()
    corr_block_diag=zeros(eltype(G),2*n,2*n)
    #plot(real.(eigenvalues_real))
    #show()
    for i in 1:div(n,2)
        ew=eigenvalues_real[i]
        norm=1+abs(ew)^2
        corr_block_diag[2*i-1,2*i-1]=1/norm
        corr_block_diag[2*i,2*i]=1/norm
        corr_block_diag[2*i-1,2*i+n]=-ew/norm
        corr_block_diag[2*i, 2*i-1+n] = ew/norm
        corr_block_diag[2*i-1+n, 2*i] = conj(ew)/norm
        corr_block_diag[2 * i + n, (2*i)-1] = - conj(ew)/norm 
        corr_block_diag[2*i+n-1, 2*i+n-1] = abs(ew)^2/norm
        corr_block_diag[2 * i + n, 2 * i + n] = abs(ew)^2/norm
    end
    double_R=zeros(eltype(G),2*n,2*n)
    double_R[1:n,1:n]=R
    double_R[n+1:end,n+1:end]=conj(R)
    #matshow(real.(double_R))
    #matshow(imag.(double_R))
    #show()
    corr_block_back_rotated = double_R * corr_block_diag * conj(transpose(double_R))
    corr_block_back_rotated_2 = zeros(ComplexF64,2*n,2*n)
    corr_block_back_rotated_2[n+1:end,n+1:end]=corr_block_back_rotated[1:n,1:n]
    corr_block_back_rotated_2[n+1:end,1:n]=transpose(corr_block_back_rotated[1:n,n+1:end])
    corr_block_back_rotated_2[1:n,n+1:end]=transpose(corr_block_back_rotated[n+1:end,1:n])
    corr_block_back_rotated_2[1:n,1:n]=corr_block_back_rotated[n+1:end,n+1:end]
    #corr_comp=h5read("/mnt/home/bkloss/.julia/v1.7/dev/ITensors/ITensorGaussianMPS/apps/influence/testcorr_Julian_unfolded.h5","c")
    #corr_comp=f["c"]
    #@show size(corr_comp),size(corr_block_back_rotated)
    #matshow(real.(corr_block_back_rotated.-corr_comp))
    #show()
    
    
    corr_rotated =ITensorGaussianMPS.interleave(corr_block_back_rotated)
    #matshow(real.(corr_rotated))
    #show()
    return corr_rotated
end


function get_correlation_matrix_Julian(Delta,dt)
    G=get_G(Delta,dt)
    #matshow(real.(G))
    #show()

    #_,S,_=svd(G)
    #plot(abs.(S[1:(size(G,1)-1)] .- S[end]),label=string(div(size(G,1),2)))
        
    return exp_bcs_julian(G)
end 

function exp_bcs_julian(G::Matrix)
    #G=transpose(G)
    n=size(G,1)
    #matshow(real.(G+transpose(G)))
    #show()
    dim_G=size(G,1)
    random_part = rand(dim_G,dim_G) * 1.e-8
    G += random_part - transpose(random_part)
    Gh=transpose(G) * conj(G)
    Gdd,R=eigen(Hermitian(Gh))
    #matshow(real.(R * R'))
    #matshow(real.(R' * R))
    
    #show()
    #@show size(R)
    #plot(real.(Gdd))
    #plot(imag.(Gdd),"--")
    #show()
    G_schur_complex = R' * G * conj(R)
    #println("Showing Schur complex")
    #matshow(real.(G_schur_complex))
    #matshow(imag.(G_schur_complex))
    #show()
    #println("A")
    eigenvalues_complex=diag(G_schur_complex,1)[1:2:end]
    #plot(real.(eigenvalues_complex))
    #plot(imag.(eigenvalues_complex),"--")
    #show()
    D_phases=diagm(exp.(0.5im*angle.(ITensorGaussianMPS.interleave(eigenvalues_complex,eigenvalues_complex))))
    R = R * D_phases
    G_schur_real=conj(transpose(R)) * G * conj(R)
    #matshow(real.(G_schur_real))
    #show()
   # println("Showing Schur real")
    #matshow(real.(G_schur_real))
    #matshow(imag.(G_schur_real))
    #show()
    #println("B")
    eigenvalues_real=real.(diag(G_schur_real,1)[1:2:end])
    #plot(eigenvalues_real)
    #show()
    #hist(log10.(abs.(diff(eigenvalues_real))))
    #plot(log10.(abs.(diff(eigenvalues_real))),'.')
    #set_yscale("log")
    #show()
    corr_block_diag=zeros(eltype(G),2*n,2*n)
    #plot(real.(eigenvalues_real))
    #show()
    #println("C")
    for i in 1:div(n,2)
        ew=eigenvalues_real[i]
        norm=1+abs(ew)^2
        corr_block_diag[2*i-1,2*i-1]=1/norm
        corr_block_diag[2*i,2*i]=1/norm
        corr_block_diag[2*i-1,2*i+n]=-ew/norm
        corr_block_diag[2*i, 2*i-1+n] = ew/norm
        corr_block_diag[2*i-1+n, 2*i] = conj(ew)/norm
        corr_block_diag[2 * i + n, (2*i)-1] = - conj(ew)/norm 
        corr_block_diag[2*i+n-1, 2*i+n-1] = abs(ew)^2/norm
        corr_block_diag[2 * i + n, 2 * i + n] = abs(ew)^2/norm
    end
    double_R=zeros(eltype(G),2*n,2*n)
    double_R[1:n,1:n]=R
    double_R[n+1:end,n+1:end]=conj(R)
    #matshow(real.(double_R))
    #matshow(imag.(double_R))
    #show()
    #println("D")
    corr_block_back_rotated = double_R * corr_block_diag * conj(transpose(double_R))
    corr_block_back_rotated_2 = zeros(ComplexF64,2*n,2*n)
    corr_block_back_rotated_2[n+1:end,n+1:end]=corr_block_back_rotated[1:n,1:n]
    corr_block_back_rotated_2[n+1:end,1:n]=transpose(corr_block_back_rotated[1:n,n+1:end])
    corr_block_back_rotated_2[1:n,n+1:end]=transpose(corr_block_back_rotated[n+1:end,1:n])
    corr_block_back_rotated_2[1:n,1:n]=corr_block_back_rotated[n+1:end,n+1:end]
    #corr_comp=h5read("/mnt/home/bkloss/.julia/v1.7/dev/ITensors/ITensorGaussianMPS/apps/influence/testcorr_Julian_unfolded.h5","c")
    #corr_comp=f["c"]
    #@show size(corr_comp),size(corr_block_back_rotated)
    #matshow(real.(corr_block_back_rotated.-corr_comp))
    #show()
    
    
    corr_rotated =ITensorGaussianMPS.interleave(corr_block_back_rotated)
    return corr_rotated
end

function get_correlation_matrix(Delta,dt)
    ###takes as input discretization timestep dt and Fourier transformed Hybridization function Delta(t)
    ###works on blocked format and interleaves creation/destruction operators at last step
    println("starting Benedikt's version of computing the correlation matrix")
    G=get_G(Delta,dt)

    n=size(G,1)
    G=G[vcat(Vector(2:n),[1]),vcat(Vector(2:n),[1])]
    #fout=h5open("testhyb_calG_fromITensor.h5","w")
    #fout["G"] = G
    #close(fout)
    Gaug=zeros(eltype(G),2*n,2*n)
    Gaug[n+1:2*n,1:n]=G
    Gexp=exp(Gaug)
    
    #matshow(real.(Gexp+Gexp'))
    #show()
    #vacvec=zeros(ComplexF64,2*n)
    vacmat=zeros(ComplexF64,2*n,2*n)
    #vacvec[1:n].=1.0
    vacmat[1:n,1:n]=Matrix(LinearAlgebra.I(n))
    #matshow(real.(vacmat))
    #show()
    #Gexpaug=Gexp .+ Gexp'
    thec=(Gexp)*vacmat*transpose(Gexp)
    println("transformed G")
    matshow(imag.(thec))
    colorbar()
    show()

    thec=ITensorGaussianMPS.interleave(thec)
    return 0.5*(thec+thec')
end

function get_correlation_matrix(G)
    ###takes as input discretization timestep dt and Fourier transformed Hybridization function Delta(t)
    ###works on blocked format and interleaves creation/destruction operators at last step
    n=size(G,1)
    Gaug=zeros(eltype(G),2*n,2*n)
    Gaug[n+1:2*n,1:n]=G
    Gexp=exp(Gaug)
    #@show Gexp
    Gaug=zeros(eltype(G),2*n,2*n)
    Gaug[1:n,n+1:2*n]=-conj(G)
    Gexp2=exp(Gaug)
    #Gexp2=
    vacmat=zeros(ComplexF64,2*n,2*n)
    vacmat[1:n,1:n]=Matrix(LinearAlgebra.I(n))
    
    thec=Gexp*vacmat*transpose(Gexp)
    println("transformed G")
    thec=ITensorGaussianMPS.interleave(thec)
    #@show thec
    return thec
end

function get_vaccuum_correlation_matrix(Nsteps::Int)
    M=zeros(ComplexF64,4*Nsteps,4*Nsteps)
    subM=[
        0.5 0 0 0.5
        0 0.5 -0.5 0
        0 -0.5 0.5 0
        0.5 0 0 0.5
        ]
    for i in 1:Nsteps
        M[(i-1)*4+1:i*4,(i-1)*4+1:i*4].=subM
    end
    inds=Vector(1:(4*Nsteps))
    ninds=vcat(inds[3:end],inds[1:2])
    M=M[ninds,ninds]
    #@show M
end