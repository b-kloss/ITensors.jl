#using F_utilities
using LinearAlgebra
using PyPlot
using QuadGK
using SkewLinearAlgebra

function apply_ph_everyother(c::AbstractMatrix)
    cb=ITensorGaussianMPS.reverse_interleave(c)
    N=size(cb,1)
    n=div(N,2)
    newc=zeros(eltype(c),n,n)
    for i in 1:div(n,2)
        for j in 1:div(n,2)
            newc[2i-1,2j-1]=cb[n+2i-1,n+2j-1]
            #newc[2i,2j]=(iszero(mod(mod(i,2)+mod(j,2),2)) ? 1 : -1) *cb[2i,2j]
            newc[2i,2j]=(iseven(i+j) ? 1 : -1) *cb[2i,2j]
            
            #newc[2i,2j]=(isodd(div(i,2)+div(i,2)) ? -1 : 1) *cb[2i,2j]
            #newc[2i-1,2j]=(iszero(mod(j,2)) ? 1 : -1)*cb[n+2i-1,2j]
            newc[2i-1,2j]=(iseven(j) ? 1 : -1)*cb[n+2i-1,2j]
            
            #newc[2i-1,2j]=(isodd(div(j,2)) ? -1 : 1)*cb[n+2i-1,2j]
            #newc[2i,2j-1]=(iszero(mod(i,2)) ? 1 : -1)*cb[2i,n+2j-1]
            newc[2i,2j-1]=(iseven(i) ? 1 : -1)*cb[2i,n+2j-1]
            
            #newc[2i,2j-1]=(isodd(div(i,2)) ? -1 : 1)*cb[2i,n+2j-1]
        end
    end
    return newc
end

function get_IM_from_corr(c::Matrix,reshuffle::Bool=true,ph::Bool=false;kwargs...)
    eigval_cutoff=get(kwargs, :eigval_cutoff, 1e-12)
    maxblocksize=get(kwargs, :maxblocksize, 8)
    minblocksize=get(kwargs, :minblocksize, 1)
    cutoff=get(kwargs,:cutoff,0.0)
    maxdim=get(kwargs,:maxdim,2^maxblocksize)
    N=size(c,1)
    if reshuffle==true
        #c=c[reverse(Vector(1:N)),reverse(Vector(1:N))]
        shuffledinds=vcat(Vector(3:N),[1,2])
        #shuffledinds=sortperm(vcat(Vector(3:N),[1,2]))
        
        c=c[shuffledinds,:][:,shuffledinds]
    end

    if !ph
        sites_r = siteinds("Fermion", div(N,2);conserve_nfparity=false,conserve_nf=false)
    else
        sites_r = siteinds("Fermion", div(N,2);conserve_nfparity=true,conserve_nf=true)
        c=apply_ph_everyother(c)
    end

    BLAS.set_num_threads(1)
    println("Converting corr to MPS")    
    @time begin
    psi=ITensorGaussianMPS.correlation_matrix_to_mps(
        sites_r,real.(c);
        eigval_cutoff=eigval_cutoff,maxblocksize=maxblocksize,minblocksize=minblocksize,cutoff=cutoff,maxdim=maxdim
    )
    end
    return psi, c
end

function get_IM_from_corr(c::Matrix,sites::Vector{<:Index};kwargs...)
    eigval_cutoff=get(kwargs, :eigval_cutoff, 1e-12)
    maxblocksize=get(kwargs, :maxblocksize, 8)
    minblocksize=get(kwargs, :minblocksize, 1)
    cutoff=get(kwargs,:cutoff,0.0)
    maxdim=get(kwargs,:maxdim,2^maxblocksize)
    N=size(c,1)
    ###assummes that reshuffling/ph transformation have already been applied
    BLAS.set_num_threads(1)
    println("Converting corr to MPS") 
    @time begin
    psi=ITensorGaussianMPS.correlation_matrix_to_mps(
        sites,real.(c);
        eigval_cutoff=eigval_cutoff,maxblocksize=maxblocksize,minblocksize=minblocksize,cutoff=cutoff,maxdim=maxdim
    )
    end
    return psi, c
end


function get_circuits(c::Matrix,reshuffle::Bool)
    if reshuffle==true
        shuffledinds=vcat(Vector(3:N),[1,2])
        c=c[shuffledinds,:][:,shuffledinds]
    end
    Λr, C, indsnext, relinds=ITensorGaussianMPS.correlation_matrix_to_gmps_brickwall_tailed
end

function get_IM(G::Matrix,reshuffle::Bool,ph::Bool;kwargs...)
    BLAS.set_num_threads(32)
    println("Exponentiating G for corr matrix")
    @time begin
        c=exp_bcs_julian(G)
    end
    return get_IM_from_corr(c,reshuffle,ph;kwargs...)
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
    #G_opt=zeros(ComplexF64,(2*Nt,2*Nt))
    #factor=1.0/(upper-lower)
    factor=1.0
    #convention="b"
    #factor=
    @assert (alpha==0.0 || alpha==1.0)
    incr=Int(alpha)
    valsg=Dict()
    valsl=Dict()
    
    for i in 0:Nt
        valsg[i]=quadgk(omega -> factor*g_greater(omega,i*dt),lower,upper)[1]
        valsl[-i]=quadgk(omega -> factor*g_lesser(omega,-i*dt),lower,upper)[1]
        
    end

    for m in 0:Nt-1   ##0 2*Nt-1 for python convention match
        #tau=m*dt
        for n in m+1:Nt-1
            #tau_p=n*dt
            G[2*m+1,2*n+1+1]+= dt^2 * valsg[n-(m+incr)] #quadgk(omega -> factor*g_greater(omega,tau_p,tau+(dt*alpha)),lower,upper)[1]
            G[2*m+1+1,2*n+1]-= dt^2 * valsl[m-(n+incr)] #quadgk(omega -> factor*g_lesser(omega,tau,tau_p+(dt*alpha)),lower,upper)[1]
        end
        G[2*m+1,2*m+1+1] += dt^2 * valsl[-incr]#quadgk(omega -> factor*g_lesser(omega,tau,tau+(dt*alpha)),lower,upper)[1]
        G[2*m+1,2*m+1+1] += - 1.
    end
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

function integrate_out_timesteps(G::AbstractMatrix,T_ren::Int)
    dim_B_temp=size(G,1)
    dim_B=div(dim_B_temp,T_ren)
    @assert iszero(dim_B_temp%T_ren)
    B_spec_dens = copy(G)
    #add intermediate integration measure to integrate out internal legs
    for i in 0:div(dim_B,2)-1
        for j in 0:T_ren-2
            B_spec_dens[2*i*T_ren+2*j+2,2*i*T_ren+3+2*j] += -1  
            B_spec_dens[2*i*T_ren+3+2*j,2*i*T_ren+2+2*j] += 1  
        end
    end
    
    #select submatrix that contains all intermediate times that are integrated out
    B_spec_dens_sub = zeros(eltype(G),dim_B_temp - dim_B, dim_B_temp - dim_B)
    @inbounds for i in 0:div(dim_B,2)-1
        @inbounds for j in 0:div(dim_B,2)-1
            B_spec_dens_sub[i*(2*T_ren-2)+1:i*(2*T_ren-2 )+2*T_ren-2,j*(2*T_ren-2)+1:j*(2*T_ren-2 )+2*T_ren-2] = B_spec_dens[2*i*T_ren+2:2*(i*T_ren + T_ren)-1,2*j*T_ren+2:2*(j*T_ren + T_ren)-1]
        end
    end
    #matrix coupling external legs to integrated (internal) legs
    B_spec_dens_coupl =  zeros(eltype(G),dim_B_temp - dim_B,dim_B)
    @inbounds for i in 0:div(dim_B,2)-1
        @inbounds for j in 0:div(dim_B,2)-1
            B_spec_dens_coupl[i*(2*T_ren-2)+1:i*(2*T_ren-2 )+2*T_ren-2,2*j+1] = B_spec_dens[2*i*T_ren+2:2*(i*T_ren + T_ren)-1,2*j*T_ren+1]
            B_spec_dens_coupl[i*(2*T_ren-2)+1:i*(2*T_ren-2 )+2*T_ren-2,2*j+1+1] = B_spec_dens[2*i*T_ren+2:2*(i*T_ren + T_ren)-1,2*(j+1)*T_ren]
        end
    end
    
    B_spec_dens_ext =zeros(eltype(G),dim_B,dim_B)
    @inbounds for i in 0:div(dim_B,2)-1
        @inbounds for j in 0:div(dim_B,2)-1
            B_spec_dens_ext[2*i+1,2*j+1] = B_spec_dens[2*i*T_ren+1,2*j*T_ren+1]
            B_spec_dens_ext[2*i+1+1,2*j+1] = B_spec_dens[2*(i+1)*T_ren,2*j*T_ren+1]
            B_spec_dens_ext[2*i+1,2*j+1+1] = B_spec_dens[2*i*T_ren+1,2*(j+1)*T_ren]
            B_spec_dens_ext[2*i+1+1,2*j+1+1] = B_spec_dens[2*(i+1)*T_ren,2*(j+1)*T_ren]
        end
    end
    
    B_spec_dens = B_spec_dens_ext .+ (transpose(B_spec_dens_coupl) * inv(B_spec_dens_sub) * B_spec_dens_coupl)
    return B_spec_dens   
end


function evaluate_ni_aim(G::AbstractMatrix,ed::Pair;convention='a')
    t=0 #hopping between spin species
    B=reverse(G)
    dim_B=size(B,1)
    exponent=zeros(ComplexF64,dim_B*4,dim_B*4)
    exponent[dim_B+1:2*dim_B,dim_B+1:2*dim_B]=B
    exponent[2*dim_B+1:3*dim_B,2*dim_B+1:3*dim_B]=B
    mu_up=ed[1]
    mu_down=ed[2]
    #return
    println("a")
    if convention=='a'
        #spin upd
        exponent[2*dim_B+1:3*dim_B-1,2:dim_B] -= diagm(ones(dim_B-1))
        exponent[3*dim_B,1] += -1
        exponent[2:dim_B,(2*dim_B+1):3*dim_B-1] += diagm(ones(dim_B-1))
        exponent[1,3*dim_B] += 1
        #spin down
        exponent[3*dim_B+2:4*dim_B,dim_B+1:2*dim_B-1] -= diagm(ones(dim_B-1))
        exponent[3*dim_B+1,2*dim_B] += -1
        exponent[dim_B+1:2*dim_B-1,3*dim_B+2:4*dim_B] += diagm(ones(dim_B-1))
        exponent[2*dim_B,3*dim_B+1] += +1
    elseif convention=='b'
        error("Not implemented yet")
    end
    T=1-tanh(t/2)^2
    println("b")
    for i in 0:div(dim_B,2)-2
        # forward 
        # (matrix elements between up -> down), last factors of (-1) are sign changes to test overlap form
        exponent[dim_B - 2 - 2*i + 1, 4*dim_B - 1 - 2*i + 1] += -1. * tanh(t/2) *2/T *exp(-1. * mu_up) 
        exponent[dim_B - 1 - 2*i + 1, 4*dim_B - 2 - 2*i + 1] -= -1. * tanh(t/2)*2/T *exp(-1. * mu_down) 
        #(matrix elements between up -> up)
        exponent[dim_B - 2 - 2*i + 1, dim_B - 1 - 2*i + 1] += 1 *cosh(t) *exp(-1 * mu_up) *(-1.) 
        #(matrix elements between down -> down)
        exponent[4*dim_B - 2 - 2*i + 1, 4*dim_B - 1 - 2*i + 1] += 1 *cosh(t) *exp(-1. * mu_down) *(-1.)

        # forward Transpose (antisymm)
        exponent[4*dim_B - 1 - 2*i + 1, dim_B - 2 - 2*i + 1] += 1 * tanh(t/2)*2/T *exp(-1 * mu_up) 
        exponent[4*dim_B - 2 - 2*i + 1, dim_B - 1 - 2*i + 1] -= 1. * tanh(t/2)*2/T *exp(-1. * mu_down)
        exponent[dim_B - 1 - 2*i + 1,dim_B - 2 - 2*i + 1] += -1 *cosh(t) *exp(-1. * mu_up) *(-1.)
        exponent[4*dim_B - 1 - 2*i + 1, 4*dim_B - 2 - 2*i + 1] += -1 *cosh(t) *exp(-1. * mu_down) *(-1.)
    end
    #last application contains antiperiodic bc.:
    println("c")
    exponent[1,3*dim_B+2] += -1. * tanh(t/2) *2/T *exp(-1. * mu_up) *(-1.) 
    exponent[2,3*dim_B+1] -= -1. * tanh(t/2)*2/T *exp(-1. * mu_down) *(-1.)
    #(matrix elements between up -> up)
    exponent[0 + 1, 1 + 1] += 1 *cosh(t) *exp(-1 * mu_up) *(-1.) *(-1.)
    #(matrix elements between down -> down)
    exponent[3*dim_B  + 1, 3*dim_B + 1 + 1] += 1 *cosh(t) *exp(-1. * mu_down) *(-1.) *(-1.)

    # forward Transpose (antisymm)
    exponent[3*dim_B+2,1] += 1 * tanh(t/2)*2/T *exp(-1 * mu_up) *(-1.) 
    exponent[3*dim_B + 1,1 + 1] -= 1. * tanh(t/2)*2/T *exp(-1. * mu_down) *(-1.)
    exponent[1 + 1,0 + 1] += -1 *cosh(t) *exp(-1. * mu_up) *(-1.) *(-1.)
    exponent[3*dim_B + 1 + 1,3*dim_B + 1] += -1 *cosh(t) *exp(-1. * mu_down) *(-1.) *(-1.)

    
    exponent_inv = inv(exponent)#this is the matrix whose elements yield the propagator
    exponent_inv_T = Matrix(transpose(exponent_inv))
    res=zeros(ComplexF64,2,div(dim_B,2))
    for tau in 0:div(dim_B,2)-1
        if convention=='a'
            m=exponent_inv_T[[1,3*dim_B-2*tau],[1,3*dim_B-2*tau]]
            @assert all(abs.(diag(m)) .< 1e-12)
            m[1,1]=0
            m[2,2]=0
            
            m[2,1]=-conj(m[1,2])
            res[1,tau+1]=pfaffian(real.(m))
            m=exponent_inv_T[[2*dim_B-2*tau,3*dim_B+1],[2*dim_B-2*tau,3*dim_B+1]]
            m[2,1]=-conj(m[1,2])
            @assert all(abs.(diag(m)) .< 1e-12)
            m[1,1]=0
            m[2,2]=0
            res[2,tau+1]=pfaffian(real.(m))
        else
            error("Not implemented")
        end
    end
    return res

end


function exp_bcs_julian(G::Matrix)
    n=size(G,1)
    dim_G=size(G,1)
    random_part = rand(dim_G,dim_G) * 1.e-8
    G += random_part - transpose(random_part)
    Gh=transpose(G) * conj(G)
    @time begin
        Gdd,R=eigen(Hermitian(Gh))
    end
    G_schur_complex = R' * G * conj(R)
    eigenvalues_complex=diag(G_schur_complex,1)[1:2:end]
    D_phases=diagm(exp.(0.5im*angle.(ITensorGaussianMPS.interleave(eigenvalues_complex,eigenvalues_complex))))
    R = R * D_phases
    G_schur_real=conj(transpose(R)) * G * conj(R)
    eigenvalues_real=real.(diag(G_schur_real,1)[1:2:end])
    corr_block_diag=zeros(eltype(G),2*n,2*n)
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
    corr_block_back_rotated = double_R * corr_block_diag * conj(transpose(double_R))
    corr_rotated =ITensorGaussianMPS.interleave(corr_block_back_rotated)
    return corr_rotated
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