
using LinearAlgebra
using ITensors
using ITensorGaussianMPS
using HDF5
using PyPlot
matplotlib.use("QtAgg")
using F_utilities
using Interpolations
#using GR
const Fu=F_utilities
ITensors.disable_contraction_sequence_optimization()
#@show ITensors.mkl_get_num_threads()

#@show ITensors.mkl_get_num_threads()

include("imp.jl")
include("bath.jl")
parameter_file="params.jl"
include(parameter_file)
#using Main.params
function get_IM(taus::Vector,Delta::Vector;kwargs...)
    Nt=length(Delta)
    Delta_ttp=zeros(Float64,Nt,Nt)
    for i in 1:Nt
        for j in 1:Nt
            Delta_ttp[i,j]=Delta[abs(i-j)+1]
        end
    end
    dt=taus[2]-taus[1]
    
    @assert all(isapprox.(dt,diff(taus)))
    c=get_correlation_matrix_Julian(Delta_ttp,dt)
    N = size(c,1)
    #sites_l = siteinds("Fermion", div(N,2); conserve_qns=false)
    sites_r = siteinds("Fermion", div(N,2); conserve_qns=false)
    eigval_cutoff=get(kwargs, :eigval_cutoff, 1e-12)
    maxblocksize=get(kwargs, :maxblocksize, 8)
    minblocksize=get(kwargs, :maxblocksize, 1)
    cutoff=get(kwargs,:cutoff,0.0)
    maxdim=get(kwargs,:maxdim,2^maxblocksize)
    psi=ITensorGaussianMPS.correlation_matrix_to_mps(
        sites_r,copy(c);
        eigval_cutoff=eigval_cutoff,maxblocksize=maxblocksize,minblocksize=minblocksize,cutoff=cutoff,maxdim=maxdim
    )    
    return psi, c
end
function get_IM_vacuum(taus::Vector,Delta::Vector;kwargs...)
    c=get_vaccuum_correlation_matrix(6)
    c=h5read("/mnt/home/bkloss/projects/IM_solver/propagator_benchmark/generic_env/correlation_matrix_Jx0.27_Jy0.11.hdf5","corr_t=6")
    #c=f["corr_t=6"][:]
   # cb=ITensorGaussianMPS.reverse_interleave(c)
    N = size(c,1)
    #sites_l = siteinds("Fermion", div(N,2); conserve_qns=false)
    sites_r = siteinds("Fermion", div(N,2); conserve_qns=false)
    eigval_cutoff=get(kwargs, :eigval_cutoff, 1e-12)
    maxblocksize=get(kwargs, :maxblocksize, 8)
    minblocksize=get(kwargs, :maxblocksize, 1)
    cutoff=get(kwargs,:cutoff,0.0)
    maxdim=get(kwargs,:maxdim,2^maxblocksize)
    psi=ITensorGaussianMPS.correlation_matrix_to_mps(
        sites_r,copy(c);
        eigval_cutoff=eigval_cutoff,maxblocksize=maxblocksize,minblocksize=minblocksize,cutoff=cutoff,maxdim=maxdim
    )    
    return psi, c
end

function get_noninteracting_bipartite_entropy(c::AbstractMatrix)
    N=size(c,1)
    subc=c[1:div(N,2),1:div(N,2)]
    subcb=ITensorGaussianMPS.reverse_interleave(subc)
    return sum(Fu.Contour(subcb))
end

function get_interacting_bipartite_entropy(psi0::MPS)
    b=div(length(psi0),2)
    psi=orthogonalize(psi0, b)
    _,S,_ = svd(psi[b], (linkind(psi, b-1), siteind(psi,b)))
    SvN = 0.0
    svals=Float64[]
    for n=1:dim(S, 1)
        push!(svals,S[n,n])
        p = S[n,n]^2
        SvN -= p * log(p)
    end
    return SvN,svals
end
let
#U=0.0  #0.23
U=params.U
ed=Pair(params.ed_u,params.ed_d)  #0.52,0.32
beta=params.beta
beta=6
Nt=params.Nt
Nt=6
maxdim=params.maxdim
minblocksize=params.minblocksize
maxblocksize=params.maxblocksize
eigval_cutoff=params.eigval_cutoff
cutoff=params.cutoff
dt=params.beta/(params.Nt-1)

D=2.0
V=0.2
Deltafn="/mnt/home/bkloss/projects/IM_solver/triqs_benchmark/Delta_beta10.h5"
reftaus=h5read(Deltafn,"taus")
@show last(reftaus)
refvals=h5read(Deltafn,"data")
Delta_inter=linear_interpolation(reftaus[:], refvals[:])
taus=Vector(LinRange(0,beta,Nt))

@show taus
#Delta_t= V^2 * D*Base.Math.sinc.(D*taus/2.0)
Delta_t=Delta_inter.(taus)


plot(taus,real.(Delta_t))
show()
using MKL
@show BLAS.set_num_threads(32)
println("getting IM")
psi_r,c=get_IM_vacuum(taus,Delta_t;eigval_cutoff=eigval_cutoff,minblocksize=minblocksize,maxblocksize=maxblocksize,maxdim=maxdim,cutoff=cutoff)

#fout=h5open("Delta_t_ref.h5","w")
#fout["taus"]=taus
#fout["c"]=c
#fout["Delta_t"] = Delta_t
#close(fout)
#return

SvN_ni=get_noninteracting_bipartite_entropy(c)
SvN_mps, SvN_spectrum=get_interacting_bipartite_entropy(psi_r)

#fout=h5open("results_beta"*string(beta)*"_Nt"*string(Nt)*"_chi"*string(maxdim)*".h5","w")
#fout["svals"] = SvN_spectrum
#plot(log10.(svals))
#show()
@show SvN_ni
@show SvN_mps
@show maxlinkdim(psi_r)
#close(fout)

##for identical bath for both spin species reuse both 
psi_l=copy(psi_r)
combined_sites_r,psi_r_fused=fuse_indices_pairwise(psi_r)
combined_sites_l,psi_l_fused=fuse_indices_pairwise(psi_l)
@show inner(psi_l_fused,psi_r_fused)

dt=1.0
@show U, dt, ed

Z_MPO=get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_Z_MPO_fun)

Z=logdot(dag((prime(psi_l_fused))),Z_MPO*psi_r_fused)
@show Z

centers=get_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_1PGreens_MPO;spin0="up",spin1="up")
results=ComplexF64[]
counter=0
BLAS.set_num_threads(1)
#@show exp(logdot(dag(prime(psi_l_fused)),centers[length(centers)]*psi_r_fused)-Z)
results=zeros(ComplexF64,length(taus))
Threads.@threads for i = 1:length(taus)
    results[i] = exp(logdot(dag(prime(psi_l_fused)),centers[i]*psi_r_fused)-Z)
end
@show results
#fout=h5open("results_beta"*string(beta)*"_Nt"*string(Nt)*"_chi"*string(maxdim)*".h5","r+")
#fout["G"] = results
#fout["t"] = taus[2:end,1:1]
#close(fout)
#plot(real.(results[1:10]),"b")
#show()
return
end
