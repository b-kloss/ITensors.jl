
using LinearAlgebra
using ITensors
using ITensorGaussianMPS
using HDF5
using PyPlot
matplotlib.use("QtAgg")
using F_utilities
#using GR
const Fu=F_utilities
ITensors.disable_contraction_sequence_optimization()

include("imp.jl")
include("bath.jl")
let
U=0.0  #0.23
ed=Pair(0.4,0.4)  #0.52,0.32
betas=Vector(LinRange(10,30,3))
#@show betas
#return
ees=Float64[]
for beta in betas
    #beta=100.0
    #Nt=Int(round(beta/0.37))
    Nt=201   
    dt=beta/(Nt-1)
    D=1.0


    taus=Vector(LinRange(0,beta,Nt))
    @show taus
    Delta_t= D*Base.Math.sinc.(D*taus/2.0)
    Delta_ttp=zeros(Float64,Nt,Nt)

    #display(gcf())
    for i in 1:Nt
        for j in 1:Nt
            Delta_ttp[i,j]=Delta_t[abs(i-j)+1]
        end
    end

    c=get_correlation_matrix(Delta_ttp,dt)
    matshow(real.(c))
    show()
    N=size(c,1)
    @show sum(diag(c))
    @show N
    subc=c[1:div(N,2),1:div(N,2)]
    subcb=ITensorGaussianMPS.reverse_interleave(subc)
    push!(ees,sum(Fu.Contour(subcb)))
end
plot(betas,ees)
show()
plot(betas,log.(ees))
show()

plot(log.(betas),log.(ees))
show()
#@show Fu.VN_entropy(subcb)

#return
#matshow(real.(c))
#show()
N = size(c,1)
sites_l = siteinds("Fermion", div(N,2); conserve_qns=false)
sites_r = siteinds("Fermion", div(N,2); conserve_qns=false)
psi_r=ITensorGaussianMPS.correlation_matrix_to_mps(sites_r,copy(c);eigval_cutoff=1e-11,maxblocksize=12,cutoff=1e-11)
return
#sianMPS.correlation_matrix_to_mps(sites_l,copy(c);eigval_cutoff=1e-10,maxblocksize=14,cutoff=1e-10)
psi_l=copy(psi_r)
#return

@show norm(psi_r)
@show inner(psi_l,psi_r)
#@show psi_r
combined_sites_r,psi_r_fused=fuse_indices_pairwise(psi_r)
combined_sites_l,psi_l_fused=fuse_indices_pairwise(psi_l)
@show inner(psi_l_fused,psi_r_fused)
Z_MPO=get_Z_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_Z_MPO_fun)
##psi_l_fused=psi_l_fused .*2
##psi_r_fused=psi_r_fused .*2

Z=logdot(dag((prime(psi_l_fused))),Z_MPO*psi_r_fused)
@show Z
#psi_l_fused=1/sqrt(Z)*psi_l_fused
#psi_r_fused=1/sqrt(Z)*psi_r_fused

centers=get_MPO(U,dt,ed,combined_sites_l,combined_sites_r,get_1PGreens_MPO;spin0="up",spin1="up")
results=ComplexF64[]
for center in centers
    push!(results,exp(logdot(dag(prime(psi_l_fused)),center*psi_r_fused)-Z))
    @show last(results)
end
fout=h5open("results.h5","w")
fout["G"] = results
fout["t"] = taus[2:end]

close(fout)
#@show reverse(results)/results[Nt-1]
plot(taus[2:end],reverse(results)/results[Nt-1])
plot(taus,Delta_t,"r")
show()
return
end
