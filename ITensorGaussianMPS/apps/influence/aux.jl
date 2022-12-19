function get_noninteracting_bipartite_entropy(c::AbstractMatrix)
    N=size(c,1)
    subc=c[1:div(N,2),1:div(N,2)]
    subcb=ITensorGaussianMPS.reverse_interleave(subc)
    return sum(Fu.Contour(subcb))
end

function get_interacting_bipartite_entropy(psi0::MPS;b=0)
    if b==0
        b=div(length(psi0),2)
    end
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
