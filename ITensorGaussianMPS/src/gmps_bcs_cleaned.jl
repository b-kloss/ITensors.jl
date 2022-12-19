#
# Single particle von Neumann entanglement entropy
#
using PyPlot
function entropy(n::Number)
    (n ≤ 0 || n ≥ 1) && return 0
    return -(n * log(n) + (1 - n) * log(1 - n))
  end
  
  entropy(ns::Vector{Float64}) = sum(entropy, ns)
  
  #
  # Linear Algebra tools
  #
  
  """
      frobenius_distance(M1::AbstractMatrix, M2::AbstractMatrix)
  
  Computes the Frobenius distance `√tr((M1-M2)'*(M1-M2))`.
  """
  function frobenius_distance(M1::AbstractMatrix, M2::AbstractMatrix)
    return sqrt(abs(tr(M1'M1) + tr(M2'M2) - tr(M1'M2) - tr(M2'M1)))
  end
  
  #
  # Rotations
  #

  
  struct DiagRotation{T} <: LinearAlgebra.AbstractRotation{T}
    i1::Int
    i2::Int
    d::T    #exp(i\theta)
    #s::T##probably interface should be similar to Givens
  end

  function diagrotation(f::T,g::T,i1::Int,i2::Int) where T<:AbstractFloat
    #determines angle of rotation of g such that it's parallel to f in the complex plane
    ##FIXME: add handling of small abs value for f and g, to avoid over/underflow
    ##FIXME: for type stability it would be desirable to handle real and complex inputs separately, such that d is real (+-1) if f,g are real 
    theta=angle(g)-angle(f)
    d=exp(-1im*theta)
    return DiagRotation{T}(i1,i2,d)
  end

  function diagrotation(f::Complex{T},g::Complex{T},i1::Int,i2::Int) where T<:AbstractFloat
    #determines angle of rotation of g such that it's parallel to f in the complex plane
    theta=angle(g)-angle(f)
    d=exp(-1im*theta)
    #d=fu/gu
    return DiagRotation{Complex{T}}(i1,i2,d)
  end
  
  
  LinearAlgebra.adjoint(G::DiagRotation) = DiagRotation(G.i1, G.i2, conj(G.d))
  #adjoint(G::Givens) = LinearAlgebra.Givens(G.i1, G.i2, G.c', -G.s)###Why do I need to redefine this
  @inline function LinearAlgebra.lmul!(G::DiagRotation, A::AbstractVecOrMat)
    Base.require_one_based_indexing(A)
    m, n = size(A, 1), size(A, 2)
    if G.i2 > m
        throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    end
    @inbounds for i = 1:n
        a1, a2 = A[G.i1,i], A[G.i2,i]
        A[G.i1,i] =       a1
        A[G.i2,i] = G.d*a2   ##figure this one out, conj proper here or not? probably not
    end
    return A
  end

  @inline function LinearAlgebra.rmul!(A::AbstractMatrix, G::DiagRotation)
    Base.require_one_based_indexing(A)
    m, n = size(A, 1), size(A, 2)
    if G.i2 > n
        throw(DimensionMismatch("column indices for rotation are outside the matrix"))
    end
    @inbounds for i = 1:m
        a1, a2 = A[i,G.i1], A[i,G.i2]
        A[i,G.i1] = a1
        A[i,G.i2] = G.d * a2
    end
    return A
  end

  
  struct Circuit{T} <: LinearAlgebra.AbstractRotation{T}
    rotations::Vector{Union{Givens{T},DiagRotation{T}}}
  end
  
  Base.adjoint(R::Circuit) = Adjoint(R)
  
  function Base.show(io::IO, ::MIME"text/plain", C::Circuit{T}) where {T}
    print(io, "Circuit{$T}:\n")
    return show(io, "text/plain", C.rotations)
  end
  
  function Base.copy(aR::Adjoint{<:Any,Circuit{T}}) where {T}
    return Circuit{T}(reverse!([r' for r in aR.parent.rotations]))
  end
  
  function LinearAlgebra.lmul!(G::Givens, R::Circuit)
    push!(R.rotations, G)
    return R
  end

  function LinearAlgebra.lmul!(G::DiagRotation, R::Circuit)
    push!(R.rotations, G)
    return R
  end

  
  function LinearAlgebra.lmul!(R::Union{Circuit}, A::AbstractArray)
    @inbounds for i in 1:length(R.rotations)
      LinearAlgebra.lmul!(R.rotations[i], A)
    end
    return A
  end
  

  function LinearAlgebra.rmul!(A::AbstractMatrix, adjR::Adjoint{<:Any,<:Union{Circuit}})
    R = adjR.parent
    @inbounds for i in 1:length(R.rotations)
      #@show R.rotations[i]
      LinearAlgebra.rmul!(A, LinearAlgebra.adjoint(R.rotations[i]))
    end
    return A
  end
  
  Base.:*(g1::Circuit, g2::Circuit) = Circuit(vcat(g2.rotations, g1.rotations))
  LinearAlgebra.lmul!(g1::Circuit, g2::Circuit) = append!(g2.rotations, g1.rotations)
  
  #LinearAlgebra.lmul!(g1::Union{BCSCircuit,Circuit}, g2::Union{Circuit,BCSCircuit}) = append!(g2.rotations, g1.rotations)   #not type stable if mixed circuit types
  

  Base.:*(A::Circuit, B::Union{<:Hermitian,<:Diagonal}) = A * convert(Matrix, B)
  Base.:*(A::Adjoint{<:Any,<:Circuit}, B::Hermitian) = copy(A) * convert(Matrix, B)
  Base.:*(A::Adjoint{<:Any,<:Circuit}, B::Diagonal) = copy(A) * convert(Matrix, B)
  function Base.:*(A::Adjoint{<:Any,<:AbstractVector}, B::Adjoint{<:Any,<:Circuit})
    return convert(Matrix, A) * B
  end
  
  function LinearAlgebra.rmul!(A::AbstractMatrix, R::Circuit)
    @inbounds for i in reverse(1:length(R.rotations))
      rmul!(A, R.rotations[i])
    end
    return A
  end
  
  function shift!(G::Circuit, i::Int)
    for (n, g) in enumerate(G.rotations)
      if typeof(g)<:Givens
        G.rotations[n] = Givens(g.i1 + i, g.i2 + i, g.c, g.s)
      elseif typeof(g)<:DiagRotation
        G.rotations[n] = DiagRotation(g.i1 + i, g.i2 + i, g.d)
      end
    end
    return G
  end
  
  ngates(G::Circuit) = length(G.rotations)
  
  #
  # Free fermion tools
  #
  
  is_creation_operator(o::Op) = is_creation_operator(ITensors.name(o))
  is_creation_operator(o::String) = is_creation_operator(OpName(o))
  is_creation_operator(::OpName) = false
  is_creation_operator(::OpName"Cdag") = true
  is_creation_operator(::OpName"Cdagup") = true
  is_creation_operator(::OpName"Cdagdn") = true
  is_creation_operator(::OpName"c†") = true
  is_creation_operator(::OpName"c†↑") = true
  is_creation_operator(::OpName"c†↓") = true
  
  is_annihilation_operator(o::Op) = is_annihilation_operator(ITensors.name(o))
  is_annihilation_operator(o::String) = is_annihilation_operator(OpName(o))
  is_annihilation_operator(::OpName) = false
  is_annihilation_operator(::OpName"C") = true
  is_annihilation_operator(::OpName"Cup") = true
  is_annihilation_operator(::OpName"Cdn") = true
  is_annihilation_operator(::OpName"c") = true
  is_annihilation_operator(::OpName"c↑") = true
  is_annihilation_operator(::OpName"c↓") = true
  
  # Make a hopping Hamiltonian from quadratic Hamiltonian
  function hopping_hamiltonian(os::OpSum)
    nterms = length(os)
    coefs = Vector{Number}(undef, nterms)
    sites = Vector{Tuple{Int,Int}}(undef, nterms)
    nsites = 0
    for n in 1:nterms
      term = os[n]
      coef = isreal(coefficient(term)) ? real(coefficient(term)) : term.coef
      coefs[n] = coef
      length(term) ≠ 2 && error("Must create hopping Hamiltonian from quadratic Hamiltonian")
      @assert is_creation_operator(term[1])
      @assert is_annihilation_operator(term[2])
      sites[n] = ntuple(n -> ITensors.site(term[n]), Val(2))
      nsites = max(nsites, maximum(sites[n]))
    end
    ElT = all(isreal(coefs)) ? Float64 : ComplexF64
    h = zeros(ElT, nsites, nsites)
    for n in 1:nterms
      h[sites[n]...] = coefs[n]
    end
    return h
  end
  
  # Make a pairing Hamiltonian from quadratic pairing only Hamiltonian
  function pairing_hamiltonian(os::OpSum)
    nterms = length(os)
    coefs_a = Vector{Number}(undef, 0)
    coefs_c = Vector{Number}(undef, 0)
    
    sites_a = Vector{Tuple{Int,Int}}(undef, 0)
    sites_c = Vector{Tuple{Int,Int}}(undef, 0)
    
    nsites_a = 0
    nsites_c = 0
    
    nterms_a = 0
    nterms_c = 0
    for n in 1:nterms
      term = os[n]
      coef = isreal(coefficient(term)) ? real(coefficient(term)) : term.coef
      
      length(term) ≠ 2 && error("Must create hopping Hamiltonian from quadratic Hamiltonian")
      cc=is_creation_operator(term[1]) && is_creation_operator(term[2])
      aa=is_annihilation_operator(term[1]) && is_annihilation_operator(term[2])
      #@show(cc,aa,cc||aa)
      @assert cc || aa 
      #@assert is_annihilation_operator(term[2])
      thesites=ntuple(n -> ITensors.site(term[n]), Val(2))
      #@show thesites
      if aa
        nterms_a+=1
        push!(coefs_a,coef)
        push!(sites_a, thesites)
        #@show last(sites_a)
        nsites_a = max(nsites_a, maximum(last(sites_a)))
      elseif cc
        nterms_c+=1
        push!(coefs_c, coef)
        push!(sites_c,ntuple(n -> ITensors.site(term[n]), Val(2)))
        nsites_c = max(nsites_c, maximum(last(sites_c)))
      end
    end
    
    ElT = all(isreal(coefs_a)) ? Float64 : ComplexF64
    h_a = zeros(ElT, nsites_a, nsites_a)
    for n in 1:nterms_a
      h_a[sites_a[n]...] = coefs_a[n]
    end
    h_c = zeros(ElT, nsites_c, nsites_c)
    for n in 1:nterms_c
      h_c[sites_c[n]...] = coefs_c[n]
    end
    @assert isapprox(h_a,-conj(h_c))
    #@assert isapprox(h_a,-transpose(h_a))
    #@show h_a
    #@show -transpose(h_a)
    return h_a
  end

  """
  pairing_hamiltonian(os_hop::OpSum,os_pair::OpSum)
  
  Assemble single-particle Hamiltonian from hopping and pairing terms.
  Returns Hamiltonian both in interlaced and blocked single particle format
  (first annhiliation operator, then creation operator)
  """

  function pairing_hamiltonian(os_hop::OpSum,os_pair::OpSum)
    hh=hopping_hamiltonian(os_hop)
    hp=pairing_hamiltonian(os_pair)
    @assert size(hh,1)==size(hh,2)
    @assert eltype(hh)==eltype(hp)
    Elt=eltype(hh)
    H=zeros(Elt, 2*size(hh,1),2*size(hh,2))
    HB=zeros(Elt, 2*size(hh,1),2*size(hh,2))
    D=size(hh,1)
    HB[1:D,1:D]=-conj(hh)
    HB[1:D,D+1:2*D]=hp
    HB[D+1:2*D,1:D]=-conj(hp)
    HB[D+1:2*D,D+1:2*D]=hh
    H.=interleave(HB)
    return H,HB
  end

  # Make a combined hopping Hamiltonian for spin up and down
  function hopping_hamiltonian(os_up::OpSum, os_dn::OpSum)
    h_up = hopping_hamiltonian(os_up)
    h_dn = hopping_hamiltonian(os_dn)
    @assert size(h_up) == size(h_dn)
    N = size(h_up, 1)
    ElT = promote_type(eltype(h_up), eltype(h_dn))
    h = zeros(ElT, 2 * N, 2 * N)
    for i in 1:(2 * N), j in 1:(2 * N)
      if isodd(i) && isodd(j)
        i_up, j_up = (i + 1) ÷ 2, (j + 1) ÷ 2
        h[i, j] = h_up[i_up, j_up]
      elseif iseven(i) && iseven(j)
        i_dn, j_dn = i ÷ 2, j ÷ 2
        h[i, j] = h_dn[i_dn, j_dn]
      end
    end
    return Hermitian(h)
  end
  
  # Make a Slater determinant matrix from a hopping Hamiltonian
  # h with Nf fermions.
  function slater_determinant_matrix(h::AbstractMatrix, Nf::Int)
    _, u = eigen(h)
    return u[:, 1:Nf]
  end
  
  #
  # Correlation matrix diagonalization
  #
  
  """
  givens_rotation_real(v::AbstractVector)
  
  For a vector `v`, return the `2*length(v)-1`
  real Givens rotations `g` and the norm `r` such that:
  ```julia
  g * v ≈ r * [n == 1 ? 1 : 0 for n in 1:length(v)]
  ```
  with `g` being composed of diagonal rotation aligning pairs
  of complex numbers in the complex plane, and Givens Rotations
  with real arguments only.
  """
  function givens_rotations_real(v0::AbstractVector{ElT}) where {ElT}
    v=copy(v0)
    N = length(v)
    gs = Circuit{ElT}([])
    r = v[1]
    for n in reverse(1:(N - 1))
      d=diagrotation(v[n],v[n+1],n,n+1)
      v=LinearAlgebra.lmul!(d,v)
      inearAlgebra.lmul!(d, gs)
      g, r = givens(convert.(ElT,abs.(v)), n, n + 1)
      v = g * v
      LinearAlgebra.lmul!(g, gs)
    end
    return gs,r
  end
  """
  givens_rotations(v::AbstractVector)

  For a vector `v`, return the `length(v)-1`
  Givens rotations `g` and the norm `r` such that:
  ```julia
  g * v ≈ r * [n == 1 ? 1 : 0 for n in 1:length(v)]
  ```
  """
  function givens_rotations(v::AbstractVector{ElT}) where {ElT}
    N = length(v)
    gs = Circuit{ElT}([])
    r = v[1]
    for n in reverse(1:(N - 1))
      g, r = givens(v, n, n + 1)
      v = g * v
      LinearAlgebra.lmul!(g, gs)
    end
    return gs, r
  end
  
  """
  generalized rotation(v::AbstractVector)
  
  For a vector `v` from a fermionic Gaussian state, return the `4*length(v)-1`
  real Givens/Boguliobov rotations `g` and the norm `r` such that:
  ```julia
  g * v ≈ r * [n == 2 ? 1 : 0 for n in 1:length(v)]
  ```
  with `g` being composed of diagonal rotation aligning pairs
  of complex numbers in the complex plane, and Givens/Boguliobov Rotations
  with real arguments only, acting on the interlaced single-particle space of
  annihilation and creation operator coefficients.
  """
  function generalized_rotations(v0::AbstractVector{ElT};do_boguliobov=true) where {ElT}
    N = div(length(v0),2)
    gs = Circuit{ElT}([])
    v=copy(v0)
    r = v[2]
    do_boguliobov=true
    for n in reverse(1:(N-1))
      d1 = diagrotation(v[2*n],v[2*(n+1)],2*n,2*(n+1))
      d2 = DiagRotation(2*n - 1, 2*n+1,conj(d1.d))
      
      v=LinearAlgebra.lmul!(d1,v)
      v=LinearAlgebra.lmul!(d2,v)
      g1, r = givens(convert.(ElT,abs.(v)), 2*n, 2*(n + 1))
      c,s=g1.c,g1.s
      @assert isapprox(imag(s),0)
      @assert isapprox(imag(c),0)
      g2 = Givens(2*n - 1, 2*n+1, c, s)
      v=g1*v
      v=g2*v
      LinearAlgebra.lmul!(d1,gs)
      LinearAlgebra.lmul!(d2,gs)
      
      LinearAlgebra.lmul!(g1,gs)
      LinearAlgebra.lmul!(g2,gs)
    end
    for n in reverse((1+Int(do_boguliobov)):(N-1))
      d2 = diagrotation(v[2*n-1],v[2*n+1],2*n-1,2*n+1)
      d1 = DiagRotation(2*n, 2*(n+1),conj(d2.d))  ##to respect convention with respect to the definition of the Angle θ, naming swapped
      v=LinearAlgebra.lmul!(d1,v)
      v=LinearAlgebra.lmul!(d2,v)
      g1, r = givens(convert.(ElT,abs.(v)), 2*n-1, 2*n+1) 
      c,s=g1.c,g1.s
      @assert isapprox(imag(s),0)
      @assert isapprox(imag(c),0)
      g2 = Givens(2*n, 2*(n+1), c, s)
      v=g2*v
      v=g1*v
      LinearAlgebra.lmul!(d1,gs)
      LinearAlgebra.lmul!(d2,gs)
      LinearAlgebra.lmul!(g2,gs)
      LinearAlgebra.lmul!(g1,gs)
    end
    ###bogoliubov
    
    if do_boguliobov
      d2=diagrotation(v[2],v[3],2,3)
      d1=DiagRotation(1,4,conj(d2.d))
      v=LinearAlgebra.lmul!(d1,v) #should have no effect
      v=LinearAlgebra.lmul!(d2,v)
      g1, r = givens(convert.(ElT,abs.(v)), 2, 3)
      g2=Givens(1,4,g1.c,g1.s)
      v=g1*v
      v=g2*v #should have no effect
      LinearAlgebra.lmul!(d1,gs)
      LinearAlgebra.lmul!(d2,gs)
      LinearAlgebra.lmul!(g2,gs)
      LinearAlgebra.lmul!(g1,gs)
    
    end
    return gs, v
    
  end
  
      
  
  
  
  """
      correlation_matrix_to_gmps(Λ::AbstractMatrix{ElT}; eigval_cutoff::Float64 = 1e-8, maxblocksize::Int = size(Λ0, 1))
  
  Diagonalize a correlation matrix, returning the eigenvalues and eigenvectors
  stored in a structure as a set of Givens rotations.
  
  The correlation matrix should be Hermitian, and will be treated as if it itensor
  in the algorithm.

  If `is_bcs`, the correlation matrix is assumed to be in interlaced annihilation operator followed by
  creation operator format."""
  function correlation_matrix_to_gmps(
    Λ0::AbstractMatrix{ElT}; eigval_cutoff::Float64=1e-8, minblocksize::Int=1,maxblocksize::Int=size(Λ0, 1),is_bcs::Bool=false,
  ) where {ElT<:Number}
    Λ = Hermitian(Λ0)
    N = size(Λ, 1)
    V = Circuit{ElT}([])
    err_tot = 0.0
    factor=is_bcs ? 2 : 1
    ns = Vector{real(ElT)}(undef,div(N,factor))
    for i in 1:div(N,factor)
      blocksize = 0
      n = 0.0
      err = 0.0
      p = Int[]
      uB = 0.0
      ΛB = 0.0
      for blocksize in minblocksize:maxblocksize
        
        j = min(factor*i + factor*blocksize, N)
        offset=factor-1
        ΛB = @view Λ[(factor*i-offset):j, (factor*i-offset):j]
        nB, uB = eigen(Hermitian(ΛB))
        if is_bcs
          p=sortperm(nB)
          n1 = nB[first(p)]  #
          n2 = nB[last(p)]
          err=n2
          n=n1
          err ≤ eigval_cutoff && break
        else
          p = sortperm(nB; by=entropy)    ###for BCS we might actually want to use a different ordering here, the manuscript takes lowest eigval
          n = nB[p[1]]
          err = min(n, 1 - n)
          err ≤ eigval_cutoff && break
        end
      end
      ns[i] = n
      err_tot += err
      v = @view uB[:, p[1]]
      if length(v)==2 && is_bcs
        @assert i ==div(N,2)
        if abs(v[1])<abs(v[2])
          @assert abs(v[1])<eigval_cutoff
          @assert abs(v[2])>1-eigval_cutoff 
        else
          @assert abs(v[2])<eigval_cutoff
          @assert abs(v[1])>1-eigval_cutoff
          ns[i]=1-ns[i]
        end
        break
      end
      if is_bcs
        g, _ = generalized_rotations(copy(v))
        ###sanity checks
        vpp=copy(v)
        vpp[1:2:end].=conj(v[2:2:end])
        vpp[2:2:end].=conj(v[1:2:end])
        #@show v
        @assert isapprox(abs((g*v)[2]),1)
        @assert isapprox(abs((g*vpp)[1]),1)
      else
        g, r = givens_rotations(v)
      end
      shift!(g, factor*(i - 1))
      
      # In-place version of:
      # V = g * V
      LinearAlgebra.lmul!(g, V)
      Λ = Hermitian(LinearAlgebra.lmul!(g,LinearAlgebra.rmul!(Matrix(Λ),g')))      
    end
    #@show ns
    return ns, V
  end
  
  function slater_determinant_to_gmps(Φ::AbstractMatrix; kwargs...)
    return correlation_matrix_to_gmps(conj(Φ) * transpose(Φ); kwargs...)
  end
  
  #
  # Turn circuit into MPS
  #
  
  function ITensors.ITensor(u::Givens, s1::Index, s2::Index)
    U = [
      1 0 0 0
      0 u.c u.s 0
      0 -conj(u.s) u.c 0
      0 0 0 1
    ]
    return itensor(U, s2', s1', dag(s2), dag(s1))
  end

  function ITensors.ITensor(u::Givens, s1::Index, s2::Index,is_boguliobov::Bool)
    if is_boguliobov
      U = [
        u.c 0 0 u.s
        0 1 0 0
        0 0 1 0
        -conj(u.s) 0 0 u.c
       ]
       return itensor(U, s2', s1', dag(s2), dag(s1))
    else
      U=ITensor(u,s1,s2)
    end 
  end
  

  function ITensors.ITensor(u::DiagRotation, s1::Index, s2::Index)
      U = [
      sqrt(u.d) 0 0 0
      0 sqrt(conj(u.d)) 0 0
      0 0 sqrt(u.d) 0
      0 0 0 sqrt(conj(u.d))
      ]    
    return itensor(U, s2', s1', dag(s2), dag(s1))
  end
  
  ITensors.ITensor(u::DiagRotation, s1::Index, s2::Index,is_boguliobov::Bool)=ITensors.ITensor(u::DiagRotation, s1::Index, s2::Index)

  function ITensors.ITensor(sites::Vector{<:Index}, u::Union{Givens,DiagRotation})
    s1 = sites[u.i1]
    s2 = sites[u.i2]
    return ITensor(u, s1, s2)
  end
  
  function ITensors.ITensor(sites::Vector{<:Index}, u::Union{Givens,DiagRotation},is_boguliubov::Bool)
    s1 = sites[u.i1]
    s2 = sites[u.i2]
    return ITensor(u, s1, s2,is_boguliobov)
  end
  
  """
      MPS(sites::Vector{<:Index}, state, U::Vector{<:ITensor}; kwargs...)
  
  Return an MPS with site indices `sites` by applying the circuit `U` to the starting state `state`.
  """
  function ITensors.MPS(sites::Vector{<:Index}, state, U::Vector{<:ITensor}; kwargs...)
    return apply(U, productMPS(sites, state); kwargs...)
  end

  function isspinful(s::Index)
    !hasqns(s) && return false
    return all(qnblock -> ITensors.hasname(qn(qnblock), ITensors.QNVal("Sz", 0)), space(s))
  end
  
  function isspinful(s::Vector{<:Index})
    return all(isspinful, s)
  end
  
  """
      correlation_matrix_to_mps(s::Vector{<:Index}, Λ::AbstractMatrix{ElT};
                                eigval_cutoff::Float64 = 1e-8,
                                maxblocksize::Int = size(Λ, 1),
                                kwargs...)
  
  Return an approximation to the state represented by the correlation matrix as
  a matrix product state (MPS).
  
  The correlation matrix should correspond to a pure state (have all eigenvalues
  of zero or one).
  """
  function correlation_matrix_to_mps(
    s::Vector{<:Index},
    Λ::AbstractMatrix;
    eigval_cutoff::Float64=1e-8,
    maxblocksize::Int=size(Λ, 1),
    minblocksize::Int=1,
    kwargs...,
  )
    Λ0=copy(Λ)
    @assert size(Λ, 1) == size(Λ, 2)
    
    ###detect whether it's a bcs state or not based on dim(Λ)/length(s) == 1 or 2
    if length(s)==size(Λ, 1)
      is_bcs=false
    elseif 2*length(s)==size(Λ, 1)
      is_bcs=true
    end
    
    ns, C = correlation_matrix_to_gmps(
      Λ; eigval_cutoff=eigval_cutoff, minblocksize=minblocksize, maxblocksize=maxblocksize,is_bcs=is_bcs
    )
    #if all(hastags("Fermion"), s)
    if true ###FIXME check for Fermion/AntiFermion combination and reinstate if statement
      if is_bcs
        is_bog = g -> abs(g.i2-g.i1)==2 ? false : true
        s1 = g -> div(g.i1-1,2)+1
        s2 = g -> div(g.i2-1,2)+1
        U = [ITensor(g,s[s1(g)],s[s2(g)],is_bog(g)) for g in reverse(C.rotations[begin:2:end])]
      else
        U = [ITensor(s,g) for g in reverse(C.rotations)]
      end
    
    if is_bcs
        #safety check that eigenvalues (0,1) are handled properly
        #FIXME: ideally this would be handled properly in the diagonalization routine
        condensed_U=ITensor[]
        for i in 1:div(length(U),2)
          push!(condensed_U,swapprime(prime(U[2*i]) * U[2*i-1],2,1))
        end
        U=condensed_U
        nscheck = real(diag(rmul!(lmul!(C,copy(Λ0)), C')))
        #@show nscheck
        #ns=-ns.+1
        
        @assert all(abs.(ns-nscheck[(1+Int(is_bcs)):(1+Int(is_bcs)):end]).<1e-10)
    end
    println("Making the MPS")
    #ψ = MPS(s, n -> round(Int, ns[n]) + 1, U; kwargs...)
    ψ = MPS(ComplexF64,s, n -> round(Int, ns[n]) + 1)
    #ψ = ψ .* 2
    #normalize!(ψ,(lognorm!)=ones(Float64,length(s)).*log(2^10))
    @show(lognorm(ψ))
    ψ=apply(U,ψ; kwargs...)
    #ψ = ψ .* 0.5
    @show(norm(ψ))
    elseif all(hastags("Electron"), s)
      @assert is_bcs==false
      isodd(length(s)) && error(
        "For Electron type, must have even number of sites of alternating up and down spins.",
      )
      N = length(s)
      if isspinful(s)
        error(
          "correlation_matrix_to_mps(Λ::AbstractMatrix) currently only supports spinless Fermions or Electrons that do not conserve Sz. Use correlation_matrix_to_mps(Λ_up::AbstractMatrix, Λ_dn::AbstractMatrix) to use spinful Fermions/Electrons.",
        )
      else
        sf = siteinds("Fermion", 2 * N; conserve_qns=true)
      end
      U = [ITensor(sf, g) for g in reverse(C.rotations)]
      ψf = MPS(sf, n -> round(Int, ns[n]) + 1, U; kwargs...)
      ψ = MPS(N)
      for n in 1:N
        i, j = 2 * n - 1, 2 * n
        C = combiner(sf[i], sf[j])
        c = combinedind(C)
        ψ[n] = ψf[i] * ψf[j] * C
        ψ[n] *= δ(dag(c), s[n])
      end
    else
      error("All sites must be Fermion or Electron type.")
    end
    @show linkdims(ψ)
    
    b=div(length(ψ),2)
    psi=orthogonalize(ψ, b)
    U,S,V = svd(psi[b], (linkind(psi, b-1), siteind(psi,b)))
    
    SvN = 0.0
    svals=Float64[]
    for n=1:dim(S, 1)
      push!(svals,S[n,n])
      p = S[n,n]^2
      SvN -= p * log(p)
    end
    #plot(log10.(svals))
    #show()
    @show SvN
    @show maxlinkdim(psi)
    return ψ
  end
  
  function slater_determinant_to_mps(s::Vector{<:Index}, Φ::AbstractMatrix; kwargs...)
    return correlation_matrix_to_mps(s, conj(Φ) * transpose(Φ); kwargs...)
  end
  
  function slater_determinant_to_mps(
    s::Vector{<:Index}, Φ_up::AbstractMatrix, Φ_dn::AbstractMatrix; kwargs...
  )
    return correlation_matrix_to_mps(
      s, conj(Φ_up) * transpose(Φ_up), conj(Φ_dn) * transpose(Φ_dn); kwargs...
    )
  end
  
  function mapindex(f::Function, C::Circuit)
    return Circuit(mapindex.(f, C.rotations))
  end
  
  function mapindex(f::Function, g::Givens)
    return Givens(f(g.i1), f(g.i2), g.c, g.s)
  end

  function mapindex(f::Function, g::DiagRotation)
    return DiagRotation(f(g.i1),f(g.i2),g.d)
  end
  
  function identity_blocks!(T::Tensor)
    for b in nzblocks(T)
      T[b] = Matrix{Float64}(I, dims(T[b]))
    end
    return T
  end
  
  # Creates an ITensor with the specified flux where each nonzero block
  # is identity
  # TODO: make a special constructor for this.
  function identity_blocks_itensor(flux::QN, i1::Index, i2::Index)
    A = ITensor(flux, i1, i2)
    identity_blocks!(tensor(A))
    return A
  end
  
  function identity_blocks_itensor(i1::ITensors.QNIndex, i2::ITensors.QNIndex)
    return identity_blocks_itensor(QN(), i1, i2)
  end
  
  function identity_blocks_itensor(i1::Index, i2::Index)
    M = Matrix{Float64}(I, dim(i1), dim(i2))
    return itensor(M, i1, i2)
  end
  
  convert_union_nothing(v::Vector{T}) where {T} = convert(Vector{Union{T,Nothing}}, v)
  
  function interleave(xs...)
    nexts = convert_union_nothing(collect(Base.iterate.(xs)))
    res = Union{eltype.(xs)...}[]
    while any(!isnothing, nexts)
      for ii in eachindex(nexts)
        if !isnothing(nexts[ii])
          (item, state) = nexts[ii]
          push!(res, item)
          nexts[ii] = iterate(xs[ii], state)
        end
      end
    end
    return res
  end
  
  function interleave(M::AbstractMatrix)
    @assert size(M,1)==size(M,2)
    n=div(size(M,1),2)
    first_half=Vector(1:n)
    second_half=Vector(n+1:2*n)
    interleaved_inds=interleave(first_half,second_half)
    return M[interleaved_inds,interleaved_inds]
  end

  function reverse_interleave(M::AbstractMatrix)
    @assert size(M,1)==size(M,2)
    n=div(size(M,1),2)
    first_half=Vector(1:n)
    second_half=Vector(n+1:2*n)
    interleaved_inds=interleave(first_half,second_half)
    ordered_inds=sortperm(interleaved_inds)
    return M[ordered_inds,ordered_inds]
  end

  function correlation_matrix_to_mps(
    s::Vector{<:Index},
    Λ_up::AbstractMatrix,
    Λ_dn::AbstractMatrix;
    eigval_cutoff::Float64=1e-8,
    maxblocksize::Int=min(size(Λ_up, 1), size(Λ_dn, 1)),
    kwargs...,
  )
    @assert size(Λ_up, 1) == size(Λ_up, 2)
    @assert size(Λ_dn, 1) == size(Λ_dn, 2)
    N_up = size(Λ_up, 1)
    N_dn = size(Λ_dn, 1)
    @assert N_up == N_dn
    # Total number of fermion sites
    N = N_up + N_dn
    ns_up, C_up = correlation_matrix_to_gmps(
      Λ_up; eigval_cutoff=eigval_cutoff, maxblocksize=maxblocksize
    )
    ns_dn, C_dn = correlation_matrix_to_gmps(
      Λ_dn; eigval_cutoff=eigval_cutoff, maxblocksize=maxblocksize
    )
    # map the up spins to the odd sites and the even spins to the even sites
    C_up = mapindex(n -> 2n - 1, C_up)
    C_dn = mapindex(n -> 2n, C_dn)
    C = Circuit(interleave(C_up.rotations, C_dn.rotations))
    ns = interleave(ns_up, ns_dn)
    if all(hastags("Fermion"), s)
      @assert length(s) == N
      U = [ITensor(s, g) for g in reverse(C.rotations)]
      ψ = MPS(s, n -> round(Int, ns[n]) + 1, U; kwargs...)
    elseif all(hastags("Electron"), s)
      @assert length(s) == N_up
      @assert length(s) == N_dn
      if isspinful(s)
        space_up = [QN(("Nf", 0, -1), ("Sz", 0)) => 1, QN(("Nf", 1, -1), ("Sz", 1)) => 1]
        space_dn = [QN(("Nf", 0, -1), ("Sz", 0)) => 1, QN(("Nf", 1, -1), ("Sz", -1)) => 1]
        sf_up = [Index(space_up, "Fermion,Site,n=$(2n-1)") for n in 1:N_up]
        sf_dn = [Index(space_dn, "Fermion,Site,n=$(2n)") for n in 1:N_dn]
        sf = collect(Iterators.flatten(zip(sf_up, sf_dn)))
      else
        sf = siteinds("Fermion", N; conserve_qns=true, conserve_sz=false)
      end
      U = [ITensor(sf, g) for g in reverse(C.rotations)]
      ψf = MPS(sf, n -> round(Int, ns[n]) + 1, U; kwargs...)
      ψ = MPS(N_up)
      for n in 1:N_up
        i, j = 2 * n - 1, 2 * n
        C = combiner(sf[i], sf[j])
        c = combinedind(C)
        ψ[n] = ψf[i] * ψf[j] * C
        ψ[n] *= identity_blocks_itensor(dag(c), s[n])
      end
    else
      error("All sites must be Fermion or Electron type.")
    end
    return ψ
  end
  