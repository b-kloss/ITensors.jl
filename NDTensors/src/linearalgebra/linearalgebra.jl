using .RankFactorization: Spectrum

#
# Linear Algebra of order 2 NDTensors
#
# Even though DenseTensor{_,2} is strided
# and passable to BLAS/LAPACK, it cannot
# be made <: StridedArray
import .Unwrap: qr_positive, ql, ql_positive

function (
  T1::Tensor{ElT1,2,StoreT1} * T2::Tensor{ElT2,2,StoreT2}
) where {ElT1,StoreT1<:Dense,ElT2,StoreT2<:Dense}
  RM = matrix(T1) * matrix(T2)
  indsR = (ind(T1, 1), ind(T2, 2))
  return tensor(Dense(vec(RM)), indsR)
end

function LinearAlgebra.exp(T::DenseTensor{ElT,2}) where {ElT<:Union{Real,Complex}}
  expTM = exp(matrix(T))
  return tensor(Dense(vec(expTM)), inds(T))
end

function LinearAlgebra.exp(
  T::Hermitian{ElT,<:DenseTensor{ElT,2}}
) where {ElT<:Union{Real,Complex}}
  # exp(::Hermitian/Symmetric) returns Hermitian/Symmetric,
  # so extract the parent matrix
  expTM = parent(exp(matrix(T)))
  return tensor(Dense(vec(expTM)), inds(T))
end

function svd_catch_error(A; kwargs...)
  USV = try
    svd(expose(A); kwargs...)
  catch
    return nothing
  end
  return USV
end

function lapack_svd_error_message(alg)
  return "The SVD algorithm `\"$alg\"` has thrown an error,\n" *
         "likely because of a convergance failure. You can try\n" *
         "other SVD algorithms that may converge better using the\n" *
         "`alg` (or `svd_alg` if called through `factorize` or MPS/MPO functionality) keyword argument:\n\n" *
         " - \"divide_and_conquer\" is a divide-and-conquer algorithm\n" *
         "   (LAPACK's `gesdd`). It is fast, but may lead to some innacurate\n" *
         "   singular values for very ill-conditioned matrices.\n" *
         "   It also may sometimes fail to converge, leading to errors\n" *
         "   (in which case `\"qr_iteration\"` or `\"recursive\"` can be tried).\n\n" *
         " - `\"qr_iteration\"` (LAPACK's `gesvd`) is typically slower \n" *
         "   than \"divide_and_conquer\", especially for large matrices,\n" *
         "   but is more accurate for very ill-conditioned matrices \n" *
         "   compared to `\"divide_and_conquer\"`.\n\n" *
         " - `\"recursive\"` is ITensor's custom SVD algorithm. It is very\n" *
         "   reliable, but may be slow if high precision is needed.\n" *
         "   To get an `svd` of a matrix `A`, an eigendecomposition of\n" *
         "   ``A^{\\dagger} A`` is used to compute `U` and then a `qr` of\n" *
         "   ``A^{\\dagger} U`` is used to compute `V`. This is performed\n" *
         "   recursively to compute small singular values.\n" *
         " - `\"qr_algorithm\"` is a CUDA.jl implemented SVD algorithm using QR.\n" *
         " - `\"jacobi_algorithm\"` is a CUDA.jl implemented SVD algorithm.\n\n" *
         "Returning `nothing`. For an output `F = svd(A, ...)` you can check if\n" *
         "`isnothing(F)` in your code and try a different algorithm.\n\n" *
         "To suppress this message in the future, you can wrap the `svd` call in the\n" *
         "`@suppress` macro from the `Suppressor` package.\n"
end

"""
    svd(T::DenseTensor{<:Number,2}; kwargs...)

svd of an order-2 DenseTensor
"""
function svd(
  T::DenseTensor{ElT,2,IndsT};
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  use_absolute_cutoff=nothing,
  use_relative_cutoff=nothing,
  alg=nothing,
  # Only used by BlockSparse svd
  min_blockdim=nothing,
) where {ElT,IndsT}
  alg = replace_nothing(alg, default_svd_alg(T))
  if alg == "divide_and_conquer"
    MUSV = svd_catch_error(matrix(T); alg=LinearAlgebra.DivideAndConquer())
    if isnothing(MUSV)
      # If "divide_and_conquer" fails, try "qr_iteration"
      alg = "qr_iteration"
      MUSV = svd_catch_error(matrix(T); alg=LinearAlgebra.QRIteration())
      if isnothing(MUSV)
        # If "qr_iteration" fails, try "recursive"
        alg = "recursive"
        MUSV = svd_recursive(matrix(T))
      end
    end
  elseif alg == "qr_iteration"
    MUSV = svd_catch_error(matrix(T); alg=LinearAlgebra.QRIteration())
    if isnothing(MUSV)
      # If "qr_iteration" fails, try "recursive"
      alg = "recursive"
      MUSV = svd_recursive(matrix(T))
    end
  elseif alg == "recursive"
    MUSV = svd_recursive(matrix(T))
  elseif alg == "qr_algorithm" || alg == "jacobi_algorithm"
    MUSV = svd_catch_error(matrix(T); alg)
  else
    error(
      "svd algorithm $alg is not currently supported. Please see the documentation for currently supported algorithms.",
    )
  end
  if isnothing(MUSV)
    if any(isnan, expose(T))
      println("SVD failed, the matrix you were trying to SVD contains NaNs.")
    else
      println(lapack_svd_error_message(alg))
    end
    return nothing
  end
  MU, MS, MV = MUSV
  conj!(MV)
  #end # @timeit_debug

  P = MS .^ 2
  if any(!isnothing, (maxdim, cutoff))
    P, truncerr, _ = truncate!!(
      P; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
    )
  else
    truncerr = 0.0
  end
  spec = Spectrum(P, truncerr)
  dS = length(P)
  if dS < length(MS)
    MU = expose(MU)[:, 1:dS]
    # Fails on some GPU backends like Metal.
    # resize!(MS, dS)
    MS = MS[1:dS]
    MV = expose(MV)[:, 1:dS]
  end

  # Make the new indices to go onto U and V
  u = eltype(IndsT)(dS)
  v = eltype(IndsT)(dS)
  Uinds = IndsT((ind(T, 1), u))
  Sinds = IndsT((u, v))
  Vinds = IndsT((ind(T, 2), v))
  U = tensor(Dense(vec(MU)), Uinds)
  S = tensor(Diag(MS), Sinds)
  V = tensor(Dense(vec(MV)), Vinds)
  return U, S, V, spec
end

function LinearAlgebra.eigen(
  T::Hermitian{ElT,<:DenseTensor{ElT,2,IndsT}};
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  use_absolute_cutoff=nothing,
  use_relative_cutoff=nothing,
) where {ElT<:Union{Real,Complex},IndsT}
  matrixT = matrix(T)
  ## TODO Here I am calling parent to ensure that the correct `any` function
  ## is envoked for non-cpu matrices
  ## TODO use expose here
  if any(!isfinite, parent(matrixT))
    throw(
      ArgumentError(
        "Trying to perform the eigendecomposition of a matrix containing NaNs or Infs"
      ),
    )
  end

  ### What do we do if DM is full of Nan or Inf?
  DM, VM = eigen(expose(matrixT))

  # Sort by largest to smallest eigenvalues
  # TODO: Replace `cpu` with `unwrap_type` dispatch.
  p = sortperm(cpu(DM); rev=true, by=abs)
  DM = DM[p]
  VM = VM[:, p]

  if any(!isnothing, (maxdim, cutoff))
    DM, truncerr, _ = truncate!!(
      DM; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
    )
    dD = length(DM)
    if dD < size(VM, 2)
      VM = VM[:, 1:dD]
    end
  else
    dD = length(DM)
    truncerr = 0.0
  end
  spec = Spectrum(DM, truncerr)

  # Make the new indices to go onto V
  l = eltype(IndsT)(dD)
  r = eltype(IndsT)(dD)
  Vinds = IndsT((dag(ind(T, 2)), dag(r)))
  Dinds = IndsT((l, dag(r)))
  V = tensor(Dense(vec(VM)), Vinds)
  D = tensor(Diag(DM), Dinds)
  return D, V, spec
end

"""
    random_unitary(n::Int,m::Int)::Matrix{ComplexF64}
    random_unitary(::Type{ElT},n::Int,m::Int)::Matrix{ElT}

Return a random matrix U of dimensions (n,m)
such that if n >= m, U'*U is the identity, or if
m > n U*U' is the identity. Optionally can pass a numeric
type as the first argument to obtain a matrix of that type.

Sampling is based on https://arxiv.org/abs/math-ph/0609050
such that in the case `n==m`, the unitary matrix will be sampled
according to the Haar measure.
"""
function random_unitary(::Type{ElT}, n::Int, m::Int) where {ElT<:Number}
  return random_unitary(Random.default_rng(), ElT, n, m)
end

function random_unitary(rng::AbstractRNG, DataT::Type{<:AbstractArray}, n::Int, m::Int)
  ElT = eltype(DataT)
  if n < m
    return DataT(random_unitary(rng, ElT, m, n)')
  end
  F = qr(randn(rng, ElT, n, m))
  Q = DataT(F.Q)
  # The upper triangle of F.factors
  # are the elements of R.
  # Multiply cols of Q by the signs
  # that would make diagonal of R
  # non-negative:
  for c in 1:size(Q, 2)
    Q[:, c] .*= sign(F.factors[c, c])
  end
  return Q
end

function random_unitary(rng::AbstractRNG, ::Type{ElT}, n::Int, m::Int) where {ElT<:Number}
  return random_unitary(rng, set_ndims(default_datatype(ElT), 2), n, m)
end

random_unitary(n::Int, m::Int) = random_unitary(ComplexF64, n, m)

"""
    random_orthog(n::Int,m::Int)::Matrix{Float64}
    random_orthog(::Type{ElT},n::Int,m::Int)::Matrix{ElT}

Return a random, real matrix O of dimensions (n,m)
such that if n >= m, transpose(O)*O is the
identity, or if m > n O*transpose(O) is the
identity. Optionally can pass a real number type
as the first argument to obtain a matrix of that type.
"""
random_orthog(::Type{ElT}, n::Int, m::Int) where {ElT<:Real} = random_unitary(ElT, n, m)

random_orthog(n::Int, m::Int) = random_orthog(Float64, n, m)

function LinearAlgebra.eigen(
  T::DenseTensor{ElT,2,IndsT};
  mindim=nothing,
  maxdim=nothing,
  cutoff=nothing,
  use_absolute_cutoff=nothing,
  use_relative_cutoff=nothing,
) where {ElT<:Union{Real,Complex},IndsT}
  matrixT = matrix(T)
  if any(!isfinite, matrixT)
    throw(
      ArgumentError(
        "Trying to perform the eigendecomposition of a matrix containing NaNs or Infs"
      ),
    )
  end

  DM, VM = eigen(expose(matrixT))

  # Sort by largest to smallest eigenvalues
  #p = sortperm(DM; rev = true)
  #DM = DM[p]
  #VM = VM[:,p]

  if any(!isnothing, (maxdim, cutoff))
    DM, truncerr, _ = truncate!!(
      DM; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff
    )
    dD = length(DM)
    if dD < size(VM, 2)
      VM = VM[:, 1:dD]
    end
  else
    dD = length(DM)
    truncerr = 0.0
  end
  spec = Spectrum(abs.(DM), truncerr)

  i1, i2 = inds(T)

  # Make the new indices to go onto D and V
  l = typeof(i1)(dD)
  r = dag(sim(l))
  Dinds = (l, r)
  Vinds = (dag(i2), r)
  D = complex(tensor(Diag(DM), Dinds))
  V = complex(tensor(Dense(vec(VM)), Vinds))
  return D, V, spec
end

# LinearAlgebra.qr
function qr(T::DenseTensor{<:Any,2}; positive=false)
  qxf = positive ? qr_positive : qr
  return qx(qxf, T)
end

# NDTensors.Unwrap.ql
function ql(T::DenseTensor{<:Any,2}; positive=false)
  qxf = positive ? ql_positive : ql
  return qx(qxf, T)
end

#
#  Generic function for qr and ql decomposition of dense matrix.
#  The X tensor = R or L.
#
function qx(qx::Function, T::DenseTensor{<:Any,2})
  QM, XM = qx(expose(matrix(T)))
  # Be aware that if positive==false, then typeof(QM)=LinearAlgebra.QRCompactWYQ, not Matrix
  # It gets converted to matrix below.
  # Make the new indices to go onto Q and R
  q, r = inds(T)
  q = dim(q) < dim(r) ? sim(q) : sim(r)
  IndsT = indstype(T) #get the index type
  Qinds = IndsT((ind(T, 1), q))
  Xinds = IndsT((q, ind(T, 2)))
  QM = convert(typeof(XM), QM)
  ## Here I convert QM twice because of an issue in CUDA where convert does not take QM to be a UnifiedBuffer array
  QM = convert(typeof(XM), QM)
  Q = tensor(Dense(vec(QM)), Qinds) #Q was strided
  X = tensor(Dense(vec(XM)), Xinds)
  return Q, X
end

# Version of `sign` that returns one
# if `x == 0`.
function nonzero_sign(x)
  iszero(x) && return one(x)
  return sign(x)
end

#
# Just flip signs between Q and R to get all the diagonals of R >=0.
# For rectangular M the indexing for "diagonal" is non-trivial.
# NDTensors.Unwrap.qr_positive and # NDTensors.Unwrap.ql_positive
#
"""
    qr_positive(M::AbstractMatrix)

Compute the QR decomposition of a matrix M
such that the diagonal elements of R are
non-negative. Such a QR decomposition of a
matrix is unique. Returns a tuple (Q,R).
"""
function qr_positive(M::AbstractMatrix)
  sparseQ, R = qr(M)
  Q = convert(typeof(R), sparseQ)
  nc = size(Q, 2)
  signs = nonzero_sign.(diag(R))
  Q = Q * Diagonal(signs)
  R = Diagonal(conj.(signs)) * R
  return (Q, R)
end

"""
    ql_positive(M::AbstractMatrix)

Compute the QL decomposition of a matrix M
such that the diagonal elements of L are
non-negative. Such a QL decomposition of a
matrix is unique. Returns a tuple (Q,L).
"""
function ql_positive(M::AbstractMatrix)
  # TODO: Change to `isgpu`, or better yet rewrite
  # in terms of broadcasting and linear algebra
  # like `qr_positive`.
  iscuda = iscu(M)
  if iscuda
    cutype = unwrap_type(M)
    M = NDTensors.cpu(M)
  end
  sparseQ, L = ql(M)
  Q = convert(typeof(L), sparseQ)
  nr, nc = size(L)
  dc = nc > nr ? nc - nr : 0 #diag is shifted over by dc if nc>nr
  for c in 1:(nc - dc)
    if L[c, c + dc] != 0.0 #sign(0.0)==0.0 so we don't want to zero out a column of Q.
      sign_Lc = sign(L[c, c + dc])
      if c <= nr && !isone(sign_Lc)
        L[c, 1:(c + dc)] *= sign_Lc #only fip non-zero portion of the column.
        Q[:, c] *= conj(sign_Lc)
      end
    end
  end
  if iscuda
    Q = adapt(cutype, Q)
    L = adapt(cutype, L)
  end
  return (Q, L)
end

#
#  Lapack replaces A with Q & R carefully packed together.  So here we just copy a
#  before letting lapack overwirte it.
#
function ql(A::AbstractMatrix)
  Base.require_one_based_indexing(A)
  T = eltype(A)
  AA = similar(A, LinearAlgebra._qreltype(T), size(A))
  copyto!(expose(AA), expose(A))
  iscuda = iscu(AA)
  if iscuda
    cutype = unwrap_type(AA)
    AA = NDTensors.cpu(AA)
  end
  Q, L = ql!(AA)
  if iscuda
    Q = adapt(cutype, Q)
    L = adapt(cutype, L)
  end
  return (Q, L)
end
#
# This is where the low level call to lapack actually occurs.  Most of the work is
# about unpacking Q and L from the A matrix.
#
function ql!(A::StridedMatrix{<:LAPACK.BlasFloat})
  if iscu(A)
    throw("Error: ql is not implemented in CUDA.jl")
  end
  tau = Base.similar(A, min(size(A)...))
  x = LAPACK.geqlf!(A, tau)
  #save L from the lower portion of A, before orgql! mangles it!
  nr, nc = size(A)
  mn = min(nr, nc)
  L = similar(A, (mn, nc))
  for r in 1:mn
    for c in 1:(r + nc - mn)
      L[r, c] = A[r + nr - mn, c]
    end
    for c in (r + 1 + nc - mn):nc
      L[r, c] = 0.0
    end
  end
  # Now we need shift the orth vectors from the right side of Q over the left side, before
  if (mn < nc)
    for r in 1:nr
      for c in 1:mn
        A[r, c] = A[r, c + nc - mn]
      end
    end
    for r in 1:nr
      A = A[:, 1:mn] #whack the extra columns in A.
    end
  end
  LAPACK.orgql!(A, tau)
  return A, L
end

# TODO: support alg keyword argument to choose the svd algorithm
function polar(T::DenseTensor{ElT,2,IndsT}) where {ElT,IndsT}
  QM, RM = polar(matrix(T))
  dim = size(QM, 2)
  # Make the new indices to go onto Q and R
  q = eltype(IndsT)(dim)
  # TODO: use push/pushfirst instead of a constructor
  # call here
  Qinds = IndsT((ind(T, 1), q))
  Rinds = IndsT((q, ind(T, 2)))
  Q = tensor(Dense(vec(QM)), Qinds)
  R = tensor(Dense(vec(RM)), Rinds)
  return Q, R
end
