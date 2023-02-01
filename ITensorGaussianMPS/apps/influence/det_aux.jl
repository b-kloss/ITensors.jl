using LinearAlgebra
using ITensors

function reshuffle(seq0::Vector, offset::Int)
  if offset == 0
    return [Vector(1:N)]
  end

  newsequ = eltype(seq0)[]
  N = length(seq0)
  seq = copy(seq0)
  cind = 1

  order = [Int[]]
  ordercounter = 1
  not_touched = Vector(1:N)
  while true
    push!(order[ordercounter], cind)
    deleteat!(not_touched, not_touched .== cind)
    #@show order
    #@show not_touched
    cind = (cind + offset) % N #+ Int((cind+offset)>N)
    if cind == 0
      cind = N
    end
    if length(not_touched) == 0
      break
    end
    for anorder in order
      if cind in anorder
        cind = not_touched[1]
        push!(order, Int[])
        ordercounter += 1
        break
      end
    end
  end
  return order
end

function reshuffle_index_vector(inds::AbstractMatrix)
  n = size(inds, 1)
  @assert inds[:, 1] == Vector(1:n)
  cind = 1
  order = [Int[]]
  ordercounter = 1
  not_touched = Vector(1:n)
  while true
    push!(order[ordercounter], cind)
    deleteat!(not_touched, not_touched .== cind)
    # @show not_touched
    #@show order
    cind = inds[cind, 2]
    if cind == 0
      cind = n
    end
    if length(not_touched) == 0
      break
    end
    for anorder in order
      if cind in anorder
        cind = not_touched[1]
        push!(order, Int[])
        ordercounter += 1
        break
      end
    end
  end
  return order
end

function restricted_determinant(A::AbstractArray; parity=true)
  n = size(A, 1)
  newM = zeros(eltype(A), n, n)
  detsum = 0.0
  if parity == true
    factor = -1
  else
    factor = 1
  end
  for z in 1:(2^n)
    newM .= 0
    bools = digits(UInt(z - 1); base=2, pad=n)
    bools = Vector(bools)
    #counter=0

    for i in 1:n
      for j in 1:n
        newM[i, j] = A[i, j, bools[i] + 1, bools[j] + 1]
      end
    end

    detsum = detsum + factor^(z - 1) % 2 * det(newM)
  end
  return detsum
end

function get_contribution_ref(A::AbstractMatrix, offset::Int, counter::Bool; parity=true)
  n = size(A, 1)
  newM = zeros(eltype(A[1, 1]), n, n)
  detsum = 0.0
  if parity == true
    factor = -1
  else
    factor = 1
  end
  parityvec = [1]
  for i in 1:n
    parityvec = outer(parityvec, [1, -1])
  end
  for z in 1:(2^n)
    newM .= 0
    bools = digits(UInt(z - 1); base=2, pad=n)
    bools = Vector(bools)
    #counter=0

    for i in 1:n
      for j in 1:n
        newM[i, j] = A[i, j][bools[i] + 1, bools[j] + 1]
      end
    end

    #detsum=detsum+ factor^((z-1)%2) * reduce(*,get_diagonal(newM;offset=offset,counter=counter))* (counter ? -1 : 1)
    detsum =
      detsum +
      (parity ? parityvec[z] : 1) *
      reduce(*, get_diagonal(newM; offset=offset, counter=counter)) *
      (counter ? -1 : 1)
  end
  return detsum
end

function get_stacked_eigvals(A::AbstractMatrix; parity=true)
  n = size(A, 1)
  newM = zeros(eltype(A[1, 1]), n, n)
  detsum = 0.0
  if parity == true
    factor = -1
  else
    factor = 1
  end
  parityvec = [1]
  for i in 1:n
    parityvec = outer(parityvec, [1, -1])
  end
  Es = zeros(ComplexF64, 2^n, n)
  for z in 1:(2^n)
    newM .= 0
    bools = digits(UInt(z - 1); base=2, pad=n)
    bools = Vector(bools)
    #counter=0

    for i in 1:n
      for j in 1:n
        newM[i, j] = A[i, j][bools[i] + 1, bools[j] + 1]
      end
    end
    Es[z, :] = eigvals(newM)
  end
  return Es
end

function get_diagonal(A::AbstractMatrix; offset=0, counter=false)
  N = size(A, 1)
  @assert N == size(A, 2)
  order = counter ? reverse(Vector(1:N)) : Vector(1:N)
  if offset == 0
    return diag(A[order, :], offset)
  else
    return vcat(diag(A[order, :], offset), diag(A[order, :], offset - N))
  end
end

function get_index_matrix(N::Int, M::Int)
  B = Matrix{Pair{Int,Int}}(undef, N, M)
  for n in 1:N
    for m in 1:M
      B[n, m] = Pair(n, m)
    end
  end
  return B
end

function pairvector_to_matrix(pairs::Vector{T}) where {T<:Pair}
  N = length(pairs)
  res = Matrix{Int}(undef, N, 2)
  for i in 1:N
    res[i, 1] = pairs[i][1]
    res[i, 2] = pairs[i][2]
  end
  return res
end

function reshuffle_matrices(vecofmat, ordering)
  reshuffled = Vector{Matrix}[]
  #@show ordering
  #@show vecofmat
  for order in ordering
    #@show order
    reshsubset = Matrix[]
    for i in order
      #@show i
      push!(reshsubset, vecofmat[i])
    end
    push!(reshuffled, reshsubset)
  end
  return reshuffled
end

function make_TN(mats::Vector{Matrix})
  cores = ITensor[]
  T = eltype(mats[1])
  if length(mats) == 1
    n = Index(2; tags="site=" * string(1))
    return [ITensor(diag(mats[1]), n)]
  end
  indstorage = Pair{Index{Int64},Index{Int64}}[]
  for amat in mats
    i = Index(2)
    j = Index(2)
    push!(indstorage, Pair(i, j))
    push!(cores, ITensor(T, amat, i, j))
  end
  copytn = ITensor[]
  N = length(cores)
  contractedtn = ITensor[]
  for i in 1:N
    l, r = indstorage[i]
    #l,r=inds(cores[i])
    othersite = (i + 1) == N ? i + 1 : (i + 1) % N
    #l2,r2=inds(cores[othersite])
    l2, r2 = indstorage[othersite]
    n = Index(2; tags="site=" * string(othersite))
    cpt = ITensors.delta(r, l2, n)
    push!(copytn, cpt)
    push!(contractedtn, cpt * cores[i])
    #@show contractedtn
  end
  return contractedtn
end

function contract_TN(tn::Vector{ITensor}; parity=true)
  N = length(tn)

  res = 1
  #@show N
  if N == 1
    res = ITensor[]
    u = inds(tn[1])[1]
    if parity == false
      cap = ITensor([1.0, 1.0], u)
    else
      cap = ITensor([1.0, -1.0], u)
    end
    #push!(res,tn[1]*cap)
    return tn[1] * cap
  end
  #return
  #full=tn[1]
  for i in 1:N
    othersite = (i + 1) == N ? i + 1 : (i + 1) % N
    anothersite = i - 1 == 0 ? N : i - 1
    #@show i, othersite
    c = commonind(tn[i], tn[othersite])
    u = uniqueind(tn[i], tn[othersite], tn[anothersite])
    #@show inds(tn[i]),inds(tn[othersite]),inds(tn[anothersite])
    #@show u, c, inds(tn[i])
    if parity == false
      cap = ITensor([1.0, 1.0], u)
    else
      cap = ITensor([1.0, -1.0], u)
    end
    #@show i, inds(cap)
    #@show i, inds(tn[i]*cap)
    #if i>1
    #    full=full*tn[i]
    #end
    tn[i] = tn[i] * cap

    #@show i, inds(tn[i])
  end
  #println("final iteration")
  res = tn[1]
  for i in 2:N
    res = res * (tn[i])
  end
  #@show inds(res)
  #@show res, sum(full)
  return res
end

function get_contribution(M::AbstractMatrix, offset::Int; counter=false, parity=false)
  N = size(M, 1)
  indm = get_index_matrix(N, N)

  C = get_diagonal(indm; offset=offset, counter=counter)
  if counter
    C = reverse(C)
  end
  D = pairvector_to_matrix(C)
  @show D
  reind = reshuffle_index_vector(D)
  @show reind
  #@show reind
  E = get_diagonal(M; offset=offset, counter=counter)
  if counter
    E = reverse(E)
  end
  #@show length(E)
  #@show size(M)
  #return
  #@show E
  Er = reshuffle_matrices(E, reind)
  #@show Er
  #@show Er[1]
  #@show Er[2]
  res = 1.0
  for item in Er
    tn = make_TN(item)
    interm_res = contract_TN(tn; parity=parity)
    res = res * interm_res
  end
  factor = counter ? -1 : 1
  #@show inds(res)
  return eltype(res).(res[1] * factor)
end
