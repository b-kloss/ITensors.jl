using ITensors
include("imp.jl")

s_up = siteinds("Fermion", 2; conserve_qns=false)     #right
s_down = siteinds("Fermion", 2; conserve_qns=false)   #left
#aliases
rs = s_up
ls = s_down
#parameters
U = 0.2
dt = 0.5im
ed = 0.1
rsc = combiner(rs[1], rs[2]; tag="right")
lsc = combiner(prime(ls[1]), prime(ls[2]); tag="left")
lscnp = combiner(ls[1], ls[2]; tag="left")

irc = combinedind(rsc)
ilc = combinedind(lsc)
ilcnp = combinedind(lscnp)
@show ilc
@show irc
@show Matrix(op("C", ls[1]), inds(op("C", ls[1])))

lccdag = op("C", ls[2]) * op("Cdag", ls[1])
M = Matrix((ITensor(lccdag) * lscnp) * lsc, (ilc, ilcnp))
@show M
#return
if false
  return nothing
else
  matricize(t::ITensor) = return Matrix((t * lsc) * rsc, (ilc, irc))

  T = get_T(U, dt, ed, rs[1], rs[2], ls[1], ls[2])
  #Tc=rsc*T
  #@show Tc
  #Tc=lsc*Tc
  #@show Tc
  @show matricize(T)

  @show T
  lc = op("C", ls[2])
  lcdag = op("Cdag", ls[2])
  rc = op("C", rs[1])
  rcdag = op("Cdag", rs[1])
  @show lc
  #rc=prime(rc)
  @show rc
  @show matricize(replaceind(T * (-rc), prime(rs[1]), rs[1]))
  return ()

  #@show matricize(replaceind(T * prime(lc),prime(prime(ls[2])),prime(ls[2])))
  #@show matricize(replaceind(T * -prime(lcdag),prime(prime(ls[2])),prime(ls[2])))
  #@show matricize(replaceind(T * rcdag,prime(rs[2]),rs[2]))

  #@show Matrix(matricize(replaceind(T * rc,prime(rs[1]),rs[1])),(ilc,irc))
  #@show T
  #"""
end
