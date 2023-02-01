using ITensors
ferm_full = QN("Sz", -1) => 1
ferm_vac = QN("Sz", 0) => 1
antiferm_full = QN("Sz", +1) => 1
antiferm_vac = QN("Sz", 0) => 1
even_link_p1 = QN("Sz", +2) => 1 # p1 = "positive 1"
even_link_0 = QN("Sz", 0) => 1
even_link_n1 = QN("Sz", -2) => 1 # n1 = "negative 1"
odd_link_p1 = QN("Sz", -2) => 1
odd_link_0 = QN("Sz", 0) => 1
odd_link_n1 = QN("Sz", +2) => 1

function Kogut_Susskind_sites(num_spatial_sites)
  sites = Index[]
  for i in 1:num_spatial_sites
    push!(sites, Index(ferm_full, ferm_vac; tags="Site,S=1/2,n=$(4*i-3)"))
    push!(sites, Index(odd_link_p1, odd_link_0, odd_link_n1; tags="Site,S=1,n=$(4*i-2)"))
    push!(sites, Index(antiferm_full, antiferm_vac; tags="Site,S=1/2,n=$(4*i-1)"))
    push!(sites, Index(even_link_p1, even_link_0, even_link_n1; tags="Site,S=1,n=$(4*i)"))
  end
  return sites
end

function Schwinger_Hamiltonian(sites, num_spatial_sites, x, μ)
  N = num_spatial_sites * 2
  ampo_H = AutoMPO()
  for j in 0:(N - 1)
    add!(ampo_H, 1, "Sz2", 2 * j + 2)
    add!(ampo_H, (μ / 2) * (-1)^j * 2, "Sz", 2 * j + 1)
    if j != N - 1
      add!(ampo_H, x * 2 * 2 / sqrt(2), "S+", 2 * j + 1, "S-", 2 * j + 2, "S-", 2 * j + 3)
      add!(ampo_H, x * 2 * 2 / sqrt(2), "S+", 2 * j + 3, "S+", 2 * j + 2, "S-", 2 * j + 1)
    end
  end
  add!(ampo_H, x * 2 * 2 / sqrt(2), "S+", 2 * N - 1, "S-", 2 * N, "S-", 1)
  add!(ampo_H, x * 2 * 2 / sqrt(2), "S+", 1, "S+", 2 * N, "S-", 2 * N - 1)
  H = MPO(ampo_H, sites)
  return H
end

function Schwinger_DMRG(num_spatial_sites)
  sites = Kogut_Susskind_sites(num_spatial_sites)
  x = 0.5
  μ = 0.0
  H = Schwinger_Hamiltonian(sites, num_spatial_sites, x, μ)
  sweeps = Sweeps(10)
  maxdim!(sweeps, 10, 20, 100, 100, 200)
  cutoff!(sweeps, 1E-10)

  # construct initial state as completely empty
  init_state = ["Up" for n in 1:(4 * num_spatial_sites)] # just to initialize
  for n in 1:(4 * num_spatial_sites)
    if n % 4 == 1
      init_state[n] = "Dn" # empty fermion sites
    elseif n % 4 == 3
      init_state[n] = "Up" # empty antifermion states
    else # n % 4 == 2 || n % 4 == 0
      init_state[n] = "Up" # flux = 0 at all links
    end
  end
  psi0 = randomMPS(sites, init_state, 100)
  energy, psi = dmrg(H, psi0, sweeps)
  println("Ground state energy = $energy")
  return energy
end
