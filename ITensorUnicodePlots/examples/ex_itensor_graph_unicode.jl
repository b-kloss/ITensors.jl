using ITensors
using ITensorUnicodePlots
using Graphs

g = grid((5,))
tn = itensornetwork(g; linkspaces=10, sitespaces=2)
@visualize fig tn

fig
