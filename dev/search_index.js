var documenterSearchIndex = {"docs":
[{"location":"DMRG.html#DMRG-1","page":"DMRG","title":"DMRG","text":"","category":"section"},{"location":"DMRG.html#","page":"DMRG","title":"DMRG","text":"dmrg","category":"page"},{"location":"DMRG.html#ITensors.dmrg","page":"DMRG","title":"ITensors.dmrg","text":"dmrg(H::MPO,psi0::MPS,sweeps::Sweeps;kwargs...)\n\nUse the density matrix renormalization group (DMRG) algorithm to optimize a matrix product state (MPS) such that it is the eigenvector of lowest eigenvalue of a Hermitian matrix H, represented as a matrix product operator (MPO). The MPS psi0 is used to initialize the MPS to be optimized, and the sweeps object determines the parameters used to  control the DMRG algorithm.\n\nReturns:\n\nenergy::Float64 - eigenvalue of the optimized MPS\npsi::MPS - optimized MPS\n\n\n\n\n\ndmrg(Hs::Vector{MPO},psi0::MPS,sweeps::Sweeps;kwargs...)\n\nUse the density matrix renormalization group (DMRG) algorithm to optimize a matrix product state (MPS) such that it is the eigenvector of lowest eigenvalue of a Hermitian matrix H. The MPS psi0 is used to initialize the MPS to be optimized, and the sweeps object determines the parameters used to  control the DMRG algorithm.\n\nThis version of dmrg accepts a representation of H as a Vector of MPOs, Hs = [H1,H2,H3,...] such that H is defined as H = H1+H2+H3+... Note that this sum of MPOs is not actually computed; rather the set of MPOs [H1,H2,H3,..] is efficiently looped over at  each step of the DMRG algorithm when optimizing the MPS.\n\nReturns:\n\nenergy::Float64 - eigenvalue of the optimized MPS\npsi::MPS - optimized MPS\n\n\n\n\n\ndmrg(H::MPO,Ms::Vector{MPS},psi0::MPS,sweeps::Sweeps;kwargs...)\n\nUse the density matrix renormalization group (DMRG) algorithm to optimize a matrix product state (MPS) such that it is the eigenvector of lowest eigenvalue of a Hermitian matrix H, subject to the constraint that the MPS is orthogonal to each of the MPS provided in the Vector Ms. The orthogonality constraint is approximately enforced by adding to H terms of  the form w|M1><M1| + w|M2><M2| + ... where Ms=[M1,M2,...] and w is the \"weight\" parameter, which can be adjusted through the optional weight keyword argument. The MPS psi0 is used to initialize the MPS to be optimized, and the sweeps object determines the parameters used to  control the DMRG algorithm.\n\nReturns:\n\nenergy::Float64 - eigenvalue of the optimized MPS\npsi::MPS - optimized MPS\n\n\n\n\n\n","category":"function"},{"location":"IndexType.html#Index-1","page":"Index","title":"Index","text":"","category":"section"},{"location":"IndexType.html#Index-object-1","page":"Index","title":"Index object","text":"","category":"section"},{"location":"IndexType.html#","page":"Index","title":"Index","text":"Index","category":"page"},{"location":"IndexType.html#ITensors.Index","page":"Index","title":"ITensors.Index","text":"An Index represents a single tensor index with fixed dimension dim. Copies of an Index compare equal unless their  tags are different.\n\nAn Index carries a TagSet, a set of tags which are small strings that specify properties of the Index to help  distinguish it from other Indices. There is a special tag which is referred to as the integer tag or prime  level which can be incremented or decremented with special priming functions.\n\nInternally, an Index has a fixed id number, which is how the ITensor library knows two indices are copies of a  single original Index. Index objects must have the same id, as well as the tags to compare equal.\n\n\n\n\n\n","category":"type"},{"location":"IndexType.html#Index-constructors-1","page":"Index","title":"Index constructors","text":"","category":"section"},{"location":"IndexType.html#","page":"Index","title":"Index","text":"Index(::Int)\nIndex(::Int, ::Union{AbstractString,TagSet})\nIndex()","category":"page"},{"location":"IndexType.html#ITensors.Index-Tuple{Int64}","page":"Index","title":"ITensors.Index","text":"Index(dim::Int; tags::Union{AbstractString,TagSet}=\"\",\n                plev::Int=0)\n\nCreate an Index with a unique id, a TagSet given by tags, and a prime level plev.\n\nExamples\n\njulia> i = Index(2; tags=\"l\", plev=1)\n(dim=2|id=818|\"l\")'\n\njulia> dim(i)\n2\n\njulia> plev(i)\n1\n\njulia> tags(i)\n\"l\"\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#ITensors.Index-Tuple{Int64,Union{TagSet, AbstractString}}","page":"Index","title":"ITensors.Index","text":"Index(dim::Integer, tags::Union{AbstractString,TagSet})\n\nCreate an Index with a unique id and a tagset given by tags.\n\nExamples\n\njulia> i = Index(2, \"l,tag\")\n(dim=2|id=58|\"l,tag\")\n\njulia> dim(i)\n2\n\njulia> plev(i)\n0\n\njulia> tags(i)\n\"l,tag\"\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#ITensors.Index-Tuple{}","page":"Index","title":"ITensors.Index","text":"Index()\n\nCreate a default Index.\n\nExamples\n\njulia> i = Index()\n(dim=0|id=0)\n\njulia> isdefault(i)\ntrue\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#Index-properties-1","page":"Index","title":"Index properties","text":"","category":"section"},{"location":"IndexType.html#","page":"Index","title":"Index","text":"id(::Index)\nhasid(::Index, ::ITensors.IDType)\ntags(::Index)\nhastags(::Index, ::Union{AbstractString,TagSet})\nplev(::Index)\nhasplev(::Index, ::Int)\ndim(::Index)\n==(::Index, ::Index)\ndir(::Index)","category":"page"},{"location":"IndexType.html#ITensors.id-Tuple{Index}","page":"Index","title":"ITensors.id","text":"id(i::Index)\n\nObtain the id of an Index, which is a unique 64 digit integer.\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#ITensors.hasid-Tuple{Index,UInt64}","page":"Index","title":"ITensors.hasid","text":"hasid(i::Index, id::ITensors.IDType)\n\nCheck if an Index i has the provided id.\n\nExamples\n\njulia> i = Index(2)\n(dim=2|id=321)\n\njulia> hasid(i, id(i))\ntrue\n\njulia> j = Index(2)\n(dim=2|id=17)\n\njulia> hasid(i, id(j))\nfalse\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#ITensors.tags-Tuple{Index}","page":"Index","title":"ITensors.tags","text":"tags(i::Index)\n\nObtain the TagSet of an Index.\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#ITensors.hastags-Tuple{Index,Union{TagSet, AbstractString}}","page":"Index","title":"ITensors.hastags","text":"hastags(i::Index, ts::Union{AbstractString,TagSet})\n\nCheck if an Index i has the provided tags, which can be a string of comma-separated tags or  a TagSet object.\n\nExamples\n\njulia> i = Index(2, \"SpinHalf,Site,n=3\")\n(dim=2|id=861|\"Site,SpinHalf,n=3\")\n\njulia> hastags(i, \"SpinHalf,Site\")\ntrue\n\njulia> hastags(i, \"Link\")\nfalse\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#ITensors.plev-Tuple{Index}","page":"Index","title":"ITensors.plev","text":"plev(i::Index)\n\nObtain the prime level of an Index.\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#ITensors.hasplev-Tuple{Index,Int64}","page":"Index","title":"ITensors.hasplev","text":"hasplev(i::Index, plev::Int)\n\nCheck if an Index i has the provided prime level.\n\nExamples\n\njulia> i = Index(2; plev=2)\n(dim=2|id=543)''\n\njulia> hasplev(i, 2)\ntrue\n\njulia> hasplev(i, 1)\nfalse\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#ITensors.NDTensors.dim-Tuple{Index}","page":"Index","title":"ITensors.NDTensors.dim","text":"dim(i::Index)\n\nObtain the dimension of an Index.\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#Base.:==-Tuple{Index,Index}","page":"Index","title":"Base.:==","text":"==(i1::Index, i1::Index)\n\nCompare indices for equality. First the id's are compared, then the prime levels are compared, and finally the tags are compared.\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#ITensors.dir-Tuple{Index}","page":"Index","title":"ITensors.dir","text":"dir(i::Index)\n\nObtain the direction of an Index (In, Out, or Neither).\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#Priming-and-tagging-methods-1","page":"Index","title":"Priming and tagging methods","text":"","category":"section"},{"location":"IndexType.html#","page":"Index","title":"Index","text":"prime(::Index, ::Int)\nsetprime(::Index, ::Int)\nnoprime(::Index)\nsettags(::Index, ::Any)\naddtags(::Index, ::Any)\nremovetags(::Index, ::Any)\nreplacetags(::Index, ::Any, ::Any)","category":"page"},{"location":"IndexType.html#ITensors.prime-Tuple{Index,Int64}","page":"Index","title":"ITensors.prime","text":"prime(i::Index, plinc::Int = 1)\n\nReturn a copy of Index i with its prime level incremented by the amount plinc\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#ITensors.setprime-Tuple{Index,Int64}","page":"Index","title":"ITensors.setprime","text":"setprime(i::Index, plev::Int)\n\nReturn a copy of Index i with its prime level set to plev\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#ITensors.noprime-Tuple{Index}","page":"Index","title":"ITensors.noprime","text":"noprime(i::Index)\n\nReturn a copy of Index i with its prime level set to zero.\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#ITensors.settags-Tuple{Index,Any}","page":"Index","title":"ITensors.settags","text":"settags(i::Index, ts)\n\nReturn a copy of Index i with tags replaced by the ones given The ts argument can be a comma-separated  string of tags or a TagSet.\n\nExamples\n\njulia> i = Index(2, \"SpinHalf,Site,n=3\")\n(dim=2|id=543|\"Site,SpinHalf,n=3\")\n\njulia> hastags(i, \"Link\")\nfalse\n\njulia> j = settags(i,\"Link,n=4\")\n(dim=2|id=543|\"Link,n=4\")\n\njulia> hastags(j, \"Link\")\ntrue\n\njulia> hastags(j, \"n=4,Link\")\ntrue\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#ITensors.addtags-Tuple{Index,Any}","page":"Index","title":"ITensors.addtags","text":"addtags(i::Index,ts)\n\nReturn a copy of Index i with the specified tags added to the existing ones. The ts argument can be a comma-separated  string of tags or a TagSet.\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#ITensors.removetags-Tuple{Index,Any}","page":"Index","title":"ITensors.removetags","text":"removetags(i::Index, ts)\n\nReturn a copy of Index i with the specified tags removed. The ts argument can be a comma-separated string of tags or a TagSet.\n\n\n\n\n\n","category":"method"},{"location":"IndexType.html#ITensors.replacetags-Tuple{Index,Any,Any}","page":"Index","title":"ITensors.replacetags","text":"replacetags(i::Index, tsold, tsnew)\n\nIf the tag set of i contains the tags specified by tsold, replaces these with the tags specified by tsnew, preserving any other tags. The arguments tsold and tsnew can be comma-separated strings of tags, or TagSet objects.\n\n\n\n\n\n","category":"method"},{"location":"ITensorType.html#ITensor-1","page":"ITensor","title":"ITensor","text":"","category":"section"},{"location":"ITensorType.html#","page":"ITensor","title":"ITensor","text":"ITensor","category":"page"},{"location":"ITensorType.html#ITensors.ITensor","page":"ITensor","title":"ITensors.ITensor","text":"An ITensor is a tensor whose interface is  independent of its memory layout. Therefore it is not necessary to know the ordering of an ITensor's indices, only which indices an ITensor has. Operations like contraction and addition of ITensors automatically handle any memory permutations.\n\n\n\n\n\n","category":"type"},{"location":"ITensorType.html#Constructors-1","page":"ITensor","title":"Constructors","text":"","category":"section"},{"location":"ITensorType.html#","page":"ITensor","title":"ITensor","text":"ITensor(::IndexSet)\nITensor(::Number, ::Index...)\nrandomITensor(::Type{<:Number}, ::IndexSet)","category":"page"},{"location":"ITensorType.html#ITensors.ITensor-Tuple{IndexSet}","page":"ITensor","title":"ITensors.ITensor","text":"ITensor(iset::IndexSet)\n\nConstruct an ITensor having indices given by the IndexSet iset\n\n\n\n\n\n","category":"method"},{"location":"ITensorType.html#ITensors.ITensor-Tuple{Number,Vararg{Index,N} where N}","page":"ITensor","title":"ITensors.ITensor","text":"ITensor(x)\n\nConstruct a scalar ITensor with value x.\n\nITensor(x,i,j,...)\n\nConstruct an ITensor with indices i,j,... and all elements set to float(x).\n\nNote that the ITensor storage will be the closest floating point version of the input value.\n\n\n\n\n\n","category":"method"},{"location":"ITensorType.html#ITensors.randomITensor-Tuple{Type{#s86} where #s86<:Number,IndexSet}","page":"ITensor","title":"ITensors.randomITensor","text":"randomITensor([S,] inds)\n\nConstruct an ITensor with type S (default Float64) and indices inds, whose elements are normally distributed random numbers.\n\n\n\n\n\n","category":"method"},{"location":"ITensorType.html#Sparse-constructors-1","page":"ITensor","title":"Sparse constructors","text":"","category":"section"},{"location":"ITensorType.html#","page":"ITensor","title":"ITensor","text":"diagITensor(::IndexSet)\ndelta(::Type{<:Number}, ::IndexSet)","category":"page"},{"location":"ITensorType.html#ITensors.diagITensor-Tuple{IndexSet}","page":"ITensor","title":"ITensors.diagITensor","text":"diagITensor(is::IndexSet)\n\nMake a sparse ITensor of element type Float64 with non-zero elements  only along the diagonal. Defaults to storing zeros along the diagonal. The storage will have Diag type.\n\n\n\n\n\n","category":"method"},{"location":"ITensorType.html#ITensors.delta-Tuple{Type{#s86} where #s86<:Number,IndexSet}","page":"ITensor","title":"ITensors.delta","text":"delta(::Type{T},inds::IndexSet)\n\nMake a diagonal ITensor with all diagonal elements 1.\n\n\n\n\n\n","category":"method"},{"location":"ITensorType.html#Operations-1","page":"ITensor","title":"Operations","text":"","category":"section"},{"location":"ITensorType.html#","page":"ITensor","title":"ITensor","text":"*(::ITensor, ::ITensor)\nexp(::ITensor, ::Any)","category":"page"},{"location":"ITensorType.html#Base.:*-Tuple{ITensor,ITensor}","page":"ITensor","title":"Base.:*","text":"*(A::ITensor, B::ITensor)\n\nContract ITensors A and B to obtain a new ITensor. This  contraction * operator finds all matching indices common to A and B and sums over them, such that the result will  have only the unique indices of A and B. To prevent indices from matching, their prime level or tags can be  modified such that they no longer compare equal - for more information see the documentation on Index objects.\n\n\n\n\n\n","category":"method"},{"location":"ITensorType.html#Base.exp-Tuple{ITensor,Any}","page":"ITensor","title":"Base.exp","text":"exp(A::ITensor, Lis::IndexSet; hermitian = false)\n\nCompute the exponential of the tensor A by treating it as a matrix A_lr with the left index l running over all indices in Lis and r running over all indices not in Lis. Must have dim(Lis) == dim(inds(A))/dim(Lis) for the exponentiation to be defined. When ishermitian=true the exponential of Hermitian(A_{lr}) is computed internally.\n\n\n\n\n\n","category":"method"},{"location":"ITensorType.html#Decompositions-1","page":"ITensor","title":"Decompositions","text":"","category":"section"},{"location":"ITensorType.html#","page":"ITensor","title":"ITensor","text":"svd(::ITensor, ::Any...)","category":"page"},{"location":"ITensorType.html#LinearAlgebra.svd-Tuple{ITensor,Vararg{Any,N} where N}","page":"ITensor","title":"LinearAlgebra.svd","text":"svd(A::ITensor,\n    leftind1::Index,\n    leftind2::Index,\n    ...\n    ;kwargs...)\n\nSingular value decomposition (SVD) of an ITensor A, computed by treating the \"left indices\" provided collectively as a row index, and the remaining \"right indices\" as a column index (matricization of a tensor).\n\nWhether the SVD performs a trunction depends on the keyword arguments provided. The following keyword arguments are recognized:\n\nmaxdim [Int]\nmindim [Int]\ncutoff [Float64]\nuse_absolute_cutoff [Bool] Default value: false.\nuse_relative_cutoff [Bool] Default value: true.\nutags [String] Default value: \"Link,u\".\nvtags [String] Default value: \"Link,v\".\nfastsvd [Bool] Defaut value: false.\n\n\n\n\n\n","category":"method"},{"location":"IndexSetType.html#IndexSet-1","page":"IndexSet","title":"IndexSet","text":"","category":"section"},{"location":"IndexSetType.html#","page":"IndexSet","title":"IndexSet","text":"IndexSet(::Vector{<:Index})","category":"page"},{"location":"IndexSetType.html#ITensors.IndexSet-Tuple{Array{#s86,1} where #s86<:Index}","page":"IndexSet","title":"ITensors.IndexSet","text":"IndexSet(inds::Vector{<:Index})\n\nConvert a Vector of indices to an IndexSet.\n\nWarning: this is not type stable, since a Vector is dynamically sized and an IndexSet is statically sized. Consider using the constructor IndexSet{N}(inds::Vector).\n\n\n\n\n\n","category":"method"},{"location":"IndexSetType.html#Priming-and-tagging-methods-1","page":"IndexSet","title":"Priming and tagging methods","text":"","category":"section"},{"location":"IndexSetType.html#","page":"IndexSet","title":"IndexSet","text":"prime(::IndexSet, ::Int)\nmap(::Function, ::IndexSet)","category":"page"},{"location":"IndexSetType.html#ITensors.prime-Tuple{IndexSet,Int64}","page":"IndexSet","title":"ITensors.prime","text":"prime(A, plinc, ...)\n\nIncrease the prime level of the indices by the specified amount. Filter which indices are primed using keyword arguments tags, plev and id.\n\n\n\n\n\n","category":"method"},{"location":"IndexSetType.html#Base.map-Tuple{Function,IndexSet}","page":"IndexSet","title":"Base.map","text":"map(f, is::IndexSet)\n\nApply the function to the elements of the IndexSet, returning a new IndexSet.\n\n\n\n\n\n","category":"method"},{"location":"IndexSetType.html#Set-operations-1","page":"IndexSet","title":"Set operations","text":"","category":"section"},{"location":"IndexSetType.html#","page":"IndexSet","title":"IndexSet","text":"intersect(::IndexSet, ::IndexSet)\nfirstintersect(::IndexSet, ::IndexSet)\nsetdiff(::IndexSet, ::IndexSet)\nfirstsetdiff(::IndexSet, ::IndexSet)","category":"page"},{"location":"IndexSetType.html#Base.intersect-Tuple{IndexSet,IndexSet}","page":"IndexSet","title":"Base.intersect","text":"intersect(A,B)\n\nOutput the IndexSet in the intersection of A and B\n\n\n\n\n\n","category":"method"},{"location":"IndexSetType.html#ITensors.firstintersect-Tuple{IndexSet,IndexSet}","page":"IndexSet","title":"ITensors.firstintersect","text":"firstintersect(Ais,Bis)\n\nOutput the Index common to Ais and Bis. If more than one Index is found, throw an error. Otherwise, return a default constructed Index.\n\n\n\n\n\n","category":"method"},{"location":"IndexSetType.html#Base.setdiff-Tuple{IndexSet,IndexSet}","page":"IndexSet","title":"Base.setdiff","text":"setdiff(A,B...)\n\nOutput the IndexSet with Indices in Ais but not in the IndexSets Bis.\n\n\n\n\n\n","category":"method"},{"location":"IndexSetType.html#ITensors.firstsetdiff-Tuple{IndexSet,IndexSet}","page":"IndexSet","title":"ITensors.firstsetdiff","text":"firstsetdiff(A,B)\n\nOutput the Index in Ais but not in the IndexSets Bis. Otherwise, return a default constructed Index.\n\nIn the future, this may throw an error if more than  one Index is found.\n\n\n\n\n\n","category":"method"},{"location":"IndexSetType.html#Subsets-1","page":"IndexSet","title":"Subsets","text":"","category":"section"},{"location":"IndexSetType.html#","page":"IndexSet","title":"IndexSet","text":"getfirst(::Function, ::IndexSet)\ngetfirst(::IndexSet)\nfilter(::Function, ::IndexSet)","category":"page"},{"location":"IndexSetType.html#ITensors.getfirst-Tuple{Function,IndexSet}","page":"IndexSet","title":"ITensors.getfirst","text":"Get the first value matching the pattern function, return nothing if not found.\n\n\n\n\n\n","category":"method"},{"location":"IndexSetType.html#ITensors.getfirst-Tuple{IndexSet}","page":"IndexSet","title":"ITensors.getfirst","text":"Like first, but if the length is 0 return nothing\n\n\n\n\n\n","category":"method"},{"location":"IndexSetType.html#Base.filter-Tuple{Function,IndexSet}","page":"IndexSet","title":"Base.filter","text":"filter(f::Function,inds::IndexSet)\n\nFilter the IndexSet by the given function (output a new IndexSet with indices i for which f(i) returns true).\n\n\n\n\n\n","category":"method"},{"location":"MPSandMPO.html#MPS-and-MPO-1","page":"MPS and MPO","title":"MPS and MPO","text":"","category":"section"},{"location":"MPSandMPO.html#Types-1","page":"MPS and MPO","title":"Types","text":"","category":"section"},{"location":"MPSandMPO.html#","page":"MPS and MPO","title":"MPS and MPO","text":"MPS\nMPO","category":"page"},{"location":"MPSandMPO.html#ITensors.MPS","page":"MPS and MPO","title":"ITensors.MPS","text":"MPS\n\nA finite size matrix product state type. Keeps track of the orthogonality center.\n\n\n\n\n\n","category":"type"},{"location":"MPSandMPO.html#ITensors.MPO","page":"MPS and MPO","title":"ITensors.MPO","text":"MPO\n\nA finite size matrix product operator type.  Keeps track of the orthogonality center.\n\n\n\n\n\n","category":"type"},{"location":"MPSandMPO.html#MPS-Constructors-1","page":"MPS and MPO","title":"MPS Constructors","text":"","category":"section"},{"location":"MPSandMPO.html#","page":"MPS and MPO","title":"MPS and MPO","text":"MPS(::Int)\nMPS(::Type{<:Number}, ::Any)\nMPS(::Any)\nrandomMPS\nproductMPS","category":"page"},{"location":"MPSandMPO.html#ITensors.MPS-Tuple{Int64}","page":"MPS and MPO","title":"ITensors.MPS","text":"MPS(N::Int)\n\nConstruct an MPS with N sites with default constructed ITensors.\n\n\n\n\n\n","category":"method"},{"location":"MPSandMPO.html#ITensors.MPS-Tuple{Type{#s86} where #s86<:Number,Any}","page":"MPS and MPO","title":"ITensors.MPS","text":"MPS(::Type{T<:Number}, sites)\n\nConstruct an MPS from a collection of indices with element type T.\n\n\n\n\n\n","category":"method"},{"location":"MPSandMPO.html#ITensors.MPS-Tuple{Any}","page":"MPS and MPO","title":"ITensors.MPS","text":"MPS(sites)\n\nConstruct an MPS from a collection of indices with element type Float64.\n\n\n\n\n\n","category":"method"},{"location":"MPSandMPO.html#ITensors.randomMPS","page":"MPS and MPO","title":"ITensors.randomMPS","text":"randomMPS(::Type{T<:Number}, sites; linkdim=1)\n\nConstruct a random MPS with link dimension linkdim of  type T.\n\n\n\n\n\nrandomMPS(sites; linkdim=1)\n\nConstruct a random MPS with link dimension linkdim of  type Float64.\n\n\n\n\n\n","category":"function"},{"location":"MPSandMPO.html#ITensors.productMPS","page":"MPS and MPO","title":"ITensors.productMPS","text":"productMPS(::Type{T<:Number}, ivals::Vector{<:IndexVal})\n\nConstruct a product state MPS with element type T and nonzero values determined from the input IndexVals.\n\n\n\n\n\nproductMPS(ivals::Vector{<:IndexVal})\n\nConstruct a product state MPS with element type Float64 and nonzero values determined from the input IndexVals.\n\n\n\n\n\n","category":"function"},{"location":"MPSandMPO.html#MPO-Constructors-1","page":"MPS and MPO","title":"MPO Constructors","text":"","category":"section"},{"location":"MPSandMPO.html#","page":"MPS and MPO","title":"MPS and MPO","text":"MPO(::Int)\nMPO(::Any, ::Vector{String})\nMPO(::Any, ::String)","category":"page"},{"location":"MPSandMPO.html#ITensors.MPO-Tuple{Int64}","page":"MPS and MPO","title":"ITensors.MPO","text":"MPO(N::Int)\n\nMake an MPO of length N filled with default ITensors.\n\n\n\n\n\n","category":"method"},{"location":"MPSandMPO.html#ITensors.MPO-Tuple{Any,Array{String,1}}","page":"MPS and MPO","title":"ITensors.MPO","text":"MPO(sites, ops::Vector{String})\n\nMake an MPO with pairs of sites s[i] and s[i]' and operators ops on each site.\n\n\n\n\n\n","category":"method"},{"location":"MPSandMPO.html#ITensors.MPO-Tuple{Any,String}","page":"MPS and MPO","title":"ITensors.MPO","text":"MPO(sites, ops::Vector{String})\n\nMake an MPO with pairs of sites s[i] and s[i]' and operators ops on each site.\n\n\n\n\n\n","category":"method"},{"location":"MPSandMPO.html#Properties-1","page":"MPS and MPO","title":"Properties","text":"","category":"section"},{"location":"MPSandMPO.html#","page":"MPS and MPO","title":"MPS and MPO","text":"length(::ITensors.AbstractMPS)\nmaxlinkdim(::ITensors.AbstractMPS)","category":"page"},{"location":"MPSandMPO.html#Base.length-Tuple{ITensors.AbstractMPS}","page":"MPS and MPO","title":"Base.length","text":"length(::MPS/MPO)\n\nThe number of sites of an MPS/MPO.\n\n\n\n\n\n","category":"method"},{"location":"MPSandMPO.html#ITensors.maxlinkdim-Tuple{ITensors.AbstractMPS}","page":"MPS and MPO","title":"ITensors.maxlinkdim","text":"maxlinkdim(M::MPS)\n\nmaxlinkdim(M::MPO)\n\nGet the maximum link dimension of the MPS or MPO.\n\n\n\n\n\n","category":"method"},{"location":"MPSandMPO.html#Priming-and-tagging-1","page":"MPS and MPO","title":"Priming and tagging","text":"","category":"section"},{"location":"MPSandMPO.html#","page":"MPS and MPO","title":"MPS and MPO","text":"prime!(::ITensors.AbstractMPS)","category":"page"},{"location":"MPSandMPO.html#ITensors.prime!-Tuple{ITensors.AbstractMPS}","page":"MPS and MPO","title":"ITensors.prime!","text":"prime!(m::MPS, args...; kwargs...)\n\nprime!(m::MPO, args...; kwargs...)\n\nPrime all ITensors of an MPS/MPO in-place.\n\n\n\n\n\n","category":"method"},{"location":"MPSandMPO.html#Operations-1","page":"MPS and MPO","title":"Operations","text":"","category":"section"},{"location":"MPSandMPO.html#","page":"MPS and MPO","title":"MPS and MPO","text":"dag(::ITensors.AbstractMPS)\northogonalize!\ntruncate!\nreplacebond!(::MPS, ::Int, ::ITensor; kwargs...)\nsample(::MPS)\nsample!(::MPS)","category":"page"},{"location":"MPSandMPO.html#ITensors.dag-Tuple{ITensors.AbstractMPS}","page":"MPS and MPO","title":"ITensors.dag","text":"dag(m::MPS)\n\ndag(m::MPO)\n\nHermitian conjugation of a matrix product state or operator m.\n\n\n\n\n\n","category":"method"},{"location":"MPSandMPO.html#ITensors.orthogonalize!","page":"MPS and MPO","title":"ITensors.orthogonalize!","text":"orthogonalize!(M::MPS, j::Int; kwargs...)\n\northogonalize!(M::MPO, j::Int; kwargs...)\n\nMove the orthogonality center of the MPS to site j. No observable property of the MPS will be changed, and no truncation of the bond indices is performed. Afterward, tensors 1,2,...,j-1 will be left-orthogonal and tensors j+1,j+2,...,N will be right-orthogonal.\n\n\n\n\n\n","category":"function"},{"location":"MPSandMPO.html#ITensors.NDTensors.truncate!","page":"MPS and MPO","title":"ITensors.NDTensors.truncate!","text":"truncate!(M::MPS; kwargs...)\n\ntruncate!(M::MPO; kwargs...)\n\nPerform a truncation of all bonds of an MPS/MPO, using the truncation parameters (cutoff,maxdim, etc.) provided as keyword arguments.\n\n\n\n\n\n","category":"function"},{"location":"MPSandMPO.html#ITensors.replacebond!-Tuple{MPS,Int64,ITensor}","page":"MPS and MPO","title":"ITensors.replacebond!","text":"replacebond!(M::MPS, b::Int, phi::ITensor; kwargs...)\n\nFactorize the ITensor phi and replace the ITensors b and b+1 of MPS M with the factors. Choose the orthogonality with ortho=\"left\"/\"right\".\n\n\n\n\n\n","category":"method"},{"location":"MPSandMPO.html#ITensors.sample-Tuple{MPS}","page":"MPS and MPO","title":"ITensors.sample","text":"sample(m::MPS)\n\nGiven a normalized MPS m with orthocenter(m)==1, returns a Vector{Int} of length(m) corresponding to one sample of the probability distribution defined by squaring the components of the tensor that the MPS represents\n\n\n\n\n\n","category":"method"},{"location":"MPSandMPO.html#ITensors.sample!-Tuple{MPS}","page":"MPS and MPO","title":"ITensors.sample!","text":"sample!(m::MPS)\n\nGiven a normalized MPS m, returns a Vector{Int} of length(m) corresponding to one sample of the probability distribution defined by squaring the components of the tensor that the MPS represents. If the MPS does not have an orthogonality center, orthogonalize!(m,1) will be called before computing the sample.\n\n\n\n\n\n","category":"method"},{"location":"MPSandMPO.html#Algebra-Operations-1","page":"MPS and MPO","title":"Algebra Operations","text":"","category":"section"},{"location":"MPSandMPO.html#","page":"MPS and MPO","title":"MPS and MPO","text":"dot(::MPS, ::MPS)\n+(::MPS, ::MPS)\n+(::MPO, ::MPO)\n*(::MPO, ::MPS)","category":"page"},{"location":"MPSandMPO.html#LinearAlgebra.dot-Tuple{MPS,MPS}","page":"MPS and MPO","title":"LinearAlgebra.dot","text":"dot(psi::MPS, phi::MPS; make_inds_match = true)\ninner(psi::MPS, phi::MPS; make_inds_match = true)\n\nCompute <psi|phi>.\n\nIf make_inds_match = true, the function attempts to make the site indices match before contracting (so for example, the inputs can have different site indices, as long as they  have the same dimensions or QN blocks).\n\n\n\n\n\n","category":"method"},{"location":"MPSandMPO.html#Base.:+-Tuple{MPS,MPS}","page":"MPS and MPO","title":"Base.:+","text":"add(A::MPS, B::MPS; kwargs...)\n+(A::MPS, B::MPS; kwargs...)\n\nadd(A::MPO, B::MPO; kwargs...)\n+(A::MPO, B::MPO; kwargs...)\n\nAdd two MPS/MPO with each other, with some optional truncation.\n\n\n\n\n\n","category":"method"},{"location":"MPSandMPO.html#Base.:+-Tuple{MPO,MPO}","page":"MPS and MPO","title":"Base.:+","text":"add(A::MPS, B::MPS; kwargs...)\n+(A::MPS, B::MPS; kwargs...)\n\nadd(A::MPO, B::MPO; kwargs...)\n+(A::MPO, B::MPO; kwargs...)\n\nAdd two MPS/MPO with each other, with some optional truncation.\n\n\n\n\n\n","category":"method"},{"location":"MPSandMPO.html#Base.:*-Tuple{MPO,MPS}","page":"MPS and MPO","title":"Base.:*","text":"contract(::MPS, ::MPO; kwargs...)\n*(::MPS, ::MPO; kwargs...)\n\ncontract(::MPO, ::MPS; kwargs...)\n*(::MPO, ::MPS; kwargs...)\n\nContract the MPO with the MPS, returning an MPS with the unique site indices of the MPO.\n\nChoose the method with the method keyword, for example \"densitymatrix\" and \"naive\".\n\n\n\n\n\n","category":"method"},{"location":"index.html#Introduction-1","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"(Image: ) (Image: Build Status) (Image: codecov)","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"PLEASE NOTE THIS IS PRE-RELEASE SOFTWARE FOR PREVIEW PURPOSES ONLY","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"THIS SOFTWARE IS SUBJECT TO BREAKING CHANGES AND NOT YET OFFICIALLY SUPPORTED","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"ITensors is a library for rapidly creating correct and efficient tensor network algorithms. ","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"An ITensor is a tensor whose interface  is independent of its memory layout. ITensor indices are objects which carry extra information and which 'recognize' each other (compare equal to each other).","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"The ITensor library also includes composable and extensible  algorithms for optimizing and transforming tensor networks, such as  matrix product state and matrix product operators, such as the DMRG algorithm.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"Development of ITensor is supported by the Flatiron Institute, a division of the Simons Foundation.","category":"page"},{"location":"index.html#Steps-to-Install-Pre-Release-Version-1","page":"Introduction","title":"Steps to Install Pre-Release Version","text":"","category":"section"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"Install the latest version of Julia: https://julialang.org/downloads/\nRun the julia command to begin an interactive Julia session (entering the so-called REPL). \nType ] on your keyboard to enter Julia's interactive package manager.\nRun the command \nadd https://github.com/ITensor/ITensors.jl\nThe package system will update itself, then install some dependencies before finally installing ITensors.jl.\nHit the backspace key to go back to the normal interactive Julia prompt, or type Ctrl+D to exit the Julia REPL.\nYou can now do using ITensors to use the ITensor library in an interactive session, or run Julia code files (.jl files) which use ITensor, with some examples given below and in our examples folder. The test folder also has many examples of ITensor code you can run.","category":"page"},{"location":"index.html#Code-Examples-1","page":"Introduction","title":"Code Examples","text":"","category":"section"},{"location":"index.html#Basic-Overview-1","page":"Introduction","title":"Basic Overview","text":"","category":"section"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"ITensor construction, setting of elements, contraction, and addition. Before constructing an ITensor, one constructs Index objects representing tensor indices.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"using ITensors\nlet\n  i = Index(3)\n  j = Index(5)\n  k = Index(2)\n  l = Index(7)\n\n  A = ITensor(i,j,k)\n  B = ITensor(j,l)\n\n  A[i=>1,j=>1,k=>1] = 11.1\n  A[i=>2,j=>1,k=>2] = -21.2\n  A[k=>1,i=>3,j=>1] = 31.1  # can provide Index values in any order\n  # ...\n\n  # A[k(1),i(3),j(1)] = 31.1  # alternative notation\n\n  # Contract over shared index j\n  C = A * B\n\n  @show hasinds(C,i,k,l) # = true\n\n  D = randomITensor(k,j,i) # ITensor with random elements\n\n  # Add two ITensors\n  # must have same set of indices\n  # but can be in any order\n  R = A + D\n\n  nothing\nend","category":"page"},{"location":"index.html#Singular-Value-Decomposition-(SVD)-of-a-Matrix-1","page":"Introduction","title":"Singular Value Decomposition (SVD) of a Matrix","text":"","category":"section"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"In this example, we create a random 10x20 matrix  and compute its SVD. The resulting factors can  be simply multiplied back together using the ITensor * operation, which automatically recognizes the matching indices between U and S, and between S and V and contracts (sums over) them.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"using ITensors\nlet\n  i = Index(10)           # index of dimension 10\n  j = Index(20)           # index of dimension 20\n  M = randomITensor(i,j)  # random matrix, indices i,j\n  U,S,V = svd(M,i)        # compute SVD with i as row index\n  @show M ≈ U*S*V         # = true\n\n  nothing\nend","category":"page"},{"location":"index.html#Singular-Value-Decomposition-(SVD)-of-a-Tensor-1","page":"Introduction","title":"Singular Value Decomposition (SVD) of a Tensor","text":"","category":"section"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"In this example, we create a random 4x4x4x4 tensor  and compute its SVD, temporarily treating the first and third indices (i and k) as the \"row\" index and the second and fourth indices (j and l) as the \"column\" index for the purposes of the SVD. The resulting factors can  be simply multiplied back together using the ITensor * operation, which automatically recognizes the matching indices between U and S, and between S and V and contracts (sums over) them.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"using ITensors\nlet\n  i = Index(4,\"i\")\n  j = Index(4,\"j\")\n  k = Index(4,\"k\")\n  l = Index(4,\"l\")\n  T = randomITensor(i,j,k,l)\n  U,S,V = svd(T,i,k)   # compute SVD with (i,k) as row indices (indices of U)\n  @show hasinds(U,i,k) # = true\n  @show hasinds(V,j,l) # = true\n  @show T ≈ U*S*V      # = true\n\n  nothing\nend\n\n# output\n\nhasinds(U,i,k) = true\nhasinds(V,j,l) = true\nM ≈ U * S * V = true","category":"page"},{"location":"index.html#Tensor-Indices:-Tags-and-Prime-Levels-1","page":"Introduction","title":"Tensor Indices: Tags and Prime Levels","text":"","category":"section"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"Before making an ITensor, you have to define its indices. Tensor Index objects carry extra information beyond just their dimension.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"All Index objects carry a permanent, immutable id number which is  determined when it is constructed, and allow it to be matched (compare equal) with copies of itself.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"Additionally, an Index can have up to four tag strings, and an integer primelevel. If two Index objects have different tags or  different prime levels, they do not compare equal even if they have the same id.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"Tags are also useful for identifying Index objects when printing tensors, and for performing certain Index manipulations (e.g. priming indices having certain sets of tags).","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"using ITensors\nlet\n  i = Index(3)     # Index of dimension 3\n  @show dim(i)     # = 3\n  @show id(i)      # = 0x5d28aa559dd13001 or similar\n\n  ci = copy(i)\n  @show ci == i    # = true\n\n  j = Index(5,\"j\") # Index with a tag \"j\"\n\n  @show j == i     # = false\n\n  s = Index(2,\"n=1,Site\") # Index with two tags,\n                          # \"Site\" and \"n=1\"\n  @show hastags(s,\"Site\") # = true\n  @show hastags(s,\"n=1\")  # = true\n\n  i1 = prime(i) # i1 has a \"prime level\" of 1\n                # but otherwise same properties as i\n  @show i1 == i # = false, prime levels do not match\n\n  nothing\nend","category":"page"},{"location":"index.html#DMRG-Calculation-1","page":"Introduction","title":"DMRG Calculation","text":"","category":"section"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"DMRG is an iterative algorithm for finding the dominant eigenvector of an exponentially large, Hermitian matrix. It originates in physics with the purpose of finding eigenvectors of Hamiltonian (energy) matrices which model the behavior of quantum systems.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"using ITensors\nlet\n  # Create 100 spin-one indices\n  N = 100\n  sites = siteinds(\"S=1\",N)\n\n  # Input operator terms which define \n  # a Hamiltonian matrix, and convert\n  # these terms to an MPO tensor network\n  # (here we make the 1D Heisenberg model)\n  ampo = AutoMPO()\n  for j=1:N-1\n    ampo +=     (\"Sz\",j,\"Sz\",j+1)\n    ampo += (0.5,\"S+\",j,\"S-\",j+1)\n    ampo += (0.5,\"S-\",j,\"S+\",j+1)\n  end\n  H = MPO(ampo,sites)\n\n  # Create an initial random matrix product state\n  psi0 = randomMPS(sites)\n\n  # Plan to do 5 passes or 'sweeps' of DMRG,\n  # setting maximum MPS internal dimensions \n  # for each sweep and maximum truncation cutoff\n  # used when adapting internal dimensions:\n  sweeps = Sweeps(5)\n  maxdim!(sweeps, 10,20,100,100,200)\n  cutoff!(sweeps, 1E-10)\n  @show sweeps\n\n  # Run the DMRG algorithm, returning energy \n  # (dominant eigenvalue) and optimized MPS\n  energy, psi = dmrg(H,psi0, sweeps)\n  println(\"Final energy = $energy\")\n\n  nothing\nend","category":"page"}]
}
