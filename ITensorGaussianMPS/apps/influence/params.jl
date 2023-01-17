module params

#export outf, eigval_cutoff,minblocksize,maxblocksize,cutoff,maxdim
#export beta, Nt, U,ed_u,ed_d

const    outf="data.h5"
const    eigval_cutoff=1e-11
const    minblocksize=8
const    maxblocksize =8  #2 or 4(4 doesn't seem to work as well as it should)
const    maxdim = 128 
const    cutoff = 0.0
const    beta =2.0 
const    Nt = 40
const    U = 0.0
const    ed_u = 0.0
const    ed_d = 0.0
const	 D = 1.0
const	 gap = 0.2
const    V = 1.0
const	 boundary = 1.0
function DOS(omega)
	res=0.0
	if -D<=omega<gap
		res=V
	elseif gap < omega <= D
		res=V
	end
	return res
end


end
