module params

#export outf, eigval_cutoff,minblocksize,maxblocksize,cutoff,maxdim
#export beta, Nt, U,ed_u,ed_d

const    outf="data.h5"
const    eigval_cutoff=1e-14
const    minblocksize=7
const    maxblocksize =20  #2 or 4(4 doesn't seem to work as well as it should)
const    maxdim=128 
const    cutoff = 0.0
const    beta =2.0 
const    Nt = 40 
const    U = 0.0
const    ed_u = -0.0
const    ed_d = -0.0
const	 D = 1.0
const	 gap = 0.0
const    nu = 1.0
const    e_c = 5.0
const    V = 1.0
const    mu = 0.0
const	 boundary = 1.0
const    ph = 1
const	 shift = 0
const 	 alpha = 1.0
const	 T_ren =1 
const	 save_psi = false
const    save_B   = false
const    save_c   = false
function DOS(omega)
	res=0.0
	if -D<=omega-mu<-gap
		res=V
	elseif gap < omega-mu <= D
		res=V
	end
	return res
end
function domain()
	return [-D-gap,D+gap]
end
end
