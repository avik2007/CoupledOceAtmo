class qg_decomp:

    """
    Vertical normal modes in Quasi-geostrophic theory
    """

    def __init__(self,n2 =0.,z=0,fo=1e-4):
        
        import numpy as np
        self.n2 = n2
        self.z = z
        self.fo = fo

        self.precondition()

    def precondition(self):
        import numpy as np
        from numpy import r_,diff,diag,matrix

        " --- check N2 "
        if self.n2.min() < 0:
            print "Error: negative values in N2"
            return
        if max(self.z) == 0:
            print("Error: do not place first N2 at z = 0")
            return
        else:
            self.nz = len(self.n2)

        z = self.z 
        nz = self.nz


        zf = r_[z[0]-(z[1]-z[0]), z, z[-1]+(z[-1]-z[-2])]
        self.zf = zf

        def mp(a,b):
            return matrix(a)*matrix(b)

        def twopave(x):
            return (x[0:-1]+x[1:])/2

        zc = r_[zf[0]-0.5*(zf[1]-zf[0]), twopave(zf), zf[-1]+0.5*(zf[-1]-zf[-2])] #Nz+3 \psi points]
        dzc = diff(zc)
        dzf = diff(zf)
        self.zc = zc
        N2 = self.fo**2 / self.n2

        A = np.zeros((nz-1,nz-1))
        for k in range(1,nz-2):
            A[k, k - 1] = - N2[k - 1] / (dzc[k - 1] * dzf[k])
            A[k, k] =  N2[k - 1] / (dzc[k - 1] * dzf[k]) \
                   + N2[k] / (dzc[k] * dzf[k])
            A[k, k + 1] = - N2[k] / (dzc[k] * dzf[k])

        A[0,0] = N2[1] / dzc[1] / dzf[1]
        A[0,1] = -A[0,0]
        A[-1,-1] = N2[-2] / dzc[-2] / dzf[-2]
        A[-1,-2] = -A[-1,-1]

        w,vi = np.linalg.eig(A)
        vi = vi[:,np.argsort(w)]
        vi = r_[vi[0:1,:],vi,vi[-1:,:]]
        self.w = np.sort(w) ## eigenvalue
        self.radii = 1./np.sqrt(np.abs(self.w))
        self.Rd = (np.abs(w))**-0.5 / np.absolute(self.fo[0])

        #### normalization:
        vi =vi- (-vi*dzf.reshape(-1,1)).sum(axis=0).reshape(1,-1)/np.abs(zf[-1])
        vi = vi/(-vi**2*dzf.reshape(-1,1)).sum(axis=0).reshape(1,-1)**0.5
        self.vi = vi*np.sign(vi[0:1,:])


        ### eigenfunction for W
	#gi = np.zeros((nz+2,nz))
        print('++++++++++ gi vi ++++++++')

        gi=((vi[1:-2,:]*dzf[1:-2].reshape(-1,1)).cumsum(axis=0)) #
        gi[0,:] = 0
        gi[-1,:] = 0
        self.gi = gi
            
