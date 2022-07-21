def raw_interp_function(data_left,data_right):
    return 0.5*(data_left+data_right)

def raw_diff_function(data_left,data_right):
    return data_right - data_left

def raw_interp_rv2tracer_points(rv):
    """
    Interpolation from vorticity points to tracer points
    or cell center point.
    """
    import numpy as np
    if rv.ndim == 3:
        rvi = np.empty(rv.shape)
        rvi[:,:,:-1] = raw_interp_function(rv[:,:,:-1],rv[:,:,1:])
        rvi[:,:-1,:] = raw_interp_function(rv[:,:-1,:],rv[:,1:,:])
        rvi[:,:,-1] = rvi[:,:,-2]
        rvi[:,-1,:] = rvi[:,-2,:]
    if rv.ndim == 2:
        rvi = np.empty(rv.shape)
        rvi[:,:-1] = raw_interp_function(rv[:,:-1],rv[:,1:])
        rvi[:-1,:] = raw_interp_function(rv[:-1,:],rv[1:,:])
        rvi[:,-1] = rvi[:,-2]
        rvi[-1,:] = rvi[-2,:]    
    return rvi

def gradvel(u,v,dxc,dyc,dxg,dyg,hFacW,hFacS,rac,raz):
    """
    Gradient of vector velocity in flux-form as in MITgcm
    """
    import numpy as np
    if u.ndim == 2:
        """
        Ux and Vy
        """
        ut = u*dyg*hFacW
        vt = v*dxg*hFacS
        ## differentiation ###
        ut = raw_diff_function(ut[:,:-1],ut[:,1:])
        vt = raw_diff_function(vt[:-1,:],vt[1:,:])
        ## at tracer-points (center of the cell)
        utx = np.empty(u.shape)
        utx[:-1,:-1] = raw_interp_function(ut[:-1,:],ut[1:,:])
        utx[-1,:] = utx[-2,:]
        utx[:,-1] = utx[:,-2]
        utx = utx/rac
        vty = np.empty(u.shape)
        vty[:-1,:-1] = raw_interp_function(vt[:,:-1],vt[:,1:])
        vty[-1,:] = vty[-2,:]
        vty[:,-1] = vty[:,-2]
        vty = vty/rac
        del ut,vt
        ############
        ########
        """
        Uy and Vx
        """
        ### Circulation theorem
        uc = u*dxc*hFacW
        vc = v*dyc*hFacS
        ### differentiation ###
        uc = raw_diff_function(uc[:-1,:],uc[1:,:])
        vc = raw_diff_function(vc[:,:-1],vc[:,1:])
        #### at tracer-points (center of the cell)
        ucy = np.empty(u.shape)
        ucy[:-1,:-1] = raw_interp_function(uc[:,:-1],uc[:,1:])
        ucy[-1,:] = ucy[-2,:]
        ucy[:,-1] = ucy[:,-2]
        ucy = ucy/raz
        vcx = np.empty(u.shape)
        vcx[:-1,:-1] = raw_interp_function(vc[:-1,:],vc[1:,:])
        vcx[-1,:] = vcx[-2,:]
        vcx[:,-1] = vcx[:,-2]
        vcx = vcx/raz
        ucy = raw_interp_rv2tracer_points(ucy)
        vcx = raw_interp_rv2tracer_points(vcx)
        ####
    elif u.ndim ==3:
        """
        Ux and Vy
        """
        ut = u*dyg*hFacW
        vt = v*dxg*hFacS
        ## differentiation ###
        ut = raw_diff_function(ut[:,:,:-1],ut[:,:,1:])
        print('u.shape ',u.shape)
        print('du.shape ',ut.shape)
        vt = raw_diff_function(vt[:,:-1,:],vt[:,1:,:])
        ## at tracer-points (center of the cell)
        utx = np.empty(u.shape)
        utx[:,:-1,:-1] = raw_interp_function(ut[:,:-1,:],ut[:,1:,:])
        utx[:,-1,:] = utx[:,-2,:]
        utx[:,:,-1] = utx[:,:,-2]
        utx = utx/rac
        vty = np.empty(u.shape)
        vty[:,:-1,:-1] = raw_interp_function(vt[:,:,:-1],vt[:,:,1:])
        vty[:,-1,:] = vty[:,-2,:]
        vty[:,:,-1] = vty[:,:,-2]
        vty = vty/rac
        del ut,vt
        ############
        ########
        """
        Uy and Vx
        """
        ### Circulation theorem
        uc = u*dxc*hFacW
        vc = v*dyc*hFacS
        ### differentiation ###
	## u_{y}
        uc = raw_diff_function(uc[:,:-1,:],uc[:,1:,:])
	## v_{x}
        vc = raw_diff_function(vc[:,:,:-1],vc[:,:,1:])
        #### at tracer-points (center of the cell)
        ucy = np.empty(u.shape)
        ucy[:,:-1,:-1] = raw_interp_function(uc[:,:,:-1],uc[:,:,1:])
        ucy[:,-1,:] = ucy[:,-2,:]
        ucy[:,:,-1] = ucy[:,:,-2]
        vcx = np.empty(u.shape)
        vcx[:,:-1,:-1] = raw_interp_function(vc[:,:-1,:],vc[:,1:,:])
        vcx[:,-1,:] = vcx[:,-2,:]
        vcx[:,:,-1] = vcx[:,:,-2]
        vcx = vcx/raz
        ucy = ucy/raz
        ucy = raw_interp_rv2tracer_points(ucy)
        vcx = raw_interp_rv2tracer_points(vcx)
        ####
        del uc,vc
    return utx,ucy,vcx,vty
   
def gradT(T,dxc,dyc):
    """
    Gradient of tracers
    """	
    import numpy as np 
    if T.ndim == 3:
    	#### T_{x} ####
        dt = raw_diff_function(T[:,:,:-1],T[:,:,1:])
        DTx = np.empty(T.shape)
        print('T.shape ',T.shape)
        DTx[:,:,1:-1] = raw_interp_function(dt[:,:,:-1],dt[:,:,1:])
        DTx[:,:,0] = DTx[:,:,1]
        DTx[:,:,-1] = DTx[:,:,-2]
        DTx = DTx/dxc[None,:,:]
        ####
	##### T_{y} ####
        del dt
        dt = raw_diff_function(T[:,:-1,:],T[:,1:,:])
        DTy = np.empty(T.shape)
        print('Ty.shape ',dt.shape)
        DTy[:,1:-1,:] = raw_interp_function(dt[:,:-1,:],dt[:,1:,:])
        DTy[:,0,:] = DTy[:,1,:]
        DTy[:,-1,:] = DTy[:,-2,:]
        DTy = DTy/dyc[None,:,:]
        del dt
    return DTx,DTy

def LapT(T,dxc,dyc,rac):
    """
    Laplacian of tracer
    """

	
