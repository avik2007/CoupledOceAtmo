import numpy as np

def gradU(uvel,vvel,dxg,dyg,dxc,dyc,dxf,dyf,dxv,dyu):
    """
    Calculates du/dx, du/dy,dv/dx,dv/dy
    at T points if u,v are given at U,V points
    in Arakawa C
    """

    dudx_xy = np.zeros(uvel.shape)
    dvdx_xy = np.zeros(uvel.shape)
    dudy_xy = np.zeros(uvel.shape)
    dvdy_xy = np.zeros(uvel.shape)
   
    dudx_xy[1:-1,1:-1] = (uvel[1:-1,1:-1] - uvel[1:-1,0:-2]) / dxg[1:-1,1:-1]
    dudy_xy[1:-1,1:-1] = 0.5 * ( (uvel[2:,1:-1] - uvel[0:-2,1:-1]) / (2*dyu[1:-1,1:-1]) + \
                                      (uvel[2:,0:-2] - uvel[0:-2,0:-2]) / (2*dyu[1:-1,1:-1]) )
    dvdx_xy[1:-1,1:-1] = 0.5 * ( (vvel[1:-1,2:] - vvel[1:-1,0:-2]) / (2*dxc[1:-1,1:-1]) + \
                                      (vvel[0:-2,2:] - vvel[0:-2,0:-2]) / (2*dxc[1:-1,1:-1]) )
    dvdy_xy[1:-1,1:-1] = (vvel[1:-1,1:-1] - vvel[0:-2,1:-1]) / dyf[1:-1,1:-1]

    data = {}
    data['dudx'] = dudx_xy
    data['dvdx'] = dvdx_xy
    data['dudy'] = dudy_xy
    data['dvdy'] = dvdy_xy
    return data


def gradT(T,dxc,dyc):
    """
    Calculates dt/dx and dtdy at T points
    using centered difference scheme
    """
    dtdx = np.zeros(T.shape)
    dtdy = np.zeros(T.shape)
    dtdx[1:-1,1:-1] = (T[1:-1,2:]-T[1:-1,0:-2])/(2*dxc[1:-1,1:-1])
    dtdy[1:-1,1:-1] = (T[2:,1:-1]-T[0:-2,1:-1])/(2*dyc[1:-1,1:-1])
    # Force to zero at boundaries
    dtdy[0,:] = 0.;dtdy[:,0] = 0.
    dtdy[-1,:]= 0.;dtdy[:,-1] = 0.
    dtdx[0,:]=0.;dtdx[:,0]=0.
    dtdx[:,-1]=0.;dtdx[-1,:]=0.
    data = {}
    data['dtdx'] = dtdx
    data['dtdy'] = dtdy
    return data

def lapl(M,dx,dy):
    """
    Calculates Laplacian of tracers at T points
    """
    mr, mc = M.shape
    D = np.zeros ((mr, mc))

    if (mc >= 3):
        ## x direction
        ## left and right boundary
        D[:, 0] = (M[:, 0] - 2 * M[:, 1] + M[:, 2]) / (dx[:,0] * dx[:,1])
        D[:, mc-1] = (M[:, mc - 3] - 2 * M[:, mc - 2] + M[:, mc-1]) \
            / (dx[:,mc - 3] * dx[:,mc - 2])

        ## interior points
        tmp1 = D[:, 1:mc - 1] 
        tmp2 = (M[:, 2:mc] - 2 * M[:, 1:mc - 1] + M[:, 0:mc - 2])
        tmp3 = (dx[:,0:mc -2] * dx[:,1:mc - 1])
        D[:, 1:mc - 1] = tmp1 + tmp2 / tmp3

    if (mr >= 3):
        ## y direction
        ## top and bottom boundary
        D[0, :] = D[0,:]  + \
            (M[0, :] - 2 * M[1, :] + M[2, :] ) / (dy[0,:] * dy[1,:])

        D[mr-1, :] = D[mr-1, :] \
            + (M[mr-3,:] - 2 * M[mr-2, :] + M[mr-1, :]) \
            / (dy[mr-3,:] * dx[:,mr-2])

        ## interior points
        tmp1 = D[1:mr-1, :] 
        tmp2 = (M[2:mr, :] - 2 * M[1:mr - 1, :] + M[0:mr-2, :])
        tmp3 = (dy[0:mr-2,:] * dy[1:mr-1,:])
        D[1:mr-1, :] = tmp1 + tmp2 / tmp3
    D = D/4
    # Force to zero at boundaries
    D[0,:]=0.;D[-1,:]=0.
    D[:,0]=0.;D[:,-1]=0.
    data = {} 
    data['Lapl'] = D  
    return data
    
