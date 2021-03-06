# -*- coding: utf-8 -*-
"""
Created on May 11 10:17

@author: htorresg
"""
"""
   This script is created in order to calculate
   the Frontogenetic vector (Q) in MITgcm.
       # At the end, each term of the equation will be co-located
       # at the intersection (*) of each numerical cell.
       #
       # -----v------------v------
       # |          |            |
       # |          |            |
       # u   rho    u     rho    u
       # |          |            |
       # |          |            |
       # -----v---- * ------v-----
       # |          |            |
       # |          |            |
       # u    rho   u     rho    u
       # |          |            |
       # |          |            |
       # -----v----------v--------

   JPL/CALTECH
   Dr. Hector S Torres
"""
def coriolis(lat):
    import numpy as np
    omg = (1.)/(86400.)
    return 2*(omg)*np.sin((lat*3.141519)/180)

def u2rho(u):
    if u.ndim == 3:
        ur = 
        u = 0.5*(u[:,1:,:]+u[:,:-1,:])
    return u

def v2rho(v):
    v = 0.5*(v[1:,:,:]+v[:-1,:,:])
    return v

def dTdX(var,dx,dy):
    """Forward """
    dvardx = (var[:,1:,:]-var[:,:-1,:])/(0.5*(dx[:,1:,:]+dx[:,:-1,:]))
    dvardy = (var[1:,:,:]-var[:-1,:,:])/(0.5*(dy[1:,:,:]+dy[:-1,:,:]))
    return dvardx,dvardy

def DVDZ(var,dz):
    # Vertical shear
    dv = var[:,:,0:-1] - var[:,:,1:]
    thkmd = 0.5*(dz[1:] + dz[0:-1])
    dvdz = dv/thkmd[None,None,:]
    return dvdz

def dT2Dx2(var,dx,dy,lon,lat):
    import numpy as np
    # Second-order derivate
    if var.ndim == 3:
        lonmd = (0.5*(lon[2:,2:]+lon[:-2,:-2]))
        latmd = (0.5*(lon[2:,2:]+lat[:-2,:-2]))
        dxmd = (0.5*(dx[:,2:]+dx[:,:-2]))
        dymd = (0.5*(dy[2:,:]+dy[:-2,:]))
        print(dxmd.shape)
        dvar2dx2 = np.diff(var,n=2,axis=2)/dxmd[None,:,:]**2
        print(dvar2dx2.shape)
        dvar2dy2 = np.diff(var,n=2,axis=1)/dymd[None,:,:]**2
        Lap = 0.5*(dvar2dx2[:,2:,:]+dvar2dx2[:,:-2,:])+0.5*(dvar2dy2[:,:,2:]+dvar2dy2[:,:,:-2])
    return Lap,lonmd,latmd

def Q(u,v,rho,dxc,dyc,thk,f):
    import numpy as np
    """
    (u,v) := horizontal velocity components
    rho := potential density
    (dx%,dy%) := spatial intervals
    thk := thickness
    f := Coriolis parameter
    """
    g = 9.81 # gravity's acceleration
    #
    # ----- collocation of U and V at rho-points ----
    u = u2rho(u)
    v = v2rho(v)
    rho = rho[1:,1:,:]
    u = u[1:,:,:]
    v = v[:,1:,:]
    dxc = dxc[1:,1:,:]
    dyc = dyc[1:,1:,:]
    f = f[1:,1:]
    #::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::: Horizontal derivatives :::::::::::
    drhodx,drhody = dTdX(rho,dxc,dyc)
    dudx,dudy    = dTdX(u,dxc,dyc)
    dvdx,dvdy    = dTdX(v,dxc,dyc)
    #::::::::::::::::::::::::::::::::::::::::::::

    # ======== Colocation at the intersection * =======
    drhodx_mid = 0.5*(drhodx[1:,:,:]+drhodx[:-1,:,:])
    drhody_mid = 0.5*(drhody[:,1:,:]+drhody[:,:-1,:])
    dudx_mid = 0.5*(dudx[1:,:,:]+dudx[:-1,:,:])
    dudy_mid = 0.5*(dudy[:,1:,:]+dudy[:,:-1,:])
    dvdx_mid = 0.5*(dvdx[1:,:,:]+dudx[:-1,:,:])
    dvdy_mid = 0.5*(dvdy[:,1:,:]+dudy[:,:-1,:])
    # =================================================

    # ======== Vertical shear =================
    thk = thk.squeeze()
    dudz = DVDZ(u,thk)
    dvdz = DVDZ(v,thk)
    dudzmd = 0.5*(dudz[1:,1:,:] + dudz[:-1,:-1,:])
    dvdzmd = 0.5*(dvdz[1:,1:,:] + dvdz[:-1,:-1,:])
    drhodx_md = 0.5*(drhodx_mid[:,:,:-1]+drhodx_mid[:,:,1:])
    drhody_md = 0.5*(drhody_mid[:,:,:-1]+drhody_mid[:,:,1:])
    # ========================

    # ========== ============
    rho = rho[:-1,:-1,:]
    factor = g/rho
    # -----------------------------------------

    # :::::::::::: Thermal-wind imbalance ::::::::
    f_md = 0.5*(f[1:,1:] + f[:-1,:-1])
    # x:
    dUadz=f_md[:,:,None]*dudzmd-0.5*(factor[:,:,:-1]+factor[:,:,1:])*drhody_md
    # y:
    dVadz=f_md[:,:,None]*dvdzmd+0.5*(factor[:,:,:-1]+factor[:,:,1:])*drhodx_md
    dUadz = dUadz/f_md[:,:,None]
    dVadz = dVadz/f_md[:,:,None]
    print('Thermal-wind def')
    # ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    # ::::::::::: Thermal-wind deformation :::::::::
    twi_x = f_md[:,:,None]*(0.5*(dvdx_mid[:,:,1:]+dvdx_mid[:,:,:-1])*dUadz)-f_md[:,:,None]*(0.5*(dudx_mid[:,:,1:]+dudx_mid[:,:,:-1])*dVadz)
    twi_y = f_md[:,:,None]*(0.5*(dvdy_mid[:,:,1:]+dvdy_mid[:,:,:-1])*dUadz)-f_md[:,:,None]*(0.5*(dudy_mid[:,:,1:]+dudy_mid[:,:,:-1])*dVadz)
    print('Thermal-wind def')
    print(twi_x.shape,twi_y.shape)
    # :::::::::::::::::::::::::::::::::::::::::::::

    # :::::::::::::: Kinematic deformation :::::::::::::::::
    Qx = factor*(drhodx_mid*dudx_mid + dvdx_mid*drhody_mid)
    Qy = factor*(dvdy_mid*drhody_mid + dudy_mid*drhodx_mid)
    # colocated at mid-cell
    Qx = 0.5*(Qx[:,:,1:]+Qx[:,:,:-1])
    Qy = 0.5*(Qy[:,:,1:]+Qy[:,:,:-1])
    print('Kinema def')
    print(Qx.shape,Qy.shape)
    # :::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    #
    drhodx_mid = 0.5*(drhodx_mid[:,:,1:]+drhodx_mid[:,:,:-1])
    drhody_mid = 0.5*(drhody_mid[:,:,1:]+drhody_mid[:,:,:-1])
    print('grad(rho)')
    print(drhodx_mid.shape,drhody_mid.shape)
    #
    return Qx,Qy,drhodx_mid,drhody_mid,twi_x,twi_y,dUadz,dVadz
