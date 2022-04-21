# -*- coding: utf-8 -*-
"""
Created on May 11 10:17

@author: htorresg
"""
"""
   Ertel Potential vorticity
   This script is created to handle the big-data size
   from any Ultra-high resulution simulation.

   JPL/CALTECH:
   Hector S. Torres
"""

def c(u,v,b,w,N2,dz,dx,dy,f):
    import numpy as np
    """
    Input:
    u,v,w :=  velocity components
    b := buoyancy
    N2 := Brunt-Vaissalla frequency
    dz := layer thicknes
    dx,dy := spatial interval

    Output:
    norm(pseudo-vorticity) := sqrt(dudz**2 + dvdz**2)
    vorticity := located at the center cell
    sqrt(bx,by) := lateral buoyancy gradients
    pv := potential vorticity centered at the cell
    |Grad(b)| := magnitude of grad(b)
    """

    #=========== vorticity ==============
    # vorticity
    ux = np.gradient(u,dx,axis=2)
    uy = np.gradient(u,dy,axis=1)
    vx = np.gradient(v,dx,axis=2)
    vy = np.gradient(v,dy,axis=1)
    vort = vx - uy
    # Location at mid-cell z-axis
    vort_md = (vort[:-1,:,:]+vort[1:,:,:])*0.5

    #========= pseudo-vorticity ========
    dudz = (u[:-1,:,:] - u[1:,:,:])/dz
    dvdz = (v[:-1,:,:] - v[1:,:,:])/dz
    wx = np.gradient(w,dx,axis=2)
    wy = np.gradient(w,dy,axis=1)
    wx_md = (wx[:-1,:,:]+wx[1:,:,:])*0.5
    wy_md = (wy[:-1,:,:]+wy[1:,:,:])*0.5
    PsVort_x =  wx_md - dvdz
    PsVort_y = dudz - wy_md

    #======== Lateral buoyancy gradients ====
    bx = np.gradient(b,dx,axis=2)
    by = np.gradient(b,dy,axis=1)
    by_md = (by[:-1,:,:]+by[1:,:,:])*0.5
    bx_md = (bx[:-1,:,:]+bx[1:,:,:])*0.5
    mag_gradb = np.sqrt(by_md**2 + bx_md**2)

    # ===== potential vorticity =======
    A = (f + vort_md)*N2
    B = PsVort_x*bx_md
    C = PsVort_y*by_md
    pv = A + B + C
    return A,B,C,pv,mag_gradb,vort_md
