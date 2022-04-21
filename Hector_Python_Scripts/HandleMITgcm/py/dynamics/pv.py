# -*- coding: utf-8 -*-
"""
Created on May 11 10:17

@author: htorresg
"""
"""
   Ertel Potential vorticity
   This script is created to handle the big-data size
   from any Ultra-high resulution.

   JPL/CALTECH:
   Dr. Hector S. Torres
"""

def c(u_up,v_up,b_up,u_lw,v_lw,b_lw,N2,dz,dx,dy,f):
    import numpy as np
    """
    Input:
    variable_up := upper layer variable
    variable_lw := lower layer variable
    u,v := horizontal velocity components
    v := buoyancy
    dz := layer thicknes
    dx,dy := spatial interval

    Output:
    norm(pseudo-vorticity) := sqrt(dudz**2 + dvdz**2)
    vorticity := located at the center cell
    sqrt(bx,by) := lateral buoyancy gradients
    pv := potential vorticity centered at the cell
    """

    #=========== vorticity ==============
    # upper layer vorticity
    uy_up,ux_up = np.gradient(u_up,dx,dy)
    vy_up,vx_up = np.gradient(v_up,dx,dy)
    vort_up = vx_up - uy_up

    # lower layer vorticity
    uy_lw,ux_lw = np.gradient(u_lw,dx,dy)
    vy_lw,vx_lw = np.gradient(v_up,dx,dy)
    vort_lw = vx_lw - uy_lw

    # vorticity at mid-cell
    vort = (vort_up + vort_lw)*0.5

    #========= pseudo-vorticity ========
    dudz = (u_up - u_lw)/dz
    dvdz = (v_up - v_lw)/dz

    #======== Lateral buoyancy gradients ====
    by_up,bx_up = np.gradient(b_up,dx,dy)
    by_lw,bx_lw = np.gradient(b_lw,dx,dy)
    bx = (bx_up + bx_lw)*0.5
    by = (by_up + bx_lw)*0.5

    # ===== potential vorticity =======
    A = (f + vort)*N2
    B = -dvdz*bx
    C = dudz*by
    pv = A + B + C
    return A,B,C,pv
