import sys
import gc
import warnings
import math
import numpy as np
import numpy.fft
from repmat import repmat
import fld_tools as ft
from compute_u_T import *

# Computes horizontal spectral energy flux, assuming nonunform regular coordinates with a finite volume formulation
# a tranfer-function approach is used that assumes energy is conserved, even though it is not, strictly speaking.
    
def compute_u_hp_FF_T_diag(U,V,W,Up,Vp,Wp,dx,dy,dz,post_taper=np.array([0,0,0]),trim_ml=0,prd=np.array([1,1,1])):
    (dU, dV,_,_,_,_) = compute_u_hp_FF_T(U,V,W,Up,Vp,Wp,dx,dy,dz,np.array([0,0,0]),trim_ml,prd);
    return (dU,dV);

def compute_u_hp_FF_T_k(U,V,W,Up,Vp,Wp,dx,dy,dz,post_taper=np.array([0,0,0]),trim_ml=0,prd=np.array([1,1,1])):
    (dhU,dhV,hU,hV,hUdivV,hVdivV) = compute_u_hp_FF_T(U,V,W,Up,Vp,Wp,dx,dy,dz,post_taper,trim_ml,prd);
    (T,k_out) = compute_u_T_corr_k(dhU,dhV,hU,hV,hUdivV,hVdivV,dx,dy,dz,post_taper,trim_ml,prd);
    return (T,k_out);

def compute_u_hp_FF_T_m(U,V,W,Up,Vp,Wp,dx,dy,dz,post_taper=np.array([0,0,0]),trim_ml=0,prd=np.array([1,1,1])):
    (dhU,dhV,hU,hV,hUdivV,hVdivV) = compute_u_hp_FF_T(U,V,W,Up,Vp,Wp,dx,dy,dz,post_taper,trim_ml,prd);
    (T,m_out) = compute_u_T_corr_m(dhU,dhV,hU,hV,hUdivV,hVdivV,dx,dy,dz,post_taper,trim_ml,prd);
                                                   
    return (T,m_out);



def compute_u_hp_FF_T(U,V,W,Up,Vp,Wp,dx,dy,dz,post_taper=np.array([0,0,0]),trim_ml=0,prd=np.array([1,1,1])):

    print ('in compute_u_hp_FF_T()')

    # Initialize Constants
    (nx, ny, nz, nt) = U.shape;
    rho0 = 1027.5; # for energy output (optional)
    
#     if (post_taper[2]):
#         raise RuntimeError('post_taper not built to handle vertical filt. of dKEdiv');

    # Add ghost cells to fields to handle periodicity in position space.
    ng = 2; # number of ghost cells, must be one larger than needed for python indexing
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits
    U = ft.pad_field_3D(U,ng,prd); V = ft.pad_field_3D(V,ng,prd);  W = ft.pad_field_3D(W,ng,prd);
    Up = ft.pad_field_3D(Up,ng,prd); Vp = ft.pad_field_3D(Vp,ng,prd);  Wp = ft.pad_field_3D(Wp,ng,prd);

    (hU,hV,_) = ft.taper_filter_3D_uvw_u(U[gx0:gxn,gy0:gyn,gz0:gzn,:],V[gx0:gxn,gy0:gyn,gz0:gzn,:],
                                  W[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper);
    (hUp,hVp,_) = ft.taper_filter_3D_uvw_u(Up[gx0:gxn,gy0:gyn,gz0:gzn,:],Vp[gx0:gxn,gy0:gyn,gz0:gzn,:],
                                  Wp[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper);

    hU = ft.pad_field_3D(hU,ng,prd); hV = ft.pad_field_3D(hV,ng,prd); 
    hUp = ft.pad_field_3D(hUp,ng,prd); hVp = ft.pad_field_3D(hVp,ng,prd); 

    # Compute divergence
    DivVcc = ((U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]-U[gx0:gxn,gy0:gyn,gz0:gzn,:])/dx
              + (V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]-V[gx0:gxn,gy0:gyn,gz0:gzn,:])/dy
              + (W[gx0:gxn,gy0:gyn,gz0:gzn,:]-W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])/dz);

    

    # initialize tendencies
    dhU = np.zeros((nx,ny,nz,nt)); dhV = np.zeros((nx,ny,nz,nt));
    
    # Compute Up advective tendency
    # dUU/dx
    dhU = dhU + 0.25*((U[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+U[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUp[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+hUp[gx0:gxn,gy0:gyn,gz0:gzn,:])
        - (U[gx0:gxn,gy0:gyn,gz0:gzn,:]+U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hUp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUp[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]))/dx;
    # dVU/dy
    dhU = dhU + 0.25*((V[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+V[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUp[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:])
        - (V[gx0-1:gxn-1,gy0+1:gyn+1,gz0:gzn,:]+V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hUp[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hUp[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dy;
    # dWU/dz
    dhU = dhU + 0.25*((W[gx0-1:gxn-1,gy0:gyn,gz0+1:gzn+1,:]+W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hUp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUp[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
        - (W[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+W[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUp[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hUp[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dz;
    
    # Compute Vp advective tendency
    # dVV/dy
    dhV = dhV + 0.25*((V[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+V[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVp[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+hVp[gx0:gxn,gy0:gyn,gz0:gzn,:])
        - (V[gx0:gxn,gy0:gyn,gz0:gzn,:]+V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hVp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVp[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]))/dy;
    # dUV/dx
    dhV = dhV + 0.25*((U[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+U[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVp[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:])
        - (U[gx0+1:gxn+1,gy0-1:gyn-1,gz0:gzn,:]+U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hVp[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hVp[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dx;
    # dWV/dz
    dhV = dhV + 0.25*((W[gx0:gxn,gy0-1:gyn-1,gz0+1:gzn+1,:]+W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hVp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVp[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
        - (W[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+W[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVp[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hVp[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dz;

    # Compute KE divergence correction (quick tapered version)...
    dKEdivhU = 0.5*(hUp[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hUp[gx0:gxn,gy0:gyn,gz0:gzn,:])*DivVcc;
    dKEdivhV = 0.5*(hVp[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hVp[gx0:gxn,gy0:gyn,gz0:gzn,:])*DivVcc;

    hU = hU[gx0:gxn,gy0:gyn,gz0:gzn,:];
    hV = hV[gx0:gxn,gy0:gyn,gz0:gzn,:];

    return (dhU,dhV,hU,hV,dKEdivhU,dKEdivhV)


    
