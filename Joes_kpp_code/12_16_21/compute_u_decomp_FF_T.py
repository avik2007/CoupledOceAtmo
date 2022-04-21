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
    
def compute_u_decomp_FF_T_diag(U,V,W,U_bp,V_bp,U_hp,V_hp,dx,dy,dz,post_taper=np.array([0,0,0]),trim_ml=0,prd=np.array([1,1,1])):
    (dUl,dVl,dUb,dVb,dUh,dVh,_,_,_,_,_,_,UdivVl,VdivVl,UdivVb,VdivVb,UdivVh,VdivVh) = compute_u_decomp_FF_T(U,V,W,U_bp,V_bp,U_hp,V_hp,dx,dy,dz,np.array([0,0,0]),trim_ml,prd);
    return (dUl,dVl,dUb,dVb,dUh,dVh,UdivVl,VdivVl,UdivVb,VdivVb,UdivVh,VdivVh);

def compute_u_decomp_FF_T_m(U,V,W,U_bp,V_bp,U_hp,V_hp,dx,dy,dz,post_taper=np.array([0,0,0]),trim_ml=0,prd=np.array([1,1,1])):
    (dhUl,dhVl,dhUb,dhVb,dhUh,dhVh,hUl,hVl,hUb,hVb,hUh,hVh,hUdivVl,hVdivVl,hUdivVb,hVdivVb,hUdivVh,hVdivVh) = compute_u_decomp_FF_T(U,V,W,U_bp,V_bp,U_hp,V_hp,dx,dy,dz,post_taper,trim_ml,prd);

    nz = W.shape[2];
    mti_max = math.floor(nz/2-1); 
    T = np.zeros((mti_max,9));
    T_KEdivV = np.zeros((mti_max,9));

    (Tl_lp,m_out) = compute_u_T_corr_m(dhUl,dhVl,hUl,hVl,hUdivVl,hVdivVl,dx,dy,dz,post_taper,trim_ml,prd);
    (Tl_bp,m_out) = compute_u_T_corr_m(dhUl,dhVl,hUb,hVb,hUdivVl,hVdivVl,dx,dy,dz,post_taper,trim_ml,prd);
    (Tl_hp,m_out) = compute_u_T_corr_m(dhUl,dhVl,hUh,hVh,hUdivVl,hVdivVl,dx,dy,dz,post_taper,trim_ml,prd);
    (Tb_lp,m_out) = compute_u_T_corr_m(dhUb,dhVb,hUl,hVl,hUdivVb,hVdivVb,dx,dy,dz,post_taper,trim_ml,prd);
    (Tb_bp,m_out) = compute_u_T_corr_m(dhUb,dhVb,hUb,hVb,hUdivVb,hVdivVb,dx,dy,dz,post_taper,trim_ml,prd);
    (Tb_hp,m_out) = compute_u_T_corr_m(dhUb,dhVb,hUh,hVh,hUdivVb,hVdivVb,dx,dy,dz,post_taper,trim_ml,prd);
    (Th_lp,m_out) = compute_u_T_corr_m(dhUh,dhVh,hUl,hVl,hUdivVh,hVdivVh,dx,dy,dz,post_taper,trim_ml,prd);
    (Th_bp,m_out) = compute_u_T_corr_m(dhUh,dhVh,hUb,hVb,hUdivVh,hVdivVh,dx,dy,dz,post_taper,trim_ml,prd);
    (Th_hp,m_out) = compute_u_T_corr_m(dhUh,dhVh,hUh,hVh,hUdivVh,hVdivVh,dx,dy,dz,post_taper,trim_ml,prd);

    T[:,0] = Tl_lp[:,0]; T_KEdivV[:,0] = Tl_lp[:,1];
    T[:,1] = Tb_lp[:,0]; T_KEdivV[:,1] = Tb_lp[:,1];
    T[:,2] = Th_lp[:,0]; T_KEdivV[:,2] = Th_lp[:,1];
    T[:,3] = Tl_bp[:,0]; T_KEdivV[:,3] = Tl_bp[:,1];
    T[:,4] = Tb_bp[:,0]; T_KEdivV[:,4] = Tb_bp[:,1];
    T[:,5] = Th_bp[:,0]; T_KEdivV[:,5] = Th_bp[:,1];
    T[:,6] = Tl_hp[:,0]; T_KEdivV[:,6] = Tl_hp[:,1];
    T[:,7] = Tb_hp[:,0]; T_KEdivV[:,7] = Tb_hp[:,1];
    T[:,8] = Th_hp[:,0]; T_KEdivV[:,8] = Th_hp[:,1];
                
    return (T,T_KEdivV,m_out);

def compute_u_decomp_FF_T(U,V,W,U_bp,V_bp,U_hp,V_hp,dx,dy,dz,post_taper=np.array([0,0,0]),trim_ml=0,prd=np.array([1,1,1])):

    print ('in compute_u_decomp_FF_T()')

    # Initialize Constants
    (nx, ny, nz, nt) = U.shape;
    rho0 = 1027.5; # for energy output (optional)
    
    # Add ghost cells to fields to handle periodicity in position space.
    ng = 2; # number of ghost cells, must be one larger than needed for python indexing
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits
    U = ft.pad_field_3D(U,ng,prd); V = ft.pad_field_3D(V,ng,prd);  W = ft.pad_field_3D(W,ng,prd);
    U_bp = ft.pad_field_3D(U_bp,ng,prd); V_bp = ft.pad_field_3D(V_bp,ng,prd);  
    U_hp = ft.pad_field_3D(U_hp,ng,prd); V_hp = ft.pad_field_3D(V_hp,ng,prd);  

    (hU,hV,_) = ft.taper_filter_3D_uvw_u(U[gx0:gxn,gy0:gyn,gz0:gzn,:],V[gx0:gxn,gy0:gyn,gz0:gzn,:],
                                  W[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper);
    (hU_bp,hV_bp,_) = ft.taper_filter_3D_uvw_u(U_bp[gx0:gxn,gy0:gyn,gz0:gzn,:],V_bp[gx0:gxn,gy0:gyn,gz0:gzn,:],
                                  W[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper);
    (hU_hp,hV_hp,_) = ft.taper_filter_3D_uvw_u(U_hp[gx0:gxn,gy0:gyn,gz0:gzn,:],V_hp[gx0:gxn,gy0:gyn,gz0:gzn,:],
                                  W[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper);

    hU = ft.pad_field_3D(hU,ng,prd); hV = ft.pad_field_3D(hV,ng,prd); 
    hU_bp = ft.pad_field_3D(hU_bp,ng,prd); hV_bp = ft.pad_field_3D(hV_bp,ng,prd); 
    hU_hp = ft.pad_field_3D(hU_hp,ng,prd); hV_hp = ft.pad_field_3D(hV_hp,ng,prd); 

    # Compute divergence
    DivVcc = ((U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]-U[gx0:gxn,gy0:gyn,gz0:gzn,:])/dx
              + (V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]-V[gx0:gxn,gy0:gyn,gz0:gzn,:])/dy
              + (W[gx0:gxn,gy0:gyn,gz0:gzn,:]-W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])/dz);

    for icase in range(0,3):
        if (icase==0):
            hU_c = hU-hU_bp-hU_hp; hV_c = hV-hV_bp-hV_hp;
        elif (icase==1):
            hU_c = hU_bp; hV_c = hV_bp;
        else:
            hU_c = hU_hp; hV_c = hV_hp;

        # initialize tendencies
        dhU = np.zeros((nx,ny,nz,nt)); dhV = np.zeros((nx,ny,nz,nt));
            
        # Computne Up advective tendency
        # dUU/dx
        dhU = dhU + 0.25*((U[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+U[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hU_c[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+hU_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                          - (U[gx0:gxn,gy0:gyn,gz0:gzn,:]+U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hU_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hU_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]))/dx;
        # dVU/dy
        dhU = dhU + 0.25*((V[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+V[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hU_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hU_c[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:])
                          - (V[gx0-1:gxn-1,gy0+1:gyn+1,gz0:gzn,:]+V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hU_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hU_c[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dy;
        # dWU/dz
        dhU = dhU + 0.25*((W[gx0-1:gxn-1,gy0:gyn,gz0+1:gzn+1,:]+W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hU_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hU_c[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
                          - (W[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+W[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hU_c[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hU_c[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dz;
    
        # Compute Vp advective tendency
        # dVV/dy
        dhV = dhV + 0.25*((V[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+V[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hV_c[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+hV_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                          - (V[gx0:gxn,gy0:gyn,gz0:gzn,:]+V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hV_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hV_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]))/dy;
        # dUV/dx
        dhV = dhV + 0.25*((U[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+U[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hV_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hV_c[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:])
                          - (U[gx0+1:gxn+1,gy0-1:gyn-1,gz0:gzn,:]+U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hV_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hV_c[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dx;
        # dWV/dz
        dhV = dhV + 0.25*((W[gx0:gxn,gy0-1:gyn-1,gz0+1:gzn+1,:]+W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hV_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hV_c[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
                          - (W[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+W[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hV_c[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hV_c[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dz;
            
        if icase==0:
            dhUl = dhU.copy();
            dhVl = dhV.copy();
            # Compute KE divergence correction (quick tapered version)...
            dKEdivhUl = 0.5*(hU_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hU_c[gx0:gxn,gy0:gyn,gz0:gzn,:])*DivVcc;
            dKEdivhVl = 0.5*(hV_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hV_c[gx0:gxn,gy0:gyn,gz0:gzn,:])*DivVcc;
        elif icase==1:
            dhUb = dhU.copy();
            dhVb = dhV.copy();
            # Compute KE divergence correction (quick tapered version)...
            dKEdivhUb = 0.5*(hU_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hU_c[gx0:gxn,gy0:gyn,gz0:gzn,:])*DivVcc;
            dKEdivhVb = 0.5*(hV_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hV_c[gx0:gxn,gy0:gyn,gz0:gzn,:])*DivVcc;
        else:
            # Compute KE divergence correction (quick tapered version)...
            dKEdivhUh = 0.5*(hU_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hU_c[gx0:gxn,gy0:gyn,gz0:gzn,:])*DivVcc;
            dKEdivhVh = 0.5*(hV_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hV_c[gx0:gxn,gy0:gyn,gz0:gzn,:])*DivVcc;
            
    hU = hU[gx0:gxn,gy0:gyn,gz0:gzn,:];
    hV = hV[gx0:gxn,gy0:gyn,gz0:gzn,:];
    hU_bp = hU_bp[gx0:gxn,gy0:gyn,gz0:gzn,:];
    hV_bp = hV_bp[gx0:gxn,gy0:gyn,gz0:gzn,:];
    hU_hp = hU_hp[gx0:gxn,gy0:gyn,gz0:gzn,:];
    hV_hp = hV_hp[gx0:gxn,gy0:gyn,gz0:gzn,:];

    return (dhUl,dhVl,dhUb,dhVb,dhU,dhV,hU-hU_bp-hU_hp,hV-hV_bp-hV_hp,hU_bp,hV_bp,hU_hp,hV_hp,dKEdivhUl,dKEdivhVl,dKEdivhUb,dKEdivhVb,dKEdivhUh,dKEdivhVh)


    
