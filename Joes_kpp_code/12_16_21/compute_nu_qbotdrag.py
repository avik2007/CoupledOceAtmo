import sys
import gc
import warnings
import math
import numpy as np
import numpy.fft
from repmat import repmat
import fld_tools as ft
from compute_nu_T import *

# Computes quadratic bottom drag from the nonuniform fld

def compute_nu_qbotdrag_diag(U,V,thknss,dxu,dyv,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    (dU,dV) = compute_nu_qbotdrag(U,V,thknss,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));

    return (dU,dV);

def compute_nu_qbotdrag_diag_u(U,V,thknss,dxu,dyv,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    (dU,dV) = compute_nu_qbotdrag(U,V,thknss,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));

    nz = U.shape[2];
    nzu = zoversamp*nz;
    thknssUni = ft.get_thknssUni(thknss);
    (uthknss, vthknss) = ft.get_uv_thknss(thknss);

    (dU,_) = ft.get_4D_vert_uniform_field(dU,thknssUni,nzu,ns=True);
    (dV,_) = ft.get_4D_vert_uniform_field(dV,thknssUni,nzu,ns=True);

    return (dU,dV);

def compute_nu_qbotdrag_k(U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    (dU,dV) = compute_nu_qbotdrag(U,V,thknss,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));
    (T,k_out) = compute_nu_T_k(dU,dV,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));

    return (T,k_out);

def compute_nu_qbotdrag_m(U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    (dU,dV) = compute_nu_qbotdrag(U,V,thknss,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));
    (T,m_out) = compute_nu_T_m(dU,dV,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));

    return (T,m_out);

def compute_nu_qbotdrag(U,V,thknss,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    print ('in compute_nu_qbotdrag()')
    
    # Initialize Constants
    Cd = 2.1e-3;
    (nx, ny, nz, nt) = U.shape;
    
    # Add ghost cells to fields to handle periodicity in position space.
    ng = 2; # number of ghost cells
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits
    U = ft.pad_field_3D(U,ng,prd); V = ft.pad_field_3D(V,ng,prd); thknss = ft.pad_field_3D(thknss,ng,prd);

    (thknssW, thknssS) = ft.get_uv_thknss(thknss);
    thknssW[:,:,0:ng,:] = 0; thknssW[:,:,gzn:gzn+ng,:] = 0;
    thknssS[:,:,0:ng,:] = 0; thknssS[:,:,gzn:gzn+ng,:] = 0;
    kbotW = ft.get_bot_Znu(thknssW); kbotS = ft.get_bot_Znu(thknssS);
    
    ### JS - both work (see above).  Remove redundant functions calls.  Keep one that works with vanishing layers if possible.
    # kbotW = ft.calc_nzmax(ft.get_hfac(thknssW)); kbotW = repmat(kbotW,(1,1,nz,1));
    # kbotS = ft.calc_nzmax(ft.get_hfac(thknssS)); kbotS = repmat(kbotS,(1,1,nz,1));

    # Not the most accurate way, but how it is computed in MITgcm.
    # (KEscheme.EQ.0):
    KE = 0.25*(U[gx0:gxn,gy0:gyn,gz0:gzn,:]**2 + 
               U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]**2 + 
               V[gx0:gxn,gy0:gyn,gz0:gzn,:]**2 + 
               V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]**2);
    KE = ft.pad_field_3D(KE,ng,prd);
    
    dU = np.zeros((nx+2*ng,ny+2*ng,nz+2*ng,nt)); dV = np.zeros((nx+2*ng,ny+2*ng,nz+2*ng,nt));
    for i in range(gx0,gxn):
        for j in range(gy0,gyn):
            for t in range(0,nt):
                k = kbotW[i,j,0,t];
                dU[i,j,k,t] = -(Cd/thknssW[i,j,k,t])*((KE[i-1,j,k,t]+KE[i,j,k,t])**(0.5))*U[i,j,k,t];
                k = kbotS[i,j,0,t];
                dV[i,j,k,t] = -(Cd/thknssS[i,j,k,t])*((KE[i,j-1,k,t]+KE[i,j,k,t])**(0.5))*V[i,j,k,t];

    dU = dU[gx0:gxn,gy0:gyn,gz0:gzn,:]; dV = dV[gx0:gxn,gy0:gyn,gz0:gzn,:];
    ft.strip_nan_inf(dU); ft.strip_nan_inf(dV);
    
    return (dU,dV)



    # KE = 0.25*(hfacW[gx0:gxn,gy0:gyn,gz0:gzn,:]*U[gx0:gxn,gy0:gyn,gz0:gzn,:]**2 + 
    #            hfacW[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]*U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]**2 + 
    #            hfacS[gx0:gxn,gy0:gyn,gz0:gzn,:]*V[gx0:gxn,gy0:gyn,gz0:gzn,:]**2 + 
    #            hfacS[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]*V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]**2)/hfacC[gx0:gxn,gy0:gyn,gz0:gzn,:];
