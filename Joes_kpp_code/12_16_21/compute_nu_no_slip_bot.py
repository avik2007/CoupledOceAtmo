import sys
import gc
import warnings
import math
import numpy as np
import numpy.fft
from repmat import repmat
import fld_tools as ft
from compute_nu_T import *

def compute_nu_no_slip_bot_diag(KappaRU,KappaRV,U,V,thknss,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    # Computes bottom drag from the nonuniform fld
    (dU,dV) = compute_nu_no_slip_bot(KappaRU,KappaRV,U,V,thknss,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));

    return (dU,dV);

def compute_nu_no_slip_bot_diag_u(KappaRU,KappaRV,U,V,thknss,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    # Computes bottom drag from the nonuniform fld
    (dU,dV) = compute_nu_no_slip_bot(KappaRU,KappaRV,U,V,thknss,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));

    nz = U.shape[2];
    nzu = zoversamp*nz;
    thknssUni = ft.get_thknssUni(thknss);
    (uthknss, vthknss) = ft.get_uv_thknss(thknss);

    (dU,_) = ft.get_4D_vert_uniform_field(dU,thknssUni,nzu,ns=True);
    (dV,_) = ft.get_4D_vert_uniform_field(dV,thknssUni,nzu,ns=True);

    return (dU,dV);

def compute_nu_no_slip_bot_k(KappaRU,KappaRV,U,V,thknss,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    # Computes bottom drag from the nonuniform fld
    (dU,dV) = compute_nu_no_slip_bot(KappaRU,KappaRV,U,V,thknss,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));
    (T,k_out) = compute_nu_T_k(dU,dV,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));

    return (T,k_out);

def compute_nu_no_slip_bot_m(KappaRU,KappaRV,U,V,thknss,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    # Computes bottom drag from the nonuniform fld
    (dU,dV) = compute_nu_no_slip_bot(KappaRU,KappaRV,U,V,thknss,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));
    (T,m_out) = compute_nu_T_m(dU,dV,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));

    return (T,m_out);

def compute_nu_no_slip_bot(KappaRU,KappaRV,U,V,thknss,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    # Computes bottom drag from the nonuniform fld
    
    # Initialize Constants
    Cd = 2.1e-3;
    (nx, ny, nz, nt) = U.shape;
    rho0 = 2027.5; # for energy output (optional)    

    # Add ghost cells to fields to handle periodicity in position space.
    ng = 2; # number of ghost cells
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits
    thknss_prd = prd; thknss_prd[2] = -1; thknss_prd = prd; thknss_prd[2] = -1; # trick dr_c into being dr_f at bottom.
    U = ft.pad_field_3D(U,ng,prd); V = ft.pad_field_3D(V,ng,prd); thknss = ft.pad_field_3D(thknss,ng,thknss_prd);
    KappaRU = ft.pad_field_3D(KappaRU,ng,prd);     KappaRV = ft.pad_field_3D(KappaRV,ng,prd); 

    (thknssW, thknssS) = ft.get_uv_thknss(thknss);
    thknssUni = ft.get_thknssUni(thknss);
    thknssW[:,:,0:ng,:] = 0; thknssW[:,:,gzn:gzn+ng,:] = 0;
    thknssS[:,:,0:ng,:] = 0; thknssS[:,:,gzn:gzn+ng,:] = 0;
    kbotW = ft.get_bot_Znu(thknssW); kbotS = ft.get_bot_Znu(thknssS);

    ### JS - both work (see above).  Remove redundant functions calls.  Keep one that works with vanishing layers if possible.
    # kbotW = ft.calc_nzmax(ft.get_hfac(thknssW)); kbotW = repmat(kbotW,(1,1,nz,1));
    # kbotS = ft.calc_nzmax(ft.get_hfac(thknssS)); kbotS = repmat(kbotS,(1,1,nz,1));


    dU = np.zeros((nx+2*ng,ny+2*ng,nz+2*ng,nt)); dV = np.zeros((nx+2*ng,ny+2*ng,nz+2*ng,nt));
    for i in range(gx0,gxn):
        for j in range(gy0,gyn):
            for t in range(0,nt):
                k = kbotW[i,j,0,t]; 
                dU[i,j,k,t] = -(4*KappaRU[i,j,k+1,t]/(thknssUni[i,j,k,t]+thknssUni[i,j,k+1,t]))*U[i,j,k,t];
                k = kbotS[i,j,0,t]; 
                dV[i,j,k,t] = -(4*KappaRV[i,j,k+1,t]/(thknssUni[i,j,k,t]+thknssUni[i,j,k+1,t]))*V[i,j,k,t];

    dU = dU[gx0:gxn,gy0:gyn,gz0:gzn,:]; dV = dV[gx0:gxn,gy0:gyn,gz0:gzn,:];
    ft.strip_nan_inf(dU); ft.strip_nan_inf(dV);
    
    return (dU,dV)
