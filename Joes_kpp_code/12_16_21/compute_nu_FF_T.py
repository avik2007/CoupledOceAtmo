import sys
import gc
import warnings
import math
import numpy as np
import numpy.fft
from repmat import repmat
import fld_tools as ft
from compute_nu_T import *

# Computes horizontal spectral energy flux, assuming nonunform regular coordinates with a finite volume formulation
# a tranfer-function approach is used that assumes energy is conserved, even though it is not, strictly speaking.
    
def compute_nu_FF_T_diag(U,V,W,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([0,0,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (dU,dV,_,_,_,_) = compute_nu_FF_T(U,V,W,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,
                                       ugrid,vgrid,pgrid,np.array([0,0,0]),trim_ml,zoversamp,prd);
    return (dU,dV);

def compute_nu_FF_T_diag_u(U,V,W,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([0,0,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (dU,dV,_,_,_,_) = compute_nu_FF_T(U,V,W,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,
                                       ugrid,vgrid,pgrid,np.array([0,0,0]),trim_ml,zoversamp,prd);

    nz = U.shape[2];
    nzu = zoversamp*nz;
    thknssUni = ft.get_thknssUni(thknss);
    (uthknss, vthknss) = ft.get_uv_thknss(thknss);

    (dU,_) = ft.get_4D_vert_uniform_field(dU,thknssUni,nzu,ns=True);
    (dV,_) = ft.get_4D_vert_uniform_field(dV,thknssUni,nzu,ns=True);

    return (dU,dV);

def compute_nu_FF_T_k(U,V,W,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (dhU,dhV,hU,hV,hUdivV,hVdivV) = compute_nu_FF_T(U,V,W,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,
                                                    ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);
    (T,k_out) = compute_nu_T_corr_k(dhU,dhV,hU,hV,hUdivV,hVdivV,thknss,dxu,dyv,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    return (T,k_out);

def compute_nu_FF_T_m(U,V,W,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (dhU,dhV,hU,hV,hUdivV,hVdivV) = compute_nu_FF_T(U,V,W,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,
                                                    ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);
    (T,m_out) = compute_nu_T_corr_m(dhU,dhV,hU,hV,hUdivV,hVdivV,thknss,dxu,dyv,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);

    return (T,m_out);



def compute_nu_FF_T(U,V,W,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    print ('in compute_nu_FF_T()')

    # Initialize Constants
    (nx, ny, nz, nt) = U.shape;
    rho0 = 1027.5; # for energy output (optional)
    
#     if (post_taper[2]):
#         raise RuntimeError('post_taper not built to handle vertical filt. of dKEdiv');

    # Add ghost cells to fields to handle periodicity in position space.
    ng = 2; # number of ghost cells, must be one larger than needed for python indexing
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits
    U = ft.pad_field_3D(U,ng,prd); V = ft.pad_field_3D(V,ng,prd);  W = ft.pad_field_3D(W,ng,prd);  thknss = ft.pad_field_3D(thknss,ng,prd); 
    dxc = ft.pad_field_2D(dxc,ng); dxu = ft.pad_field_2D(dxu,ng); dxv = ft.pad_field_2D(dxv,ng); dxq = ft.pad_field_2D(dxq,ng);
    dyc = ft.pad_field_2D(dyc,ng); dyu = ft.pad_field_2D(dyu,ng); dyv = ft.pad_field_2D(dyv,ng); dyq = ft.pad_field_2D(dyq,ng); 

    (hU,hV,_) = ft.taper_filter_3D_uvw_nu(U[gx0:gxn,gy0:gyn,gz0:gzn,:],V[gx0:gxn,gy0:gyn,gz0:gzn,:],W[gx0:gxn,gy0:gyn,gz0:gzn,:],
                                      dxu[gx0:gxn,gy0:gyn],dyu[gx0:gxn,gy0:gyn],dxv[gx0:gxn,gy0:gyn],dyv[gx0:gxn,gy0:gyn],
                                      dxc[gx0:gxn,gy0:gyn],dyc[gx0:gxn,gy0:gyn],thknss[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper);
    hU = ft.pad_field_3D(hU,ng,prd); hV = ft.pad_field_3D(hV,ng,prd); 

    (uthknss, vthknss) = ft.get_uv_f(thknss); uthknss = uthknss[0:-1,:,:,:]; vthknss = vthknss[:,0:-1,:,:];

    dxq3D = repmat(dxq,(1,1,nz+2*ng,nt));    dyq3D = repmat(dyq,(1,1,nz+2*ng,nt));
    Ut = U[gx0:gxn,gy0:gyn,gz0:gzn,:]*dyq3D[gx0:gxn,gy0:gyn,gz0:gzn,:]*uthknss[gx0:gxn,gy0:gyn,gz0:gzn,:]; 
    Vt = V[gx0:gxn,gy0:gyn,gz0:gzn,:]*dxq3D[gx0:gxn,gy0:gyn,gz0:gzn,:]*vthknss[gx0:gxn,gy0:gyn,gz0:gzn,:]; 
    Wt = W[gx0:gxn,gy0:gyn,gz0:gzn,:]*dxq3D[gx0:gxn,gy0:gyn,gz0:gzn,:]*dyq3D[gx0:gxn,gy0:gyn,gz0:gzn,:];
    Ut = ft.pad_field_3D(Ut,ng,prd); Vt = ft.pad_field_3D(Vt,ng,prd); Wt = ft.pad_field_3D(Wt,ng,prd);
    del dxq3D, dyq3D;

    

    # initialize tendencies
    dhU = np.zeros((nx,ny,nz,nt)); dhV = np.zeros((nx,ny,nz,nt));
    
    # Compute Up advective tendency
    # d(UU)/dx
    dhU = dhU + 0.25*((Ut[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+Ut[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hU[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+hU[gx0:gxn,gy0:gyn,gz0:gzn,:])
        - (Ut[gx0:gxn,gy0:gyn,gz0:gzn,:]+Ut[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hU[gx0:gxn,gy0:gyn,gz0:gzn,:]+hU[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]));
    # dVU/dy
    dhU = dhU + 0.25*((Vt[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+Vt[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hU[gx0:gxn,gy0:gyn,gz0:gzn,:]+hU[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:])
        - (Vt[gx0-1:gxn-1,gy0+1:gyn+1,gz0:gzn,:]+Vt[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hU[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hU[gx0:gxn,gy0:gyn,gz0:gzn,:]));
    # dWU/dz
    dhU = dhU + 0.25*((Wt[gx0-1:gxn-1,gy0:gyn,gz0+1:gzn+1,:]+Wt[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hU[gx0:gxn,gy0:gyn,gz0:gzn,:]+hU[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
        - (Wt[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+Wt[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hU[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hU[gx0:gxn,gy0:gyn,gz0:gzn,:]));
    
    # Compute Vp advective tendency
    # dVV/dy
    dhV = dhV + 0.25*((Vt[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+Vt[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hV[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+hV[gx0:gxn,gy0:gyn,gz0:gzn,:])
        - (Vt[gx0:gxn,gy0:gyn,gz0:gzn,:]+Vt[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hV[gx0:gxn,gy0:gyn,gz0:gzn,:]+hV[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]));
    # dUV/dx
    dhV = dhV + 0.25*((Ut[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+Ut[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hV[gx0:gxn,gy0:gyn,gz0:gzn,:]+hV[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:])
        - (Ut[gx0+1:gxn+1,gy0-1:gyn-1,gz0:gzn,:]+Ut[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hV[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hV[gx0:gxn,gy0:gyn,gz0:gzn,:]));
    # dWV/dz
    dhV = dhV + 0.25*((Wt[gx0:gxn,gy0-1:gyn-1,gz0+1:gzn+1,:]+Wt[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hV[gx0:gxn,gy0:gyn,gz0:gzn,:]+hV[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
        - (Wt[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+Wt[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hV[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hV[gx0:gxn,gy0:gyn,gz0:gzn,:]));

#################DEBUG###################    del Ut, Vt, Wt;
    
    dxq3D = repmat(dxq,(1,1,nz+2*ng,nt));    dyq3D = repmat(dyq,(1,1,nz+2*ng,nt));
    dxu3D = repmat(dxu,(1,1,nz+2*ng,nt));    dyv3D = repmat(dyv,(1,1,nz+2*ng,nt));
    dhU = dhU/dxq3D[gx0:gxn,gy0:gyn,gz0:gzn,:]/dyv3D[gx0:gxn,gy0:gyn,gz0:gzn,:]/uthknss[gx0:gxn,gy0:gyn,gz0:gzn,:];
    dhV = dhV/dyq3D[gx0:gxn,gy0:gyn,gz0:gzn,:]/dxu3D[gx0:gxn,gy0:gyn,gz0:gzn,:]/vthknss[gx0:gxn,gy0:gyn,gz0:gzn,:];
    ft.strip_nan_inf(dhU); ft.strip_nan_inf(dhV);
    del dxq3D, dyq3D, dxu3D, dyv3D;

    # Compute Flux-Form Divergence Correction Term
    DivVcc = ft.get_divergence_nu(U[gx0:gxn,gy0:gyn,gz0:gzn,:],V[gx0:gxn,gy0:gyn,gz0:gzn,:],W[gx0:gxn,gy0:gyn,gz0:gzn,:],dxu[gx0:gxn,gy0:gyn],dyv[gx0:gxn,gy0:gyn],dxq[gx0:gxn,gy0:gyn],dyq[gx0:gxn,gy0:gyn],thknss[gx0:gxn,gy0:gyn,gz0:gzn,:]);
    hUdivV = 0.5*ft.taper_filter_3Dcc_nu((U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+U[gx0:gxn,gy0:gyn,gz0:gzn,:]),dxc[gx0:gxn,gy0:gyn],dyc[gx0:gxn,gy0:gyn],thknss[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper)*DivVcc;
    hVdivV = 0.5*ft.taper_filter_3Dcc_nu((V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+V[gx0:gxn,gy0:gyn,gz0:gzn,:]),dxc[gx0:gxn,gy0:gyn],dyc[gx0:gxn,gy0:gyn],thknss[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper)*DivVcc;
    del DivVcc;
        
    hU = hU[gx0:gxn,gy0:gyn,gz0:gzn,:];
    hV = hV[gx0:gxn,gy0:gyn,gz0:gzn,:];

    return (dhU,dhV,hU,hV,hUdivV,hVdivV)


    
