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
    
def compute_nu_decomp_FF_T_diag(U,V,W,U_bp,V_bp,U_hp,V_hp,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([0,0,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (dUl,dVl,dUb,dVb,dUh,dVh,_,_,_,_,_,_,hUdivVl,hVdivVl,hUdivVb,hVdivVb,hUdivVh,hVdivVh) = compute_nu_decomp_FF_T(U,V,W,U_bp,V_bp,U_hp,V_hp,
                                                                                                                   thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,
                                                                                                                   ugrid,vgrid,pgrid,np.array([0,0,0]),
                                                                                                                   trim_ml,zoversamp,prd);
    return (dUl,dVl,dUb,dVb,dUh,dVh,hUdivVl,hVdivVl,hUdivVb,hVdivVb,hUdivVh,hVdivVh);

def compute_nu_decomp_FF_T_diag_u(U,V,W,U_bp,V_bp,U_hp,V_hp,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([0,0,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (dUl,dVl,dUb,dVb,dUh,dVh,_,_,_,_,_,_,hUdivVl,hVdivVl,hUdivVb,hVdivVb,hUdivVh,hVdivVh) = compute_nu_decomp_FF_T(U,V,W,U_bp,V_bp,U_hp,V_hp,
                                                                                                                   thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,
                                                                                                                   ugrid,vgrid,pgrid,np.array([0,0,0]),
                                                                                                                   trim_ml,zoversamp,prd);
    nz = U.shape[2];
    nzu = zoversamp*nz;
    thknssUni = ft.get_thknssUni(thknss);
    (uthknss, vthknss) = ft.get_uv_thknss(thknss);

    (dUl,_) = ft.get_4D_vert_uniform_field(dUl,thknssUni,nzu,ns=True);
    (dVl,_) = ft.get_4D_vert_uniform_field(dVl,thknssUni,nzu,ns=True);
    (dUb,_) = ft.get_4D_vert_uniform_field(dUb,thknssUni,nzu,ns=True);
    (dVb,_) = ft.get_4D_vert_uniform_field(dVb,thknssUni,nzu,ns=True);
    (dUh,_) = ft.get_4D_vert_uniform_field(dUh,thknssUni,nzu,ns=True);
    (dVh,_) = ft.get_4D_vert_uniform_field(dVh,thknssUni,nzu,ns=True);
    (hUdivVl,_) = ft.get_4D_vert_uniform_field(hUdivVl,thknssUni,nzu,ns=True);
    (hVdivVl,_) = ft.get_4D_vert_uniform_field(hVdivVl,thknssUni,nzu,ns=True);
    (hUdivVb,_) = ft.get_4D_vert_uniform_field(hUdivVb,thknssUni,nzu,ns=True);
    (hVdivVb,_) = ft.get_4D_vert_uniform_field(hVdivVb,thknssUni,nzu,ns=True);
    (hUdivVh,_) = ft.get_4D_vert_uniform_field(hUdivVh,thknssUni,nzu,ns=True);
    (hVdivVh,_) = ft.get_4D_vert_uniform_field(hVdivVh,thknssUni,nzu,ns=True);

    return (dUl,dVl,dUb,dVb,dUh,dVh,hUdivVl,hVdivVl,hUdivVb,hVdivVb,hUdivVh,hVdivVh);

def compute_nu_decomp_FF_T_m(U,V,W,U_bp,V_bp,U_hp,V_hp,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (dUl,dVl,dUb,dVb,dUh,dVh,hUl,hVl,hUb,hVb,hUh,hVh,hUdivVl,hVdivVl,hUdivVb,hVdivVb,hUdivVh,hVdivVh) = compute_nu_decomp_FF_T(U,V,W,U_bp,V_bp,U_hp,V_hp,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,
                                                    ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);

    nz = W.shape[2];
    mti_max = math.floor(nz/2-1); 
    T = np.zeros((mti_max,9));
    T_KEdivV = np.zeros((mti_max,9));

    (Tl_lp,m_out) = compute_nu_T_corr_m(dUl,dVl,hUl,hVl,hUdivVl,hVdivVl,thknss,dxu,dyv,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    (Tl_bp,m_out) = compute_nu_T_corr_m(dUl,dVl,hUb,hVb,hUdivVl,hVdivVl,thknss,dxu,dyv,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    (Tl_hp,m_out) = compute_nu_T_corr_m(dUl,dVl,hUh,hVh,hUdivVl,hVdivVl,thknss,dxu,dyv,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    (Tb_lp,m_out) = compute_nu_T_corr_m(dUb,dVb,hUl,hVl,hUdivVb,hVdivVb,thknss,dxu,dyv,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    (Tb_bp,m_out) = compute_nu_T_corr_m(dUb,dVb,hUb,hVb,hUdivVb,hVdivVb,thknss,dxu,dyv,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    (Tb_hp,m_out) = compute_nu_T_corr_m(dUb,dVb,hUh,hVh,hUdivVb,hVdivVb,thknss,dxu,dyv,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    (Th_lp,m_out) = compute_nu_T_corr_m(dUh,dVh,hUl,hVl,hUdivVh,hVdivVh,thknss,dxu,dyv,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    (Th_bp,m_out) = compute_nu_T_corr_m(dUh,dVh,hUb,hVb,hUdivVh,hVdivVh,thknss,dxu,dyv,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    (Th_hp,m_out) = compute_nu_T_corr_m(dUh,dVh,hUh,hVh,hUdivVh,hVdivVh,thknss,dxu,dyv,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);

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

def compute_nu_decomp_FF_T(U,V,W,U_bp,V_bp,U_hp,V_hp,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    print ('in compute_nu_decomp_FF_T()')

    # Initialize Constants
    (nx, ny, nz, nt) = U.shape;
    rho0 = 1027.5; # for energy output (optional)
    
    # Add ghost cells to fields to handle periodicity in position space.
    ng = 2; # number of ghost cells, must be one larger than needed for python indexing
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits
    U = ft.pad_field_3D(U,ng,prd); V = ft.pad_field_3D(V,ng,prd);  W = ft.pad_field_3D(W,ng,prd);  thknss = ft.pad_field_3D(thknss,ng,prd); 
    U_bp = ft.pad_field_3D(U_bp,ng,prd); V_bp = ft.pad_field_3D(V_bp,ng,prd);
    U_hp = ft.pad_field_3D(U_hp,ng,prd); V_hp = ft.pad_field_3D(V_hp,ng,prd);
    dxc = ft.pad_field_2D(dxc,ng); dxu = ft.pad_field_2D(dxu,ng); dxv = ft.pad_field_2D(dxv,ng); dxq = ft.pad_field_2D(dxq,ng);
    dyc = ft.pad_field_2D(dyc,ng); dyu = ft.pad_field_2D(dyu,ng); dyv = ft.pad_field_2D(dyv,ng); dyq = ft.pad_field_2D(dyq,ng); 

    (hU,hV,_) = ft.taper_filter_3D_uvw_nu(U[gx0:gxn,gy0:gyn,gz0:gzn,:],V[gx0:gxn,gy0:gyn,gz0:gzn,:],W[gx0:gxn,gy0:gyn,gz0:gzn,:],
                                      dxu[gx0:gxn,gy0:gyn],dyu[gx0:gxn,gy0:gyn],dxv[gx0:gxn,gy0:gyn],dyv[gx0:gxn,gy0:gyn],
                                      dxc[gx0:gxn,gy0:gyn],dyc[gx0:gxn,gy0:gyn],thknss[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper);
    (hU_bp,hV_bp,_) = ft.taper_filter_3D_uvw_nu(U_bp[gx0:gxn,gy0:gyn,gz0:gzn,:],V_bp[gx0:gxn,gy0:gyn,gz0:gzn,:],W[gx0:gxn,gy0:gyn,gz0:gzn,:],
                                      dxu[gx0:gxn,gy0:gyn],dyu[gx0:gxn,gy0:gyn],dxv[gx0:gxn,gy0:gyn],dyv[gx0:gxn,gy0:gyn],
                                      dxc[gx0:gxn,gy0:gyn],dyc[gx0:gxn,gy0:gyn],thknss[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper);
    (hU_hp,hV_hp,_) = ft.taper_filter_3D_uvw_nu(U_hp[gx0:gxn,gy0:gyn,gz0:gzn,:],V_hp[gx0:gxn,gy0:gyn,gz0:gzn,:],W[gx0:gxn,gy0:gyn,gz0:gzn,:],
                                      dxu[gx0:gxn,gy0:gyn],dyu[gx0:gxn,gy0:gyn],dxv[gx0:gxn,gy0:gyn],dyv[gx0:gxn,gy0:gyn],
                                      dxc[gx0:gxn,gy0:gyn],dyc[gx0:gxn,gy0:gyn],thknss[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper);
    hU = ft.pad_field_3D(hU,ng,prd); hV = ft.pad_field_3D(hV,ng,prd); 
    hU_bp = ft.pad_field_3D(hU_bp,ng,prd); hV_bp = ft.pad_field_3D(hV_bp,ng,prd); 
    hU_hp = ft.pad_field_3D(hU_hp,ng,prd); hV_hp = ft.pad_field_3D(hV_hp,ng,prd); 

    (uthknss, vthknss) = ft.get_uv_f(thknss); uthknss = uthknss[0:-1,:,:,:]; vthknss = vthknss[:,0:-1,:,:];

    dxq3D = repmat(dxq,(1,1,nz+2*ng,nt));    dyq3D = repmat(dyq,(1,1,nz+2*ng,nt));
    Ut = U[gx0:gxn,gy0:gyn,gz0:gzn,:]*dyq3D[gx0:gxn,gy0:gyn,gz0:gzn,:]*uthknss[gx0:gxn,gy0:gyn,gz0:gzn,:]; 
    Vt = V[gx0:gxn,gy0:gyn,gz0:gzn,:]*dxq3D[gx0:gxn,gy0:gyn,gz0:gzn,:]*vthknss[gx0:gxn,gy0:gyn,gz0:gzn,:]; 
    Wt = W[gx0:gxn,gy0:gyn,gz0:gzn,:]*dxq3D[gx0:gxn,gy0:gyn,gz0:gzn,:]*dyq3D[gx0:gxn,gy0:gyn,gz0:gzn,:];
    Ut = ft.pad_field_3D(Ut,ng,prd); Vt = ft.pad_field_3D(Vt,ng,prd); Wt = ft.pad_field_3D(Wt,ng,prd);
    del dxq3D, dyq3D;

    DivVcc = ft.get_divergence_nu(U[gx0:gxn,gy0:gyn,gz0:gzn,:],V[gx0:gxn,gy0:gyn,gz0:gzn,:],W[gx0:gxn,gy0:gyn,gz0:gzn,:],dxu[gx0:gxn,gy0:gyn],dyv[gx0:gxn,gy0:gyn],dxq[gx0:gxn,gy0:gyn],dyq[gx0:gxn,gy0:gyn],thknss[gx0:gxn,gy0:gyn,gz0:gzn,:]);

    for icase in range(0,3):
        if (icase==0):
            hU_c = hU-hU_bp-hU_hp; hV_c = hV-hV_bp-hV_hp;
            U_c = U-U_bp-U_hp; V_c = V-V_bp-V_hp;
        elif (icase==1):
            hU_c = hU_bp; hV_c = hV_bp;
            U_c = U_bp; V_c = V_bp;
        else:
            hU_c = hU_hp; hV_c = hV_hp;
            U_c = U_hp; V_c = V_hp;

        # initialize tendencies
        dhU = np.zeros((nx,ny,nz,nt)); dhV = np.zeros((nx,ny,nz,nt));
    
        # Compute Up advective tendency
        # d(UU)/dx
        dhU = dhU + 0.25*((Ut[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+Ut[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hU_c[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+hU_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                          - (Ut[gx0:gxn,gy0:gyn,gz0:gzn,:]+Ut[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hU_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hU_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]));
        # dVU/dy
        dhU = dhU + 0.25*((Vt[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+Vt[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hU_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hU_c[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:])
                          - (Vt[gx0-1:gxn-1,gy0+1:gyn+1,gz0:gzn,:]+Vt[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hU_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hU_c[gx0:gxn,gy0:gyn,gz0:gzn,:]));
        # dWU/dz
        dhU = dhU + 0.25*((Wt[gx0-1:gxn-1,gy0:gyn,gz0+1:gzn+1,:]+Wt[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hU_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hU_c[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
                          - (Wt[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+Wt[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hU_c[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hU_c[gx0:gxn,gy0:gyn,gz0:gzn,:]));
        
        # Compute Vp advective tendency
        # dVV/dy
        dhV = dhV + 0.25*((Vt[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+Vt[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hV_c[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+hV_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                          - (Vt[gx0:gxn,gy0:gyn,gz0:gzn,:]+Vt[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hV_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hV_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]));
        # dUV/dx
        dhV = dhV + 0.25*((Ut[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+Ut[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hV_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hV_c[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:])
                          - (Ut[gx0+1:gxn+1,gy0-1:gyn-1,gz0:gzn,:]+Ut[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hV_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hV_c[gx0:gxn,gy0:gyn,gz0:gzn,:]));
        # dWV/dz
        dhV = dhV + 0.25*((Wt[gx0:gxn,gy0-1:gyn-1,gz0+1:gzn+1,:]+Wt[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hV_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hV_c[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
                          - (Wt[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+Wt[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hV_c[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hV_c[gx0:gxn,gy0:gyn,gz0:gzn,:]));
        
#        del Ut, Vt, Wt;
        
        dxq3D = repmat(dxq,(1,1,nz+2*ng,nt));    dyq3D = repmat(dyq,(1,1,nz+2*ng,nt));
        dxu3D = repmat(dxu,(1,1,nz+2*ng,nt));    dyv3D = repmat(dyv,(1,1,nz+2*ng,nt));
        dhU = dhU/dxq3D[gx0:gxn,gy0:gyn,gz0:gzn,:]/dyv3D[gx0:gxn,gy0:gyn,gz0:gzn,:]/uthknss[gx0:gxn,gy0:gyn,gz0:gzn,:];
        dhV = dhV/dyq3D[gx0:gxn,gy0:gyn,gz0:gzn,:]/dxu3D[gx0:gxn,gy0:gyn,gz0:gzn,:]/vthknss[gx0:gxn,gy0:gyn,gz0:gzn,:];
        ft.strip_nan_inf(dhU); ft.strip_nan_inf(dhV);
        del dxq3D, dyq3D, dxu3D, dyv3D;
        
        if icase==0:
            dhUl = dhU.copy();
            dhVl = dhV.copy();
            # Compute Flux-Form Divergence Correction Term
            hUdivVl = 0.5*ft.taper_filter_3Dcc_nu((U_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+U_c[gx0:gxn,gy0:gyn,gz0:gzn,:]),dxc[gx0:gxn,gy0:gyn],dyc[gx0:gxn,gy0:gyn],thknss[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper)*DivVcc;
            hVdivVl = 0.5*ft.taper_filter_3Dcc_nu((V_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+V_c[gx0:gxn,gy0:gyn,gz0:gzn,:]),dxc[gx0:gxn,gy0:gyn],dyc[gx0:gxn,gy0:gyn],thknss[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper)*DivVcc;
        elif icase==1:
            dhUb = dhU.copy();
            dhVb = dhV.copy();
            # Compute Flux-Form Divergence Correction Term
            hUdivVb = 0.5*ft.taper_filter_3Dcc_nu((U_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+U_c[gx0:gxn,gy0:gyn,gz0:gzn,:]),dxc[gx0:gxn,gy0:gyn],dyc[gx0:gxn,gy0:gyn],thknss[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper)*DivVcc;
            hVdivVb = 0.5*ft.taper_filter_3Dcc_nu((V_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+V_c[gx0:gxn,gy0:gyn,gz0:gzn,:]),dxc[gx0:gxn,gy0:gyn],dyc[gx0:gxn,gy0:gyn],thknss[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper)*DivVcc;
        else:
            # Compute KE divergence correction (quick tapered version)...
            hUdivVh = 0.5*ft.taper_filter_3Dcc_nu((U_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+U_c[gx0:gxn,gy0:gyn,gz0:gzn,:]),dxc[gx0:gxn,gy0:gyn],dyc[gx0:gxn,gy0:gyn],thknss[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper)*DivVcc;
            hVdivVh = 0.5*ft.taper_filter_3Dcc_nu((V_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+V_c[gx0:gxn,gy0:gyn,gz0:gzn,:]),dxc[gx0:gxn,gy0:gyn],dyc[gx0:gxn,gy0:gyn],thknss[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper)*DivVcc;

    hU = hU[gx0:gxn,gy0:gyn,gz0:gzn,:];
    hV = hV[gx0:gxn,gy0:gyn,gz0:gzn,:];
    hU_bp = hU_bp[gx0:gxn,gy0:gyn,gz0:gzn,:];
    hV_bp = hV_bp[gx0:gxn,gy0:gyn,gz0:gzn,:];
    hU_hp = hU_hp[gx0:gxn,gy0:gyn,gz0:gzn,:];
    hV_hp = hV_hp[gx0:gxn,gy0:gyn,gz0:gzn,:];

    return (dhUl,dhVl,dhUb,dhVb,dhU,dhV,hU-hU_bp-hU_hp,hV-hV_bp-hV_hp,hU_bp,hV_bp,hU_hp,hV_hp,hUdivVl,hVdivVl,hUdivVb,hVdivVb,hUdivVh,hVdivVh)


    
