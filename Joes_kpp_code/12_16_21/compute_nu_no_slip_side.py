import sys
import gc
import warnings
import math
import numpy as np
import numpy.fft
from repmat import repmat
import fld_tools as ft
from compute_nu_T import *

def compute_nu_no_slip_side_diag(viscA4_Z,U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,
                                 post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    # Computes bottom drag from the nonuniform fld
    (dU,dV) = compute_nu_no_slip_side(viscA4_Z,U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,
                                      post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));

    return (dU,dV);

def compute_nu_no_slip_side_diag_u(viscA4_Z,U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,
                                 post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    # Computes bottom drag from the nonuniform fld
    (dU,dV) = compute_nu_no_slip_side(viscA4_Z,U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,
                                      post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));

    nz = U.shape[2];
    nzu = zoversamp*nz;
    thknssUni = ft.get_thknssUni(thknss);
    (uthknss, vthknss) = ft.get_uv_thknss(thknss);

    (dU,_) = ft.get_4D_vert_uniform_field(dU,thknssUni,nzu,ns=True);
    (dV,_) = ft.get_4D_vert_uniform_field(dV,thknssUni,nzu,ns=True);

    return (dU,dV);

def compute_nu_no_slip_side_k(viscA4_Z,U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,
                              post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    # Computes bottom drag from the nonuniform fld
    (dU,dV) = compute_nu_no_slip_side(viscA4_Z,U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,
                                      post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));
    (T,k_out) = compute_nu_T_k(dU,dV,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));

    return (T,k_out);

def compute_nu_no_slip_side_m(viscA4_Z,U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,
                              post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    # Computes bottom drag from the nonuniform fld
    (dU,dV) = compute_nu_no_slip_side(viscA4_Z,U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,
                                      post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));
    (T,m_out) = compute_nu_T_m(dU,dV,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1]));

    return (T,m_out);

def compute_nu_no_slip_side(viscA4_Z,U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,
                            post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    print ('in compute_nu_no_slip_side()')

    # Computes bottom drag from the nonuniform fld
    
    # Initialize Constants
    Cd = 2.1e-3;
    (nx, ny, nz, nt) = U.shape;
    rho0 = 2027.5; # for energy output (optional)    
    
    # Add ghost cells to fields to handle periodicity in position space.
    ng = 2; # number of ghost cells
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits
    U = ft.pad_field_3D(U,ng,prd); V = ft.pad_field_3D(V,ng,prd); thknss = ft.pad_field_3D(thknss,ng,prd);
    dxc = ft.pad_field_2D(dxc,ng); dxu = ft.pad_field_2D(dxu,ng); dxv = ft.pad_field_2D(dxv,ng); dxq = ft.pad_field_2D(dxq,ng);
    dyc = ft.pad_field_2D(dyc,ng); dyu = ft.pad_field_2D(dyu,ng); dyv = ft.pad_field_2D(dyv,ng); dyq = ft.pad_field_2D(dyq,ng); 

    thknssUni = ft.get_thknssUni(thknss);
    thknssZ = ft.get_q_thknss(thknss);
    hfacC = ft.get_hfac(thknss);
    hfacZ = ft.get_q_hfac(thknss);
    (hfacW,hfacS) = ft.get_uv_hfac(thknss);

    hfacZClosedE = hfacS[gx0:gxn,gy0:gyn,gz0:gzn,:] - hfacZ[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:];
    hfacZClosedW = hfacS[gx0:gxn,gy0:gyn,gz0:gzn,:] - hfacZ[gx0:gxn,gy0:gyn,gz0:gzn,:];
    hfacZClosedN = hfacW[gx0:gxn,gy0:gyn,gz0:gzn,:] - hfacZ[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:];
    hfacZClosedS = hfacW[gx0:gxn,gy0:gyn,gz0:gzn,:] - hfacZ[gx0:gxn,gy0:gyn,gz0:gzn,:];

    # eventually, write option to pad_field_3D with 0's instead of periodic.
    hfacZClosedE = ft.pad_field_3D(hfacZClosedE,ng,prd); hfacZClosedE[:,:,0:ng,:] = 0; hfacZClosedE[:,:,gzn:gzn+ng,:] = 0;
    hfacZClosedW = ft.pad_field_3D(hfacZClosedW,ng,prd); hfacZClosedW[:,:,0:ng,:] = 0; hfacZClosedW[:,:,gzn:gzn+ng,:] = 0;
    hfacZClosedN = ft.pad_field_3D(hfacZClosedN,ng,prd); hfacZClosedN[:,:,0:ng,:] = 0; hfacZClosedN[:,:,gzn:gzn+ng,:] = 0;
    hfacZClosedS = ft.pad_field_3D(hfacZClosedS,ng,prd); hfacZClosedS[:,:,0:ng,:] = 0; hfacZClosedS[:,:,gzn:gzn+ng,:] = 0;

    hdiv = ft.get_nu_hdiv(U,V,hfacC,hfacW,hfacS,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng);
    vort3 = ft.get_nu_relvort3(U,V,thknssZ,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng,no_slip_sides=1);

    del2u = ft.get_nu_vi_laplacian_U(U,hdiv,vort3,hfacZ,hfacW,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng);
    del2v = ft.get_nu_vi_laplacian_V(V,hdiv,vort3,hfacZ,hfacS,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng);

    viscA4_Z = ft.pad_field_3D(viscA4_Z,ng,prd); 

    dU = 2*(viscA4_Z[gx0:gxn,gy0:gyn,gz0:gzn,:]*del2u[gx0:gxn,gy0:gyn,gz0:gzn,:]*hfacZClosedS[gx0:gxn,gy0:gyn,gz0:gzn,:]*
            repmat(np.reshape(dxv[gx0:gxn,gy0:gyn]/(dyu[gx0:gxn,gy0:gyn]*dxc[gx0:gxn,gy0:gyn]*dyq[gx0:gxn,gy0:gyn]),(nx,ny,1,1)),(1,1,nz,nt))
            + viscA4_Z[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]*del2u[gx0:gxn,gy0:gyn,gz0:gzn,:]*hfacZClosedN[gx0:gxn,gy0:gyn,gz0:gzn,:]*
            repmat(np.reshape(dxv[gx0:gxn,gy0+1:gyn+1]/(dyu[gx0:gxn,gy0+1:gyn+1]*dxc[gx0:gxn,gy0:gyn]*dyq[gx0:gxn,gy0:gyn]),(nx,ny,1,1)),(1,1,nz,nt))
        )/hfacW[gx0:gxn,gy0:gyn,gz0:gzn,:];
    dV = 2*(viscA4_Z[gx0:gxn,gy0:gyn,gz0:gzn,:]*del2v[gx0:gxn,gy0:gyn,gz0:gzn,:]*hfacZClosedW[gx0:gxn,gy0:gyn,gz0:gzn,:]*
            repmat(np.reshape(dyu[gx0:gxn,gy0:gyn]/(dyc[gx0:gxn,gy0:gyn]*dxq[gx0:gxn,gy0:gyn]*dxv[gx0:gxn,gy0:gyn]),(nx,ny,1,1)),(1,1,nz,nt))
            + viscA4_Z[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]*del2v[gx0:gxn,gy0:gyn,gz0:gzn,:]*hfacZClosedE[gx0:gxn,gy0:gyn,gz0:gzn,:]*
            repmat(np.reshape(dyu[gx0+1:gxn+1,gy0:gyn]/(dyc[gx0:gxn,gy0:gyn]*dxq[gx0:gxn,gy0:gyn]*dxv[gx0+1:gxn+1,gy0:gyn]),(nx,ny,1,1)),(1,1,nz,nt))
        )/hfacS[gx0:gxn,gy0:gyn,gz0:gzn,:];

    ft.strip_nan_inf(dU); ft.strip_nan_inf(dV);
    
    return (dU,dV)
