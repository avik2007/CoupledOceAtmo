import sys
import gc
import math
import warnings
import numpy as np
import numpy.fft
from repmat import repmat
import fld_tools as ft
from compute_nu_T import *
import global_vars as glb
from global_vars import ng

# Computes Leith dissipation on a nonuniform grid
    
def compute_nu_hp_leith_sep_diag(U,V,Up,Vp,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    (dU_D,dV_D,dU_Z,dV_Z,ViscA4D,ViscA4Z) = compute_nu_hp_leith_sep(U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);

    return (dU_D,dV_D,dU_Z,dV_Z,ViscA4D,ViscA4Z);

def compute_nu_hp_leith_sep_diag_u(U,V,Up,Vp,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    (dU_D,dV_D,dU_Z,dV_Z,_,_) = compute_nu_hp_leith_sep(U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);

    nz = U.shape[2];
    nzu = zoversamp*nz;
    thknssUni = ft.get_thknssUni(thknss);
    (uthknss, vthknss) = ft.get_uv_thknss(thknss);

    (dU_D,_) = ft.get_4D_vert_uniform_field(dU_D,thknssUni,nzu,ns=True);
    (dV_D,_) = ft.get_4D_vert_uniform_field(dV_D,thknssUni,nzu,ns=True);
    (dU_Z,_) = ft.get_4D_vert_uniform_field(dU_Z,thknssUni,nzu,ns=True);
    (dV_Z,_) = ft.get_4D_vert_uniform_field(dV_Z,thknssUni,nzu,ns=True);

    return (dU_D,dV_D,dU_Z,dV_Z);

def compute_nu_hp_leith_sep_viscA4(U,V,Up,Vp,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    (_,_,_,_,viscA4_D,viscA4_Z) = compute_nu_hp_leith_sep(U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);

    return (viscA4_D,viscA4_Z);

def compute_nu_hp_leith_sep_k(U,V,Up,Vp,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    (dU_D,dV_D,dU_Z,dV_Z,_,_) = compute_nu_hp_leith_sep(U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);
    (T_D,k_out) = compute_nu_T_k(dU_D,dV_D,Up,Vp,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    (T_Z,k_out) = compute_nu_T_k(dU_Z,dV_Z,Up,Vp,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);

    return (T_D,T_Z,k_out);

def compute_nu_hp_leith_sep_m(U,V,Up,Vp,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    (dU_D,dV_D,dU_Z,dV_Z,_,_) = compute_nu_hp_leith_sep(U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);
    (T_D,m_out) = compute_nu_T_m(dU_D,dV_D,Up,Vp,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    (T_Z,m_out) = compute_nu_T_m(dU_Z,dV_Z,Up,Vp,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);

    return (T_D,T_Z,m_out);

def compute_nu_hp_leith_sep_k_z(U,V,Up,Vp,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    (dU_D,dV_D,dU_Z,dV_Z,_,_) = compute_nu_hp_leith_sep(U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);
    (T_D,k_out) = compute_nu_T_k_z(dU_D,dV_D,Up,Vp,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    (T_Z,k_out) = compute_nu_T_k_z(dU_Z,dV_Z,Up,Vp,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);

    return (T_D,T_Z,k_out);

def compute_nu_hp_leith_sep_m_xy(U,V,Up,Vp,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    (dU_D,dV_D,dU_Z,dV_Z,_,_) = compute_nu_hp_leith_sep(U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);
    (T_D,m_out) = compute_nu_T_m_xy(dU_D,dV_D,Up,Vp,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    (T_Z,m_out) = compute_nu_T_m_xy(dU_Z,dV_Z,Up,Vp,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);

    return (T_D,T_Z,m_out);



def compute_nu_hp_leith_sep(U,V,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    print('in compute_nu_hp_leith_sep()')
    print('viscC4leith = ' + str(glb.viscC4leith))
    print('viscC4leith = ' + str(glb.viscC4leithD))
    print('leithDivOff = ' + str(glb.leithDivOff));
    print('dt = ' + str(glb.dt))

    (nx, ny, nz, nt) = U.shape;
    
    # if (post_taper[2]):
    #     raise RuntimeError('post_taper not built to handle vertical filt. of dKEdiv');

    leith4fac =0.125*(glb.viscC4leith/np.pi)**3;
    leithD4fac =0.125*(glb.viscC4leithD/np.pi)**3;

    # Add ghost cells to fields to handle periodicity in position space.
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits
    U = ft.pad_field_3D(U,ng,prd); V = ft.pad_field_3D(V,ng,prd);  thknss = ft.pad_field_3D(thknss,ng,prd);
    dxc = ft.pad_field_2D(dxc,ng); dxu = ft.pad_field_2D(dxu,ng); dxv = ft.pad_field_2D(dxv,ng); dxq = ft.pad_field_2D(dxq,ng);
    dyc = ft.pad_field_2D(dyc,ng); dyu = ft.pad_field_2D(dyu,ng); dyv = ft.pad_field_2D(dyv,ng); dyq = ft.pad_field_2D(dyq,ng); 

    thknssUni = ft.get_thknssUni(thknss);
    (hfacW, hfacS) = ft.get_uv_hfac(thknss);
    hfacC = ft.get_hfac(thknss);
    thknssZ = ft.get_q_thknss(thknss);

    hdiv = ft.get_nu_hdiv(U,V,hfacC,hfacW,hfacS,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng);
    vort3 = ft.get_nu_relvort3(U,V,thknssZ,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng,glb.no_slip_sides);

    # calc dhdiv/dx
    dxc3D = repmat(dxc,(1,1,nz+2*ng,nt));
    dyc3D = repmat(dyc,(1,1,nz+2*ng,nt));
    divDx = (hdiv[gx0:gxn,gy0:gyn,gz0:gzn,:]-hdiv[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:])/dxc3D[gx0:gxn,gy0:gyn,gz0:gzn,:];
    divDy = (hdiv[gx0:gxn,gy0:gyn,gz0:gzn,:]-hdiv[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:])/dyc3D[gx0:gxn,gy0:gyn,gz0:gzn,:];
    divDx = ft.pad_field_3D(divDx,ng,prd);
    divDy = ft.pad_field_3D(divDy,ng,prd);
    del dxc3D, dyc3D;

    # dvort/dx
    dxq3D = repmat(dxq,(1,1,nz+2*ng,nt));
    dyq3D = repmat(dyq,(1,1,nz+2*ng,nt));
    maskS = ft.get_mask_from_hfac(hfacS);
    vrtDx = maskS[gx0:gxn,gy0:gyn,gz0:gzn,:]*(vort3[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]-vort3[gx0:gxn,gy0:gyn,gz0:gzn,:])/dxq3D[gx0:gxn,gy0:gyn,gz0:gzn,:];
    vrtDx = ft.pad_field_3D(vrtDx,ng,prd);
    del maskS;
            
    # dvort/dy   
    maskW = ft.get_mask_from_hfac(hfacW);
    vrtDy = maskW[gx0:gxn,gy0:gyn,gz0:gzn,:]*(vort3[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]-vort3[gx0:gxn,gy0:gyn,gz0:gzn,:])/dyq3D[gx0:gxn,gy0:gyn,gz0:gzn,:];
    vrtDy = ft.pad_field_3D(vrtDy,ng,prd);
    del maskW, hdiv, vort3;
    
##### Calc On Div Points #####
    
    L2 = dxu*dyv;
    L3 = L2**(1.5);
    L4rdt = 0.03125*L2**2/glb.dt;
    L5 = (L2*L3);
            
    grdVrt = np.maximum( np.abs(vrtDx[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]), np.abs(vrtDx[gx0:gxn,gy0:gyn,gz0:gzn,:]) );
    grdVrt = np.maximum( grdVrt, np.abs(vrtDy[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]) );
    grdVrt = np.maximum( grdVrt, np.abs(vrtDy[gx0:gxn,gy0:gyn,gz0:gzn,:])   );
          
    grdDiv = np.maximum( np.abs(divDx[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]), np.abs(divDx[gx0:gxn,gy0:gyn,gz0:gzn,:]) );
    grdDiv = np.maximum( grdDiv, np.abs(divDy[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]) );
    grdDiv = np.maximum( grdDiv, np.abs(divDy[gx0:gxn,gy0:gyn,gz0:gzn,:])   );

    L5_3D = repmat(L5,(1,1,nz+2*ng,nt));
    viscA4_D = (leith4fac*grdVrt + leithD4fac*grdDiv)*L5_3D[gx0:gxn,gy0:gyn,gz0:gzn,:];
    del L5_3D;

    # BiHarmonic on Div.u points
    L4rdt3D = repmat(L4rdt,(1,1,nz+2*ng,nt));
    viscA4_D = glb.viscA4D + glb.viscA4Grid*L4rdt3D[gx0:gxn,gy0:gyn,gz0:gzn,:] + viscA4_D;
    ft.strip_nan_inf(viscA4_D);
    viscA4_D = np.minimum(viscA4_D,glb.viscA4Max);
    viscA4_D = np.minimum(glb.viscA4GridMax*L4rdt3D[gx0:gxn,gy0:gyn,gz0:gzn,:],viscA4_D);

    viscA4_D = ft.pad_field_3D(viscA4_D,ng,prd);
    del L4rdt3D;



##### Calc On Vort Points #####
    
    L2 = dxv*dyu;
    L3 = L2**(1.5);
    L4rdt = 0.03125*L2**2/glb.dt;
    L5 = (L2*L3);
    
    grdVrt = np.maximum( np.abs(vrtDx[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]), np.abs(vrtDx[gx0:gxn,gy0:gyn,gz0:gzn,:]) );
    grdVrt = np.maximum( grdVrt, np.abs(vrtDy[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]) );
    grdVrt = np.maximum( grdVrt, np.abs(vrtDy[gx0:gxn,gy0:gyn,gz0:gzn,:])   );

    grdDiv = np.maximum( np.abs(divDx[gx0:gxn,gy0:gyn,gz0:gzn,:]), np.abs(divDx[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]) );
    grdDiv = np.maximum( grdDiv, np.abs(divDy[gx0:gxn,gy0:gyn,gz0:gzn,:])   );
    grdDiv = np.maximum( grdDiv, np.abs(divDy[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]) );
    
    L5_3D = repmat(L5,(1,1,nz+2*ng,nt));
    viscA4_Z = (leith4fac*grdVrt + leithD4fac*grdDiv)*L5_3D[gx0:gxn,gy0:gyn,gz0:gzn,:];
    del L5_3D;

    L4rdt3D = repmat(L4rdt,(1,1,nz+2*ng,nt));
    viscA4_Z = glb.viscA4Z + glb.viscA4Grid*L4rdt3D[gx0:gxn,gy0:gyn,gz0:gzn,:] + viscA4_Z;
    ft.strip_nan_inf(viscA4_Z);
    viscA4_Z = np.minimum(viscA4_Z,glb.viscA4Max);
    viscA4_Z = np.minimum(glb.viscA4GridMax*L4rdt3D[gx0:gxn,gy0:gyn,gz0:gzn,:],viscA4_Z);

    viscA4_Z = ft.pad_field_3D(viscA4_Z,ng,prd);
    
    del divDx, divDy, vrtDx, vrtDy, L2, L3, L4rdt, L5, L4rdt3D;



    ########## Calc Tendencies ##########

    if (glb.leithDivOff):
        viscA4_D=0*viscA4_D;

    (dU_D,dV_D) = ft.get_nu_vi_hdissip(U,V,viscA4_D,0*viscA4_Z,thknssZ,hfacC,ft.get_q_hfac(thknss),hfacW,hfacS,
                                   dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,
                                   nx,ny,nz,nt,ng);
    (dU_Z,dV_Z) = ft.get_nu_vi_hdissip(U,V,0*viscA4_D,viscA4_Z,thknssZ,hfacC,ft.get_q_hfac(thknss),hfacW,hfacS,
                                   dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,
                                   nx,ny,nz,nt,ng);

    return (dU_D[gx0:gxn,gy0:gyn,gz0:gzn,:],dV_D[gx0:gxn,gy0:gyn,gz0:gzn,:],dU_Z[gx0:gxn,gy0:gyn,gz0:gzn,:],dV_Z[gx0:gxn,gy0:gyn,gz0:gzn,:],viscA4_D[gx0:gxn,gy0:gyn,gz0:gzn,:],viscA4_Z[gx0:gxn,gy0:gyn,gz0:gzn,:])






