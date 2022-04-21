import sys
import gc
import warnings
import math
import numpy as np
from repmat import repmat
from pad_field import *
from strip_nan_inf import *
from vert_grid_mitgcm import *
from get_nu_laplacian import *
from get_nu_relvort3 import *
from get_nu_hdiv import *



def get_nu_vi_hdissip(U,V,viscA4_D,viscA4_Z,thknssZ,hfacC,hfacZ,hfacW,hfacS,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng,no_slip_sides=1,prd=np.array([1,1,1])):

    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    hdiv = get_nu_hdiv(U,V,hfacC,hfacW,hfacS,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng);
    vort3 = get_nu_relvort3(U,V,thknssZ,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng,no_slip_sides);

    del2u = get_nu_vi_laplacian_U(U,hdiv,vort3,hfacZ,hfacW,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng);
    del2v = get_nu_vi_laplacian_V(V,hdiv,vort3,hfacZ,hfacS,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng);

    del hdiv,vort3;

    dstar = get_nu_hdiv(del2u,del2v,hfacC,hfacW,hfacS,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng);
    zstar = get_nu_relvort3(del2u,del2v,thknssZ,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng,no_slip_sides);

    del del2u,del2v;

    dxc3D = repmat(dxc,(1,1,nz+2*ng,nt));
    dyc3D = repmat(dyc,(1,1,nz+2*ng,nt));
    dxq3D = repmat(dxq,(1,1,nz+2*ng,nt));
    dyq3D = repmat(dyq,(1,1,nz+2*ng,nt));

    uD4 = - (hfacZ[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]*zstar[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]*viscA4_Z[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:] 
              - hfacZ[gx0:gxn,gy0:gyn,gz0:gzn,:]*zstar[gx0:gxn,gy0:gyn,gz0:gzn,:]*viscA4_Z[gx0:gxn,gy0:gyn,gz0:gzn,:])/(hfacW[gx0:gxn,gy0:gyn,gz0:gzn,:]*dyq3D[gx0:gxn,gy0:gyn,gz0:gzn,:]);
    strip_nan_inf(uD4);
    uD4 = uD4 + (dstar[gx0:gxn,gy0:gyn,gz0:gzn,:]*viscA4_D[gx0:gxn,gy0:gyn,gz0:gzn,:] - dstar[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]*viscA4_D[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:])/dxc3D[gx0:gxn,gy0:gyn,gz0:gzn,:] 
    uD4 = uD4*get_mask_from_hfac(hfacW[gx0:gxn,gy0:gyn,gz0:gzn,:]);

    vD4 = (hfacZ[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]*zstar[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]*viscA4_Z[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:] 
           - hfacZ[gx0:gxn,gy0:gyn,gz0:gzn,:]*zstar[gx0:gxn,gy0:gyn,gz0:gzn,:]*viscA4_Z[gx0:gxn,gy0:gyn,gz0:gzn,:])/(hfacS[gx0:gxn,gy0:gyn,gz0:gzn,:]*dxq3D[gx0:gxn,gy0:gyn,gz0:gzn,:]);
    strip_nan_inf(vD4);
    vD4 = vD4 + (dstar[gx0:gxn,gy0:gyn,gz0:gzn,:]*viscA4_D[gx0:gxn,gy0:gyn,gz0:gzn,:] - dstar[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]*viscA4_D[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:])/dyc3D[gx0:gxn,gy0:gyn,gz0:gzn,:] 
    vD4 = vD4*get_mask_from_hfac(hfacS[gx0:gxn,gy0:gyn,gz0:gzn,:]);

    uD4 = pad_field_3D(uD4,ng,prd);     
    vD4 = pad_field_3D(vD4,ng,prd);

    del dxc3D,dyc3D,dxq3D,dyq3D;

    return (-uD4,-vD4);



def get_nu_ff_hdissip(U,V,hdiv,vort3,viscA4_D,viscA4_Z,thknss,thknssZ,thknssUni,hfacW,hfacS,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng,prd=np.array([1,1,1])):

    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

### Calc U Tendency 

    del2u = get_nu_ff_laplacian_U(U,thknss,thknssZ,thknssUni,hfacW,hfacS,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng);
    # del2u = get_nu_vi_laplacian_U(U,hdiv,vort3,hfacZ,hfacW,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng);

    # (note: cos factors are 1 unless using spherical-polar grid)
    fZon = np.zeros((nx,ny,nz,nt)); 
    for k in range(gz0,gzn):
        for t in range(0,nt):
            fZon[:,:,k-ng,t] = (
                dyq[gx0:gxn,gy0:gyn]*thknss[gx0:gxn,gy0:gyn,k,t]
                *( 
                   viscA4_D[gx0:gxn,gy0:gyn,k,t]*(del2u[gx0+1:gxn+1,gy0:gyn,k,t]-del2u[gx0:gxn,gy0:gyn,k,t])
                )/dxv[gx0:gxn,gy0:gyn]);
    fZon = pad_field_3D(fZon,ng,prd);
                
    fMer = np.zeros((nx,ny,nz,nt));
    for k in range(gz0,gzn):
        for t in range(0,nt):
            fMer[:,:,k-ng,t] = dxv[gx0:gxn,gy0:gyn]*thknssZ[gx0:gxn,gy0:gyn,k,t]*(
                viscA4_Z[gx0:gxn,gy0:gyn,k,t]*(del2u[gx0:gxn,gy0:gyn,k,t]-del2u[gx0:gxn,gy0-1:gyn-1,k,t])
                )/dyu[gx0:gxn,gy0:gyn];
    fMer = pad_field_3D(fMer,ng,prd);
    
    del del2u;

    dU = np.zeros((nx,ny,nz,nt));
    for k in range(gz0,gzn):
        for t in range(0,nt):
            dU[:,:,k-ng,t] = -(( ( fZon[gx0:gxn,gy0:gyn,k,t] - fZon[gx0-1:gxn-1,gy0:gyn,k,t] )
                                 +( fMer[gx0:gxn,gy0+1:gyn+1,k,t] - fMer[gx0:gxn,gy0:gyn,k,t] )) 
                               /(thknssUni[gx0:gxn,gy0:gyn,k,t]*dxc[gx0:gxn,gy0:gyn]*dyq[gx0:gxn,gy0:gyn]*hfacW[gx0:gxn,gy0:gyn,k,t]));
    strip_nan_inf(dU);
    
    del fZon, fMer;



### Calc V Tendency 

    del2v = get_nu_ff_laplacian_V(V,thknss,thknssZ,thknssUni,hfacW,hfacS,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng);
    # del2v = get_nu_vi_laplacian_V(V,hdiv,vort3,get_q_hfac(thknss),hfacS,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng);
    del hdiv, vort3;

    fZon = np.zeros((nx,ny,nz,nt));
    for k in range(gz0,gzn):
        for t in range(0,nt):
            fZon[:,:,k-ng,t] = (dyu[gx0:gxn,gy0:gyn]*thknssZ[gx0:gxn,gy0:gyn,k,t]*(
                + viscA4_Z[gx0:gxn,gy0:gyn,k,t]*(del2v[gx0:gxn,gy0:gyn,k,t]-del2v[gx0-1:gxn-1,gy0:gyn,k,t])
                )/dxv[gx0:gxn,gy0:gyn]);
    fZon = pad_field_3D(fZon,ng,prd);
  
    fMer = np.zeros((nx,ny,nz,nt));
    for k in range(gz0,gzn):
        for t in range(0,nt):
            fMer[:,:,k-ng,t] = dxu[gx0:gxn,gy0:gyn]*thknss[gx0:gxn,gy0:gyn,k,t]*(
                + viscA4_D[gx0:gxn,gy0:gyn,k,t]*(del2v[gx0:gxn,gy0+1:gyn+1,k,t]-del2v[gx0:gxn,gy0:gyn,k,t])
                )/dyv[gx0:gxn,gy0:gyn];
    fMer = pad_field_3D(fMer,ng,prd);

    del del2v;
    
    dV = np.zeros((nx,ny,nz,nt));
    for k in range(gz0,gzn):
        for t in range(0,nt):
            dV[:,:,k-ng,t] = -(( ( fZon[gx0+1:gxn+1,gy0:gyn,k,t]  - fZon[gx0:gxn,gy0:gyn,k,t]) 
                +( fMer[gx0:gxn,gy0:gyn,k,t]  - fMer[gx0:gxn,gy0-1:gyn-1,k,t]) ) 
                /(thknssUni[gx0:gxn,gy0:gyn,k,t]*dxq[gx0:gxn,gy0:gyn]*dyc[gx0:gxn,gy0:gyn]*hfacS[gx0:gxn,gy0:gyn,k,t]));
    strip_nan_inf(dV);

    del fZon, fMer;
    del hfacS, hfacW, thknssZ, hfacC;

    dU = pad_field_3D(dU,ng,prd);     
    dV = pad_field_3D(dV,ng,prd);
 
    return (dU,dV);
