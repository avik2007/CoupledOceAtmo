import sys
import gc
import warnings
import math
import numpy as np
from repmat import repmat
from pad_field import *
from strip_nan_inf import *
from vert_grid_mitgcm import *

# Compute the laplacian of vel fields, assume I/O has padding

def get_nu_ff_laplacian_U(U,thknss,thknssZ,thknssUni,hfacW,hfacS,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng):
    
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    dyv3D = repmat(dyv,(1,1,nz+2*ng,nt));    
    dxu3D = repmat(dxu,(1,1,nz+2*ng,nt));    
    fZon = (thknss[gx0:gxn,gy0:gyn,gz0:gzn,:]
                *(U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]-U[gx0:gxn,gy0:gyn,gz0:gzn,:])
                *dyv3D[gx0:gxn,gy0:gyn,gz0:gzn,:]/dxu3D[gx0:gxn,gy0:gyn,gz0:gzn,:]);
    fZon = pad_field_3D(fZon,ng);
    del dyv3D, dxu3D;

    dxv3D = repmat(dxv,(1,1,nz+2*ng,nt));    
    dyu3D = repmat(dyu,(1,1,nz+2*ng,nt));    
    fMer = (thknssZ[gx0:gxn,gy0:gyn,gz0:gzn,:]
                *(U[gx0:gxn,gy0:gyn,gz0:gzn,:]-U[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:])
                *dxv3D[gx0:gxn,gy0:gyn,gz0:gzn,:]/dyu3D[gx0:gxn,gy0:gyn,gz0:gzn,:]);
    fMer = pad_field_3D(fMer,ng);
    del dxv3D, dyu3D;

    dxc3D = repmat(dxc,(1,1,nz+2*ng,nt));    
    dyq3D = repmat(dyq,(1,1,nz+2*ng,nt));    
    del2u = ((fZon[gx0:gxn,gy0:gyn,gz0:gzn,:]    - fZon[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]
                + fMer[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]    - fMer[gx0:gxn,gy0:gyn,gz0:gzn,:])
            /(thknssUni[gx0:gxn,gy0:gyn,gz0:gzn,:]*hfacW[gx0:gxn,gy0:gyn,gz0:gzn,:]
              *dxc3D[gx0:gxn,gy0:gyn,gz0:gzn,:]*dyq3D[gx0:gxn,gy0:gyn,gz0:gzn,:]));
    del2u = pad_field_3D(del2u,ng);
    strip_nan_inf(del2u);
    del fZon, fMer, dxc3D, dyq3D;
    return(del2u);

def get_nu_ff_laplacian_V(V,thknss,thknssZ,thknssUni,hfacW,hfacS,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng):
    
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    dxv3D = repmat(dxv,(1,1,nz+2*ng,nt));    
    dyu3D = repmat(dyu,(1,1,nz+2*ng,nt));    
    fZon = (thknssZ[gx0:gxn,gy0:gyn,gz0:gzn,:]
                                *dyu3D[gx0:gxn,gy0:gyn,gz0:gzn,:]
                                *(V[gx0:gxn,gy0:gyn,gz0:gzn,:]-V[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:])
                                /dxv3D[gx0:gxn,gy0:gyn,gz0:gzn,:]);
    fZon = pad_field_3D(fZon,ng);
    del dyu3D, dxv3D;

    dxu3D = repmat(dxu,(1,1,nz+2*ng,nt));    
    dyv3D = repmat(dyv,(1,1,nz+2*ng,nt));    
    fMer = (thknss[gx0:gxn,gy0:gyn,gz0:gzn,:]
                *dxu3D[gx0:gxn,gy0:gyn,gz0:gzn,:]
                *(V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]-V[gx0:gxn,gy0:gyn,gz0:gzn,:])
                /dyv3D[gx0:gxn,gy0:gyn,gz0:gzn,:]);
    fMer = pad_field_3D(fMer,ng);
    del dxu3D, dyv3D;
    
    dxq3D = repmat(dxq,(1,1,nz+2*ng,nt));    
    dyc3D = repmat(dyc,(1,1,nz+2*ng,nt));    
    del2v = ((fZon[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]    - fZon[gx0:gxn,gy0:gyn,gz0:gzn,:]
                                  +fMer[gx0:gxn,gy0:gyn,gz0:gzn,:]    - fMer[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:])
                                 /(thknssUni[gx0:gxn,gy0:gyn,gz0:gzn,:]*hfacS[gx0:gxn,gy0:gyn,gz0:gzn,:]
                                   *dxq3D[gx0:gxn,gy0:gyn,gz0:gzn,:]*dyc3D[gx0:gxn,gy0:gyn,gz0:gzn,:]));
    del2v = pad_field_3D(del2v,ng);
    strip_nan_inf(del2v);
    del fZon, fMer, dxq3D, dyc3D;
    return(del2v);

def get_nu_vi_laplacian_U(U,hdiv,vort3,hfacZ,hfacW,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng):
    
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    maskW = get_mask_from_hfac(hfacW);

    dxc3D = repmat(dxc,(1,1,nz+2*ng,nt));    
    dyq3D = repmat(dyq,(1,1,nz+2*ng,nt));    
    del2u = maskW[gx0:gxn,gy0:gyn,gz0:gzn,:]*((hdiv[gx0:gxn,gy0:gyn,gz0:gzn,:]-hdiv[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:])/dxc3D[gx0:gxn,gy0:gyn,gz0:gzn,:]
                                              - ((hfacZ[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]*vort3[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:] - 
                                                  hfacZ[gx0:gxn,gy0:gyn,gz0:gzn,:]*vort3[gx0:gxn,gy0:gyn,gz0:gzn,:])/(hfacW[gx0:gxn,gy0:gyn,gz0:gzn,:]*dyq3D[gx0:gxn,gy0:gyn,gz0:gzn,:])))

    del2u = pad_field_3D(del2u,ng);
    strip_nan_inf(del2u);
    return(del2u);

def get_nu_vi_laplacian_V(V,hdiv,vort3,hfacZ,hfacS,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng):
    
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    maskS = get_mask_from_hfac(hfacS);

    dxq3D = repmat(dxq,(1,1,nz+2*ng,nt));    
    dyc3D = repmat(dyc,(1,1,nz+2*ng,nt));    
    del2v = maskS[gx0:gxn,gy0:gyn,gz0:gzn,:]*((hdiv[gx0:gxn,gy0:gyn,gz0:gzn,:]-hdiv[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:])/dyc3D[gx0:gxn,gy0:gyn,gz0:gzn,:]
                                              + ((hfacZ[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]*vort3[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:] - 
                                                  hfacZ[gx0:gxn,gy0:gyn,gz0:gzn,:]*vort3[gx0:gxn,gy0:gyn,gz0:gzn,:])/(hfacS[gx0:gxn,gy0:gyn,gz0:gzn,:]*dxq3D[gx0:gxn,gy0:gyn,gz0:gzn,:])))

    del2v = pad_field_3D(del2v,ng);
    strip_nan_inf(del2v);
    return(del2v);
