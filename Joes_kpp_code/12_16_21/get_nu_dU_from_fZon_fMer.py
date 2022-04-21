import sys
import gc
import warnings
import math
import numpy as np
import numpy.fft
from repmat import repmat
import fld_tools as ft

def get_nu_dU_from_fZon_fMer(fZonU,fMerU,fZonV,fMerV,fnu,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    thknss = fnu.thknss;
    (nx, ny, nz, nt) = thknss.shape;
    
    if (post_taper[2]):
        raise RuntimeError('post_taper not built to handle vertical filt. of dKEdiv');

    # Add ghost cells to fields to handle periodicity in position space.
    ng = 2; # number of ghost cells, must be one larger than needed for python indexing
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits
    fZonU = ft.pad_field_3D(fZonU,ng,prd); fZonV = ft.pad_field_3D(fZonV,ng,prd);  thknss = ft.pad_field_3D(thknss,ng,prd);
    fMerU = ft.pad_field_3D(fMerU,ng,prd); fMerV = ft.pad_field_3D(fMerV,ng,prd);  
    dxc = ft.pad_field_2D(fnu.dxc,ng); dxu = ft.pad_field_2D(fnu.dxu,ng); dxv = ft.pad_field_2D(fnu.dxv,ng); dxq = ft.pad_field_2D(fnu.dxq,ng);
    dyc = ft.pad_field_2D(fnu.dyc,ng); dyu = ft.pad_field_2D(fnu.dyu,ng); dyv = ft.pad_field_2D(fnu.dyv,ng); dyq = ft.pad_field_2D(fnu.dyq,ng); 

    (hfacW, hfacS) = ft.get_uv_hfac(thknss);
    thknssUni = ft.get_thknssUni(thknss);

    dU = np.zeros((nx,ny,nz,nt));
    for k in range(gz0,gzn):
        for t in range(0,nt):
            dU[:,:,k-ng,t] = -(( ( fZonU[gx0:gxn,gy0:gyn,k,t] - fZonU[gx0-1:gxn-1,gy0:gyn,k,t] )
                                 +( fMerU[gx0:gxn,gy0+1:gyn+1,k,t] - fMerU[gx0:gxn,gy0:gyn,k,t] )) 
                               /(thknssUni[gx0:gxn,gy0:gyn,k,t]*dxc[gx0:gxn,gy0:gyn]*dyq[gx0:gxn,gy0:gyn]*hfacW[gx0:gxn,gy0:gyn,k,t]));
    ft.strip_nan_inf(dU);

    dV = np.zeros((nx,ny,nz,nt));
    for k in range(gz0,gzn):
        for t in range(0,nt):
            dV[:,:,k-ng,t] = -(( ( fZonV[gx0+1:gxn+1,gy0:gyn,k,t]  - fZonV[gx0:gxn,gy0:gyn,k,t]) 
                                 +( fMerV[gx0:gxn,gy0:gyn,k,t]  - fMerV[gx0:gxn,gy0-1:gyn-1,k,t]) ) 
                               /(thknssUni[gx0:gxn,gy0:gyn,k,t]*dxq[gx0:gxn,gy0:gyn]*dyc[gx0:gxn,gy0:gyn]*hfacS[gx0:gxn,gy0:gyn,k,t]));
    ft.strip_nan_inf(dV);

    return (dU,dV);
