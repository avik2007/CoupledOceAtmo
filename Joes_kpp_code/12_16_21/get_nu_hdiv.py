import sys
import gc
import warnings
import math
import numpy as np
from repmat import repmat
from pad_field import *
from strip_nan_inf import *

def get_nu_hdiv(U,V,hfacC,hfacW,hfacS,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng,prd=np.array([1,1,1])):

    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    # MITgcm's algorithm
    dxq3D = repmat(dxq,(1,1,nz+2*ng,nt));    
    dyq3D = repmat(dyq,(1,1,nz+2*ng,nt));    
    dxu3D = repmat(dxu,(1,1,nz+2*ng,nt));    
    dyv3D = repmat(dyv,(1,1,nz+2*ng,nt));    
    hdiv = (((U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]*dyq3D[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]*hfacW[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:] - U[gx0:gxn,gy0:gyn,gz0:gzn,:]*dyq3D[gx0:gxn,gy0:gyn,gz0:gzn,:]*hfacW[gx0:gxn,gy0:gyn,gz0:gzn,:])
                                +(V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]*dxq3D[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]*hfacS[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:] - V[gx0:gxn,gy0:gyn,gz0:gzn,:]*dxq3D[gx0:gxn,gy0:gyn,gz0:gzn,:]*hfacS[gx0:gxn,gy0:gyn,gz0:gzn,:]))
            /(dxu3D[gx0:gxn,gy0:gyn,gz0:gzn,:]*dyv3D[gx0:gxn,gy0:gyn,gz0:gzn,:]*hfacC[gx0:gxn,gy0:gyn,gz0:gzn,:]));
    hdiv = pad_field_3D(hdiv,ng,prd);
    strip_nan_inf(hdiv);

    del dxq3D, dyq3D, dxu3D, dyv3D;
      
    return hdiv;
