import sys
import gc
import warnings
import math
import numpy as np
from repmat import repmat
from pad_field import *

def get_nu_relvort3(U,V,thknssZ,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,nx,ny,nz,nt,ng,no_slip_sides=1,prd=np.array([1,1,1])):

    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    # Compute Vort3
    dxc3D = repmat(dxc,(1,1,nz+2*ng,nt));    
    dyc3D = repmat(dyc,(1,1,nz+2*ng,nt));    
    rAz3D = repmat(dxv,(1,1,nz+2*ng,nt))*repmat(dyu,(1,1,nz+2*ng,nt));
    vort3 = ((V[gx0:gxn,gy0:gyn,gz0:gzn,:]*dyc3D[gx0:gxn,gy0:gyn,gz0:gzn,:] - V[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]*dyc3D[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:])
             -(U[gx0:gxn,gy0:gyn,gz0:gzn,:]*dxc3D[gx0:gxn,gy0:gyn,gz0:gzn,:] - U[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]*dxc3D[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]))/rAz3D[gx0:gxn,gy0:gyn,gz0:gzn,:];
    vort3 = pad_field_3D(vort3,ng,prd);
    del dxc3D, dyc3D, rAz3D;
    filt = (thknssZ==0);
    vort3 = vort3 + no_slip_sides*filt*vort3; # vorticity is double at the BC's when no-slip sides are enabled in vecinv version.
    del filt

    return vort3;
