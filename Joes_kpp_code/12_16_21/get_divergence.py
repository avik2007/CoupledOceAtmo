import numpy as np
from repmat import repmat
from pad_field import *
from strip_nan_inf import *

def get_divergence(u_in,v_in,w_in,dx,dy,dz):

    (nx,ny,nz,_) = w_in.shape;
    
    u = pad_field_3D(u_in[0:nx,0:ny,0:nz,:]);
    v = pad_field_3D(v_in[0:nx,0:ny,0:nz,:]);
    w = pad_field_3D(w_in[0:nx,0:ny,0:nz,:]);
    
    d = -((u[1:-1,1:-1,1:-1,:]-u[2:,1:-1,1:-1,:])/dx 
        +   (v[1:-1,1:-1,1:-1,:]-v[1:-1,2:,1:-1,:])/dy 
        +   (w[1:-1,1:-1,2:,:]-w[1:-1,1:-1,1:-1,:])/dz);

    return d;

def get_divergence_nu(u_in,v_in,w_in,dxu_in,dyv_in,dxq_in,dyq_in,thknss_in):

    (nx,ny,nz,nt) = w_in.shape;

    ng = 2; # number of ghost cells
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    u = pad_field_3D(u_in[:nx,:ny,:nz,:],ng);
    v = pad_field_3D(v_in[:nx,:ny,:nz,:],ng);
    w = pad_field_3D(w_in[:nx,:ny,:nz,:],ng);
    thknss = pad_field_3D(thknss_in[:nx,:ny,:nz,:],ng);

    dxu = pad_field_2D(dxu_in,ng); dxq = pad_field_2D(dxq_in,ng);
    dyv = pad_field_2D(dyv_in,ng); dyq = pad_field_2D(dyq_in,ng);

    dxu = repmat(dxu,(1,1,nz+2*ng,nt)); dxq = repmat(dxq,(1,1,nz+2*ng,nt));
    dyv = repmat(dyv,(1,1,nz+2*ng,nt)); dyq = repmat(dyq,(1,1,nz+2*ng,nt));

    d = -(
        0.5*u[gx0:gxn,gy0:gyn,gz0:gzn,:]*dyq[gx0:gxn,gy0:gyn,gz0:gzn,:]*(thknss[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+thknss[gx0:gxn,gy0:gyn,gz0:gzn,:])
        - 0.5*u[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]*dyq[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]*(thknss[gx0:gxn,gy0:gyn,gz0:gzn,:]+thknss[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])
        + 0.5*v[gx0:gxn,gy0:gyn,gz0:gzn,:]*dxq[gx0:gxn,gy0:gyn,gz0:gzn,:]*(thknss[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+thknss[gx0:gxn,gy0:gyn,gz0:gzn,:])
        - 0.5*v[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]*dxq[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]*(thknss[gx0:gxn,gy0:gyn,gz0:gzn,:]+thknss[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])
        + w[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:]*dyv[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:]*dxu[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:]
        - w[gx0:gxn,gy0:gyn,gz0:gzn,:]*dyv[gx0:gxn,gy0:gyn,gz0:gzn,:]*dxu[gx0:gxn,gy0:gyn,gz0:gzn,:]
    )/(dxu[gx0:gxn,gy0:gyn,gz0:gzn,:]*dyv[gx0:gxn,gy0:gyn,gz0:gzn,:]*thknss[gx0:gxn,gy0:gyn,gz0:gzn,:]);

    strip_nan_inf(d);

    return d;
