import numpy as np
from pad_field import pad_field_3D

def get_shsq(U,V):
    (nx,ny,nz,nt) = U.shape; ng = 2;  prd = np.array([1,1,1]);
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    U = pad_field_3D(U,ng,prd); V = pad_field_3D(V,ng,prd);

    shsq = 0.5*( 
        (U[gx0:gxn,gy0:gyn,gz0:gzn,:] - U[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:]) * 
        (U[gx0:gxn,gy0:gyn,gz0:gzn,:] - U[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:]) + 
        (U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:] - U[gx0+1:gxn+1,gy0:gyn,gz0+1:gzn+1,:])* 
        (U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:] - U[gx0+1:gxn+1,gy0:gyn,gz0+1:gzn+1,:]) + 
        (V[gx0:gxn,gy0:gyn,gz0:gzn,:] - V[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])* 
        (V[gx0:gxn,gy0:gyn,gz0:gzn,:] - V[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:]) + 
        (V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:] - V[gx0:gxn,gy0+1:gyn+1,gz0+1:gzn+1,:])* 
        (V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:] - V[gx0:gxn,gy0+1:gyn+1,gz0+1:gzn+1,:]));
    
    shsq = 0.5*shsq + 0.125*( 
        (U[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]-U[gx0:gxn,gy0-1:gyn-1,gz0+1:gzn+1,:])* 
        (U[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]-U[gx0:gxn,gy0-1:gyn-1,gz0+1:gzn+1,:]) + 
        (U[gx0+1:gxn+1,gy0-1:gyn-1,gz0:gzn,:]-U[gx0+1:gxn+1,gy0-1:gyn-1,gz0+1:gzn+1,:])* 
        (U[gx0+1:gxn+1,gy0-1:gyn-1,gz0:gzn,:]-U[gx0+1:gxn+1,gy0-1:gyn-1,gz0+1:gzn+1,:]) + 
        (U[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]-U[gx0:gxn,gy0+1:gyn+1,gz0+1:gzn+1,:])* 
        (U[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]-U[gx0:gxn,gy0+1:gyn+1,gz0+1:gzn+1,:]) + 
        (U[gx0+1:gxn+1,gy0+1:gyn+1,gz0:gzn,:]-U[gx0+1:gxn+1,gy0+1:gyn+1,gz0+1:gzn+1,:])* 
        (U[gx0+1:gxn+1,gy0+1:gyn+1,gz0:gzn,:]-U[gx0+1:gxn+1,gy0+1:gyn+1,gz0+1:gzn+1,:]) + 
        (V[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]-V[gx0-1:gxn-1,gy0:gyn,gz0+1:gzn+1,:])* 
        (V[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]-V[gx0-1:gxn-1,gy0:gyn,gz0+1:gzn+1,:]) + 
        (V[gx0-1:gxn-1,gy0+1:gyn+1,gz0:gzn,:]-V[gx0-1:gxn-1,gy0+1:gyn+1,gz0+1:gzn+1,:])* 
        (V[gx0-1:gxn-1,gy0+1:gyn+1,gz0:gzn,:]-V[gx0-1:gxn-1,gy0+1:gyn+1,gz0+1:gzn+1,:]) + 
        (V[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]-V[gx0+1:gxn+1,gy0:gyn,gz0+1:gzn+1,:])* 
        (V[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]-V[gx0+1:gxn+1,gy0:gyn,gz0+1:gzn+1,:]) + 
        (V[gx0+1:gxn+1,gy0+1:gyn+1,gz0:gzn,:]-V[gx0+1:gxn+1,gy0+1:gyn+1,gz0+1:gzn+1,:])* 
        (V[gx0+1:gxn+1,gy0+1:gyn+1,gz0:gzn,:]-V[gx0+1:gxn+1,gy0+1:gyn+1,gz0+1:gzn+1,:]));
        
    return shsq;
