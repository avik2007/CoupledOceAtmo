import warnings
import numpy as np
from repmat import repmat
from pad_field import pad_field_3D
from strip_nan_inf import *

def get_hfac(thknss):
    thknssUni = get_thknssUni(thknss);
    hfac = thknss/thknssUni;
    strip_nan_inf(hfac)
    return hfac;
    
def get_mask_from_hfac(hfacP):
    maskP = (hfacP!=0);
    return maskP;

def get_thknssUni(thknss):
    warnings.warn("thknss needs adapting for HYCOM, etc.");
    (nx,ny,nz,nt) = thknss.shape;
    thknssUni = np.max(np.max(thknss,axis=0,keepdims=True),axis=1,keepdims=True);
    thknssUni = repmat(thknssUni,(nx,ny,1,1));
    return thknssUni;

def get_uv_hfac(thknss):
    (nx,ny,nz,nt) = thknss.shape;
    thknssUni = np.amax(np.amax(thknss,axis=0,keepdims=True),axis=1,keepdims=True);
    thknssUni = repmat(thknssUni,(nx,ny,1,1));
    
    # use one-sided 1st-order for now.
    uthknss = np.zeros([nx,ny,nz,nt]); vthknss = np.zeros([nx,ny,nz,nt]);
    
    uthknss[1:,:,:,:] = np.minimum(thknss[0:-1,:,:,:],thknss[1:,:,:,:]);
    uthknss[0,:,:,:] = thknss[0,:,:,:];
    hfacW = uthknss/thknssUni;
    
    vthknss[:,1:,:,:] = np.minimum(thknss[:,0:-1,:,:],thknss[:,1:,:,:]);
    vthknss[:,0,:,:] = thknss[:,0,:,:];
    hfacS = vthknss/thknssUni;

    strip_nan_inf(hfacW); strip_nan_inf(hfacS);
    return (hfacW,hfacS);
    
def get_q_hfac(thknss_in):

    (nx,ny,nz,nt) = thknss_in.shape;
    thknss = thknss_in.copy();
    ng = 2;  gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits
    thknss = pad_field_3D(thknss,ng);

    thknss_q = np.minimum(thknss[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:],thknss[gx0:gxn,gy0:gyn,gz0:gzn,:]);
    thknss_q = np.minimum(thknss_q,thknss[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]);
    thknss_q = np.minimum(thknss_q,thknss[gx0-1:gxn-1,gy0-1:gyn-1,gz0:gzn,:]);

    return get_hfac(thknss_q);

def get_uv_thknss(thknss):
    (nx,ny,nz,nt) = thknss.shape;

    # use one-sided 1st-order for now.
    uthknss = np.zeros([nx,ny,nz,nt]); vthknss = np.zeros([nx,ny,nz,nt]);
    
    uthknss[1:,:,:,:] = np.minimum(thknss[0:-1,:,:,:],thknss[1:,:,:,:]);
    uthknss[0,:,:,:] = thknss[0,:,:,:];
    
    vthknss[:,1:,:,:] = np.minimum(thknss[:,0:-1,:,:],thknss[:,1:,:,:]);
    vthknss[:,0,:,:] = thknss[:,0,:,:];

    return (uthknss,vthknss);
    
def get_q_thknss(thknss_in):

    (nx,ny,nz,nt) = thknss_in.shape;
    thknss = thknss_in.copy();
    ng = 2;  gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits
    thknss = pad_field_3D(thknss,ng);

    thknss_q = np.minimum(thknss[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:],thknss[gx0:gxn,gy0:gyn,gz0:gzn,:]);
    thknss_q = np.minimum(thknss_q,thknss[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]);
    thknss_q = np.minimum(thknss_q,thknss[gx0-1:gxn-1,gy0-1:gyn-1,gz0:gzn,:]);

    return thknss_q;

def calc_nzmax(hfacC):
    (nx,ny,nz,nt) = hfacC.shape;

    ### JS - Probably SLOW!
    f = np.zeros([nx,ny,1,nt]);
    for i in range(0,nx):
        for j in range(0,ny):
            for k in range(0,nz):
                for t in range(0,nt):
                    if (hfacC[i,j,k,t]!=0):
                        f[i,j,0,t] = k;

    return f;

def get_dzgrid(thknss):
    (nx,ny,nz,nt) = thknss.shape;

    dzgrid = np.zeros([1,1,nz+1,nt]);
    tmp_grid = np.amax(np.amax(thknss,axis=0,keepdims=True),axis=1,keepdims=True);
    dzgrid[0,0,0,:] = 0.5*tmp_grid[0,0,0,:]; 
    dzgrid[0,0,1:-1,:] = 0.5*(tmp_grid[0,0,0:-1,:]+tmp_grid[0,0,1:,:]);
    dzgrid[0,0,-1,:] = 100*0.5*(tmp_grid[0,0,-1,:]);
    dzgrid = repmat(dzgrid,(nx,ny,1,1));

    return dzgrid;

def get_thknssC(thknss):
    (nx,ny,nz,nt) = thknss.shape;

    thknssC = np.zeros((nx,ny,nz+1,nt));

    thknssC[:,:,0,:] = 0.5*thknss[:,:,0,:];
    thknssC[:,:,1:-1,:] = 0.5*(thknss[:,:,0:-1,:]+thknss[:,:,1:,:]);
    thknssC[:,:,-1,:] = 0.5*thknss[:,:,-1,:];

    return thknssC;

def get_volume_average(f,thknss,dx,dy):

    (nx,ny,nz,nt) = f.shape;
    return np.sum(f*thknss*repmat(dx*dy,(1,1,nz,nt)))/np.sum(thknss*repmat(dx*dy,(1,1,nz,nt)));










