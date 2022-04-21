import numpy as np

def get_uv_f(f):

    # use one-sided 1st-order for now.
    (nx,ny,nz,nt) = f.shape;
    uf = np.zeros([nx+1,ny,nz,nt]); vf = np.zeros([nx,ny+1,nz,nt]);
    uf[1:-1,:,:,:] = (0.5*f[0:-1,:,:,:]+0.5*f[1:,:,:,:]);
    uf[0,:,:,:] = f[0,:,:,:];
    uf[-1,:,:,:] = f[-1,:,:,:];
    vf[:,1:-1,:,:] = (0.5*f[:,0:-1,:,:]+0.5*f[:,1:,:,:]);
    vf[:,0,:,:] = f[:,0,:,:];
    vf[:,-1,:,:] = f[:,-1,:,:];

    return (uf,vf)
