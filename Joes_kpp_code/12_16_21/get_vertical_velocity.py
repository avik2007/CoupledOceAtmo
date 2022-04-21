import numpy as np

def get_uniform_vertical_vel(u,v,dx,dy,dz):

    nx = u.shape[0]-1; ny = u.shape[1]; nz = u.shape[2]; nt = u.shape[3];

    # allocate:
    w = np.zeros([nx,ny,nz,nt]);

    for k in range(0,nz):
        for t in range(0,nt):
            w[:,:,k,t] = dz*(
                          (u[0:-1,:,k,t]-u[1:,:,k,t])/dx 
                        + (v[:,0:-1,k,t]-v[:,1:,k,t])/dy);

    w = np.flip(np.cumsum(np.flip(w,axis=2),axis=2),axis=2);
    return w;

def get_uniform_vertical_vel_mod_grid(u_in,v_in,dx,dy,dz):

    u = u_in.copy(); v = v_in.copy();
    nx = u.shape[0]; ny = u.shape[1]; nz = u.shape[2]; nt = u.shape[3];
    u = np.concatenate((u,u[0:1,:,:,:]),axis=0);
    v = np.concatenate((v,v[:,0:1,:,:]),axis=1);

    # allocate:
    w = np.zeros([nx,ny,nz,nt]);

    for k in range(0,nz):
        for t in range(0,nt):
            w[:,:,k,t] = dz*(
                          (u[0:-1,:,k,t]-u[1:,:,k,t])/dx 
                        + (v[:,0:-1,k,t]-v[:,1:,k,t])/dy);

    w = np.flip(np.cumsum(np.flip(w,axis=2),axis=2),axis=2);
    return w;

