import numpy as np

def get_bot_Zu(Zu,depth2d):
    (nx,ny,_,nt) = depth2d.shape;
    botZ = np.zeros([nx,ny,1,nt]).astype(int);
    for i in range(0,nx):
        for j in range(0,ny):
            for t in range(0,nt):
                botZ[i,j,0,t] = np.sum(np.squeeze(Zu[i,j,:]<depth2d[i,j,0,t]));

    return botZ;

def get_bot_Znu(f):
    (nx,ny,_,nt) = f.shape;
    botZ = np.zeros([nx,ny,1,nt]).astype(int); maxf = np.nanmax(f); thrs = 1e-6;
    for i in range(0,nx):
        for j in range(0,ny):
            for t in range(0,nt):
                tmp = np.cumsum(np.squeeze(f[i,j,:,t]));
                botZ[i,j,0,t] = np.nansum(tmp<(np.nanmax(tmp)-maxf*thrs));

    return botZ;
