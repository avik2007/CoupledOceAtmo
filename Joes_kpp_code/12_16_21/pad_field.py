import numpy as np

def pad_field_3D(f,ng=1,pbool=np.array([1,1,1])):

    if (ng<1) or (ng>np.amax(f.shape)):
        raise RuntimeError('ng < 1 or ng > max dimension!.')
    
    # initialize
    fp = np.zeros([f.shape[0]+2*ng,f.shape[1]+2*ng,f.shape[2]+2*ng,f.shape[3]]);
    fp[ng:-ng,ng:-ng,ng:-ng,:] = f;
    
    # x pad
    if (pbool[0]==1):
        fp[0:ng,:,:,:] = fp[-2*ng:-ng,:,:,:];
        fp[-ng:,:,:,:] = fp[ng:2*ng,:,:,:];
    elif (pbool[0]<0):
        fp[ng-1,:,:,:] = fp[ng,:,:,:]
        fp[-ng,:,:,:] = fp[-ng-1,:,:,:]
    else:
        fp[ng-1,:,:,:] = 2*fp[ng,:,:,:]-fp[ng+1,:,:,:];
        fp[-ng,:,:,:] = 2*fp[-ng-1,:,:,:]-fp[-ng-2,:,:,:];
    
    # y pad
    if (pbool[1]==1):
        fp[:,0:ng,:,:] = fp[:,-2*ng:-ng,:,:];
        fp[:,-ng:,:,:] = fp[:,ng:2*ng,:,:];
    elif (pbool[1]<0):
        fp[:,ng-1,:,:] = fp[:,ng,:,:]
        fp[:,-ng,:,:] = fp[:,-ng-1,:,:]
    else:
        fp[:,ng-1,:,:] = 2*fp[:,ng,:,:]-fp[:,ng+1,:,:];
        fp[:,-ng,:,:] = 2*fp[:,-ng-1,:,:]-fp[:,-ng-2,:,:];
    
    # z pad
    if (pbool[2]==1):
        fp[:,:,0:ng,:] = fp[:,:,-2*ng:-ng,:];
        fp[:,:,-ng:,:] = fp[:,:,ng:2*ng,:];
    elif (pbool[2]<0):
        fp[:,:,ng-1,:] = fp[:,:,ng,:]
        fp[:,:,-ng,:] = fp[:,:,-ng-1,:]
    else:
        fp[:,:,ng-1,:] = 2*fp[:,:,ng,:]-fp[:,:,ng+1,:];
        fp[:,:,-ng,:] = 2*fp[:,:,-ng-1,:]-fp[:,:,-ng-2,:];

    return fp

def pad_field_2D(f,ng=1,pbool=np.array([1,1])):

    if (ng<1) or (ng>np.amax(f.shape)):
        raise RuntimeError('ng < 1 or ng > max dimension!.')
    
    # initialize
    fp = np.zeros([f.shape[0]+2*ng,f.shape[1]+2*ng]);
    fp[ng:-ng,ng:-ng] = f;
    
    # x pad
    if (pbool[0]):
        fp[0:ng,:] = fp[-2*ng:-ng,:];
        fp[-ng:,:] = fp[ng:2*ng,:];
    else:
        fp[ng-1,:] = 2*fp[ng,:]-fp[ng+1,:];
        fp[-ng,:] = 2*fp[-ng-1,:]-fp[-ng-2,:];
    
    # y pad
    if (pbool[1]):
        fp[:,0:ng] = fp[:,-2*ng:-ng];
        fp[:,-ng:] = fp[:,ng:2*ng];
    else:
        fp[:,ng-1] = 2*fp[:,ng]-fp[:,ng+1];
        fp[:,-ng] = 2*fp[:,-ng-1]-fp[:,-ng-2];

    return fp
