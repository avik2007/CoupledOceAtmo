import numpy as np
from repmat import repmat

def get_horz_correlation_coeff(f,g):
    (ns,nx,ny) = f.shape;
    f_mean = np.mean(np.mean(f,axis=2,keepdims=True),axis=1,keepdims=True);
    f_mean = repmat(f_mean,(1,nx,ny));
    g_mean = np.mean(np.mean(g,axis=2,keepdims=True),axis=1,keepdims=True);
    g_mean = repmat(g_mean,(1,nx,ny));

    cc = (np.mean(np.mean((f-f_mean)*(g-g_mean),axis=2),axis=1)
          /((np.mean(np.mean((f-f_mean)*(f-f_mean),axis=2),axis=1)*np.mean(np.mean((g-g_mean)*(g-g_mean),axis=2),axis=1))**(0.5)))
    
    return cc;

def get_horz_correlation_coeff_4D(f,g):
    (ns,nx,ny,nb) = f.shape;
    f_mean = np.mean(np.mean(f,axis=2,keepdims=True),axis=1,keepdims=True);
    f_mean = repmat(f_mean,(1,nx,ny,1));
    g_mean = np.mean(np.mean(g,axis=2,keepdims=True),axis=1,keepdims=True);
    g_mean = repmat(g_mean,(1,nx,ny,1));

    cc = (np.mean(np.mean((f-f_mean)*(g-g_mean),axis=2),axis=1)
          /((np.mean(np.mean((f-f_mean)*(f-f_mean),axis=2),axis=1)*np.mean(np.mean((g-g_mean)*(g-g_mean),axis=2),axis=1))**(0.5)))
    
    return cc;

def average_horizontal(f,keepdims=False):
    (nx,nx,ny) = f.shape;
    if (keepdims):
        f_mean = np.mean(np.mean(f,axis=2,keepdims=True),axis=1,keepdims=True);
        f_mean = repmat(f_mean,(1,nx,ny));
    else:
        f_mean = np.mean(np.mean(f,axis=2),axis=1);
    return f_mean;
    
def weighted_average_horizontal(f,w_in,keepdims=False):
    (nx,nx,ny) = f.shape;
    
    # normalize weights
    w_norm = np.sum(np.sum(w_in,axis=2,keepdims=True),axis=1,keepdims=True);
    w_norm = repmat(w_norm,(1,nx,ny));
    w = w_in/w_norm;

    if (keepdims):
        f_mean = np.sum(np.sum(f*w,axis=2,keepdims=True),axis=1,keepdims=True);
        f_mean = repmat(f_mean,(1,nx,ny));
    else:
        f_mean = np.sum(np.sum(f*w,axis=2),axis=1);
    return f_mean;
    
