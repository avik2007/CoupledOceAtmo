import warnings
import numpy as np
from repmat import repmat
from pad_field import pad_field_3D
from strip_nan_inf import *

def trim_ml(f,thknss,trim_ml):
    (nx,ny,nz,nt) = f.shape;
    if (thknss.ndim==1):
        depth = np.cumsum(thknss,axis=0) - 0.5*thknss;
        depth = repmat(np.reshape(depth,(1,1,nz,1)),(nx,ny,1,nt));
    else:
        depth = np.cumsum(thknss,axis=2) - 0.5*thknss;
    filt = depth>trim_ml;
    return filt*f;

