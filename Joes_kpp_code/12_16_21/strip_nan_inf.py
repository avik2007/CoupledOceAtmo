import numpy as np

def strip_nan_inf(f):
    f[np.isnan(f)] = 0;
    f[np.isinf(f)] = 0;
