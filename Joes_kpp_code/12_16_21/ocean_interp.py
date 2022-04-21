import warnings
import numpy as np
from repmat import repmat
from pad_field import pad_field_3D
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from strip_nan_inf import *

def ocean_interpZ(Z,f,Zq,k,kq):
    fq = 0*Zq;
    if (k>3):
        # intrp = interp1d(Z[:k],f[:k],'cubic',fill_value="extrapolate");
        intrp = CubicSpline(Z[:k],f[:k],bc_type="natural");
#        intrp = InterpolatedUnivariateSpline(Z[:k],f[:k])
        fq[:kq] = intrp(Zq[:kq]);

    return fq;

def ocean_interpZ_from_fixed(Z,f,Zq,k,kq):
    fq = np.zeros((Zq.shape[0],f.shape[1]))
    if (k>3):
        intrp = interp1d(Z[:k],f[:k,:],'cubic',0,fill_value="extrapolate");
#         intrp = CubicSpline(Z[:k],f[:k,:],axis=0,bc_type="natural");
        fq[:kq,:] = intrp(Zq[:kq]);

    return fq;
