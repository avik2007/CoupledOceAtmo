import sys
import gc
import warnings
import math
import numpy as np
import numpy.fft
from repmat import repmat
import fld_tools as ft
from compute_nu_T import *

# Computes horizontal spectral energy flux, assuming nonunform regular coordinates with a finite volume formulation
# a tranfer-function approach is used that assumes energy is conserved, even though it is not, strictly speaking.
    
def compute_nu_KE_k(U,V,W,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    print ('in compute_nu_KE_k()')
#     (hU,hV,_) = ft.taper_filter_3D_uvw_nu(U,V,W,dxu,dyu,dxv,dyv,dxc,dyc,thknss,post_taper);
    (T,k_out) = compute_nu_T_k(U,V,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    return (T,k_out);

def compute_nu_KE_m(U,V,W,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    print ('in compute_nu_KE_m()')
#     (hU,hV,_) = ft.taper_filter_3D_uvw_nu(U,V,W,dxu,dyu,dxv,dyv,dxc,dyc,thknss,post_taper);
    (T,m_out) = compute_nu_T_m(U,V,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    return (T,m_out);



