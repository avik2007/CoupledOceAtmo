import sys
import gc
import warnings
import math
import numpy as np
import numpy.fft
from repmat import repmat
import fld_tools as ft
from compute_u_T import *

# Computes horizontal spectral energy flux, assuming nonunform regular coordinates with a finite volume formulation
# a tranfer-function approach is used that assumes energy is conserved, even though it is not, strictly speaking.
    
def compute_u_hp_KE_k(U,V,W,Up,Vp,Wp,dx,dy,dz,post_taper=np.array([0,0,0]),trim_ml=0,prd=np.array([1,1,1])):
    print ('in compute_u_KE_k()')
    (hU,hV,_) = ft.taper_filter_3D_uvw_u(U,V,W,post_taper);
    (hUp,hVp,_) = ft.taper_filter_3D_uvw_u(Up,Vp,Wp,post_taper);
    (T,k_out) = compute_u_T_k(hUp,hVp,hUp,hVp,dx,dy,dz,post_taper,trim_ml,prd); # takes filtered fields as arguments, diff from nu_T_k
    return (T,k_out);

def compute_u_hp_KE_m(U,V,W,Up,Vp,Wp,dx,dy,dz,post_taper=np.array([0,0,0]),trim_ml=0,prd=np.array([1,1,1])):
    print ('in compute_u_KE_m()')
    (hU,hV,_) = ft.taper_filter_3D_uvw_u(U,V,W,post_taper);
    (hUp,hVp,_) = ft.taper_filter_3D_uvw_u(Up,Vp,Wp,post_taper);
    (T,m_out) = compute_u_T_m(hUp,hVp,hUp,hVp,dx,dy,dz,post_taper,trim_ml,prd);
                                                   
    return (T,m_out);



    
