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
    
def compute_u_decomp_KE_m(U,V,W,Ubp,Vbp,Uhp,Vhp,dx,dy,dz,post_taper=np.array([0,0,0]),trim_ml=0,prd=np.array([1,1,1])):
    print ('in compute_u_decomp_KE_m()')

    nz = W.shape[2];
    mti_max = math.floor(nz/2-1); 
    T = np.zeros((mti_max,3));

#     (hU,hV,_) = ft.taper_filter_3D_uvw_u(U,V,W,post_taper);
#     (hUbp,hVbp,_) = ft.taper_filter_3D_uvw_u(Ubp,Vbp,Wbp,post_taper);
#     (hUhp,hVhp,_) = ft.taper_filter_3D_uvw_u(Uhp,Vhp,Whp,post_taper);
    (Tl,m_out) = compute_u_T_m(U-Ubp-Uhp,V-Vbp-Vhp,U-Ubp-Uhp,V-Vbp-Vhp,dx,dy,dz,post_taper,trim_ml,prd);
    (Tb,m_out) = compute_u_T_m(Ubp,Vbp,Ubp,Vbp,dx,dy,dz,post_taper,trim_ml,prd);
    (Th,m_out) = compute_u_T_m(Uhp,Vhp,Uhp,Vhp,dx,dy,dz,post_taper,trim_ml,prd);

    T[:,0:1] = Tl; T[:,1:2] = Tb; T[:,2:3] = Th;
                                                   
    return (T,m_out);



    
