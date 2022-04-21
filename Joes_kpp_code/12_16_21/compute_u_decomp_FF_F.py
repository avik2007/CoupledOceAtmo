import gc
import sys
import math
import time
import warnings
import numpy as np
import numpy.fft
from repmat import repmat
import fld_tools as ft
from compute_u_T import *

# Computes horizontal spectral energy flux, assuming nonunform regular coordinates with a finite volume formulation
# a tranfer-function approach is used that assumes energy is conserved, even though it is not, strictly speaking.
    
def compute_u_decomp_FF_F_m(U,V,W,U_bp,V_bp,U_hp,V_hp,dx,dy,dz,post_taper=np.array([0,0,0]),trim_ml=0,prd=np.array([1,1,1])):

    print ('in compute_u_decomp_FF_F_m()')

    # Initialize Constants
    (nx, ny, nz, nt) = U.shape;
    Lz = dz*nz; ColumnVol = dx*dy*Lz;
    rho0 = 1027.5; # for energy output (optional)
    
    if (post_taper[2]):
        raise RuntimeError('post_taper not built to handle vertical filt. of dKEdiv');

    # Add ghost cells to fields to handle periodicity in position space.
    ng = 2; # number of ghost cells, must be one larger than needed for python indexing
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits
    U = ft.pad_field_3D(U,ng,prd); V = ft.pad_field_3D(V,ng,prd);  W = ft.pad_field_3D(W,ng,prd);
    U_hp = ft.pad_field_3D(U_hp,ng,prd); V_hp = ft.pad_field_3D(V_hp,ng,prd);  
    U_bp = ft.pad_field_3D(U_bp,ng,prd); V_bp = ft.pad_field_3D(V_bp,ng,prd);  

    (hU,hV,_) = ft.taper_filter_3D_uvw_u(U[gx0:gxn,gy0:gyn,gz0:gzn,:],V[gx0:gxn,gy0:gyn,gz0:gzn,:],
                                  W[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper);
    (hU_bp,hV_bp,_) = ft.taper_filter_3D_uvw_u(U_bp[gx0:gxn,gy0:gyn,gz0:gzn,:],V_bp[gx0:gxn,gy0:gyn,gz0:gzn,:],
                                  W[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper);
    (hU_hp,hV_hp,_) = ft.taper_filter_3D_uvw_u(U_hp[gx0:gxn,gy0:gyn,gz0:gzn,:],V_hp[gx0:gxn,gy0:gyn,gz0:gzn,:],
                                  W[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper);

    FhUb = np.fft.fft(hU,axis=2); FhUp = 0*FhUb;
    FhVb = np.fft.fft(hV,axis=2); FhVp = 0*FhVb;
    FhUb_bp = np.fft.fft(hU_bp,axis=2); FhUp_bp = 0*FhUb_bp;
    FhVb_bp = np.fft.fft(hV_bp,axis=2); FhVp_bp = 0*FhVb_bp;
    FhUb_hp = np.fft.fft(hU_hp,axis=2); FhUp_hp = 0*FhUb_hp;
    FhVb_hp = np.fft.fft(hV_hp,axis=2); FhVp_hp = 0*FhVb_hp;
    
    # Compute filtered divergence
    DivVcc = ((U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]-U[gx0:gxn,gy0:gyn,gz0:gzn,:])/dx
              + (V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]-V[gx0:gxn,gy0:gyn,gz0:gzn,:])/dy
              + (W[gx0:gxn,gy0:gyn,gz0:gzn,:]-W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])/dz);

    # Initialize Output   
    mti_max = math.floor(nz/2-1); 
    Lt = Lz;
    F = np.empty((mti_max,nx,ny,9,4)); F[:] = np.NaN;
    
    dftnz = math.floor(nz/2)+1;
    spec_dist = np.reshape(np.linspace(1,nz,nz),(1,1,nz));
    spec_dist = np.abs(np.mod(spec_dist-2+dftnz,nz)-dftnz+1)+1;
    spec_dist = repmat(spec_dist,(nx,ny,1,nt));
    
    # Main Loop
    mt_offset = 0;
    for mti in range(mt_offset,mti_max-mt_offset):
        t1 = time.time() 

        FhUp = FhUp + FhUb; FhVp = FhVp + FhVb; 
        FhUb = FhUp; FhVb = FhVp; 
        FhUp_bp = FhUp_bp + FhUb_bp; FhVp_bp = FhVp_bp + FhVb_bp; 
        FhUb_bp = FhUp_bp; FhVb_bp = FhVp_bp; 
        FhUp_hp = FhUp_hp + FhUb_hp; FhVp_hp = FhVp_hp + FhVb_hp; 
        FhUb_hp = FhUp_hp; FhVb_hp = FhVp_hp; 
        
        filt = (spec_dist<(mti+1));
        
        FhUp = (1-filt)*FhUp;
        FhVp = (1-filt)*FhVp;
        FhUb= filt*FhUb;
        FhVb= filt*FhVb;
        FhUp_bp = (1-filt)*FhUp_bp;
        FhVp_bp = (1-filt)*FhVp_bp;
        FhUb_bp= filt*FhUb_bp;
        FhVb_bp= filt*FhVb_bp;
        FhUp_hp = (1-filt)*FhUp_hp;
        FhVp_hp = (1-filt)*FhVp_hp;
        FhUb_hp= filt*FhUb_hp;
        FhVb_hp= filt*FhVb_hp;

        hUp = np.real(np.fft.ifft(FhUp,axis=2)); hUb = hU - hUp;
        hVp = np.real(np.fft.ifft(FhVp,axis=2)); hVb = hV - hVp;
        hUp_bp = np.real(np.fft.ifft(FhUp_bp,axis=2)); hUb_bp = hU_bp - hUp_bp;
        hVp_bp = np.real(np.fft.ifft(FhVp_bp,axis=2)); hVb_bp = hV_bp - hVp_bp;
        hUp_hp = np.real(np.fft.ifft(FhUp_hp,axis=2)); hUb_hp = hU_hp - hUp_hp;
        hVp_hp = np.real(np.fft.ifft(FhVp_hp,axis=2)); hVb_hp = hV_hp - hVp_hp;

        for icase in range(0,3):
            if (icase==0):
                hUb_c = ft.pad_field_3D(hUb-hUb_bp-hUb_hp,ng,prd); hVb_c = ft.pad_field_3D(hVb-hVb_bp-hVb_hp,ng,prd);
                hUp_c = ft.pad_field_3D(hUp-hUp_bp-hUp_hp,ng,prd); hVp_c = ft.pad_field_3D(hVp-hVp_bp-hVp_hp,ng,prd);
            elif (icase==1):
                hUb_c = ft.pad_field_3D(hUb_bp,ng,prd); hVb_c = ft.pad_field_3D(hVb_bp,ng,prd);
                hUp_c = ft.pad_field_3D(hUp_bp,ng,prd); hVp_c = ft.pad_field_3D(hVp_bp,ng,prd);
            else:
                hUb_c = ft.pad_field_3D(hUb_hp,ng,prd); hVb_c = ft.pad_field_3D(hVb_hp,ng,prd);
                hUp_c = ft.pad_field_3D(hUp_hp,ng,prd); hVp_c = ft.pad_field_3D(hVp_hp,ng,prd);

            hUb = ft.pad_field_3D(hUb,ng,prd); hVb = ft.pad_field_3D(hVb,ng,prd);
            hUp = ft.pad_field_3D(hUp,ng,prd); hVp = ft.pad_field_3D(hVp,ng,prd);
            hUb_bp = ft.pad_field_3D(hUb_bp,ng,prd); hVb_bp = ft.pad_field_3D(hVb_bp,ng,prd);
            hUp_bp = ft.pad_field_3D(hUp_bp,ng,prd); hVp_bp = ft.pad_field_3D(hVp_bp,ng,prd);
            hUb_hp = ft.pad_field_3D(hUb_hp,ng,prd); hVb_hp = ft.pad_field_3D(hVb_hp,ng,prd);
            hUp_hp = ft.pad_field_3D(hUp_hp,ng,prd); hVp_hp = ft.pad_field_3D(hVp_hp,ng,prd);

            # Zero Tendency
            dhUp = np.zeros((nx,ny,nz,nt)); dhVp = np.zeros((nx,ny,nz,nt));
            dhUb = np.zeros((nx,ny,nz,nt)); dhVb = np.zeros((nx,ny,nz,nt));

            # Compute Up advective tendency
            # dUU/dx
            dhUp = dhUp + 0.25*((U[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+U[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUb_c[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+hUb_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                                - (U[gx0:gxn,gy0:gyn,gz0:gzn,:]+U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hUb_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUb_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]))/dx;
            # dVU/dy
            dhUp = dhUp + 0.25*((V[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+V[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUb_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUb_c[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:])
                                - (V[gx0-1:gxn-1,gy0+1:gyn+1,gz0:gzn,:]+V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hUb_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hUb_c[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dy;
            # dWU/dz
            dhUp = dhUp + 0.25*((W[gx0-1:gxn-1,gy0:gyn,gz0+1:gzn+1,:]+W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hUb_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUb_c[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
                                - (W[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+W[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUb_c[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hUb_c[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dz;
            
            # Compute Vp advective tendency
            # dVV/dy
            dhVp = dhVp + 0.25*((V[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+V[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVb_c[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+hVb_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                                - (V[gx0:gxn,gy0:gyn,gz0:gzn,:]+V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hVb_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVb_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]))/dy;
            # dUV/dx
            dhVp = dhVp + 0.25*((U[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+U[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVb_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVb_c[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:])
                                - (U[gx0+1:gxn+1,gy0-1:gyn-1,gz0:gzn,:]+U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hVb_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hVb_c[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dx;
            # dWV/dz
            dhVp = dhVp + 0.25*((W[gx0:gxn,gy0-1:gyn-1,gz0+1:gzn+1,:]+W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hVb_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVb_c[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
                                - (W[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+W[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVb_c[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hVb_c[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dz;
            
            # Compute Ub advective tendency
            # dUU/dx
            dhUb = dhUb + 0.25*((U[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+U[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUp_c[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+hUp_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                                - (U[gx0:gxn,gy0:gyn,gz0:gzn,:]+U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hUp_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUp_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]))/dx;
            # dVU/dy
            dhUb = dhUb + 0.25*((V[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+V[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUp_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUp_c[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:])
                                - (V[gx0-1:gxn-1,gy0+1:gyn+1,gz0:gzn,:]+V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hUp_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hUp_c[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dy;
            # dWU/dz
            dhUb = dhUb + 0.25*((W[gx0-1:gxn-1,gy0:gyn,gz0+1:gzn+1,:]+W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hUp_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUp_c[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
                                - (W[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+W[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUp_c[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hUp_c[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dz;
            
            # Compute Vb advective tendency
            # dVV/dy
            dhVb = dhVb + 0.25*((V[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+V[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVp_c[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+hVp_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                                - (V[gx0:gxn,gy0:gyn,gz0:gzn,:]+V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hVp_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVp_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]))/dy;
            # dUV/dxn
            dhVb = dhVb + 0.25*((U[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+U[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVp_c[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:])
                                - (U[gx0+1:gxn+1,gy0-1:gyn-1,gz0:gzn,:]+U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hVp_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hVp_c[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dx;
            # dWV/dzn
            dhVb = dhVb + 0.25*((W[gx0:gxn,gy0-1:gyn-1,gz0+1:gzn+1,:]+W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hVp_c[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVp_c[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
                                - (W[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+W[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVp_c[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hVp_c[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dz;

            
            
            # Compute KE divergence correction (quick tapered version)...
            dh2KEdiv_hp = 0.25*((hUb_hp[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hUb_hp[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUp_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hUp_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                             + (hVb_hp[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hVb_hp[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVp_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hVp_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                         )*DivVcc;
            dh2KEdiv_bp = 0.25*((hUb_bp[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hUb_bp[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUp_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hUp_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                             + (hVb_bp[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hVb_bp[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVp_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hVp_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                         )*DivVcc;
            dh2KEdiv_lp = 0.25*((hUb[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hUb[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUp_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hUp_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                             + (hVb[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hVb[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVp_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hVp_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                         )*DivVcc;
            dh2KEdiv_lp = dh2KEdiv_lp - dh2KEdiv_bp - dh2KEdiv_hp;



            dh2KEdiv_hp_switch = 0.25*((hUp_hp[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hUp_hp[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUb_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hUb_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                             + (hVp_hp[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hVp_hp[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVb_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hVb_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                         )*DivVcc;
            dh2KEdiv_bp_switch = 0.25*((hUp_bp[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hUp_bp[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUb_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hUb_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                             + (hVp_bp[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hVp_bp[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVb_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hVb_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                         )*DivVcc;
            dh2KEdiv_lp_switch = 0.25*((hUp[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hUp[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUb_c[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hUb_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                             + (hVp[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hVp[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVb_c[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hVb_c[gx0:gxn,gy0:gyn,gz0:gzn,:])
                         )*DivVcc;
            dh2KEdiv_lp_switch = dh2KEdiv_lp_switch - dh2KEdiv_bp_switch - dh2KEdiv_hp_switch;


            
            hUp_c = hUp_c[gx0:gxn,gy0:gyn,gz0:gzn,:];
            hUb_c = hUb_c[gx0:gxn,gy0:gyn,gz0:gzn,:];
            hVp_c = hVp_c[gx0:gxn,gy0:gyn,gz0:gzn,:];
            hVb_c = hVb_c[gx0:gxn,gy0:gyn,gz0:gzn,:];
            hUp = hUp[gx0:gxn,gy0:gyn,gz0:gzn,:];
            hUb = hUb[gx0:gxn,gy0:gyn,gz0:gzn,:];
            hVp = hVp[gx0:gxn,gy0:gyn,gz0:gzn,:];
            hVb = hVb[gx0:gxn,gy0:gyn,gz0:gzn,:];
            hUp_bp = hUp_bp[gx0:gxn,gy0:gyn,gz0:gzn,:];
            hUb_bp = hUb_bp[gx0:gxn,gy0:gyn,gz0:gzn,:];
            hVp_bp = hVp_bp[gx0:gxn,gy0:gyn,gz0:gzn,:];
            hVb_bp = hVb_bp[gx0:gxn,gy0:gyn,gz0:gzn,:];
            hUp_hp = hUp_hp[gx0:gxn,gy0:gyn,gz0:gzn,:];
            hUb_hp = hUb_hp[gx0:gxn,gy0:gyn,gz0:gzn,:];
            hVp_hp = hVp_hp[gx0:gxn,gy0:gyn,gz0:gzn,:];
            hVb_hp = hVb_hp[gx0:gxn,gy0:gyn,gz0:gzn,:];
        
            # Compute the spectral energy flux.  3 and 4 are corrected forward and backward, 1 and 2 are exact
            if (icase==0): # from lp
                F[mti,:,:,0,0] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*(hUp-hUp_bp-hUp_hp) + dhVp*(hVp-hVp_bp-hVp_hp) + 1.0*dh2KEdiv_lp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,0,1] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*(hUb-hUb_bp-hUb_hp) + dhVb*(hVb-hVb_bp-hVb_hp) + 1.0*dh2KEdiv_lp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,0,2] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*(hUp-hUp_bp-hUp_hp) + dhVp*(hVp-hVp_bp-hVp_hp) + 0.5*dh2KEdiv_lp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,0,3] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*(hUb-hUb_bp-hUb_hp) + dhVb*(hVb-hVb_bp-hVb_hp) + 0.5*dh2KEdiv_lp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,3,0] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*(hUp_bp) + dhVp*(hVp_bp) + 1.0*dh2KEdiv_bp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,3,1] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*(hUb_bp) + dhVb*(hVb_bp) + 1.0*dh2KEdiv_bp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,3,2] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*(hUp_bp) + dhVp*(hVp_bp) + 0.5*dh2KEdiv_bp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,3,3] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*(hUb_bp) + dhVb*(hVb_bp) + 0.5*dh2KEdiv_bp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,6,0] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*(hUp_hp) + dhVp*(hVp_hp) + 1.0*dh2KEdiv_hp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,6,1] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*(hUb_hp) + dhVb*(hVb_hp) + 1.0*dh2KEdiv_hp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,6,2] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*(hUp_hp) + dhVp*(hVp_hp) + 0.5*dh2KEdiv_hp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,6,3] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*(hUb_hp) + dhVb*(hVb_hp) + 0.5*dh2KEdiv_hp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
            elif (icase==1): # from bp
                F[mti,:,:,1,0] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*(hUp-hUp_bp-hUp_hp) + dhVp*(hVp-hVp_bp-hVp_hp) + 1.0*dh2KEdiv_bp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,1,1] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*(hUb-hUb_bp-hUb_hp) + dhVb*(hVb-hVb_bp-hVb_hp) + 1.0*dh2KEdiv_bp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,1,2] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*(hUp-hUp_bp-hUp_hp) + dhVp*(hVp-hVp_bp-hVp_hp) + 0.5*dh2KEdiv_bp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,1,3] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*(hUb-hUb_bp-hUb_hp) + dhVb*(hVb-hVb_bp-hVb_hp) + 0.5*dh2KEdiv_bp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,4,0] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*hUp_bp + dhVp*hVp_bp + 1.0*dh2KEdiv_bp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,4,1] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*hUb_bp + dhVb*hVb_bp + 1.0*dh2KEdiv_bp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,4,2] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*hUp_bp + dhVp*hVp_bp + 0.5*dh2KEdiv_bp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,4,3] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*hUb_bp + dhVb*hVb_bp + 0.5*dh2KEdiv_bp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,7,0] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*hUp_hp + dhVp*hVp_hp + 1.0*dh2KEdiv_hp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,7,1] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*hUb_hp + dhVb*hVb_hp + 1.0*dh2KEdiv_hp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,7,2] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*hUp_hp + dhVp*hVp_hp + 0.5*dh2KEdiv_hp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,7,3] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*hUb_hp + dhVb*hVb_hp + 0.5*dh2KEdiv_hp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
            else: # from hp
                F[mti,:,:,2,0] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*(hUp-hUp_bp-hUp_hp) + dhVp*(hVp-hVp_bp-hVp_hp) + 1.0*dh2KEdiv_lp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,2,1] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*(hUb-hUb_bp-hUb_hp) + dhVb*(hVb-hVb_bp-hVb_hp) + 1.0*dh2KEdiv_lp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,2,2] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*(hUp-hUp_bp-hUp_hp) + dhVp*(hVp-hVp_bp-hVp_hp) + 0.5*dh2KEdiv_lp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,2,3] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*(hUb-hUb_bp-hUb_hp) + dhVb*(hVb-hVb_bp-hVb_hp) + 0.5*dh2KEdiv_lp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,5,0] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*hUp_bp + dhVp*hVp_bp + 1.0*dh2KEdiv_bp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,5,1] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*hUb_bp + dhVb*hVb_bp + 1.0*dh2KEdiv_bp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,5,2] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*hUp_bp + dhVp*hVp_bp + 0.5*dh2KEdiv_bp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,5,3] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*hUb_bp + dhVb*hVb_bp + 0.5*dh2KEdiv_bp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,8,0] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*hUp_hp + dhVp*hVp_hp + 1.0*dh2KEdiv_hp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,8,1] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*hUb_hp + dhVb*hVb_hp + 1.0*dh2KEdiv_hp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,8,2] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*hUp_hp + dhVp*hVp_hp + 0.5*dh2KEdiv_hp_switch,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
                F[mti,:,:,8,3] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*hUb_hp + dhVb*hVb_hp + 0.5*dh2KEdiv_hp,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
            
        print('comp. u_decomp_FF_F: mti = ' + str(mti) + ' of ' + str(mti_max) + ', time = ' + str(time.time() - t1)) 

    # Compute m
    m_out = np.linspace(0,mti_max-1,mti_max)/Lt;     # cycles / meter

    return (F,m_out)


    
