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
    
def compute_u_hp_FF_F_k(U,V,W,Uhp,Vhp,Whp,dx,dy,dz,post_taper=np.array([0,0,0]),trim_ml=0,prd=np.array([1,1,1])):

    print ('in compute_u_hp_FF_F_k()')

    # Initialize Constants
    (nx, ny, nz, nt) = U.shape;
    Lx = dx*nx; Ly = dy*ny; LayerVol = Lx*Ly*dz;
    rho0 = 1027.5; # for energy output (optional)
    
    # if (post_taper[2]):
    #     raise RuntimeError('post_taper not built to handle vertical filt. of dKEdiv');

    # Add ghost cells to fields to handle periodicity in position space.
    ng = 2; # number of ghost cells, must be one larger than needed for python indexing
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits
    U = ft.pad_field_3D(U,ng,prd); V = ft.pad_field_3D(V,ng,prd);  W = ft.pad_field_3D(W,ng,prd);
    Uhp = ft.pad_field_3D(Uhp,ng,prd); Vhp = ft.pad_field_3D(Vhp,ng,prd);  Whp = ft.pad_field_3D(Whp,ng,prd);

    (hU,hV,_) = ft.taper_filter_3D_uvw_u(U[gx0:gxn,gy0:gyn,gz0:gzn,:],V[gx0:gxn,gy0:gyn,gz0:gzn,:],
                                  W[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper);
    (hUhp,hVhp,_) = ft.taper_filter_3D_uvw_u(Uhp[gx0:gxn,gy0:gyn,gz0:gzn,:],Vhp[gx0:gxn,gy0:gyn,gz0:gzn,:],
                                  Whp[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper);

    FhUb_hp = np.fft.fft(np.fft.fft(hUhp,axis=0),axis=1); FhUp_hp = 0*FhUb_hp;
    FhVb_hp = np.fft.fft(np.fft.fft(hVhp,axis=0),axis=1); FhVp_hp = 0*FhVb_hp;
    FhUb = np.fft.fft(np.fft.fft(hU,axis=0),axis=1); FhUp = 0*FhUb;
    FhVb = np.fft.fft(np.fft.fft(hV,axis=0),axis=1); FhVp = 0*FhVb;
    
    # Compute filtered divergence
    DivVcc = ((U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]-U[gx0:gxn,gy0:gyn,gz0:gzn,:])/dx
              + (V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]-V[gx0:gxn,gy0:gyn,gz0:gzn,:])/dy
              + (W[gx0:gxn,gy0:gyn,gz0:gzn,:]-W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])/dz);

    # Initialize Output
    if (1/dx) < (1/dy):
        kti_max = math.floor(nx/2-1);
        Lt = Lx;
    else:
        kti_max = math.floor(ny/2-1);
        Lt = Ly;
    
    # Initialize Output
    F = np.empty((kti_max,nz,4));
    F[:] = np.NaN;
    
    dftnx = math.floor(nx/2)+1;
    dftny = math.floor(ny/2)+1;
    ki_mod = repmat(np.reshape(np.linspace(1,nx,nx),(nx,1)),(1,ny));
    kj_mod = repmat(np.reshape(np.linspace(1,ny,ny),(1,ny)),(nx,1));
    ki_mod = np.abs(np.mod(ki_mod-2+dftnx,nx)-dftnx+1)/Lx;
    kj_mod = np.abs(np.mod(kj_mod-2+dftny,ny)-dftny+1)/Ly;
    spec_dist = (ki_mod**2+kj_mod**2)**(0.5);
    spec_dist = repmat(spec_dist,(1,1,nz,nt));
    del ki_mod, kj_mod

    # Main Loop
    kt_offset = 0;
    for kti in range(kt_offset,kti_max-kt_offset):
        t1 = time.time() 
        kt = kti/Lt;
        ktp = (kti+1)/Lt;

        FhUp_hp = FhUp_hp + FhUb_hp; FhVp_hp = FhVp_hp + FhVb_hp; 
        FhUb_hp = FhUp_hp; FhVb_hp = FhVp_hp; 
        FhUp = FhUp + FhUb; FhVp = FhVp + FhVb; 
        FhUb = FhUp; FhVb = FhVp; 
        
        filt = spec_dist<ktp;
        
        FhUp_hp = (1-filt)*FhUp_hp;
        FhVp_hp = (1-filt)*FhVp_hp;
        FhUb_hp= filt*FhUb_hp;
        FhVb_hp= filt*FhVb_hp;
        FhUp = (1-filt)*FhUp;
        FhVp = (1-filt)*FhVp;
        FhUb= filt*FhUb;
        FhVb= filt*FhVb;

        hUp = np.real(np.fft.ifft(np.fft.ifft(FhUp_hp,axis=0),axis=1)); hUb = hU_hp - hUp;
        hVp = np.real(np.fft.ifft(np.fft.ifft(FhVp_hp,axis=0),axis=1)); hVb = hV_hp - hVp;
        hUpp = np.real(np.fft.ifft(np.fft.ifft(FhUp,axis=0),axis=1)); hUbb = hU - hUpp;
        hVpp = np.real(np.fft.ifft(np.fft.ifft(FhVp,axis=0),axis=1)); hVbb = hV - hVpp;

        # Add periodicity
        hUb = ft.pad_field_3D(hUb,ng,prd); hVb = ft.pad_field_3D(hVb,ng,prd);
        hUp = ft.pad_field_3D(hUp,ng,prd); hVp = ft.pad_field_3D(hVp,ng,prd);
        hUbb = ft.pad_field_3D(hUbb,ng,prd); hVbb = ft.pad_field_3D(hVbb,ng,prd);
        hUpp = ft.pad_field_3D(hUpp,ng,prd); hVpp = ft.pad_field_3D(hVpp,ng,prd);
        
        # Zero Tendency
        dhUp = np.zeros((nx,ny,nz,nt)); dhVp = np.zeros((nx,ny,nz,nt));
        dhUb = np.zeros((nx,ny,nz,nt)); dhVb = np.zeros((nx,ny,nz,nt));
    
        # Compute Up advective tendency
        # dUU/dx
        dhUp = dhUp + 0.25*((U[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+U[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUb[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+hUb[gx0:gxn,gy0:gyn,gz0:gzn,:])
                          - (U[gx0:gxn,gy0:gyn,gz0:gzn,:]+U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hUb[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUb[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]))/dx;
        # dVU/dy
        dhUp = dhUp + 0.25*((V[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+V[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUb[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUb[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:])
                          - (V[gx0-1:gxn-1,gy0+1:gyn+1,gz0:gzn,:]+V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hUb[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hUb[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dy;
        # dWU/dz
        dhUp = dhUp + 0.25*((W[gx0-1:gxn-1,gy0:gyn,gz0+1:gzn+1,:]+W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hUb[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUb[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
                          - (W[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+W[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUb[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hUb[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dz;
    
        # Compute Vp advective tendency
        # dVV/dy
        dhVp = dhVp + 0.25*((V[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+V[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVb[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+hVb[gx0:gxn,gy0:gyn,gz0:gzn,:])
                          - (V[gx0:gxn,gy0:gyn,gz0:gzn,:]+V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hVb[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVb[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]))/dy;
        # dUV/dx
        dhVp = dhVp + 0.25*((U[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+U[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVb[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVb[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:])
                          - (U[gx0+1:gxn+1,gy0-1:gyn-1,gz0:gzn,:]+U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hVb[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hVb[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dx;
        # dWV/dz
        dhVp = dhVp + 0.25*((W[gx0:gxn,gy0-1:gyn-1,gz0+1:gzn+1,:]+W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hVb[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVb[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
                          - (W[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+W[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVb[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hVb[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dz;

        # Compute Ub advective tendency
        # dUU/dx
        dhUb = dhUb + 0.25*((U[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+U[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUp[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+hUp[gx0:gxn,gy0:gyn,gz0:gzn,:])
                          - (U[gx0:gxn,gy0:gyn,gz0:gzn,:]+U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hUp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUp[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]))/dx;
        # dVU/dy
        dhUb = dhUb + 0.25*((V[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+V[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUp[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:])
                          - (V[gx0-1:gxn-1,gy0+1:gyn+1,gz0:gzn,:]+V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hUp[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hUp[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dy;
        # dWU/dz
        dhUb = dhUb + 0.25*((W[gx0-1:gxn-1,gy0:gyn,gz0+1:gzn+1,:]+W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hUp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUp[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
                          - (W[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+W[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUp[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hUp[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dz;
    
        # Compute Vb advective tendency
        # dVV/dy
        dhVb = dhVb + 0.25*((V[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+V[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVp[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+hVp[gx0:gxn,gy0:gyn,gz0:gzn,:])
                          - (V[gx0:gxn,gy0:gyn,gz0:gzn,:]+V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hVp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVp[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]))/dy;
        # dUV/dx
        dhVb = dhVb + 0.25*((U[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+U[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVp[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:])
                          - (U[gx0+1:gxn+1,gy0-1:gyn-1,gz0:gzn,:]+U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hVp[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hVp[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dx;
        # dWV/dz
        dhVb = dhVb + 0.25*((W[gx0:gxn,gy0-1:gyn-1,gz0+1:gzn+1,:]+W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hVp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVp[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
                          - (W[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+W[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVp[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hVp[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dz;



        # Compute KE divergence correction (quick tapered version)...
        dh2KEdiv = 0.25*((hUb[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hUb[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUp[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hUp[gx0:gxn,gy0:gyn,gz0:gzn,:])
                         + (hVb[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hVb[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVp[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hVp[gx0:gxn,gy0:gyn,gz0:gzn,:])
                     )*DivVcc;
        
        hUp = hUp[gx0:gxn,gy0:gyn,gz0:gzn,:];
        hUb = hUb[gx0:gxn,gy0:gyn,gz0:gzn,:];
        hVp = hVp[gx0:gxn,gy0:gyn,gz0:gzn,:];
        hVb = hVb[gx0:gxn,gy0:gyn,gz0:gzn,:];
        hUpp = hUpp[gx0:gxn,gy0:gyn,gz0:gzn,:];
        hUbb = hUbb[gx0:gxn,gy0:gyn,gz0:gzn,:];
        hVpp = hVpp[gx0:gxn,gy0:gyn,gz0:gzn,:];
        hVbb = hVbb[gx0:gxn,gy0:gyn,gz0:gzn,:];
        
        # Compute the spectral energy flux.  3 and 4 are corrected forward and backward, 1 and 2 are exact
        F[kti,:,0] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(np.sum(dhUp*hUpp + dhVp*hVpp + 1.0*dh2KEdiv,axis=0,keepdims=True),axis=1,keepdims=True),axis=3,keepdims=True)/nt/LayerVol,(nz));
        F[kti,:,1] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(np.sum(dhUb*hUbb + dhVb*hVbb + 1.0*dh2KEdiv,axis=0,keepdims=True),axis=1,keepdims=True),axis=3,keepdims=True)/nt/LayerVol,(nz));
        F[kti,:,2] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(np.sum(dhUp*hUpp + dhVp*hVpp + 0.5*dh2KEdiv,axis=0,keepdims=True),axis=1,keepdims=True),axis=3,keepdims=True)/nt/LayerVol,(nz));
        F[kti,:,3] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(np.sum(dhUb*hUbb + dhVb*hVbb + 0.5*dh2KEdiv,axis=0,keepdims=True),axis=1,keepdims=True),axis=3,keepdims=True)/nt/LayerVol,(nz));

        print('comp. u_hp_FF_F: kti = ' + str(kti) + ' of ' + str(kti_max) + ', time = ' + str(time.time() - t1)) # DEBUG

    # Compute k
    k_out = np.linspace(0,kti_max-1,kti_max)/Lt;     # cycles / meter

    return (F,k_out)





def compute_u_hp_FF_F_m(U,V,W,Uhp,Vhp,Whp,dx,dy,dz,post_taper=np.array([0,0,0]),trim_ml=0,prd=np.array([1,1,1])):

    print ('in compute_u_hp_FF_F_m()')

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
    Uhp = ft.pad_field_3D(Uhp,ng,prd); Vhp = ft.pad_field_3D(Vhp,ng,prd);  Whp = ft.pad_field_3D(Whp,ng,prd);

    (hU,hV,_) = ft.taper_filter_3D_uvw_u(U[gx0:gxn,gy0:gyn,gz0:gzn,:],V[gx0:gxn,gy0:gyn,gz0:gzn,:],
                                  W[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper);
    (hUhp,hVhp,_) = ft.taper_filter_3D_uvw_u(Uhp[gx0:gxn,gy0:gyn,gz0:gzn,:],Vhp[gx0:gxn,gy0:gyn,gz0:gzn,:],
                                  Whp[gx0:gxn,gy0:gyn,gz0:gzn,:],post_taper);

    FhUb_hp = np.fft.fft(hUhp,axis=2); FhUp_hp = 0*FhUb_hp;
    FhVb_hp = np.fft.fft(hVhp,axis=2); FhVp_hp = 0*FhVb_hp;
    FhUb = np.fft.fft(hU,axis=2); FhUp = 0*FhUb;
    FhVb = np.fft.fft(hV,axis=2); FhVp = 0*FhVb;
    
    # Compute filtered divergence
    DivVcc = ((U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]-U[gx0:gxn,gy0:gyn,gz0:gzn,:])/dx
              + (V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]-V[gx0:gxn,gy0:gyn,gz0:gzn,:])/dy
              + (W[gx0:gxn,gy0:gyn,gz0:gzn,:]-W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])/dz);

    # Initialize Output   
    mti_max = math.floor(nz/2-1); 
    Lt = Lz;
    F = np.empty((mti_max,nx,ny,4));
    F[:] = np.NaN;
    
    dftnz = math.floor(nz/2)+1;
    spec_dist = np.reshape(np.linspace(1,nz,nz),(1,1,nz));
    spec_dist = np.abs(np.mod(spec_dist-2+dftnz,nz)-dftnz+1)+1;
    spec_dist = repmat(spec_dist,(nx,ny,1,nt));
    
    # Main Loop
    mt_offset = 0;
    for mti in range(mt_offset,mti_max-mt_offset):
        t1 = time.time() 

        FhUp_hp = FhUp_hp + FhUb_hp; FhVp_hp = FhVp_hp + FhVb_hp; 
        FhUb_hp = FhUp_hp; FhVb_hp = FhVp_hp; 
        FhUp = FhUp + FhUb; FhVp = FhVp + FhVb; 
        FhUb = FhUp; FhVb = FhVp; 
        
        filt = (spec_dist<(mti+1));
        
        FhUp_hp = (1-filt)*FhUp_hp;
        FhVp_hp = (1-filt)*FhVp_hp;
        FhUb_hp= filt*FhUb_hp;
        FhVb_hp= filt*FhVb_hp;
        FhUp = (1-filt)*FhUp;
        FhVp = (1-filt)*FhVp;
        FhUb= filt*FhUb;
        FhVb= filt*FhVb;

        hUp = np.real(np.fft.ifft(FhUp_hp,axis=2)); hUb = hU - hUp;
        hVp = np.real(np.fft.ifft(FhVp_hp,axis=2)); hVb = hV - hVp;
        hUpp = np.real(np.fft.ifft(FhUp,axis=2)); hUbb = hU - hUpp;
        hVpp = np.real(np.fft.ifft(FhVp,axis=2)); hVbb = hV - hVpp;

        # Add periodicity
        hUb = ft.pad_field_3D(hUb,ng,prd); hVb = ft.pad_field_3D(hVb,ng,prd);
        hUp = ft.pad_field_3D(hUp,ng,prd); hVp = ft.pad_field_3D(hVp,ng,prd);
        hUbb = ft.pad_field_3D(hUbb,ng,prd); hVbb = ft.pad_field_3D(hVbb,ng,prd);
        hUpp = ft.pad_field_3D(hUpp,ng,prd); hVpp = ft.pad_field_3D(hVpp,ng,prd);
        
        # Zero Tendency
        dhUp = np.zeros((nx,ny,nz,nt)); dhVp = np.zeros((nx,ny,nz,nt));
        dhUb = np.zeros((nx,ny,nz,nt)); dhVb = np.zeros((nx,ny,nz,nt));
    


        # Compute Up advective tendency
        # dUU/dx
        dhUp = dhUp + 0.25*((U[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+U[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUb[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+hUb[gx0:gxn,gy0:gyn,gz0:gzn,:])
                          - (U[gx0:gxn,gy0:gyn,gz0:gzn,:]+U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hUb[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUb[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]))/dx;
        # dVU/dy
        dhUp = dhUp + 0.25*((V[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+V[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUb[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUb[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:])
                          - (V[gx0-1:gxn-1,gy0+1:gyn+1,gz0:gzn,:]+V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hUb[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hUb[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dy;
        # dWU/dz
        dhUp = dhUp + 0.25*((W[gx0-1:gxn-1,gy0:gyn,gz0+1:gzn+1,:]+W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hUb[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUb[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
                          - (W[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+W[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUb[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hUb[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dz;
    
        # Compute Vp advective tendency
        # dVV/dy
        dhVp = dhVp + 0.25*((V[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+V[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVb[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+hVb[gx0:gxn,gy0:gyn,gz0:gzn,:])
                          - (V[gx0:gxn,gy0:gyn,gz0:gzn,:]+V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hVb[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVb[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]))/dy;
        # dUV/dx
        dhVp = dhVp + 0.25*((U[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+U[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVb[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVb[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:])
                          - (U[gx0+1:gxn+1,gy0-1:gyn-1,gz0:gzn,:]+U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hVb[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hVb[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dx;
        # dWV/dz
        dhVp = dhVp + 0.25*((W[gx0:gxn,gy0-1:gyn-1,gz0+1:gzn+1,:]+W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hVb[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVb[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
                          - (W[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+W[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVb[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hVb[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dz;

        # Compute Ub advective tendency
        # dUU/dx
        dhUb = dhUb + 0.25*((U[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+U[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUp[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+hUp[gx0:gxn,gy0:gyn,gz0:gzn,:])
                          - (U[gx0:gxn,gy0:gyn,gz0:gzn,:]+U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hUp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUp[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]))/dx;
        # dVU/dy
        dhUb = dhUb + 0.25*((V[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+V[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUp[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:])
                          - (V[gx0-1:gxn-1,gy0+1:gyn+1,gz0:gzn,:]+V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hUp[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hUp[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dy;
        # dWU/dz
        dhUb = dhUb + 0.25*((W[gx0-1:gxn-1,gy0:gyn,gz0+1:gzn+1,:]+W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hUp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hUp[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
                          - (W[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]+W[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUp[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hUp[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dz;
    
        # Compute Vb advective tendency
        # dVV/dy
        dhVb = dhVb + 0.25*((V[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+V[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVp[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+hVp[gx0:gxn,gy0:gyn,gz0:gzn,:])
                          - (V[gx0:gxn,gy0:gyn,gz0:gzn,:]+V[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:])*(hVp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVp[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]))/dy;
        # dUV/dx
        dhVb = dhVb + 0.25*((U[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+U[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVp[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:])
                          - (U[gx0+1:gxn+1,gy0-1:gyn-1,gz0:gzn,:]+U[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:])*(hVp[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hVp[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dx;
        # dWV/dz
        dhVb = dhVb + 0.25*((W[gx0:gxn,gy0-1:gyn-1,gz0+1:gzn+1,:]+W[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])*(hVp[gx0:gxn,gy0:gyn,gz0:gzn,:]+hVp[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:])
                          - (W[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]+W[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVp[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]+hVp[gx0:gxn,gy0:gyn,gz0:gzn,:]))/dz;



        # Compute KE divergence correction (quick tapered version)...
        dh2KEdiv = 0.25*((hUb[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hUb[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hUp[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hUp[gx0:gxn,gy0:gyn,gz0:gzn,:])
                         + (hVb[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hVb[gx0:gxn,gy0:gyn,gz0:gzn,:])*(hVp[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hVp[gx0:gxn,gy0:gyn,gz0:gzn,:])
                     )*DivVcc;
        
        hUp = hUp[gx0:gxn,gy0:gyn,gz0:gzn,:];
        hUb = hUb[gx0:gxn,gy0:gyn,gz0:gzn,:];
        hVp = hVp[gx0:gxn,gy0:gyn,gz0:gzn,:];
        hVb = hVb[gx0:gxn,gy0:gyn,gz0:gzn,:];
        hUpp = hUpp[gx0:gxn,gy0:gyn,gz0:gzn,:];
        hUbb = hUbb[gx0:gxn,gy0:gyn,gz0:gzn,:];
        hVpp = hVpp[gx0:gxn,gy0:gyn,gz0:gzn,:];
        hVbb = hVbb[gx0:gxn,gy0:gyn,gz0:gzn,:];
        
        # Compute the spectral energy flux.  3 and 4 are corrected forward and backward, 1 and 2 are exact
        F[mti,:,:,0] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*hUpp + dhVp*hVpp + 1.0*dh2KEdiv,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
        F[mti,:,:,1] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*hUbb + dhVb*hVbb + 1.0*dh2KEdiv,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
        F[mti,:,:,2] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUp*hUpp + dhVp*hVpp + 0.5*dh2KEdiv,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));
        F[mti,:,:,3] = np.reshape(rho0*dx*dy*dz*np.sum(np.sum(dhUb*hUbb + dhVb*hVbb + 0.5*dh2KEdiv,axis=2,keepdims=True),axis=3,keepdims=True)/nt/ColumnVol,(nx,ny));

        print('comp. u_hp_FF_F: mti = ' + str(mti) + ' of ' + str(mti_max) + ', time = ' + str(time.time() - t1)) # DEBUG

    # Compute k
    m_out = np.linspace(0,mti_max-1,mti_max)/Lt;     # cycles / meter

    return (F,m_out)


    
