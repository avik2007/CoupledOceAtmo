import gc
import math
import numpy as np
import numpy.fft
from repmat import repmat
import fld_tools as ft

def compute_u_kpp_k(U,V,rho,fnu_thknss,Aew,Ans,dx,dy,dz,post_taper=np.array([1,1,0]),trim_ml=0,prd=np.array([1,1,1])):

    # Computes vertical dissipation from the uniform flds
    
    # Initialize Constants
    (nx, ny, nz, nt) = U.shape;
    bot_filt_iter = 3;
    g = 9.81;
    rho0 = 1027.5; 
    viscArNr = 5.6614e-4;
    Riinfty = 0.6998;
    BVSQcon = -2e-5;
    difm0 = 5e-3;
    difs0 = 5e-3;
    dift0 = 5e-3;
    difmcon = 1e-1;
    difscon = 1e-1;
    diftcon = 1e-1;
    Trho0 = 1.9;
    dsfmax = 1e-2;
    cstar = 1e1;

    Lx = dx*nx; Ly = dy*ny; Lz = dz*nz; Vol = Lx*Ly*Lz;
    rho0 = 1035; # for energy output (optional)    
    
    # Add ghost cells to fields to handle periodicity in position space.
    ng = 2; # number of ghost cells: need 2 because Python cannot evaluate [:0]
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits
    U = ft.pad_field_3D(U,ng,prd); V = ft.pad_field_3D(V,ng,prd); rho = ft.pad_field_3D(rho,ng,prd);
    
    nzmax = np.ceil(np.sum(fnu_thknss,axis=2,keepdims=True)/dz);
    nzmax = repmat(nzmax,(1,1,nz,1));
    
    # find local buoyancy gradient (NOTE: should be 0 below bottom, worth checking...)
    dblocSm = g*( (rho[gx0:gxn,gy0:gyn,gz0:gzn,:] - rho[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:])
                  /(rho[gx0:gxn,gy0:gyn,gz0:gzn,:]
                + rho0)
                 );

    # compute diffus2 using unsmoothed buoyancy gradianet
    diffus2 = dblocSm/dz; # surf layer is just 0
    diffus2[nzmax<=1] = 0.0;

    # apply smoothing to dbloc
    dblocSm = ft.smooth_horiz(dblocSm,0*dblocSm+1);

    # find the local velocity shear.
    shsq = ft.get_shsq(U,V);

    # compute diffusivity
    diffus = dz*dblocSm/(np.maximum(shsq[gx0:gxn,gy0:gyn,gz0:gzn,:],1e-10));
    diffus[nzmax<=1] = 0.0;
    
    for i in range(0,nx):
        for j in range(0,ny):
            for k in range(0,nz):
                for t in range(0,nt):
                    if (k>=nzmax[i,j,k,t]) and (nzmax[i,j,k,t]>=2):
                        diffus[i,j,k,t] = diffus[i,j,k-1,t];
                        diffus2[i,j,k,t] = diffus2[i,j,k-1,t];

    Rig   = np.maximum(diffus2 , BVSQcon );
    ratio = np.minimum((BVSQcon-Rig)/BVSQcon, 1.0);
    fcon  = 1.0 - ratio*ratio;
    fcon  = fcon*fcon*fcon;

    # evaluate f of smooth Ri for shear instability, store in fRi

    Rig  = np.maximum(diffus, 0.0);
    ratio = np.minimum(Rig/Riinfty, 1.0);
    fRi   = 1.0 - ratio*ratio;
    fRi   = fRi*fRi*fRi;

    diffus = viscArNr + fcon*difmcon + fRi*difm0;

    for i in range(0,nx):
        for j in range(0,ny):
            for k in range(0,nz):
                for t in range(0,nt):
                    if (k>=nzmax[i,j,k,t]):
                        diffus[i,j,k,t] = 0.0;


    
    ### JS NOTE: for now, skip the BL mixing enhancment (requires BL impl.)
    
    diffus = ft.pad_field_3D(diffus,ng,prd); diffus[:,:,0,:] = 0*diffus[:,:,0,:]; # handled earlier in the code as a 0 index.
    KPPviscAz = diffus[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:];

    del dblocSm, shsq, diffus, diffus2, nzmax;
    gc.collect()
    
    KPPviscAz = ft.pad_field_3D(KPPviscAz,ng,prd);
    KappaRU = np.maximum(viscArNr,0.5*(KPPviscAz[gx0:gxn,gy0:gyn,gz0:gzn,:]+KPPviscAz[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]));
    KappaRV = np.maximum(viscArNr,0.5*(KPPviscAz[gx0:gxn,gy0:gyn,gz0:gzn,:]+KPPviscAz[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]));
    
    rViscFluxU = dx*dy*KappaRU*(U[gx0:gxn,gy0:gyn,gz0:gzn,:]-U[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:])/dz;
    rViscFluxU[:,:,0,:] = 0*rViscFluxU[:,:,0,:]; rViscFluxU[:,:,-1,:] = 0*rViscFluxU[:,:,-1,:];
    rViscFluxV = dx*dy*KappaRV*(V[gx0:gxn,gy0:gyn,gz0:gzn,:]-V[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:])/dz;    
    rViscFluxV[:,:,0,:] = 0*rViscFluxV[:,:,0,:]; rViscFluxV[:,:,-1,:] = 0*rViscFluxV[:,:,-1,:];
    
    rViscFluxU = ft.pad_field_3D(rViscFluxU,ng,prd); rViscFluxV = ft.pad_field_3D(rViscFluxV,ng,prd);
    dU = (rViscFluxU[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:]-rViscFluxU[gx0:gxn,gy0:gyn,gz0:gzn,:])/(dx*dy*dz);
    dV = (rViscFluxV[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:]-rViscFluxU[gx0:gxn,gy0:gyn,gz0:gzn,:])/(dx*dy*dz);

    del rViscFluxU, rViscFluxV;
    gc.collect()


    
    ### JS NOTE: for now, skip side drag and bottom drag...

    # Compute the filtered velocity and tendencies
    (dU,dV,_) = ft.taper_filter_3D_uvw_u(dU,dV,dU,post_taper);
    (U,V,_) = ft.taper_filter_3D_uvw_u(U[gx0:gxn,gy0:gyn,gz0:gzn,:],V[gx0:gxn,gy0:gyn,gz0:gzn,:],dU,post_taper);
    
    # trim the mixed layer
    k_mld = math.ceil(trim_ml/dz);
    dU[:,:,0:k_mld,:] = 0*dU[:,:,0:k_mld,:];
    dV[:,:,0:k_mld,:] = 0*dV[:,:,0:k_mld,:];

    # trim the bottom cells
    dU = ft.remove_bottom_adjacent_cells(dU,dz,fnu_thknss,bot_filt_iter);
    dV = ft.remove_bottom_adjacent_cells(dV,dz,fnu_thknss,bot_filt_iter);

    # Take Fourier transform to get flux
    FdhU = np.fft.fft(np.fft.fft(dU,axis=0),axis=1); FdhV = np.fft.fft(np.fft.fft(dV,axis=0),axis=1);
    FhU = np.fft.fft(np.fft.fft(U,axis=0),axis=1); FhV = np.fft.fft(np.fft.fft(V,axis=0),axis=1);

    
    
    ### Compute the "Spectral Flux" via the Transfer Function
    
    # Initialize Output
    if (1/dx) < (1/dy):
        kti_max = math.floor(nx/2-1);
        Lt = Lx;
    else:
        kti_max = math.floor(ny/2-1);
        Lt = Ly;

    T = np.empty((kti_max,1));
    T[:] = np.NaN;
    
    dftnx = math.floor(nx/2)+1;
    dftny = math.floor(ny/2)+1;
    ki_mod = repmat(np.reshape(np.linspace(1,nx,nx),(nx,1)),(1,ny));
    kj_mod = repmat(np.reshape(np.linspace(1,ny,ny),(1,ny)),(nx,1));
    ki_mod = np.abs(np.mod(ki_mod-2+dftnx,nx)-dftnx+1)/Lx;
    kj_mod = np.abs(np.mod(kj_mod-2+dftny,ny)-dftny+1)/Ly;
    spec_dist = (ki_mod**2+kj_mod**2)**(0.5);
    spec_dist = repmat(spec_dist,(1,1,nz,nt));
    del ki_mod, kj_mod
    
    kt_offset = 0;
    for kti in range(kt_offset,kti_max-kt_offset):
        kt = kti/Lt;
        ktp = (kti+1)/Lt;
        
        filt = (spec_dist>=kt) * (spec_dist<ktp);
        
        T[kti] = dx*dy*dz*Lt*rho0*np.sum(np.real(FhU[filt]*np.conj(FdhU[filt]) + FhV[filt]*np.conj(FdhV[filt])))/nx/ny/nt/Vol;
    
    # Compute k_out
    k_out = np.linspace(0,kti_max-1,kti_max)/Lt;     # cycles / meter

    return (T, k_out)







 



















