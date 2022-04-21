import sys
import gc
import warnings
import math
import numpy as np
import numpy.fft
from repmat import repmat
import fld_tools as ft
import global_vars as glb
from global_vars import ng

### JS - Lots of repeated code - compress when you get a chance
def compute_nu_T_k(dU,dV,U,V,thknss,dxu,dyu,dxv,dyv,dxw,dyw,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    # Add ghost cells to fields to handle periodicity in position space.
    (nx,ny,nz,nt) = U.shape; # nf = 1;

    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    thknssUni = ft.get_thknssUni(thknss);
    (uthknss, vthknss) = ft.get_uv_thknss(thknss);
    
    # Compute the filtered velocity and tendencies
    (dU,dV,_) = ft.taper_filter_3D_uvw_nu(dU,dV,dU,dxu,dyu,dxv,dyv,dxw,dyw,thknss,post_taper);
    (U,V,_) = ft.taper_filter_3D_uvw_nu(U,V,U,dxu,dyu,dxv,dyv,dxw,dyw,thknss,post_taper);    

    # Interpolate dU, dV, U, V to uniform grids...
    # May be time consuming - hopefully just use this to validate other method
    nzu = zoversamp*nz;
    (dU,_) = ft.get_4D_vert_uniform_field(dU,thknssUni,nzu,ns=True);
    (dV,_) = ft.get_4D_vert_uniform_field(dV,thknssUni,nzu,ns=True);
    (U,_) = ft.get_4D_vert_uniform_field(U,uthknss,nzu,ns=True);
    (V,dz) = ft.get_4D_vert_uniform_field(V,vthknss,nzu,ns=True);
    Lx = np.sum(dxu[:,0]); Ly = np.sum(dyv[0,:]); dx = Lx/nx; dy = Ly/ny;
    Vol = dx*dy*dz*nx*ny*nzu; Lz = dz*nzu;
    del uthknss, vthknss, thknssUni;
    gc.collect();
    
    # trim the mixed layer
    k_mld = math.ceil(trim_ml/dz);
    dU[:,:,0:k_mld,:] = 0*dU[:,:,0:k_mld,:];
    dV[:,:,0:k_mld,:] = 0*dV[:,:,0:k_mld,:];
    
#     dU = ft.reshape(dU,(nx,ny,nz,nt,nf)); dV = ft.reshape(dV,(nx,ny,nz,nt,nf));

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
    spec_dist = repmat(spec_dist,(1,1,nzu,nt));
    del ki_mod, kj_mod
    
    kt_offset = 0;
    for kti in range(kt_offset,kti_max-kt_offset):
        kt = (kti+1)/Lt;
        ktp = (kti+2)/Lt;
        
        filt = (spec_dist>=kt) * (spec_dist<ktp);
        T[kti] = Lt*dz*dx*dy*np.sum(np.real(FhU[filt]*np.conj(FdhU[filt]) + FhV[filt]*np.conj(FdhV[filt])))*glb.rho0/nt/Vol/nx/ny;

    filt = (spec_dist>=kt);
    T[kti] = Lt*dz*dx*dy*np.sum(np.real(FhU[filt]*np.conj(FdhU[filt]) + FhV[filt]*np.conj(FdhV[filt])))*glb.rho0/nt/Vol/nx/ny;
    
    # Compute k_out
    k_out = np.linspace(0,kti_max-1,kti_max)/Lt;     # cycles / meter

    return (T,k_out);



def compute_nu_T_m(dU,dV,U,V,thknss,dxu,dyu,dxv,dyv,dxw,dyw,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    # Add ghost cells to fields to handle periodicity in position space.
    (nx,ny,nz,nt) = U.shape;
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    thknssUni = ft.get_thknssUni(thknss);
    (uthknss, vthknss) = ft.get_uv_thknss(thknss);
    
    # Compute the filtered velocity and tendencies
    (dU,dV,_) = ft.taper_filter_3D_uvw_nu(dU,dV,dU,dxu,dyu,dxv,dyv,dxw,dyw,thknss,post_taper);
    (dU,dV,_) = ft.mask_boundary_3D(dU,dV,dU,nmask=3,dmask=np.array([1,1,0]));
    (U,V,_) = ft.taper_filter_3D_uvw_nu(U,V,dU,dxu,dyu,dxv,dyv,dxw,dyw,thknss,post_taper);    

    # Interpolate dU, dV, U, V to uniform grids...
    # May be time consuming - hopefully just use this to validate other method
    nzu = zoversamp*nz;
    (dU,_) = ft.get_4D_vert_uniform_field(dU,thknssUni,nzu,ns=True);
    (dV,_) = ft.get_4D_vert_uniform_field(dV,thknssUni,nzu,ns=True);
    (U,_) = ft.get_4D_vert_uniform_field(U,uthknss,nzu,ns=True);
    (V,dz) = ft.get_4D_vert_uniform_field(V,vthknss,nzu,ns=True);
    Lx = np.sum(dxu[:,0]); Ly = np.sum(dyv[0,:]); dx = Lx/nx; dy = Ly/ny;
    Vol = dx*dy*dz*nx*ny*nzu; Lz = dz*nzu;
    del uthknss, vthknss, thknssUni;
    gc.collect();

    # trim the mixed layer
    k_mld = math.ceil(trim_ml/dz);
    dU[:,:,0:k_mld,:] = 0*dU[:,:,0:k_mld,:];
    dV[:,:,0:k_mld,:] = 0*dV[:,:,0:k_mld,:];
    
    # Take Fourier transform to get flux
    FdhU = np.fft.fft(dU,axis=2); FdhV = np.fft.fft(dV,axis=2);
    FhU = np.fft.fft(U,axis=2); FhV = np.fft.fft(V,axis=2);
    
    
    
    ### Compute the "Spectral Flux" via the Transfer Function
    
    # Initialize Output   
    mti_max = math.floor(nz/2-1); 
    Lt = Lz;
    T = np.empty((mti_max,1));
    T[:] = np.NaN;
    
    dftnz = math.floor(nzu/2)+1;
    spec_dist = np.reshape(np.linspace(1,nzu,nzu),(1,1,nzu));
    spec_dist = np.abs(np.mod(spec_dist-2+dftnz,nzu)-dftnz+1)+1;
    spec_dist = repmat(spec_dist,(nx,ny,1,nt));
    
    mt_offset = 0;
    for mti in range(mt_offset,mti_max-mt_offset):
        filt = (spec_dist==(mti+1));
        T[mti] = Lt*dz*dx*dy*np.sum(np.real(FhU[filt]*np.conj(FdhU[filt]) + FhV[filt]*np.conj(FdhV[filt])))*glb.rho0/nt/Vol/nzu;
    filt = (spec_dist>=(mti+1));
    T[mti] = Lt*dz*dx*dy*np.sum(np.real(FhU[filt]*np.conj(FdhU[filt]) + FhV[filt]*np.conj(FdhV[filt])))*glb.rho0/nt/Vol/nzu;
    
    # Compute m_out
    m_out = np.linspace(0,mti_max-1,mti_max)/Lt;     # cycles / meter

    return (T,m_out)



def compute_nu_T_corr_k(dhU,dhV,hU,hV,hUdivV,hVdivV,thknss,dxu,dyv,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    # Add ghost cells to fields to handle periodicity in position space.
    (nx,ny,nz,nt) = hU.shape;
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    # Interpolate (Periodic Assumptions Inherited from Taper Taper.)    
    nzu = zoversamp*nz;
    thknssUni = ft.get_thknssUni(thknss);
    (uthknss, vthknss) = ft.get_uv_thknss(thknss); # BC's don't make a difference because gh¯ost cells get stripped.
    (uthknssUni, vthknssUni) = ft.get_uv_thknss(thknssUni); # BC's don't make a difference because ghost cells get stripped.
    (hUdivV,_) = ft.get_4D_vert_uniform_field(hUdivV,thknss,nzu,ns=True);
    (hVdivV,_) = ft.get_4D_vert_uniform_field(hVdivV,thknss,nzu,ns=True);
    (dhU,_) = ft.get_4D_vert_uniform_field(dhU,uthknssUni,nzu,ns=True);
    (dhV,_) = ft.get_4D_vert_uniform_field(dhV,vthknssUni,nzu,ns=True);
    (hU,_) = ft.get_4D_vert_uniform_field(hU,uthknss,nzu,ns=True);
    (hV,dz) = ft.get_4D_vert_uniform_field(hV,vthknss,nzu,ns=True);
    Lx = np.sum(dxu[:,0]); Ly = np.sum(dyv[0,:]); dx = Lx/nx; dy = Ly/ny;
    Vol = dx*dy*dz*nx*ny*nzu; Lz = dz*nzu;
    del uthknss, vthknss, uthknssUni, vthknssUni;

    gzu0 = ng; gzun = nzu+ng;
    hU = ft.pad_field_3D(hU,ng,prd);
    hV = ft.pad_field_3D(hV,ng,prd);
    hUcc = 0.5*(hU[gx0+1:gxn+1,gy0:gyn,gzu0:gzun,:]+hU[gx0:gxn,gy0:gyn,gzu0:gzun,:]);
    hVcc = 0.5*(hV[gx0:gxn,gy0+1:gyn+1,gzu0:gzun,:]+hV[gx0:gxn,gy0:gyn,gzu0:gzun,:]);
    hU = hU[gx0:gxn,gy0:gyn,gzu0:gzun,:];
    hV = hV[gx0:gxn,gy0:gyn,gzu0:gzun,:];

    # trim the mixed layer
    k_mld = math.ceil(trim_ml/dz);
    dhU[:,:,0:k_mld,:] = 0*dhU[:,:,0:k_mld,:];
    dhV[:,:,0:k_mld,:] = 0*dhV[:,:,0:k_mld,:];
    
    # FFT
    FhU = np.fft.fft(np.fft.fft(hU,axis=0),axis=1); del hU;
    FhV = np.fft.fft(np.fft.fft(hV,axis=0),axis=1); del hV;
    FdhU = np.fft.fft(np.fft.fft(dhU,axis=0),axis=1); del dhU;
    FdhV = np.fft.fft(np.fft.fft(dhV,axis=0),axis=1); del dhV;
    FhUcc = np.fft.fft(np.fft.fft(hUcc,axis=0),axis=1); del hUcc;
    FhVcc = np.fft.fft(np.fft.fft(hVcc,axis=0),axis=1); del hVcc;
    FdKEdivhU = np.fft.fft(np.fft.fft(hUdivV,axis=0),axis=1); del hUdivV;
    FdKEdivhV = np.fft.fft(np.fft.fft(hVdivV,axis=0),axis=1); del hVdivV;
    

    
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
    spec_dist = repmat(spec_dist,(1,1,nzu,nt));
    del ki_mod, kj_mod
    
    kt_offset = 0;
    for kti in range(kt_offset,kti_max-kt_offset):
        kt = (kti+1)/Lt; ktp = (kti+2)/Lt;
        
        filt = (spec_dist>=kt) * (spec_dist<ktp);
        T[kti] = Lt*dz*dx*dy*np.sum(np.real(FhU[filt]*np.conj(FdhU[filt]) 
                                            + FhV[filt]*np.conj(FdhV[filt])
                                            + 0.5*FhUcc[filt]*np.conj(FdKEdivhU[filt]) 
                                            + 0.5*FhVcc[filt]*np.conj(FdKEdivhV[filt])))*glb.rho0/nt/Vol/nx/ny;

    filt = (spec_dist>=kt);
    T[kti] = Lt*dz*dx*dy*np.sum(np.real(FhU[filt]*np.conj(FdhU[filt]) 
                                        + FhV[filt]*np.conj(FdhV[filt])
                                        + 0.5*FhUcc[filt]*np.conj(FdKEdivhU[filt]) 
                                        + 0.5*FhVcc[filt]*np.conj(FdKEdivhV[filt])))*glb.rho0/nt/Vol/nx/ny;

    # Compute k
    k_out = np.linspace(0,kti_max-1,kti_max)/Lt;     # cycles / meter

    return (T,k_out);



def compute_nu_T_corr_m(dhU,dhV,hU,hV,hUdivV,hVdivV,thknss,dxu,dyv,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    # Add ghost cells to fields to handle periodicity in position space.
    (nx,ny,nz,nt) = hU.shape;
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

#     dmask = np.logical_not(post_taper); dmask[2] = 0;
#     (dhU,dhV,_) = ft.mask_boundary_3D(dhU,dhV,dhU,nmask=3,dmask=dmask);
#     (hUdivV,hVdivV,_) = ft.mask_boundary_3D(hUdivV,hVdivV,hUdivV,nmask=3,dmask=dmask);

    thknssUni = ft.get_thknssUni(thknss);
        
    # Interpolate (Periodic Assumptions Inherited from Taper.)    
    nzu = zoversamp*nz;
    (uthknss, vthknss) = ft.get_uv_thknss(thknss); # BC's don't make a difference because ghost cells get stripped.
    (uthknssUni, vthknssUni) = ft.get_uv_thknss(thknssUni); # BC's don't make a difference because ghost cells get stripped.
    (hUdivV,_) = ft.get_4D_vert_uniform_field(hUdivV,thknss,nzu,ns=True);
    (hVdivV,_) = ft.get_4D_vert_uniform_field(hVdivV,thknss,nzu,ns=True);
    (dhU,_) = ft.get_4D_vert_uniform_field(dhU,uthknssUni,nzu,ns=True);
    (dhV,_) = ft.get_4D_vert_uniform_field(dhV,vthknssUni,nzu,ns=True);
    (hU,_) = ft.get_4D_vert_uniform_field(hU,uthknss,nzu,ns=True);
    (hV,dz) = ft.get_4D_vert_uniform_field(hV,vthknss,nzu,ns=True);
    Lx = np.sum(dxu[:,0]); Ly = np.sum(dyv[0,:]); dx = Lx/nx; dy = Ly/ny;
    Vol = dx*dy*dz*nx*ny*nzu; Lz = dz*nzu;
    del uthknss, vthknss, uthknssUni, vthknssUni;

    gzu0 = ng; gzun = nzu+ng;
    hU = ft.pad_field_3D(hU,ng,prd);
    hV = ft.pad_field_3D(hV,ng,prd);
    hUcc = 0.5*(hU[gx0+1:gxn+1,gy0:gyn,gzu0:gzun,:]+hU[gx0:gxn,gy0:gyn,gzu0:gzun,:]);
    hVcc = 0.5*(hV[gx0:gxn,gy0+1:gyn+1,gzu0:gzun,:]+hV[gx0:gxn,gy0:gyn,gzu0:gzun,:]);
    hU = hU[gx0:gxn,gy0:gyn,gzu0:gzun,:];
    hV = hV[gx0:gxn,gy0:gyn,gzu0:gzun,:];

    # trim the mixed layer
    k_mld = math.ceil(trim_ml/dz);
    dhU[:,:,0:k_mld,:] = 0*dhU[:,:,0:k_mld,:];
    dhV[:,:,0:k_mld,:] = 0*dhV[:,:,0:k_mld,:];

    # FFT
    FhU = np.fft.fft(hU,axis=2); del hU;
    FhV = np.fft.fft(hV,axis=2); del hV;
    FdhU = np.fft.fft(dhU,axis=2); del dhU;
    FdhV = np.fft.fft(dhV,axis=2); del dhV;
    FhUcc = np.fft.fft(hUcc,axis=2); del hUcc;
    FhVcc = np.fft.fft(hVcc,axis=2); del hVcc;
    FdKEdivhU = np.fft.fft(hUdivV,axis=2); del hUdivV;
    FdKEdivhV = np.fft.fft(hVdivV,axis=2); del hVdivV;
    
    
    
    ### Compute the "Spectral Flux" via the Transfer Function
    
    # Initialize Output
    mti_max = math.floor(nz/2-1); 
    Lt = Lz;
    T = np.empty((mti_max,2));
    T[:] = np.NaN;

    dftnz = math.floor(nzu/2)+1;
    spec_dist = np.reshape(np.linspace(1,nzu,nzu),(1,1,nzu));
    spec_dist = np.abs(np.mod(spec_dist-2+dftnz,nzu)-dftnz+1)+1;
    spec_dist = repmat(spec_dist,(nx,ny,1,nt));
    
    mt_offset = 0;
    for mti in range(mt_offset,mti_max-mt_offset):
        filt = (spec_dist==(mti+1));
        T[mti,0] = Lt*dz*dx*dy*np.sum(np.real(FhU[filt]*np.conj(FdhU[filt]) 
                                            + FhV[filt]*np.conj(FdhV[filt])
                                            + 0.5*FhUcc[filt]*np.conj(FdKEdivhU[filt]) 
                                            + 0.5*FhVcc[filt]*np.conj(FdKEdivhV[filt])))*glb.rho0/nt/Vol/nzu;        
        T[mti,1] = Lt*dz*dx*dy*np.sum(np.real(0.5*FhUcc[filt]*np.conj(FdKEdivhU[filt]) 
                                            + 0.5*FhVcc[filt]*np.conj(FdKEdivhV[filt])))*glb.rho0/nt/Vol/nzu;        
    filt = (spec_dist>=(mti+1));
    T[mti] = Lt*dz*dx*dy*np.sum(np.real(FhU[filt]*np.conj(FdhU[filt]) 
                                        + FhV[filt]*np.conj(FdhV[filt])
                                            + 0.5*FhUcc[filt]*np.conj(FdKEdivhU[filt]) 
                                            + 0.5*FhVcc[filt]*np.conj(FdKEdivhV[filt])))*glb.rho0/nt/Vol/nzu;        
    T[mti,1] = Lt*dz*dx*dy*np.sum(np.real(0.5*FhUcc[filt]*np.conj(FdKEdivhU[filt]) 
                                          + 0.5*FhVcc[filt]*np.conj(FdKEdivhV[filt])))*glb.rho0/nt/Vol/nzu;        

    # Compute m_out
    m_out = np.linspace(0,mti_max-1,mti_max)/Lt;     # cycles / meter

    return (T,m_out);



def compute_nu_T_corr_full_k(dhU,dhV,hU,hV,hUdivV,hVdivV,thknss,dxu,dyv,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    # Add ghost cells to fields to handle periodicity in position space.
    (nx,ny,nz,nt) = hU.shape;
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    # Interpolate (Periodic Assumptions Inherited from Taper Taper.)    
    nzu = zoversamp*nz;
    thknssUni = ft.get_thknssUni(thknss);
    (uthknss, vthknss) = ft.get_uv_thknss(thknss); # BC's don't make a difference because ghost cells get stripped.
    (uthknssUni, vthknssUni) = ft.get_uv_thknss(thknssUni); # BC's don't make a difference because ghost cells get stripped.
    (hUdivV,_) = ft.get_4D_vert_uniform_field(hUdivV,thknss,nzu,ns=True);
    (hVdivV,_) = ft.get_4D_vert_uniform_field(hVdivV,thknss,nzu,ns=True);
    (dhU,_) = ft.get_4D_vert_uniform_field(dhU,uthknssUni,nzu,ns=True);
    (dhV,_) = ft.get_4D_vert_uniform_field(dhV,vthknssUni,nzu,ns=True);
    (hU,_) = ft.get_4D_vert_uniform_field(hU,uthknss,nzu,ns=True);
    (hV,dz) = ft.get_4D_vert_uniform_field(hV,vthknss,nzu,ns=True);
    Lx = np.sum(dxu[:,0]); Ly = np.sum(dyv[0,:]); dx = Lx/nx; dy = Ly/ny;
    Vol = dx*dy*dz*nx*ny*nzu; Lz = dz*nzu;
    del uthknss, vthknss, uthknssUni, vthknssUni;

    gzu0 = ng; gzun = nzu+ng;
    hU = ft.pad_field_3D(hU,ng,prd);
    hV = ft.pad_field_3D(hV,ng,prd);
    hUcc = 0.5*(hU[gx0+1:gxn+1,gy0:gyn,gzu0:gzun,:]+hU[gx0:gxn,gy0:gyn,gzu0:gzun,:]);
    hVcc = 0.5*(hV[gx0:gxn,gy0+1:gyn+1,gzu0:gzun,:]+hV[gx0:gxn,gy0:gyn,gzu0:gzun,:]);
    hU = hU[gx0:gxn,gy0:gyn,gzu0:gzun,:];
    hV = hV[gx0:gxn,gy0:gyn,gzu0:gzun,:];

    # trim the mixed layer
    k_mld = math.ceil(trim_ml/dz);
    dhU[:,:,0:k_mld,:] = 0*dhU[:,:,0:k_mld,:];
    dhV[:,:,0:k_mld,:] = 0*dhV[:,:,0:k_mld,:];
    
    # FFT
    FhU = np.fft.fft(np.fft.fft(hU,axis=0),axis=1); del hU;
    FhV = np.fft.fft(np.fft.fft(hV,axis=0),axis=1); del hV;
    FdhU = np.fft.fft(np.fft.fft(dhU,axis=0),axis=1); del dhU;
    FdhV = np.fft.fft(np.fft.fft(dhV,axis=0),axis=1); del dhV;
    FhUcc = np.fft.fft(np.fft.fft(hUcc,axis=0),axis=1); del hUcc;
    FhVcc = np.fft.fft(np.fft.fft(hVcc,axis=0),axis=1); del hVcc;
    FdKEdivhU = np.fft.fft(np.fft.fft(hUdivV,axis=0),axis=1); del hUdivV;
    FdKEdivhV = np.fft.fft(np.fft.fft(hVdivV,axis=0),axis=1); del hVdivV;
    

    
    ### Compute the "Spectral Flux" via the Transfer Function
    
    # Initialize Output
    if (1/dx) < (1/dy):
        kti_max = math.floor(nx/2-1);
        Lt = Lx;
    else:
        kti_max = math.floor(ny/2-1);
        Lt = Ly;

    T = np.empty((kti_max,4));
    T[:,:] = np.NaN;
    
    FhUback = 0*FhU;
    FhVback = 0*FhV;
    FhUccback = 0*FhUcc;
    FhVccback = 0*FhVcc;
    
    dftnx = math.floor(nx/2)+1;
    dftny = math.floor(ny/2)+1;
    ki_mod = repmat(np.reshape(np.linspace(1,nx,nx),(nx,1)),(1,ny));
    kj_mod = repmat(np.reshape(np.linspace(1,ny,ny),(1,ny)),(nx,1));
    ki_mod = np.abs(np.mod(ki_mod-2+dftnx,nx)-dftnx+1)/Lx;
    kj_mod = np.abs(np.mod(kj_mod-2+dftny,ny)-dftny+1)/Ly;
    spec_dist = (ki_mod**2+kj_mod**2)**(0.5);
    spec_dist = repmat(spec_dist,(1,1,nzu,nt));
    del ki_mod, kj_mod
    
    kt_offset = 0;
    for kti in range(kt_offset,kti_max-kt_offset):
        kt = (kti+1)/Lt;
        
        # reset fields
        FhU = FhU + FhUback;
        FhV = FhV + FhVback;
        FhUcc = FhUcc + FhUccback;
        FhVcc = FhVcc + FhVccback;

        FhUback = 1*FhU;
        FhVback = 1*FhV;
        FhUccback = 1*FhUcc;
        FhVccback = 1*FhVcc;
        
        filt = spec_dist<kt;

        FhU[filt] = 0;
        FhV[filt] = 0;
        FhUcc[filt] = 0;
        FhVcc[filt] = 0;

        FhUback[np.logical_not(filt)] = 0;
        FhVback[np.logical_not(filt)] = 0;
        FhUccback[np.logical_not(filt)] = 0;
        FhVccback[np.logical_not(filt)] = 0;
        
        dE1 = dx*dy*np.sum(np.sum(
            FhU*np.conj(FdhU) + FhV*np.conj(FdhV)
            + FhUcc*np.conj(FdKEdivhU) + FhVcc*np.conj(FdKEdivhV)
            ,0,keepdims=True),1,keepdims=True)/nx/ny;
        dE1 = dz*np.sum(np.real(dE1))*glb.rho0/nt;
        
        dE2 = dx*dy*np.sum(np.sum(
            FhUback*np.conj(FdhU) + FhVback*np.conj(FdhV)
            + FhUccback*np.conj(FdKEdivhU) + FhVccback*np.conj(FdKEdivhV)
            ,0,keepdims=True),1,keepdims=True)/nx/ny;
        dE2 = dz*np.sum(np.real(dE2))*glb.rho0/nt;
        
        dE3 = dx*dy*np.sum(np.sum(
            FhU*np.conj(FdhU) + FhV*np.conj(FdhV)
            + FhUcc*np.conj(FdKEdivhU) + FhVcc*np.conj(FdKEdivhV)
            - 0.5*FhUcc*np.conj(FdKEdivhU) - 0.5*FhVcc*np.conj(FdKEdivhV)
            ,0,keepdims=True),1,keepdims=True)/nx/ny;
        dE3 = dz*np.sum(np.real(dE3))*glb.rho0/nt;
        
        dE4 = dx*dy*np.sum(np.sum(
            FhUback*np.conj(FdhU) + FhVback*np.conj(FdhV)
            + FhUccback*np.conj(FdKEdivhU) + FhVccback*np.conj(FdKEdivhV)
            - 0.5*FhUccback*np.conj(FdKEdivhU) - 0.5*FhVccback*np.conj(FdKEdivhV)
            ,0,keepdims=True),1,keepdims=True)/nx/ny;
        dE4 = dz*np.sum(np.real(dE4))*glb.rho0/nt;
        
        T[kti,0] = dE1/Vol;
        T[kti,1] = dE2/Vol;
        T[kti,2] = dE3/Vol;
        T[kti,3] = dE4/Vol;

    # Compute k
    k_out = np.linspace(0,kti_max-1,kti_max)/Lt;     # cycles / meter

    return (T,k_out);



def compute_nu_T_corr_full_m(dhU,dhV,hU,hV,hUdivV,hVdivV,thknss,dxu,dyv,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    (nx,ny,nz,nt) = hU.shape;
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

#     dmask = np.logical_not(post_taper); dmask[2] = 0;
#     (dhU,dhV,_) = ft.mask_boundary_3D(dhU,dhV,dhU,nmask=3,dmask=dmask);
#     (hUdivV,hVdivV,_) = ft.mask_boundary_3D(hUdivV,hVdivV,hUdivV,nmask=3,dmask=dmask);

    # Interpolate (Periodic Assumptions Inherited from Taper Taper.)    
    nzu = zoversamp*nz;
    (uthknss, vthknss) = ft.get_uv_thknss(thknss); # BC's don't make a difference because gh¯ost cells get stripped.
    (uthknssUni, vthknssUni) = ft.get_uv_thknss(thknssUni); # BC's don't make a difference because ghost cells get stripped.
    (hUdivV,_) = ft.get_4D_vert_uniform_field(hUdivV,thknss,nzu,ns=True);
    (hVdivV,_) = ft.get_4D_vert_uniform_field(hVdivV,thknss,nzu,ns=True);
    (dhU,_) = ft.get_4D_vert_uniform_field(dhU,uthknssUni,nzu,ns=True);
    (dhV,_) = ft.get_4D_vert_uniform_field(dhV,vthknssUni,nzu,ns=True);
    (hU,_) = ft.get_4D_vert_uniform_field(hU,uthknss,nzu,ns=True);
    (hV,dz) = ft.get_4D_vert_uniform_field(hV,vthknss,nzu,ns=True);
    Lx = np.sum(dxu[:,0]); Ly = np.sum(dyv[0,:]); dx = Lx/nx; dy = Ly/ny;
    Vol = dx*dy*dz*nx*ny*nzu; Lz = dz*nzu;
    del uthknss, vthknss, uthknssUni, vthknssUni;

    gzu0 = ng; gzun = nzu+ng;
    hU = ft.pad_field_3D(hU,ng,prd);
    hV = ft.pad_field_3D(hV,ng,prd);
    hUcc = 0.5*(hU[gx0+1:gxn+1,gy0:gyn,gzu0:gzun,:]+hU[gx0:gxn,gy0:gyn,gzu0:gzun,:]);
    hVcc = 0.5*(hV[gx0:gxn,gy0+1:gyn+1,gzu0:gzun,:]+hV[gx0:gxn,gy0:gyn,gzu0:gzun,:]);
    hU = hU[gx0:gxn,gy0:gyn,gzu0:gzun,:];
    hV = hV[gx0:gxn,gy0:gyn,gzu0:gzun,:];
    
    # FFT
    FhU = np.fft.fft(hU,axis=2); del hU;
    FhV = np.fft.fft(hV,axis=2); del hV;
    FdhU = np.fft.fft(dhU,axis=2); del dhU;
    FdhV = np.fft.fft(dhV,axis=2); del dhV;
    FhUcc = np.fft.fft(hUcc,axis=2); del hUcc;
    FhVcc = np.fft.fft(hVcc,axis=2); del hVcc;
    FdKEdivhU = np.fft.fft(hUdivV,axis=2); del hUdivV;
    FdKEdivhV = np.fft.fft(hVdivV,axis=2); del hVdivV;
    
    
    
    ### Compute the "Spectral Flux" via the Transfer Function
    
    # Initialize Output
    mti_max = math.floor(nz/2-1); 
    Lt = Lz;

    T = np.empty((mti_max,4));
    T[:,:] = np.NaN;
    
    FhUback = 0*FhU;
    FhVback = 0*FhV;
    FhUccback = 0*FhUcc;
    FhVccback = 0*FhVcc;
    
    dftnz = math.floor(nz/2)+1;
    spec_dist = np.reshape(np.linspace(1,nz,nz),(1,1,nz));
    spec_dist = np.abs(np.mod(spec_dist-2+dftnz,nz)-dftnz+1)+1;
    spec_dist = repmat(spec_dist,(nx,ny,1,nt));
    
    mt_offset = 0;
    for mti in range(mt_offset,mti_max-mt_offset):
        
        # reset fields
        FhU = FhU + FhUback;
        FhV = FhV + FhVback;
        FhUcc = FhUcc + FhUccback;
        FhVcc = FhVcc + FhVccback;

        FhUback = 1*FhU;
        FhVback = 1*FhV;
        FhUccback = 1*FhUcc;
        FhVccback = 1*FhVcc;
        
        filt = spec_dist<(mti+1);
        
        FhU[filt] = 0;
        FhV[filt] = 0;
        FhUcc[filt] = 0;
        FhVcc[filt] = 0;
        
        FhUback[np.logical_not(filt)] = 0;
        FhVback[np.logical_not(filt)] = 0;
        FhUccback[np.logical_not(filt)] = 0;
        FhVccback[np.logical_not(filt)] = 0;

        dE1 = dz*np.sum(
            FhU*np.conj(FdhU) + FhV*np.conj(FdhV)
            + FhUcc*np.conj(FhUdivV) + FhVcc*np.conj(FhVdivV)
            ,2,keepdims=True)/nzu;
        dE1 = dx*dy*np.sum(np.real(dE1))*glb.rho0/nt;
        
        dE2 = dz*np.sum(
            FhUback*np.conj(FdhU) + FhVback*np.conj(FdhV)
            + FhUccback*np.conj(FhUdivV) + FhVccback*np.conj(FhVdivV)
            ,2,keepdims=True)/nzu;
        dE2 = dx*dy*np.sum(np.real(dE2))*glb.rho0/nt;
        
        dE3 = dz*np.sum(
            FhU*np.conj(FdhU) + FhV*np.conj(FdhV)
            + FhUcc*np.conj(FhUdivV) + FhVcc*np.conj(FhVdivV)
            - 0.5*FhUcc*np.conj(FhUdivV) - 0.5*FhVcc*np.conj(FhVdivV)
            ,2,keepdims=True)/nzu;
        dE3 = dx*dy*np.sum(np.real(dE3))*glb.rho0/nt;
        
        dE4 = dz*np.sum(
            FhUback*np.conj(FdhU) + FhVback*np.conj(FdhV)
            + FhUccback*np.conj(FhUdivV) + FhVccback*np.conj(FhVdivV)
            - 0.5*FhUccback*np.conj(FhUdivV) - 0.5*FhVccback*np.conj(FhVdivV)
            ,2,keepdims=True)/nzu;
        dE4 = dx*dy*np.sum(np.real(dE4))*glb.rho0/nt;
        
        T[mti,0] = dE1/Vol;
        T[mti,1] = dE2/Vol;
        T[mti,2] = dE3/Vol;
        T[mti,3] = dE4/Vol;

    # Compute m_out
    m_out = linspace(0,mti_max-1,mti_max)/Lt;     # cycles / meter

    return (T,m_out);



def compute_nu_T_k_z(dU,dV,U,V,thknss,dxu,dyu,dxv,dyv,dxw,dyw,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    # Add ghost cellps to fields to handle periodicity in position space.
    (nx,ny,nz,nt) = U.shape; # nf = 1;

    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    thknssUni = ft.get_thknssUni(thknss);
    (uthknss, vthknss) = ft.get_uv_thknss(thknss);
    
    # Compute the filtered velocity and tendencies
    (dU,dV,_) = ft.taper_filter_3D_uvw_nu(dU,dV,dU,dxu,dyu,dxv,dyv,dxw,dyw,thknss,post_taper);
    (U,V,_) = ft.taper_filter_3D_uvw_nu(U,V,U,dxu,dyu,dxv,dyv,dxw,dyw,thknss,post_taper);    

    # Interpolate dU, dV, U, V to uniform grids...
    # May be time consuming - hopefully just use this to validate other method
    nzu = zoversamp*nz;
    (dU,_) = ft.get_4D_vert_uniform_field(dU,thknssUni,nzu,ns=True);
    (dV,_) = ft.get_4D_vert_uniform_field(dV,thknssUni,nzu,ns=True);
    (U,_) = ft.get_4D_vert_uniform_field(U,uthknss,nzu,ns=True);
    (V,dz) = ft.get_4D_vert_uniform_field(V,vthknss,nzu,ns=True);
    Lx = np.sum(dxu[:,0]); Ly = np.sum(dyv[0,:]); dx = Lx/nx; dy = Ly/ny;
    Vol = dx*dy*dz*nx*ny*nzu; Lz = dz*nzu; LayerVol = Lx*Ly*dz;
    del uthknss, vthknss, thknssUni;
    gc.collect();
    
    # trim the mixed layer
    k_mld = math.ceil(trim_ml/dz);
    dU[:,:,0:k_mld,:] = 0*dU[:,:,0:k_mld,:];
    dV[:,:,0:k_mld,:] = 0*dV[:,:,0:k_mld,:];
    
    # Take Fourier transform to get flux
    FdhU = np.fft.fft(np.fft.fft(dU,axis=0),axis=1); FdhV = np.fft.fft(np.fft.fft(dV,axis=0),axis=1);
    FhU = np.fft.fft(np.fft.fft(U,axis=0),axis=1); FhV = np.fft.fft(np.fft.fft(V,axis=0),axis=1);
    
    
    
    ### Compute the "Spectral Flux" via the Transfer Function
    
    # Initialize Ouptput
    if (1/dx) < (1/dy):
        kti_max = math.floor(nx/2-1);
        Lt = Lx;
    else:
        kti_max = math.floor(ny/2-1);
        Lt = Ly;

    T = np.empty((kti_max,nz));
    T[:] = np.NaN;
    
    dftnx = math.floor(nx/2)+1;
    dftny = math.floor(ny/2)+1;
    ki_mod = repmat(np.reshape(np.linspace(1,nx,nx),(nx,1)),(1,ny));
    kj_mod = repmat(np.reshape(np.linspace(1,ny,ny),(1,ny)),(nx,1));
    ki_mod = np.abs(np.mod(ki_mod-2+dftnx,nx)-dftnx+1)/Lx;
    kj_mod = np.abs(np.mod(kj_mod-2+dftny,ny)-dftny+1)/Ly;
    spec_dist = (ki_mod**2+kj_mod**2)**(0.5);
    spec_dist = repmat(spec_dist,(1,1,nzu,nt));
    del ki_mod, kj_mod
    
    kt_offset = 0;
    for kti in range(kt_offset,kti_max-kt_offset):
        kt = (kti+1)/Lt;
        ktp = (kti+2)/Lt;
        
        filt = (spec_dist>=kt) * (spec_dist<ktp);
        T[kti,:] = np.reshape(Lt*dz*dx*dy*np.sum(np.sum(np.sum(np.real(filt*(FhU*np.conj(FdhU) + FhV*np.conj(FdhV))),axis=0,keepdims=True),axis=1,keepdims=True),axis=3,keepdims=True)*glb.rho0/nt/LayerVol/nx/ny,(nz));

    filt = (spec_dist>=kt);
    T[kti,:] = np.reshape(Lt*dz*dx*dy*np.sum(np.sum(np.sum(np.real(filt*(FhU*np.conj(FdhU) + FhV*np.conj(FdhV))),axis=0,keepdims=True),axis=1,keepdims=True),axis=3,keepdims=True)*glb.rho0/nt/LayerVol/nx/ny,(nz));
    
    # Compute k_out
    k_out = np.linspace(0,kti_max-1,kti_max)/Lt;     # cycles / meter

    return (T,k_out);



def compute_nu_T_m_xy(dU,dV,U,V,thknss,dxu,dyu,dxv,dyv,dxw,dyw,ugrid,vgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    # Add ghost cells to fields to handle periodicity in position space.
    (nx,ny,nz,nt) = U.shape;
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    thknssUni = ft.get_thknssUni(thknss);
    (uthknss, vthknss) = ft.get_uv_thknss(thknss);
    
    # Compute the filtered velocity and tendencies
    (dU,dV,_) = ft.taper_filter_3D_uvw_nu(dU,dV,dU,dxu,dyu,dxv,dyv,dxw,dyw,thknss,post_taper);
    (dU,dV,_) = ft.mask_boundary_3D(dU,dV,dU,nmask=3,dmask=np.array([1,1,0]));
    (U,V,_) = ft.taper_filter_3D_uvw_nu(U,V,dU,dxu,dyu,dxv,dyv,dxw,dyw,thknss,post_taper);    

    # Interpolate dU, dV, U, V to uniform grids...
    # May be time consuming - hopefully just use this to validate other method
    nzu = zoversamp*nz;
    (dU,_) = ft.get_4D_vert_uniform_field(dU,thknssUni,nzu,ns=True);
    (dV,_) = ft.get_4D_vert_uniform_field(dV,thknssUni,nzu,ns=True);
    (U,_) = ft.get_4D_vert_uniform_field(U,uthknss,nzu,ns=True);
    (V,dz) = ft.get_4D_vert_uniform_field(V,vthknss,nzu,ns=True);
    Lx = np.sum(dxu[:,0]); Ly = np.sum(dyv[0,:]); dx = Lx/nx; dy = Ly/ny;
    Vol = dx*dy*dz*nx*ny*nzu; Lz = dz*nzu; ColumnVol = dx*dy*Lz;
    del uthknss, vthknss, thknssUni;
    gc.collect();

    # trim the mixed layer
    k_mld = math.ceil(trim_ml/dz);
    dU[:,:,0:k_mld,:] = 0*dU[:,:,0:k_mld,:];
    dV[:,:,0:k_mld,:] = 0*dV[:,:,0:k_mld,:];
    
    # Take Fourier transform to get flux
    FdhU = np.fft.fft(dU,axis=2); FdhV = np.fft.fft(dV,axis=2);
    FhU = np.fft.fft(U,axis=2); FhV = np.fft.fft(V,axis=2);
    
    
    
    ### Compute the "Spectral Flux" via the Transfer Function
    
    # Initialize Output   
    mti_max = math.floor(nz/2-1); 
    Lt = Lz;
    T = np.empty((mti_max,nx,ny));
    T[:] = np.NaN;
    
    dftnz = math.floor(nzu/2)+1;
    spec_dist = np.reshape(np.linspace(1,nzu,nzu),(1,1,nzu));
    spec_dist = np.abs(np.mod(spec_dist-2+dftnz,nzu)-dftnz+1)+1;
    spec_dist = repmat(spec_dist,(nx,ny,1,nt));
    
    mt_offset = 0;
    for mti in range(mt_offset,mti_max-mt_offset):
        filt = (spec_dist==(mti+1));
        T[mti,:,:] = np.reshape(Lt*dz*dx*dy*np.sum(np.sum(np.real(filt*(FhU*np.conj(FdhU) + FhV*np.conj(FdhV))),axis=2,keepdims=True),axis=3,keepdims=True)*glb.rho0/nt/ColumnVol/nzu,(nx,ny));
    filt = (spec_dist>=(mti+1));
    T[mti,:,:] = np.reshape(Lt*dz*dx*dy*np.sum(np.sum(np.real(filt*(FhU*np.conj(FdhU) + FhV*np.conj(FdhV))),axis=2,keepdims=True),axis=3,keepdims=True)*glb.rho0/nt/ColumnVol/nzu,(nx,ny));
    
    # Compute m_out
    m_out = np.linspace(0,mti_max-1,mti_max)/Lt;     # cycles / meter

    return (T,m_out)



