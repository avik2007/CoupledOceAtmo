import sys
import gc
import warnings
import math
import numpy as np
import numpy.fft
from repmat import repmat
import fld_tools as ft

def compute_u_T_corr_k(dhU,dhV,hU,hV,hUdivV,hVdivV,dx,dy,dz,post_taper=np.array([1,1,0]),trim_ml=0,prd=np.array([1,1,1])):

    # Add ghost cells to fields to handle periodicity in position space.
    (nx,ny,nz,nt) = hU.shape;
    ng = 2; # number of ghost cells, must be one larger than needed for python indexing
    rho0 = 1027.5;
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    Lx = nx*dx; Ly = ny*dy; Lz = dz*nz;
    Vol = dx*dy*dz*nx*ny*nz;

    # trim the mixed layer
    k_mld = math.ceil(trim_ml/dz);
    dhU[:,:,0:k_mld,:] = 0*dhU[:,:,0:k_mld,:];
    dhV[:,:,0:k_mld,:] = 0*dhV[:,:,0:k_mld,:];

    hU = ft.pad_field_3D(hU,ng,prd);
    hV = ft.pad_field_3D(hV,ng,prd);
    hUcc = 0.5*(hU[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hU[gx0:gxn,gy0:gyn,gz0:gzn,:]);
    hVcc = 0.5*(hV[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hV[gx0:gxn,gy0:gyn,gz0:gzn,:]);
    hU = hU[gx0:gxn,gy0:gyn,gz0:gzn,:];
    hV = hV[gx0:gxn,gy0:gyn,gz0:gzn,:];
    
    # FFT
    FhU = np.fft.fft(np.fft.fft(hU,axis=0),axis=1); del hU;
    FhV = np.fft.fft(np.fft.fft(hV,axis=0),axis=1); del hV;
    FdhU = np.fft.fft(np.fft.fft(dhU,axis=0),axis=1); del dhU;
    FdhV = np.fft.fft(np.fft.fft(dhV,axis=0),axis=1); del dhV;
    FhUcc = np.fft.fft(np.fft.fft(hUcc,axis=0),axis=1); del hUcc;
    FhVcc = np.fft.fft(np.fft.fft(hVcc,axis=0),axis=1); del hVcc;
    FdKEdivhU = np.fft.fft(np.fft.fft(hUdivV,axis=0),axis=1); del hUdivV;
    FdKEdivhV = np.fft.fft(np.fft.fft(hVdivV,axis=0),axis=1); del hVdivV;
    
    # print(np.sum(np.abs(FhU)))
    # print(np.sum(np.abs(FdhU)))
    # print(np.sum(np.abs(FhUcc)))
    # print(np.sum(np.abs(FdKEdivhU)))


    
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
        kt = (kti+1)/Lt; ktp = (kti+2)/Lt;
        
        filt = (spec_dist>=kt) * (spec_dist<ktp);
        T[kti] = Lt*dz*dx*dy*np.sum(np.real(FhU[filt]*np.conj(FdhU[filt]) 
                                            + FhV[filt]*np.conj(FdhV[filt])
                                            + 0.5*FhUcc[filt]*np.conj(FdKEdivhU[filt]) 
                                            + 0.5*FhVcc[filt]*np.conj(FdKEdivhV[filt])))*rho0/nt/Vol/nx/ny;

    filt = (spec_dist>=kt);
    T[kti] = Lt*dz*dx*dy*np.sum(np.real(FhU[filt]*np.conj(FdhU[filt]) 
                                        + FhV[filt]*np.conj(FdhV[filt])
                                        + 0.5*FhUcc[filt]*np.conj(FdKEdivhU[filt]) 
                                        + 0.5*FhVcc[filt]*np.conj(FdKEdivhV[filt])))*rho0/nt/Vol/nx/ny;

    # Compute k
    k_out = np.linspace(0,kti_max-1,kti_max)/Lt;     # cycles / meter

    return (T,k_out);



def compute_u_T_corr_m(dhU,dhV,hU,hV,hUdivV,hVdivV,dx,dy,dz,post_taper=np.array([1,1,0]),trim_ml=0,prd=np.array([1,1,1])):

    # Add ghost cells to fields to handle periodicity in position space.
    (nx,ny,nz,nt) = hU.shape;
    ng = 2; # number of ghost cells, must be one larger than needed for python indexing
    rho0 = 1027.5;
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    Lx = nx*dx; Ly = ny*dy; Lz = dz*nz;
    Vol = dx*dy*dz*nx*ny*nz;

    # trim the mixed layer
    k_mld = math.ceil(trim_ml/dz);
    dhU[:,:,0:k_mld,:] = 0*dhU[:,:,0:k_mld,:];
    dhV[:,:,0:k_mld,:] = 0*dhV[:,:,0:k_mld,:];

    hU = ft.pad_field_3D(hU,ng,prd);
    hV = ft.pad_field_3D(hV,ng,prd);
    hUcc = 0.5*(hU[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]+hU[gx0:gxn,gy0:gyn,gz0:gzn,:]);
    hVcc = 0.5*(hV[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]+hV[gx0:gxn,gy0:gyn,gz0:gzn,:]);
    hU = hU[gx0:gxn,gy0:gyn,gz0:gzn,:];
    hV = hV[gx0:gxn,gy0:gyn,gz0:gzn,:];
    
    # FFT
    FhU = np.fft.fft(hU,axis=2); del hU;
    FhV = np.fft.fft(hV,axis=2); del hV;
    FdhU = np.fft.fft(dhU,axis=2); del dhU;
    FdhV = np.fft.fft(dhV,axis=2); del dhV;
    FhUcc = np.fft.fft(hUcc,axis=2); del hUcc;
    FhVcc = np.fft.fft(hVcc,axis=2); del hVcc;
    FdKEdivhU = np.fft.fft(hUdivV,axis=2); del hUdivV;
    FdKEdivhV = np.fft.fft(hVdivV,axis=2); del hVdivV;

    # print(np.sum(np.abs(FhU)))
    # print(np.sum(np.abs(FdhU)))
    # print(np.sum(np.abs(FhUcc)))
    # print(np.sum(np.abs(FdKEdivhU)))
    
    
    
    ### Compute the "Spectral Flux" via the Transfer Function
    
    # Initialize Output
    mti_max = math.floor(nz/2-1); 
    Lt = Lz;
    T = np.empty((mti_max,2));
    T[:] = np.NaN;

    dftnz = math.floor(nz/2)+1;
    spec_dist = np.reshape(np.linspace(1,nz,nz),(1,1,nz));
    spec_dist = np.abs(np.mod(spec_dist-2+dftnz,nz)-dftnz+1)+1;
    spec_dist = repmat(spec_dist,(nx,ny,1,nt));
    
    mt_offset = 0;
    for mti in range(mt_offset,mti_max-mt_offset):
        filt = (spec_dist==(mti+1));
        T[mti,0] = Lt*dz*dx*dy*np.sum(np.real(FhU[filt]*np.conj(FdhU[filt]) 
                                            + FhV[filt]*np.conj(FdhV[filt])
                                            + 0.5*FhUcc[filt]*np.conj(FdKEdivhU[filt]) 
                                            + 0.5*FhVcc[filt]*np.conj(FdKEdivhV[filt])))*rho0/nt/Vol/nz;        
        T[mti,1] = Lt*dz*dx*dy*np.sum(np.real(0.5*FhUcc[filt]*np.conj(FdKEdivhU[filt]) 
                                            + 0.5*FhVcc[filt]*np.conj(FdKEdivhV[filt])))*rho0/nt/Vol/nz;        
    filt = (spec_dist>=(mti+1));
    T[mti,0] = Lt*dz*dx*dy*np.sum(np.real(FhU[filt]*np.conj(FdhU[filt]) 
                                        + FhV[filt]*np.conj(FdhV[filt])
                                            + 0.5*FhUcc[filt]*np.conj(FdKEdivhU[filt]) 
                                            + 0.5*FhVcc[filt]*np.conj(FdKEdivhV[filt])))*rho0/nt/Vol/nz;        
    T[mti,1] = Lt*dz*dx*dy*np.sum(np.real(0.5*FhUcc[filt]*np.conj(FdKEdivhU[filt]) 
                                          + 0.5*FhVcc[filt]*np.conj(FdKEdivhV[filt])))*rho0/nt/Vol/nz;        

    # Compute m_out
    m_out = np.linspace(0,mti_max-1,mti_max)/Lt;     # cycles / meter

    # print(np.sum(T))

    return (T,m_out);




def compute_u_T_k(dhU,dhV,hU,hV,dx,dy,dz,post_taper=np.array([1,1,0]),trim_ml=0,prd=np.array([1,1,1])):

    # Add ghost cells to fields to handle periodicity in position space.
    (nx,ny,nz,nt) = hU.shape;
    ng = 2; # number of ghost cells, must be one larger than needed for python indexing
    rho0 = 1027.5;
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    Lx = nx*dx; Ly = ny*dy; Lz = dz*nz;
    Vol = dx*dy*dz*nx*ny*nz;

    # trim the mixed layer
    k_mld = math.ceil(trim_ml/dz);
    dhU[:,:,0:k_mld,:] = 0*dhU[:,:,0:k_mld,:];
    dhV[:,:,0:k_mld,:] = 0*dhV[:,:,0:k_mld,:];

    # FFT
    FhU = np.fft.fft(np.fft.fft(hU,axis=0),axis=1); del hU;
    FhV = np.fft.fft(np.fft.fft(hV,axis=0),axis=1); del hV;
    FdhU = np.fft.fft(np.fft.fft(dhU,axis=0),axis=1); del dhU;
    FdhV = np.fft.fft(np.fft.fft(dhV,axis=0),axis=1); del dhV;


    
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
        kt = (kti+1)/Lt; ktp = (kti+2)/Lt;
        
        filt = (spec_dist>=kt) * (spec_dist<ktp);
        T[kti] = Lt*dz*dx*dy*np.sum(np.real(FhU[filt]*np.conj(FdhU[filt]) 
                                            + FhV[filt]*np.conj(FdhV[filt])))*rho0/nt/Vol/nx/ny;

    filt = (spec_dist>=kt);
    T[kti] = Lt*dz*dx*dy*np.sum(np.real(FhU[filt]*np.conj(FdhU[filt]) 
                                        + FhV[filt]*np.conj(FdhV[filt])))*rho0/nt/Vol/nx/ny;

    # Compute k
    k_out = np.linspace(0,kti_max-1,kti_max)/Lt;     # cycles / meter

    return (T,k_out);



def compute_u_T_m(dhU,dhV,hU,hV,dx,dy,dz,post_taper=np.array([1,1,0]),trim_ml=0,prd=np.array([1,1,1])):

    # Add ghost cells to fields to handle periodicity in position space.
    (nx,ny,nz,nt) = hU.shape;
    ng = 2; # number of ghost cells, must be one larger than needed for python indexing
    rho0 = 1027.5;
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    Lx = nx*dx; Ly = ny*dy; Lz = dz*nz;
    Vol = dx*dy*dz*nx*ny*nz;

    # trim the mixed layer
    k_mld = math.ceil(trim_ml/dz);
    dhU[:,:,0:k_mld,:] = 0*dhU[:,:,0:k_mld,:];
    dhV[:,:,0:k_mld,:] = 0*dhV[:,:,0:k_mld,:];

    # FFT
    FhU = np.fft.fft(hU,axis=2); del hU;
    FhV = np.fft.fft(hV,axis=2); del hV;
    FdhU = np.fft.fft(dhU,axis=2); del dhU;
    FdhV = np.fft.fft(dhV,axis=2); del dhV;
    
    
    
    ### Compute the "Spectral Flux" via the Transfer Function
    
    # Initialize Output
    mti_max = math.floor(nz/2-1); 
    Lt = Lz;
    T = np.empty((mti_max,1));
    T[:] = np.NaN;

    dftnz = math.floor(nz/2)+1;
    spec_dist = np.reshape(np.linspace(1,nz,nz),(1,1,nz));
    spec_dist = np.abs(np.mod(spec_dist-2+dftnz,nz)-dftnz+1)+1;
    spec_dist = repmat(spec_dist,(nx,ny,1,nt));
    
    mt_offset = 0;
    for mti in range(mt_offset,mti_max-mt_offset):
        filt = (spec_dist==(mti+1));
        T[mti] = Lt*dz*dx*dy*np.sum(np.real(FhU[filt]*np.conj(FdhU[filt]) 
                                            + FhV[filt]*np.conj(FdhV[filt])))*rho0/nt/Vol/nz;        
    filt = (spec_dist>=(mti+1));
    T[mti] = Lt*dz*dx*dy*np.sum(np.real(FhU[filt]*np.conj(FdhU[filt]) 
                                        + FhV[filt]*np.conj(FdhV[filt])))*rho0/nt/Vol/nz;        

    # Compute m_out
    m_out = np.linspace(0,mti_max-1,mti_max)/Lt;     # cycles / meter

    # print(np.sum(T))

    return (T,m_out);












