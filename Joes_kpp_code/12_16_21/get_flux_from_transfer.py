import numpy as np

def get_flux_from_transfer(T,L):

    if (len(T.shape)==1):
        nk = T.size;
        F = T.copy();
        for ki in range(nk-2,-1,-1):
            F[ki] = F[ki+1] + F[ki];
    else:
        nk = T.shape[0]; nd = T.shape[1];
        F = T.copy();
        for ki in range(nk-2,-1,-1):
            for di in range(0,nd):
                F[ki,di] = F[ki+1,di] + F[ki,di];

    return F/L;

def get_neg_bw_flux_from_transfer(T,L):

    if (len(T.shape)==1):
        nk = T.size;
        F = T.copy();
        F[0] = -F[0]
        for ki in range(1,nk):
            F[ki] = F[ki-1] - F[ki];
    elif (len(T.shape)==2):
        (nk,nb) = T.shape;
        F = T.copy();
        F[0,:] = -F[0,:]
        for ki in range(1,nk):
            F[ki,:] = F[ki-1,:] - F[ki,:];
    else:
        raise RuntimeError('get_neg_bw_flux_from_transfer() only supports 1D or 2D arrays...');

    return F/L;

def get_1D_flux_from_transfer(T,L):

    (nk,nz) = T.shape;
    F = T.copy();
    for ki in range(nk-2,-1,-1):
        F[ki,:] = F[ki+1,:] + F[ki,:];

    return F/L;

def get_2D_flux_from_transfer(T,L):

    if (len(T.shape)==3):
        (nk,nx,ny) = T.shape;
        F = T.copy();
        for ki in range(nk-2,-1,-1):
            F[ki,:,:] = F[ki+1,:,:] + F[ki,:,:];
    elif (len(T.shape)==4):
        (nk,nx,ny,nb) = T.shape;
        F = T.copy();
        for ki in range(nk-2,-1,-1):
            F[ki,:,:,:] = F[ki+1,:,:,:] + F[ki,:,:,:];
    else:
        raise RuntimeError('get_2D_flux_from_transfer() only supports 3D or 4D arrays...');
    return F/L;

def get_transfer_from_flux(F,L):

    nk = F.size;
    T = F.copy();
    for ki in range(nk-2,-1,-1):
        T[ki] = T[ki] - T[ki+1];

    return T*L;

def get_1D_transfer_from_flux(F,L):

    (nk,nz) = F.shape;
    T = F.copy();
    for ki in range(nk-2,-1,-1):
        T[ki,:] = T[ki,:] - T[ki+1,:];

    return F*L;

def get_2D_transfer_from_flux(F,L):

    if (len(F.shape)==3):
        (nk,nx,ny) = F.shape;
        T = F.copy();
        for ki in range(nk-2,-1,-1):
            T[ki,:,:] = T[ki,:,:] - T[ki+1,:,:];
    elif (len(F.shape)==4):
        (nk,nx,ny,nb) = F.shape;
        T = F.copy();
        for ki in range(nk-2,-1,-1):
            T[ki,:,:,:] = T[ki,:,:,:] - T[ki+1,:,:,:];
    else:
        raise RuntimeError('get_2D_transfer_from_flux() only supports 3D or 4D arrays...');
    return T*L;

def get_flux_from_transfer_2D(T,L1,L2):

    if len(T.shape)==3:
        (nm,nomega,nf) = T.shape;
        F = T.copy();
        for mi in range(nm-2,-1,-1):
            F[mi,:,:] = F[mi+1,:,:] + F[mi,:,:];
            
        for oi in range(nomega-2,-1,-1):
            F[:,oi,:] = F[:,oi+1,:] + F[:,oi,:];
            
    elif (len(T.shape)==4):
        (nm,nomega,nf,nb) = T.shape;
        F = T.copy();
        for mi in range(nm-2,-1,-1):
            F[mi,:,:,:] = F[mi+1,:,:,:] + F[mi,:,:,:];
            
        for oi in range(nomega-2,-1,-1):
            F[:,oi,:,:] = F[:,oi+1,:,:] + F[:,oi,:,:];

    else:
        raise RuntimeError('get_flux_from_transfer_2D() only supports 3D or 4D arrays...');

    return F/L1/L2;

def get_flux_from_transfer_2D_forward_backward(T,L1,L2):

    if len(T.shape)==3:
        (nm,nomega,nf) = T.shape;
        F = T.copy();
        for mi in range(nm-2,-1,-1):
            F[mi,:,:] = F[mi+1,:,:] + F[mi,:,:];
            
        for oi in range(1,nomega):
            F[:,oi,:] = F[:,oi-1,:] + F[:,oi,:];
    elif (len(T.shape)==4):
        (nm,nomega,nf,nb) = T.shape;
        F = T.copy();
        for mi in range(nm-2,-1,-1):
            F[mi,:,:,:] = F[mi+1,:,:,:] + F[mi,:,:,:];
            
        for oi in range(1,nomega):
            F[:,oi,:,:] = F[:,oi-1,:,:] + F[:,oi,:,:];
    else:
        raise RuntimeError('get_flux_from_transfer_2D_forward_backward() only supports 3D or 4D arrays...');

    return F/L1/L2;

