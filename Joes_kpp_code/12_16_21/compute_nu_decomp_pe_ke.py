import sys
import gc
import warnings
import math
import numpy as np
import numpy.fft
from repmat import repmat
import fld_tools as ft
from compute_nu_T import *
import global_vars as glb
from global_vars import ng

# Computes horizontal spectral energy flux, assuming nonunform regular coordinates with a finite volume formulation
# a tranfer-function approach is used that assumes energy is conserved, even though it is not, strictly speaking.
    
def compute_nu_decomp_pe_ke_diag(U,V,Eta,rho,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([0,0,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (dU,dV,phiHyd) = compute_nu_decomp_pe_ke(U,V,Eta,rho,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,
                                       ugrid,vgrid,pgrid,np.array([0,0,0]),trim_ml,zoversamp,prd);
    return (dU,dV,phiHyd);

def compute_nu_decomp_pe_ke_diag_u(U,V,Eta,rho,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([0,0,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (dU,dV,_) = compute_nu_decomp_pe_ke(U,V,Eta,rho,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,
                                       ugrid,vgrid,pgrid,np.array([0,0,0]),trim_ml,zoversamp,prd);

    nz = U.shape[2];
    nzu = zoversamp*nz;
    thknssUni = ft.get_thknssUni(thknss);
    (uthknss, vthknss) = ft.get_uv_thknss(thknss);

    (dU,_) = ft.get_4D_vert_uniform_field(dU,thknssUni,nzu,ns=True);
    (dV,_) = ft.get_4D_vert_uniform_field(dV,thknssUni,nzu,ns=True);

    return (dU,dV);

def compute_nu_decomp_pe_ke_k(U,V,Eta,rho,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (dU,dV,_) = compute_nu_decomp_pe_ke(U,V,Eta,rho,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,
                                                    ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);
    (T,k_out) = compute_nu_T_k(dU,dV,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    return (T,k_out);

def compute_nu_decomp_pe_ke_m(U,V,Ub,Vb,Uh,Vh,Eta,rho,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (dU,dV,_) = compute_nu_decomp_pe_ke(U,V,Eta,rho,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,
                                                    ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);

    nz = thknss.shape[2];
    mti_max = math.floor(nz/2-1); 
    T = np.zeros((mti_max,3));

    (Tl,m_out) = compute_nu_T_m(dU,dV,U-Ub-Uh,V-Vb-Vh,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    (Tb,m_out) = compute_nu_T_m(dU,dV,Ub,Vb,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);
    (Th,m_out) = compute_nu_T_m(dU,dV,Uh,Vh,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,trim_ml,zoversamp,prd);

    T[:,0:1] = Tl; T[:,1:2] = Tb; T[:,2:3] = Th;

    return (T,m_out);

def compute_nu_decomp_pe_ke(U,V,Eta,rho,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    print ('in compute_nu_pe_ke()')

    # Initialize Constants
    (nx, ny, nz, nt) = U.shape;
    g = glb.g;
    
#     if (post_taper[2]):
#         raise RuntimeError('post_taper not built to handle vertical filt. of dKEdiv');

    thknssC = ft.get_thknssC(thknss);

    # Add ghost cells to fields to handle periodicity in position space.
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits
    U = ft.pad_field_3D(U,ng,prd); V = ft.pad_field_3D(V,ng,prd);  
#    thknss = ft.pad_field_3D(thknss,ng,prd); thknssC = ft.pad_field_3D(thknssC,ng,prd); rho = ft.pad_field_3D(rho,ng,prd); 
    dxc = ft.pad_field_2D(dxc,ng); dxu = ft.pad_field_2D(dxu,ng); dxv = ft.pad_field_2D(dxv,ng); dxq = ft.pad_field_2D(dxq,ng);
    dyc = ft.pad_field_2D(dyc,ng); dyu = ft.pad_field_2D(dyu,ng); dyv = ft.pad_field_2D(dyv,ng); dyq = ft.pad_field_2D(dyq,ng); 

#     (uthknss, vthknss) = ft.get_uv_f(thknss); uthknss = uthknss[0:-1,:,:,:]; vthknss = vthknss[:,0:-1,:,:];

    # initialize hydrostatic pressures
    phiHydF = np.zeros((nx,ny,nz,nt)); phiHydC = np.zeros((nx,ny,nz,nt));
    
    # compute phi_hydrostatic
    dRlocM = thknssC[:,:,0:1,:];
    dRlocP = 0.5*thknssC[:,:,1:2,:];

    phiHydC[:,:,0:1,:] = (rho[:,:,0:1,:]-glb.rho0)*dRlocM[:,:,:,:]*g/glb.rho0 + g*np.reshape(Eta,(nx,ny,1,1));
    phiHydF[:,:,0:1,:] = phiHydC[:,:,0:1,:] + (rho[:,:,0:1,:]-glb.rho0)*dRlocP[:,:,:,:]*g/glb.rho0;

    for k in range(1,nz):

        dRlocM = 0.5*thknssC[:,:,k:k+1,:];
        if (k==(nz-1)):
            dRlocP = thknssC[:,:,k+1:k+2,:];
        else: 
            dRlocP = 0.5*thknssC[:,:,k+1:k+2,:];

        phiHydC[:,:,k:k+1,:] = phiHydF[:,:,k-1:k,:] + (rho[:,:,k:k+1,:]-glb.rho0)*dRlocM[:,:,:,:]*g/glb.rho0;
        phiHydF[:,:,k:k+1,:] = phiHydC[:,:,k:k+1,:] + (rho[:,:,k:k+1,:]-glb.rho0)*dRlocP[:,:,:,:]*g/glb.rho0;

    thknss = ft.pad_field_3D(thknss,ng,prd); thknssC = ft.pad_field_3D(thknssC,ng,prd); rho = ft.pad_field_3D(rho,ng,prd); 
    phiHydC = ft.pad_field_3D(phiHydC,ng,prd); phiHydF = ft.pad_field_3D(phiHydF,ng,prd);

    # initialize hydrostatic pressure gradients
    dPhiHydX = np.zeros((nx,ny,nz,nt)); dPhiHydY = np.zeros((nx,ny,nz,nt));

    for k in range(gz0,gzn):

        # compute grad_phi_hydrostatic
        dPhiHydX[:,:,k-ng:k-ng+1,:] = (phiHydC[gx0:gxn,gy0:gyn,k:k+1,:]-phiHydC[gx0-1:gxn-1,gy0:gyn,k:k+1,:])/(
            repmat(dxc[gx0:gxn,gy0:gyn],(1,1,1,nt))*rho[gx0:gxn,gy0:gyn,k:k+1,:])
        dPhiHydY[:,:,k-ng:k-ng+1,:] = (phiHydC[gx0:gxn,gy0:gyn,k:k+1,:]-phiHydC[gx0:gxn,gy0-1:gyn-1,k:k+1,:])/(
            repmat(dyc[gx0:gxn,gy0:gyn],(1,1,1,nt))*rho[gx0:gxn,gy0:gyn,k:k+1,:])

    return (-dPhiHydX,-dPhiHydY,phiHydC[gx0:gxn,gy0:gyn,gz0:gzn,:])


    
