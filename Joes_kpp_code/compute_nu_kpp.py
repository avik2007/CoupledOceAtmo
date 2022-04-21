import gc
import sys
import math
import time
import warnings
import numpy as np
import numpy.fft
from repmat import repmat
import fld_tools as ft
from compute_nu_T import *
import global_vars as glb
from global_vars import ng

# Computes vertical dissipation from the nonuniform flds
    
def compute_nu_kpp_diag(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (dU,dV,_,_,_,_,_,_,KPPviscA,dbloc,_,_) = compute_nu_kpp(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);
    return (dU,dV,KPPviscA,dbloc);

def compute_nu_kpp_sep_diag(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (_,_,dUb,dUc,dUs,dVb,dVc,dVs,_,_,_,_) = compute_nu_kpp(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);

    return (dUb,dVb,dUc,dVc,dUs,dVs);

def compute_nu_kpp_sep_diag_u(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (_,_,dUb,dUc,dUs,dVb,dVc,dVs,_,_,_,_) = compute_nu_kpp(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);

    nz = U.shape[2];
    nzu = zoversamp*nz;
    thknssUni = ft.get_thknssUni(thknss);
    (uthknss, vthknss) = ft.get_uv_thknss(thknss);

    (dUb,_) = ft.get_4D_vert_uniform_field(dUb,thknssUni,nzu,ns=True);
    (dVb,_) = ft.get_4D_vert_uniform_field(dVb,thknssUni,nzu,ns=True);
    (dUc,_) = ft.get_4D_vert_uniform_field(dUc,thknssUni,nzu,ns=True);
    (dVc,_) = ft.get_4D_vert_uniform_field(dVc,thknssUni,nzu,ns=True);
    (dUs,_) = ft.get_4D_vert_uniform_field(dUs,thknssUni,nzu,ns=True);
    (dVs,_) = ft.get_4D_vert_uniform_field(dVs,thknssUni,nzu,ns=True);

    return (dUb,dVb,dUc,dVc,dUs,dVs);

def compute_nu_kpp_benchmark(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    (dU,dV,_,_,_,_,_,_,_,_,_,_) = compute_nu_kpp(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);
    t1 = time.time() 
    (_,_) = compute_nu_T_k(dU,dV,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,0,zoversamp,prd);

def compute_nu_kpp_kappa(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (_,_,_,_,_,_,_,_,_,_,KappaRU,KappaRV) = compute_nu_kpp(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,0,zoversamp,prd);
    return (KappaRU,KappaRV);

def compute_nu_kpp_k(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (dU,dV,_,_,_,_,_,_,_,_,_,_) = compute_nu_kpp(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);
    (T,k_out) = compute_nu_T_k(dU,dV,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,0,zoversamp,prd);
    return (T,k_out);

def compute_nu_kpp_m(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (dU,dV,_,_,_,_,_,_,_,_,_,_) = compute_nu_kpp(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);
    (T,m_out) = compute_nu_T_m(dU,dV,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,0,zoversamp,prd);
    return (T,m_out);

def compute_nu_kpp_k_z(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (dU,dV,_,_,_,_,_,_,_,_,_,_) = compute_nu_kpp(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);
    (T,k_out) = compute_nu_T_k_z(dU,dV,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,0,zoversamp,prd);
    return (T,k_out);

def compute_nu_kpp_m_xy(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (dU,dV,_,_,_,_,_,_,_,_,_,_) = compute_nu_kpp(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);
    (T,m_out) = compute_nu_T_m_xy(dU,dV,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,0,zoversamp,prd);
    return (T,m_out);

def compute_nu_kpp_sep_k(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (_,_,dUb,dUc,dUs,dVb,dVc,dVs,_,_,_,_) = compute_nu_kpp(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);

    (Ts,_) = compute_nu_T_k(dUs,dVs,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,0,zoversamp,prd);
    (Tb,k_out) = compute_nu_T_k(dUb,dVb,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,0,zoversamp,prd);
    (Tc,_) = compute_nu_T_k(dUc,dVc,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,0,zoversamp,prd);
    return (Tb,Tc,Ts,k_out);

def compute_nu_kpp_sep_m(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):
    (_,_,dUb,dUc,dUs,dVb,dVc,dVs,_,_,_,_) = compute_nu_kpp(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);

    (Ts,_) = compute_nu_T_m(dUs,dVs,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,0,zoversamp,prd);
    (Tb,m_out) = compute_nu_T_m(dUb,dVb,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,0,zoversamp,prd);
    (Tc,_) = compute_nu_T_m(dUc,dVc,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,0,zoversamp,prd);
    return (Tb,Tc,Ts,m_out);

def compute_nu_kpp_sep_k_z(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    (_,_,dUb,dUc,dUs,dVb,dVc,dVs,_,_,_,_) = compute_nu_kpp(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);

    (Ts,_) = compute_nu_T_k_z(dUs,dVs,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,0,zoversamp,prd);
    (Tb,k_out) = compute_nu_T_k_z(dUb,dVb,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,0,zoversamp,prd);
    (Tc,_) = compute_nu_T_k_z(dUc,dVc,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,0,zoversamp,prd);
    return (Tb,Tc,Ts,k_out);

def compute_nu_kpp_sep_m_xy(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    (_,_,dUb,dUc,dUs,dVb,dVc,dVs,_,_,_,_) = compute_nu_kpp(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper,trim_ml,zoversamp,prd);

    (Ts,_) = compute_nu_T_m_xy(dUs,dVs,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,0,zoversamp,prd);
    (Tb,m_out) = compute_nu_T_m_xy(dUb,dVb,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,0,zoversamp,prd);
    (Tc,_) = compute_nu_T_m_xy(dUc,dVc,U,V,thknss,dxu,dyu,dxv,dyv,dxc,dyc,ugrid,vgrid,post_taper,0,zoversamp,prd);
    return (Tb,Tc,Ts,m_out);






def compute_nu_kpp(U,V,rho,pot_rho_down,thknss,dxc,dyc,dxu,dyu,dxv,dyv,dxq,dyq,ugrid,vgrid,pgrid,post_taper=np.array([1,1,0]),
                     trim_ml=0,zoversamp=1,prd=np.array([1,1,1])):

    print ('in compute_nu_kpp()')
    print('viscArNr = ' + str(glb.viscArNr)) 
    print('Riinfty = ' + str(glb.Riinfty)) 

    # Initialize Constants
    (nx, ny, nz, nt) = U.shape;
    
    # if (post_taper[2]):
    #     raise RuntimeError('post_taper not built to handle vertical filt. of dKEdiv');
    
    # Add ghost cells to fields to handle periodicity in position space.
    ng = 2; # number of ghost cells, must be one larger than needed for python indexing
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits
    dzgrid = ft.get_dzgrid(thknss);
    U = ft.pad_field_3D(U,ng,prd); V = ft.pad_field_3D(V,ng,prd);  thknss = ft.pad_field_3D(thknss,ng,prd); 
    rho = ft.pad_field_3D(rho,ng,prd); pot_rho_down = ft.pad_field_3D(pot_rho_down,ng,prd); 
    dxc = ft.pad_field_2D(dxc,ng); dxu = ft.pad_field_2D(dxu,ng); dxv = ft.pad_field_2D(dxv,ng); dxq = ft.pad_field_2D(dxq,ng);
    dyc = ft.pad_field_2D(dyc,ng); dyu = ft.pad_field_2D(dyu,ng); dyv = ft.pad_field_2D(dyv,ng); dyq = ft.pad_field_2D(dyq,ng); 

    warnings.warn("thknss needs adapting for HYCOM, etc.");
    hfacC = ft.get_hfac(thknss);
    (hfacW, hfacS) = ft.get_uv_hfac(thknss);
    maskW = ft.get_mask_from_hfac(hfacW); maskS = ft.get_mask_from_hfac(hfacS);
    del hfacW, hfacS;
    
    nzmax = ft.calc_nzmax(hfacC[gx0:gxn,gy0:gyn,gz0:gzn,:]); 
    nzmax = repmat(nzmax,(1,1,nz,1));

    # find local buoyancy gradient
    dbloc = glb.g*(rho[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:] - pot_rho_down[gx0:gxn,gy0:gyn,gz0:gzn,:])/(rho[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:]);
    dbloc[:,:,-1,:] = 0; #what's this for?         

    # compute diffus2 using unsmoothed buoyancy gradianet
    diffus2 = dbloc/dzgrid[:,:,1:,:]; # surf layer is just 0
    diffus2[nzmax<=0] = 0.0; #what's this mean?
    ft.strip_nan_inf(diffus2);
    
    # apply smoothing to dbloc
    dblocSm = ft.smooth_horiz(dbloc,ft.get_mask_from_hfac(hfacC[gx0:gxn,gy0:gyn,gz0:gzn,:]));
 	#what's the smoothing here?

    # find the local velocity shear.
    shsq = ft.get_shsq(U,V);

    # compute diffusivity
    diffus = dblocSm*dzgrid[:,:,1:,:]/np.maximum(shsq[gx0:gxn,gy0:gyn,gz0:gzn,:],1e-10);
    diffus[nzmax<=0] = 0.0;
    
    Ks = repmat(np.reshape(np.linspace(0,nz-1,nz),(1,1,nz,1)),(nx,ny,1,nt));
    Kfilt = (Ks>=nzmax)*(nzmax>=2);
	#What's going on here?
    for k in range(1,nz):
        diffus[:,:,k,:] = (1-Kfilt[:,:,k,:])*diffus[:,:,k,:] + Kfilt[:,:,k,:]*diffus[:,:,k-1,:];
        diffus2[:,:,k,:] = (1-Kfilt[:,:,k,:])*diffus2[:,:,k,:] + Kfilt[:,:,k,:]*diffus2[:,:,k-1,:];
    del Kfilt, Ks;

    Rig   = np.maximum(diffus2, glb.BVSQcon);
    ratio = np.minimum((glb.BVSQcon-Rig)/glb.BVSQcon, 1.0);
    fcon  = 1.0 - ratio*ratio;
    fcon  = fcon*fcon*fcon;

    #  evaluate f of smooth Ri for shear instability, store in fRi

    Rig  = np.maximum(diffus, 0.0);
    ratio = np.minimum(Rig/glb.Riinfty, 1.0);
    fRi   = 1.0 - ratio*ratio;
    fRi   = fRi*fRi*fRi;

    del dblocSm, shsq, diffus, diffus2;
     .collect();

    # diffus = viscArNr + fcon*difmcon + fRi*difm0;
    KPPviscA = np.zeros((nx,ny,nz,nt))
    maskC = ft.get_mask_from_hfac(hfacC[gx0:gxn,gy0:gyn,gz0:gzn,:]);
    maskC = np.maximum(np.minimum(ft.pad_field_3D(maskC,ng=2,pbool=np.array([1,1,0])),1),0); 
    thknssUni = ft.get_thknssUni(thknss);
    (thknssW, thknssS) = ft.get_uv_thknss(thknss);
    for icase in range(0,3):
        if (icase==0):
            diffus = glb.viscArNr + np.zeros((nx,ny,nz,nt));
        elif (icase==1):
            diffus = fcon*glb.difmcon;
        elif (icase==2):
            diffus = fRi*glb.difm0;
        else:
            raise RuntimeError('should never get here!!');

        Ks = repmat(np.reshape(np.linspace(0,nz-1,nz),(1,1,nz,1)),(nx,ny,1,nt));
        diffus[Ks>=nzmax] = 0.0;
        del Ks;

        diffus = ft.pad_field_3D(diffus,ng,prd); diffus[:,:,0,:] = 0*diffus[:,:,0,:]; # handled earlier in the code as a 0 index.
        KPPviscAz = diffus[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]*maskC[gx0:gxn,gy0:gyn,gz0:gzn,:]*maskC[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:];

        KPPviscA = KPPviscA + KPPviscAz;
        KPPviscAz = ft.pad_field_3D(KPPviscAz,ng=ng);
        KappaRU = np.maximum(0,0.5*maskW[gx0:gxn,gy0:gyn,gz0:gzn,:]*(KPPviscAz[gx0:gxn,gy0:gyn,gz0:gzn,:]+KPPviscAz[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]));
        KappaRV = np.maximum(0,0.5*maskS[gx0:gxn,gy0:gyn,gz0:gzn,:]*(KPPviscAz[gx0:gxn,gy0:gyn,gz0:gzn,:]+KPPviscAz[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]));

        # trim the mixed layer in the vertical viscosity, not the tendency
        KappaRU = ft.trim_ml(KappaRU,thknss[gx0:gxn,gy0:gyn,gz0:gzn,:],trim_ml)
        KappaRV = ft.trim_ml(KappaRV,thknss[gx0:gxn,gy0:gyn,gz0:gzn,:],trim_ml)
        
        rViscFluxU = KappaRU*(repmat(np.reshape(dxc[gx0:gxn,gy0:gyn]*dyq[gx0:gxn,gy0:gyn],(nx,ny,1,1)),(1,1,nz,nt))*
                          maskW[gx0:gxn,gy0:gyn,gz0:gzn,:]*maskW[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]*
                          (U[gx0:gxn,gy0:gyn,gz0:gzn,:]-U[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:])/dzgrid[:,:,0:-1,:]);
        rViscFluxU[:,:,0,:] = 0*rViscFluxU[:,:,0,:]; rViscFluxU[:,:,-1,:] = 0*rViscFluxU[:,:,-1,:];
    
        rViscFluxV = KappaRV*(repmat(np.reshape(dxq[gx0:gxn,gy0:gyn]*dyc[gx0:gxn,gy0:gyn],(nx,ny,1,1)),(1,1,nz,nt))*
                          maskS[gx0:gxn,gy0:gyn,gz0:gzn,:]*maskS[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]*
                          (V[gx0:gxn,gy0:gyn,gz0:gzn,:]-V[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:])/dzgrid[:,:,0:-1,:]);
        rViscFluxV[:,:,0,:] = 0*rViscFluxV[:,:,0,:]; rViscFluxV[:,:,-1,:] = 0*rViscFluxV[:,:,-1,:];
    
        ft.strip_nan_inf(rViscFluxU);
        ft.strip_nan_inf(rViscFluxV);
    
        rViscFluxU = ft.pad_field_3D(rViscFluxU,ng=ng); rViscFluxV = ft.pad_field_3D(rViscFluxV,ng=ng);
        dU = ((rViscFluxU[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:]-rViscFluxU[gx0:gxn,gy0:gyn,gz0:gzn,:])
              /(thknssW[gx0:gxn,gy0:gyn,gz0:gzn,:]*repmat(dxc[gx0:gxn,gy0:gyn]*dyq[gx0:gxn,gy0:gyn],(1,1,nz,nt))));
        dV = ((rViscFluxV[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:]-rViscFluxV[gx0:gxn,gy0:gyn,gz0:gzn,:])
              /(thknssS[gx0:gxn,gy0:gyn,gz0:gzn,:]*repmat(dxq[gx0:gxn,gy0:gyn]*dyc[gx0:gxn,gy0:gyn],(1,1,nz,nt))));
    
        ft.strip_nan_inf(dU);
        ft.strip_nan_inf(dV);
        
        if (icase==0):
            dUb = dU; dVb = dV;
        elif (icase==1):
            dUc = dU; dVc = dV;
        elif (icase==2): 
            dUs = dU; dVs = dV;
        else:
            raise RuntimeError('should never get here!!');

    return (dUb+dUc+dUs,dVb+dVc+dVs,dUb,dUc,dUs,dVb,dVc,dVs,KPPviscA,dbloc,KappaRU,KappaRV)





