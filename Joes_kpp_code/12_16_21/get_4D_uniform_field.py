import sys
import time
import warnings
import numpy as np
from repmat import repmat
from get_bot import get_bot_Zu
from get_bot import get_bot_Znu
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from ocean_interp import *

def get_4D_uniform_field(f_in,grid,thknss):

    f = f_in.copy();
    
    interp = 'cubic';
    # interp = 'pchip'; # do not use - stick to linear interpolation...

    # Organize Grids
    (nx,ny,nz,nt) = f.shape;
    X = repmat(grid[:,:,0],(1,1,nz)); Y = repmat(grid[:,:,1],(1,1,nz));
    Lx = X[-1,0,0]-X[0,0,0];  Ly = Y[0,-1,0]-Y[0,0,0];
    Z = np.cumsum(thknss,axis=2) - 0.5*thknss; # data should be at layer center.
    depth2d = np.sum(thknss,axis=2,keepdims=True); zBot = np.amax(depth2d); zSurf = 0; # zSurf = mean(mean(mean(thknss(:,:,1,:),1),2),4)/2;
    botZ = get_bot_Znu(thknss); xStart = np.amax(X[0,:,0]); xEnd = np.amin(X[-1,:,0]);
    dxu = Lx/(nx-1); dyu = Ly/(ny-1); dzu = (zBot-zSurf)/(nz); # in Z, due to high nonuniformity, inter. to new cell interiors

    # Interpolate to a Uniform Grid
    Xq = np.linspace(xStart,xEnd,nx); Yq = np.linspace(Y[0,0,0],Y[0,-1,0],ny); Zq = np.linspace(zSurf+dzu/2,zBot-dzu/2,nz);
    Xq = np.reshape(Xq,(nx,1,1,1)); Yq = np.reshape(Yq,(1,ny,1,1)); Zq = np.reshape(Zq,(1,1,nz,1));
    Xq = repmat(Xq,(1,ny,nz,1)); Yq = repmat(Yq,(nx,1,nz,1)); Zq = repmat(Zq,(nx,ny,1,1));
    botZq = get_bot_Zu(Zq,depth2d);
    
    t1 = time.time() # DEBUG

    # Handle flattened Layers in Z
    thr = 1e-6*np.nanmax(thknss);
    incr = 1e-3;
    for i in range(0,nx):
        for j in range(0,ny):
            for k in range(0,nz-1):
                for t in range(0,nt):
                    if (thknss[i,j,k,t]<thr):
                        if (k<botZ[i,j,0,t]):
                            Z[i,j,k,t] = Z[i,j,k,t] + incr*k;
                            f[i,j,k,t] = np.nan;
                        else:
                            f[i,j,k,t] = 0; # interior cells

    f[np.isinf(f)] = np.NaN;  # zero thickness and spurrious field are skipped.

    # interpolate Z
    for i in range(0,nx):
        for j in range(0,ny):
            for t in range(0,nt):
                # if (np.squeeze(Z[i,j,0:botZ[i,j,0,t],t]).size==1):
                if (botZ[i,j,0,t]>2):
                    # intrp = interp1d(np.squeeze(Z[i,j,0:botZ[i,j,0,t],t]),np.squeeze(f[i,j,0:botZ[i,j,0,t],t]),interp,fill_value="extrapolate");
                    intrp = CubicSpline(np.squeeze(Z[i,j,0:botZ[i,j,0,t],t]),np.squeeze(f[i,j,0:botZ[i,j,0,t],t]),axis=0,bc_type="natural");
                    f[i,j,0:botZq[i,j,0,t],t] = intrp(np.squeeze(Zq[i,j,0:botZq[i,j,0,t],0]));
                    f[i,j,botZq[i,j,0,t]:,t] = 0*f[i,j,botZq[i,j,0,t]:,t]; # interior cells below the diagnosed bottom at botZq.
                    # f[i,j,0:botZq[i,j,0,t],t] = np.reshape(intrp(np.squeeze(Zq[i,j,0:botZq[i,j,0,t],0])),(1,1,botZq[i,j,0,t],1));
                else:
                    f[i,j,:,t] = 0;

    # interpolate X
    for j in range(0,ny):
        for k in range(0,nz):
            for t in range(0,nt):
                if np.sum(np.isnan(f[:,j,k,t]))<(nx-1):
                    intrp = interp1d(X[:,j,k],f[:,j,k,t],interp);
                    # f[:,j,k,t] = np.reshape(intrp(Xq[:,j,k]),(nx,));
                    f[:,j,k,t] = np.squeeze(intrp(Xq[:,j,k]));
                else:
                    f[:,j,k,t] = np.nan*f[:,j,k,t]; 

    # interpolate Y
    for i in range(0,nx):
        for k in range(0,nz):
            for t in range(0,nt):
                if sum(np.isnan(f[i,:,k,t]))<(ny-1):
                    intrp = interp1d(Y[i,:,k],f[i,:,k,t],interp);
                    # f[i,:,k,t] = np.reshape(intrp(Yq[i,:,k]),(ny,));
                    f[i,:,k,t] = np.squeeze(intrp(Yq[i,:,k]));
                else:
                    f[i,:,k,t] = np.nan*f[i,:,k,t];

    print('full interp field time = ' + str(time.time() - t1)) # DEBUG
                    
    # After Interpolation, All NaNs are returned to Zero for the FFTN.
    tmp = f[:,:,0:nz-1,:];
    if np.isnan(tmp).any(): # these should be in the bottom layers.
        warnings.warn("nans above bottom boundary in f");

    f[np.isnan(f)] = 0; # JS: does this work in python? 
    
    return (f,dxu,dyu,dzu)



def get_4D_vert_uniform_field(f_in,thknss,nzu=0,ns=False):

    # interpolates to a vert. unifrom grid from a vert. nonuniform grid that is variable in time.

#     ns = False; print('DEBUG Mode for no-slip interpolation'); # DEBUG
    f = f_in.copy();

    # Organize Grids
    (nx,ny,nz,nt) = f.shape;
    Z = np.cumsum(thknss,axis=2) - 0.5*thknss; # data should be at layer center.
    depth2d = np.sum(thknss,axis=2,keepdims=True); zBot = np.amax(depth2d); zSurf = 0; # zSurf = np.mean(np.mean(np.mean(thknss(:,:,0,:),axis=0),axis=1),axis=3)/2;
    botZ = get_bot_Znu(thknss);

    if (nzu == 0):
        nzu = 1*nz;

    dzu = (zBot-zSurf)/(nzu); # in Z, due to high nonuniformity, interpolate to new cell interiors
 
    # Interpolate to a Uniform Grid
    Zq = np.reshape(np.linspace(zSurf+dzu/2,zBot-dzu/2,nzu),(1,1,nzu,1));
    Zq = repmat(Zq,(nx,ny,1,nt));
    botZq = get_bot_Zu(Zq,depth2d);
    
    # Handle flattened Layers in Z
    thr = 1e-6*np.nanmax(thknss);
    incr = 1e-3;
    
    Ks = repmat(np.reshape(np.linspace(0,nz-1,nz),(1,1,nz,1)),(nx,ny,1,nt));
    botZ3D = repmat(botZ,(1,1,nz,1));
    TKfilt = (Ks<botZ3D)*(thknss<thr);
    TnKfilt = (Ks>=botZ3D)*(thknss<thr);
    Z = Z + TKfilt*incr*Ks;
    f[TKfilt] = np.NaN;
    f[TnKfilt] = 0.0;
    f[np.isinf(f)] = np.NaN;  # zero thickness and spurrious field are skipped.
    del Ks, TKfilt, TnKfilt, botZ3D;

    t1 = time.time() 

    # tack on a buffer layer to handle possible no-slip condition
    Z = np.concatenate((Z,Z[:,:,-1:,:]),axis=2);
    f = np.concatenate((f,0*f[:,:,-1:,:]),axis=2);

    # interpolate Z
    fu = np.zeros([nx,ny,nzu,nt]);
    for i in range(0,nx):
        for j in range(0,ny):
            kbot = botZ[i,j,0,0];
            kbotq = botZq[i,j,0,0];
            if (ns):
                Z[i,j,kbot+1,:] = depth2d[i,j,0,:];
                f[i,j,kbot+1,:] = 0;
                kbot += 1;

            if (nt==1):
                fu[i,j,:,0] = ocean_interpZ(Z[i,j,:,0],f[i,j,:,0],Zq[i,j,:,0],kbot,kbotq);
            else:
                fu[i,j,:,:] = ocean_interpZ_from_fixed(Z[i,j,:,0],f[i,j,:,:],Zq[i,j,:,0],kbot,kbotq);

            # if (i==39) and (j==39):
            #     print(fu[i,j,:,0]);


    print('interp field time = ' + str(time.time() - t1)) # DEBUG

    del f;

    # After Interpolation, All NaNs are returned to Zero for the FFTN.
    tmp = fu[:,:,0:nz-1,:];
    if np.isnan(tmp).any(): # these should be in the bottom layers.
        warnings.warn("nans above bottom boundary in f");

    fu[np.isnan(fu)] = 0;

    return (fu,dzu)



def get_4D_vert_uniform_field_flx(f,thknss,nzu=0,ns=False):

    # converts to an integrated spatial flux from the surface then
    # interpolates to a vert. unifrom grid from a vert. nonuniform grid that is variable in time.
    # then interpolates back to a velocity field.

    if (ns):
        warnings.warn("flx interpolation does not yet utilize no-slip bottom");

    if (nzu == 0):
        nzu = 1*nz;

    frc = 0.01;

    # Organize Grids
    (nx,ny,nz,nt) = f.shape;
    Z = np.cumsum(thknss,axis=2) - 0.5*thknss; # data should be at layer center.
    depth2d = np.sum(thknss,axis=2,keepdims=True); zBot = np.amax(depth2d); zSurf = 0; # zSurf = np.mean(np.mean(np.mean(thknss(:,:,0,:),axis=0),axis=1),axis=3)/2;
    botZ = get_bot_Znu(thknss);

    # Compute nonuniform flux
    flx = np.zeros((nx,ny,nz+1,nt));
    flx[:,:,0,:] = 0.5*f[:,:,0,:]*thknss[:,:,0,:];
    flx[:,:,1:-2,:] = 0.5*f[:,:,0:-2,:]*thknss[:,:,0:-2,:] + 0.5*f[:,:,1:-1,:]*thknss[:,:,1:-1,:];
    flx[:,:,-2,:] = 0.5*f[:,:,-2,:]*thknss[:,:,-2,:] + 0.5*f[:,:,-1,:]*(1-frc)*thknss[:,:,-1,:]; # bump up the layer a bit
    flx[:,:,-1,:] = 0.5*f[:,:,-1,:]*thknss[:,:,-1,:];

    (flxu,dzu) = get_4D_vert_uniform_field(flx,np.concatenate((thknss[:,:,:-1,:],(1-frc)*thknss[:,:,-1:,:],frc*thknss[:,:,-1:,:]),axis=2),nzu,ns=False);

    del flx;

    # Compute uniform field from flux
    fu = np.zeros((nx,ny,nz,nt));
    fu[:,:,0,:] = 2*flxu[:,:,0,:]/dzu;
    for k in range(1,nzu):
        fu[:,:,k,:] = 2*(flxu[:,:,k,:] - 0.5*fu[:,:,k-1,:]*dzu)/dzu;

    # Set interior field to zero
    depth3d = repmat(depth2d,(1,1,nzu,1));
    Zq = np.reshape(np.linspace(zSurf+dzu/2,zBot-dzu/2,nzu),(1,1,nzu,1));
    Zq = repmat(Zq,(nx,ny,1,nt));
    fu[Zq>depth3d] = 0;
    
    del depth3d,Zq,flxu;

    return (fu,dzu)
















