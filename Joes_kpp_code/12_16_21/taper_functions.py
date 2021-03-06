import numpy as np
import global_vars as glb
from repmat import repmat
from vert_grid_mitgcm import *

# main taper function
def taper(f):
    # input f: values between 0 and 1
    if (glb.taper_function==1):
        ft = np.sin(f*np.pi)**2;
    elif (glb.taper_function==2):
        ft_left = 0.5*(1-np.cos(2*np.pi*f/glb.tukey_alpha));
        ft_left[f>=(glb.tukey_alpha/2)] = 0;
        ft_right = 0.5*(1-np.cos(2*np.pi*(1-f)/glb.tukey_alpha));
        ft_right[f<=(1-glb.tukey_alpha/2)] = 0;
        ft_middle = 0*f;
        ft_middle[(f>=(glb.tukey_alpha/2))*(f<=(1-glb.tukey_alpha/2))] = 1;
        ft = ft_left + ft_right + ft_middle;
    else:
        raise RuntimeError('taper function ' + str(glb.taper_function) + ' not supported!');

    return ft;

# single-field isotropic taper window
def get_sclfct(pbool):
    sclfct = 1;
    if (glb.taper_function==1):
        sclfct = (8/3)**(np.sum(pbool)/2);
    elif (glb.taper_function==2):
        sclfct = (1-5*glb.tukey_alpha/8)**(-np.sum(pbool)/2);
    else:
        raise RuntimeError('taper function ' + str(glb.taper_function) + ' not supported!');

    return sclfct;


def taper_function_5D(f,pbool=np.array([0,0,0,1,0])):

    (nx,ny,nz,nt,nf) = f.shape;
    sclfct = get_sclfct(pbool);
    
    norm = np.sum(f*f);

    # x filter
    if (pbool[0]):
        taperX = taper((np.linspace(1,nx,nx)-0.5)/nx);
        taperX = np.reshape(taperX,(nx,1,1,1,1));
        taperX = repmat(taperX,(1,ny,nz,nt,nf));
    else:
        taperX = np.ones([nx,ny,nz,nt,nf]);
    
    # y filter
    if (pbool[1]):
        taperY = taper((np.linspace(1,ny,ny)-0.5)/ny);
        taperY = np.reshape(taperY,(1,ny,1,1,1));
        taperY = repmat(taperY,(nx,1,nz,nt,nf));
    else:
        taperY = np.ones([nx,ny,nz,nt,nf]);
    
    # z filter
    if (pbool[2]):
        taperZ = taper((np.linspace(1,nz,nz)-0.5)/nz);
        taperZ = np.reshape(taperZ,(1,1,nz,1,1));
        taperZ = repmat(taperZ,(nx,ny,1,nt,nf));
    else:
        taperZ = np.ones([nx,ny,nz,nt,nf]);

    # t filter
    if (pbool[3]):
        taperT = taper((np.linspace(1,nt,nt)-0.5)/nt);
        taperT = np.reshape(taperT,(1,1,1,nt,1));
        taperT = repmat(taperT,(nx,ny,nz,1,nf));
    else:
        taperT = np.ones([nx,ny,nz,nt,nf]);

    # f filter
    if (pbool[4]):
        taperF = taper((np.linspace(1,nf,nf)-0.5)/nf);
        taperF = np.reshape(taperF,(1,1,1,1,nf));
        taperF = repmat(taperF,(nx,ny,nz,nt,1));
    else:
        taperF = np.ones([nx,ny,nz,nt,nf]);

    # apply filter.  Exact cons. is off (as it should be).
    f[0:nx,0:ny,0:nz,0:nt,0:nf] = f[0:nx,0:ny,0:nz,0:nt,0:nf]*taperX*taperY*taperZ*taperT*taperF;
    f = f*sclfct;

#     print('taper changes field norm by ' + str((np.sum(f*f)/norm)**(0.5)))
    
def taper_filter_3D_uvw_u(u_in,v_in,w_in,pbool=np.array([1,1,0])):

    u = u_in.copy(); v = v_in.copy(); w = w_in.copy();

    (nx,ny,nz,nt) = w.shape;
    sclfct = get_sclfct(pbool);

    # x filter
    if (pbool[0]):
        taperXu = taper((np.linspace(1,nx,nx)-1)/nx);
        taperXu = np.reshape(taperXu,(nx,1,1,1));
        taperXu = repmat(taperXu,(1,ny,nz,nt));
        
        taperXvw = taper((np.linspace(1,nx,nx)-0.5)/nx);
        taperXvw = np.reshape(taperXvw,(nx,1,1,1));
        taperXvw = repmat(taperXvw,(1,ny,nz,nt));
    else:
        taperXu = np.ones([nx,ny,nz,nt]);
        taperXvw = np.ones([nx,ny,nz,nt]);
    
    # y filter
    if (pbool[1]):
        taperYv = taper((np.linspace(1,ny,ny)-1)/ny);
        taperYv = np.reshape(taperYv,(1,ny,1,1));
        taperYv = repmat(taperYv,(nx,1,nz,nt));
        
        taperYuw = taper((np.linspace(1,ny,ny)-0.5)/ny);
        taperYuw = np.reshape(taperYuw,(1,ny,1,1));
        taperYuw = repmat(taperYuw,(nx,1,nz,nt));
    else:
        taperYv = np.ones([nx,ny,nz,nt]);
        taperYuw = np.ones([nx,ny,nz,nt]);
    
    # z filter
    if (pbool[2]):
        taperZw = taper((np.linspace(1,nz,nz)-1)/nz);
        taperZw = np.reshape(taperZw,(1,1,nz,1));
        taperZw = repmat(taperZw,(nx,ny,1,nt));
        
        taperZuv = taper((np.linspace(1,nz,nz)-0.5)/nz);
        taperZuv = np.reshape(taperZuv,(1,1,nz,1));
        taperZuv = repmat(taperZuv,(nx,ny,1,nt));
    else:
        taperZw = np.ones([nx,ny,nz,nt]);
        taperZuv = np.ones([nx,ny,nz,nt]);

    # apply filter.  Exact cons. is off (as it should be).
    u[0:nx,0:ny,0:nz,0:nt] = u[0:nx,0:ny,0:nz,0:nt]*taperXu*taperYuw*taperZuv;
    v[0:nx,0:ny,0:nz,0:nt] = v[0:nx,0:ny,0:nz,0:nt]*taperXvw*taperYv*taperZuv;
    w = w*taperXvw*taperYuw*taperZw;

    u = u*sclfct; v = v*sclfct; w = w*sclfct;
    
    # periodize by carrying over zeros, if necessary.
    if (u.shape[0]>nx):
        u[-1,:,:,:] = u[0,:,:,:];

    if (v.shape[1]>ny):
        v[:,-1,:,:] = v[:,0,:,:];

    return (u,v,w)



def taper_filter_3D_uvw_nu(u_in,v_in,w_in,dxu,dyu,dxv,dyv,dxw,dyw,thknss,pbool=np.array([1,1,0])):

    u = u_in.copy(); v = v_in.copy(); w = w_in.copy();

    ### JS - Make sure that the u/v fields are nx x ny x nz x nt       
    (nx,ny,nz,nt) = w.shape;
    if (u.shape[0]!=v.shape[0]): raise RuntimeError('incompatible sizes!');
    if (dxu.shape[0]!=u.shape[0]): raise RuntimeError('incorrectly sized grid!');
    if (thknss.shape[0]!=u.shape[0]): raise RuntimeError('incorrectly sized thknss!');

    sclfct = (8/3)**(np.sum(pbool)/2);

    Lx = np.amin(np.squeeze(np.sum(dxu,0,keepdims=True))); # should be a scalar
    Ly = np.amin(np.sum(dyv,1,keepdims=True)); # should be a scalar
    Lxs = np.sum(dxu,0,keepdims=True); dLx = (Lxs-repmat(Lx,(1,ny)))/2;
    Lys = np.sum(dyv,1,keepdims=True); dLy = (Lys-repmat(Ly,(nx,1)))/2;

    Xu = np.cumsum(dxu,0)-repmat(dLx,(nx,1))-1.0*repmat(dxu[0:1,:],(nx,1));
    Yu = np.cumsum(dyu,1)-repmat(dLy,(1,ny))-0.5*repmat(dyu[:,0:1],(1,ny));
    Xv = np.cumsum(dxv,0)-repmat(dLx,(nx,1))-0.5*repmat(dxv[0:1,:],(nx,1));
    Yv = np.cumsum(dyv,1)-repmat(dLy,(1,ny))-1.0*repmat(dyv[:,0:1],(1,ny));
    Xw = np.cumsum(dxw,0)-repmat(dLx,(nx,1))-0.5*repmat(dxw[0:1,:],(nx,1));
    Yw = np.cumsum(dyw,1)-repmat(dLy,(1,ny))-0.5*repmat(dyw[:,0:1],(1,ny));

    Xu = Xu/Lx; Xv = Xv/Lx; Xw = Xw/Lx;
    Yu = Yu/Ly; Yv = Yv/Ly; Yw = Yw/Ly;

    Xu[Xu<0] = 0; Xu[Xu>1] = 1;
    Yu[Yu<0] = 0; Yu[Yu>1] = 1;
    Xv[Xv<0] = 0; Xv[Xv>1] = 1;
    Yv[Yv<0] = 0; Yv[Yv>1] = 1;
    Xw[Xw<0] = 0; Xw[Xw>1] = 1;
    Yw[Yw<0] = 0; Yw[Yw>1] = 1;

    # build filters
    if (pbool[0]):
        taperXu = taper(Xu);
        taperXu = repmat(taperXu,(1,1,nz,nt));

        taperXv = taper(Xv);
        taperXv = repmat(taperXv,(1,1,nz,nt));

        taperXw = taper(Xw);
        taperXw = repmat(taperXw,(1,1,nz,nt));
    else:
        taperXu = np.ones((nx,ny,nz,nt));
        taperXv = np.ones((nx,ny,nz,nt));
        taperXw = np.ones((nx,ny,nz,nt));

    if pbool[1]:
        taperYv = taper(Yv);
        taperYv = repmat(taperYv,(1,1,nz,nt));

        taperYu = taper(Yu);
        taperYu = repmat(taperYu,(1,1,nz,nt));

        taperYw = taper(Yw);
        taperYw = repmat(taperYw,(1,1,nz,nt));
    else:
        taperYv = np.ones((nx,ny,nz,nt));
        taperYu = np.ones((nx,ny,nz,nt));
        taperYw = np.ones((nx,ny,nz,nt));

    if pbool[2]:
        depth2d = np.sum(thknss,2,keepdims=True); Lz = np.amax(depth2d);

        (uthknss, vthknss) = get_uv_thknss(thknss);

        Zu = (np.cumsum(uthknss,2) - 0.5*uthknss)/Lz;
        Zv = (np.cumsum(vthknss,2) - 0.5*vthknss)/Lz;
        Zw = (np.cumsum(thknss,2) - thknss)/Lz;

        taperZu = taper(Zu);
        taperZv = taper(Zv);
        taperZw = taper(Zw);
    else:
        taperZu = np.ones((nx,ny,nz,nt));
        taperZv = np.ones((nx,ny,nz,nt));
        taperZw = np.ones((nx,ny,nz,nt));

    # apply filter
    u[0:nx,0:ny,0:nz,0:nt] = u[0:nx,0:ny,0:nz,0:nt]*taperXu*taperYu*taperZu;
    v[0:nx,0:ny,0:nz,0:nt] = v[0:nx,0:ny,0:nz,0:nt]*taperXv*taperYv*taperZv;
    w[0:nx,0:ny,0:nz,0:nt] = w[0:nx,0:ny,0:nz,0:nt]*taperXw*taperYw*taperZw;
    u = u*sclfct; v = v*sclfct; w = w*sclfct;

    return (u,v,w);



def taper_filter_3Dcc_nu(f_in,dxc,dyc,thknss,pbool=np.array([1,1,0])):

    f = f_in.copy();

    (nx,ny,nz,nt) = f.shape;
    if (dxc.shape[0]!=f.shape[0]):        raise RuntimeError('incorrectly sized grid!');
    if (thknss.shape[1]!=f.shape[1]):         raise RuntimeError('incorrectly sized thknss!');

    sclfct = (8/3)**(np.sum(pbool)/2);

    Lx = min(np.squeeze(np.sum(dxc,0,keepdims=True)));
    Ly = min(np.sum(dyc,1,keepdims=True));
    Lxs = np.sum(dxc,0,keepdims=True); dLx = (Lxs-repmat(Lx,(1,ny)))/2;
    Lys = np.sum(dyc,1,keepdims=True); dLy = (Lys-repmat(Ly,(nx,1)))/2;

    Xc = np.cumsum(dxc,0)-repmat(dLx,(nx,1))-0.5*repmat(dxc[0:1,:],(nx,1));
    Yc = np.cumsum(dyc,1)-repmat(dLy,(1,ny))-0.5*repmat(dyc[:,0:1],(1,ny));

    Xc = Xc/Lx; Yc = Yc/Ly;

    Xc[Xc<0] = 0; Xc[Xc>1] = 1;
    Yc[Yc<0] = 0; Yc[Yc>1] = 1;

    # x-filter
    if (pbool[0]):
        taperX = taper(Xc);
        taperX = repmat(taperX,(1,1,nz,nt));
    else:
        taperX = np.ones((nx,ny,nz,nt));

    # y-filter
    if (pbool[1]):
        taperY = taper(Yc);
        taperY = repmat(taperY,(1,1,nz,nt));
    else:
        taperY = np.ones((nx,ny,nz,nt));

    # z-filter
    if (pbool[2]):
        depth2d = np.sum(thknss,2,keepdims=True); Lz = np.amax(depth2d);
        Z = (np.cumsum(thknss,2) - 0.5*thknss)/Lz;
        taperZ = taper(Z);
    else:
        taperZ = np.ones((nx,ny,nz,nt));

    # apply filter.  Exact cons. is off (as it should be).
    return f*taperX*taperY*taperZ*sclfct;

def mask_boundary_3D(u_in,v_in,w_in,nmask=2,dmask=np.array([1,1,0])):

    u = u_in.copy(); v = v_in.copy(); w = w_in.copy();

    (nx,ny,nz,nt) = w.shape;

    # compute mask correction
    if (np.sum(dmask)==0):
        sclfct = 1;
    elif ((dmask[0]==1) and (dmask[1]==0) and (dmask[2]==0)):
        sclfct = nx/(nx - 2*nmask);
    elif ((dmask[0]==0) and (dmask[1]==1) and (dmask[2]==0)):
        sclfct = ny/(ny - 2*nmask);
    elif ((dmask[0]==0) and (dmask[1]==0) and (dmask[2]==1)):
        sclfct = nz/(nz - 2*nmask);
    elif ((dmask[0]==1) and (dmask[1]==1) and (dmask[2]==0)):
        sclfct = nx*ny/(nx*ny - (2*nmask*ny + 2*nmask*nx - 4*nmask*nmask));
    elif ((dmask[0]==1) and (dmask[1]==0) and (dmask[2]==1)):
        sclfct = nx*nz/(nx*nz - (2*nmask*nz + 2*nmask*nx - 4*nmask*nmask));
    elif ((dmask[0]==0) and (dmask[1]==1) and (dmask[2]==1)):
        sclfct = ny*nz/(ny*nz - (2*nmask*nz + 2*nmask*ny - 4*nmask*nmask));
    elif ((dmask[0]==1) and (dmask[1]==1) and (dmask[2]==1)):
        sclfct = nx*ny*nz/(nx*ny*nz - (2*nmask*ny*nz + 2*nmask*nx*nz + 2*nmask*nx*ny 
                                       - 4*nmask*nmask*nx - 4*nmask*nmask*ny - 4*nmask*nmask*nz
                                       + 16*nmask*nmask*nmask));
    else:
        raise RuntimeError('shouldnt be here!');
    
    # x mask
    if (dmask[0]):
        u[:nmask,:,:,:] = 0; u[-nmask:,:,:,:] = 0;
        v[:nmask,:,:,:] = 0; v[-nmask:,:,:,:] = 0;
        w[:nmask,:,:,:] = 0; w[-nmask:,:,:,:] = 0;
    
    # y mask
    if (dmask[1]):
        u[:,:nmask,:,:] = 0; u[:,-nmask:,:,:] = 0;
        v[:,:nmask,:,:] = 0; v[:,-nmask:,:,:] = 0;
        w[:,:nmask,:,:] = 0; w[:,-nmask:,:,:] = 0;
    
    # z mask
    if (dmask[2]):
        u[:,:,:nmask,:] = 0; u[:,:,-nmask:,:] = 0;
        v[:,:,:nmask,:] = 0; v[:,:,-nmask:,:] = 0;
        w[:,:,:nmask,:] = 0; w[:,:,-nmask:,:] = 0;

    u = sclfct*u;
    v = sclfct*v;
    w = sclfct*w;

    return (u,v,w)



