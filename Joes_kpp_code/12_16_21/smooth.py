import numpy as np
from repmat import repmat
from pad_field import pad_field_3D
from strip_nan_inf import strip_nan_inf

def smooth_horiz(f_in,maskC=np.NaN):
    f = f_in.copy();
    (nx,ny,nz,nt) = f.shape; ng = 2;  prd = np.array([1,1,1]);
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits

    if (np.isnan(maskC)).any():
        maskC=np.ones(f.shape);
        print('warning: replacing smoothing mask with ones')
        
    f = pad_field_3D(f,ng,prd);
    maskC = pad_field_3D(maskC,ng,prd);
    
    tmpVar = (0.25*maskC[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:] 
        + 0.125*maskC[gx0+1:gxn+1,gy0:gyn,gz0+1:gzn+1,:] 
        + 0.125*maskC[gx0:gxn,gy0+1:gyn+1,gz0+1:gzn+1,:] 
        + 0.125*maskC[gx0-1:gxn-1,gy0:gyn,gz0+1:gzn+1,:] 
        + 0.125*maskC[gx0:gxn,gy0-1:gyn-1,gz0+1:gzn+1,:] 
        + 0.0625*maskC[gx0+1:gxn+1,gy0+1:gyn+1,gz0+1:gzn+1,:] 
        + 0.0625*maskC[gx0+1:gxn+1,gy0-1:gyn-1,gz0+1:gzn+1,:] 
        + 0.0625*maskC[gx0-1:gxn-1,gy0+1:gyn+1,gz0+1:gzn+1,:] 
        + 0.0625*maskC[gx0-1:gxn-1,gy0-1:gyn-1,gz0+1:gzn+1,:]);
    
    tmpVarBool = tmpVar>0.25;
    tmpVarBool[:,:,-1,:]=0; # exclude bottom layer from smoothing
   
    ftmp = (0.25*f[gx0:gxn,gy0:gyn,gz0:gzn,:]*maskC[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:] 
        + 0.125*f[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]*maskC[gx0+1:gxn+1,gy0:gyn,gz0+1:gzn+1,:] 
        + 0.125*f[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]*maskC[gx0:gxn,gy0+1:gyn+1,gz0+1:gzn+1,:] 
        + 0.125*f[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]*maskC[gx0-1:gxn-1,gy0:gyn,gz0+1:gzn+1,:] 
        + 0.125*f[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]*maskC[gx0:gxn,gy0-1:gyn-1,gz0+1:gzn+1,:] 
        + 0.0625*f[gx0+1:gxn+1,gy0+1:gyn+1,gz0:gzn,:]*maskC[gx0+1:gxn+1,gy0+1:gyn+1,gz0+1:gzn+1,:] 
        + 0.0625*f[gx0+1:gxn+1,gy0-1:gyn-1,gz0:gzn,:]*maskC[gx0+1:gxn+1,gy0-1:gyn-1,gz0+1:gzn+1,:] 
        + 0.0625*f[gx0-1:gxn-1,gy0+1:gyn+1,gz0:gzn,:]*maskC[gx0-1:gxn-1,gy0+1:gyn+1,gz0+1:gzn+1,:] 
        + 0.0625*f[gx0-1:gxn-1,gy0-1:gyn-1,gz0:gzn,:]*maskC[gx0-1:gxn-1,gy0-1:gyn-1,gz0+1:gzn+1,:]);
    
    f = tmpVarBool*ftmp/tmpVar + (np.logical_not(tmpVarBool))*f[gx0:gxn,gy0:gyn,gz0:gzn,:];
#     f[:,:,-1,:] = f_in[:,:,-1,:]; # double check exclude bottom layer from filtering...
    strip_nan_inf(f);

    return f;



def remove_bottom_adjacent_cells(f_in,dz,thknss,itr=1):
    f = f_in.copy();
    (nx,ny,nz,nt) = f.shape; ng = 2;  prd = np.array([1,1,1]);
    gx0 = ng; gxn = nx+ng;    gy0 = ng; gyn = ny+ng;    gz0 = ng; gzn = nz+ng; # interior cell limits
    
    thknss = pad_field_3D(thknss,ng,prd);
    thknssUni = np.amax(np.amax(thknss,axis=0,keepdims=True),axis=1,keepdims=True);
    thknssUni = repmat(thknssUni,(nx+2*ng,ny+2*ng,1,1));
    
#     dpth = pad_field_3D(repmat(sum(thknss(gx0:gxn,gy0+1:gyn+1,gz0:gzn,:),3),1,1,nz,1),ng,prd);
#     z = pad_field_3D(cumsum(thknss(gx0:gxn,gy0+1:gyn+1,gz0:gzn,:),3),ng,prd);
    
    filt = np.equal(thknss,thknssUni);
    for i in range(0,itr):
        filt = (filt[gx0:gxn,gy0:gyn,gz0:gzn,:]
            * filt[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]
            * filt[gx0:gxn,gy0:gyn,gz0+1:gzn+1,:]
            * filt[gx0:gxn,gy0-1:gyn-1,gz0:gzn,:]
            * filt[gx0:gxn,gy0:gyn,gz0-1:gzn-1,:]
            * filt[gx0:gxn,gy0+1:gyn+1,gz0+1:gzn+1,:]
            * filt[gx0:gxn,gy0-1:gyn-1,gz0-1:gzn-1,:]
            * filt[gx0:gxn,gy0+1:gyn+1,gz0-1:gzn-1,:]
            * filt[gx0:gxn,gy0-1:gyn-1,gz0+1:gzn+1,:]
            * filt[gx0+1:gxn+1,gy0:gyn,gz0:gzn,:]
            * filt[gx0+1:gxn+1,gy0+1:gyn+1,gz0:gzn,:]
            * filt[gx0+1:gxn+1,gy0:gyn,gz0+1:gzn+1,:]
            * filt[gx0+1:gxn+1,gy0-1:gyn-1,gz0:gzn,:]
            * filt[gx0+1:gxn+1,gy0:gyn,gz0-1:gzn-1,:]
            * filt[gx0+1:gxn+1,gy0+1:gyn+1,gz0+1:gzn+1,:]
            * filt[gx0+1:gxn+1,gy0-1:gyn-1,gz0-1:gzn-1,:]
            * filt[gx0+1:gxn+1,gy0+1:gyn+1,gz0-1:gzn-1,:]
            * filt[gx0+1:gxn+1,gy0-1:gyn-1,gz0+1:gzn+1,:]
            * filt[gx0-1:gxn-1,gy0:gyn,gz0:gzn,:]
            * filt[gx0-1:gxn-1,gy0+1:gyn+1,gz0:gzn,:]
            * filt[gx0-1:gxn-1,gy0:gyn,gz0+1:gzn+1,:]
            * filt[gx0-1:gxn-1,gy0-1:gyn-1,gz0:gzn,:]
            * filt[gx0-1:gxn-1,gy0:gyn,gz0-1:gzn-1,:]
            * filt[gx0-1:gxn-1,gy0+1:gyn+1,gz0+1:gzn+1,:]
            * filt[gx0-1:gxn-1,gy0-1:gyn-1,gz0-1:gzn-1,:]
            * filt[gx0-1:gxn-1,gy0+1:gyn+1,gz0-1:gzn-1,:]
            * filt[gx0-1:gxn-1,gy0-1:gyn-1,gz0+1:gzn+1,:]);
        filt = pad_field_3D(filt,ng,prd);

    filt = filt[gx0:gxn,gy0:gyn,gz0:gzn,:];
    
    blkdpth = repmat(np.sum(thknss[gx0:gxn,gy0+1:gyn+1,gz0:gzn,:]*filt,axis=2,keepdims=True),(1,1,nz,1));
    Udpth = np.cumsum(dz*np.ones([nx,ny,nz,nt]),axis=2);
    
    Ufilt = (Udpth<blkdpth);
    
    f = f*Ufilt;

    return f;
