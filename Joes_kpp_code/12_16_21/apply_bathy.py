import numpy as np

def apply_bathy_mitgcm(thknss_in,bathy,hFacMinSz=0.3):

    thknss = thknss_in.copy();

    # Note, hFacMinDr = 0...

    # Get Dims
    (nx,ny,nz,nt) = thknss.shape;
    
    depth = np.cumsum(thknss[:,:,:,:],axis=2);
    for k in range(0,nz):
        hFacC = 1+np.maximum(np.minimum((np.reshape(bathy,(nx,ny,1)) - depth[:,:,k,:])/thknss[:,:,k,:],0),-1);
        hFacC[hFacC<(0.5*hFacMinSz)] = 0.0;
        hFacC[(hFacC>=(0.5*hFacMinSz))*(hFacC<hFacMinSz)] = hFacMinSz;
        thknss[:,:,k,:] = hFacC*thknss[:,:,k,:];

    return thknss;
       
    # thknss = repmat(bathy,1,1,nz)-cumsum(thknss,3);
    # thknss(thknss<0)=0;
    # thknss = repmat(bathy,1,1,nz)-thknss;
    # top_layer = thknss(:,:,1,:);
    # thknss = diff(thknss,1,3);
    # thknss = cat(3,top_layer,thknss);

