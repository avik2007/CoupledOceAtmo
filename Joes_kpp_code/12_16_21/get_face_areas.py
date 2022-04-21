import numpy as np
from pad_field import pad_field_3D

def get_face_areas(thknss,dx,dy):

    thknss = pad_field_3D(thknss);
    Aew = dy*np.minimum(thknss[0:-1,1:-1,1:-1,:],thknss[1:,1:-1,1:-1,:]);
    Ans = dx*np.minimum(thknss[1:-1,0:-1,1:-1,:],thknss[1:-1,1:,1:-1,:]);
    thknss = thknss[1:-1,1:-1,1:-1,:]
    
    return (Aew,Ans);

def get_face_areas_mod_grid(thknss,dx,dy):

    thknss = pad_field_3D(thknss);
    Aew = dy*np.minimum(thknss[0:-1,1:-1,1:-1,:],thknss[1:,1:-1,1:-1,:]);
    Ans = dx*np.minimum(thknss[1:-1,0:-1,1:-1,:],thknss[1:-1,1:,1:-1,:]);
    thknss = thknss[1:-1,1:-1,1:-1,:]
    
    Aew = Aew[:-1,:,:,:]; Ans = Ans[:,:-1,:,:];

    return (Aew,Ans);

