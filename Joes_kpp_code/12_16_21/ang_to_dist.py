import sys
import numpy as np

def ang_to_dist(a_in):
    a_work = a_in.copy()
    Rearth = 6370000;
    mLon = np.mean(a_work[:,0,0]);
    a_work[:,:,0] = a_work[:,:,0]-mLon;
    a_work[:,:,0] = a_work[:,:,0]*np.cos(a_work[:,:,1]*np.pi/180);
    d = a_work*(Rearth*np.pi/180);
    return d;

def get_spacing_from_grid(grid):
    dx = np.squeeze(np.diff(grid[:,:,0],1,0));
    dy = np.squeeze(np.diff(grid[:,:,1],1,1));
    return (dx,dy);
