import os, glob, sys
import scipy.io
sys.path.append("//nobackup//amondal//Python//Hector_Python_Scripts")
import numpy as np
import xarray as xr
import dask.array as da
import netCDF4

"""
depthToZlayer takes in a depth in meters and tells you which layer in the LLC2160
MITgcm grid that depth corresponds to
"""
def depthToZlayer(depth):
    cdir = os.getcwd()
    os.chdir("//nobackup//amondal/Python//Hector_Python_Scripts") 
    thk = scipy.io.loadmat('thk90')
    d = thk['dpt90'][0,0:90]
    os.chdir(cdir)
    return np.max(np.nonzero(np.where(d < depth, d,0)))

"""
zlayerToDepth takes in the zlayer number and tells you how deep down it is
"""
def zlayerToDepth(z):
    cdir = os.getcwd()
    os.chdir("//nobackup//amondal/Python//Hector_Python_Scripts") 
    thk = scipy.io.loadmat('thk90')
    d = thk['dpt90'][0,0:90]
    os.chdir(cdir)
    return d[z]
"""
getZlayerThickness tells you how thick a specific Zlayer is
"""
def getZlayerThickness(z):
    cdir = os.getcwd()
    os.chdir("//nobackup//amondal/Python//Hector_Python_Scripts") 
    thk = scipy.io.loadmat('thk90')
    thickness = thk['thk90'][0,0:90]
    os.chdir(cdir)
    return thickness[z]
