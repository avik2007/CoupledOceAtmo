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
    t = thk['thk90'][0,0:90]
    os.chdir(cdir)
    zdim = np.max(np.nonzero(np.where(d < depth, d,0)))
    # we have to add a conditional because the above zdim is defined by looking at
    # where the values in the dpt array are greater than our desired depth. However
    # this does not tell us which cell the depth is located in, since the reported
    # depths in dpt are the depths of the tracer points, and not the cells.
    # thus, if the tracer depth plus half the cell thickness is less than the 
    # depth we are looking for, than we add 1 to the zlayer
    if (d[zdim]+t[zdim]/2 < depth):
        zdim = zdim + 1
    return zdim
    

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
"""
getThickWeightedVector returns a vector that acts like a probability density function weighted averages over
the depth of a set of zlayers. It takes depth assuming that one is trying to average over that depth
"""
def getThickWeightedVector(depth):
    cdir = os.getcwd()
    os.chdir("//nobackup//amondal/Python//Hector_Python_Scripts") 
    thk = scipy.io.loadmat('thk90')
    t = thk['thk90'][0,0:90]
    d = thk['dpt90'][0,0:90]
    os.chdir(cdir)
    maxdepth = np.max(depth)
    zdim = np.max(np.nonzero(np.where(d<maxdepth,d,0)))
    if ((d[zdim] + t[zdim]/2) < maxdepth):
        zdim = zdim + 1
    weightedthickvector = np.empty(depth.shape+(zdim+1,))
    for di in range(0,len(depth)):
        zlayer = np.max(np.nonzero(np.where(d<depth,d,0)))
        if ((d[zlayer]+t[zlayer]/2) < depth[di]):
            zlayer = zlayer+1
        firstbit = t[0:zlayer]/depth[di]
        #"firstbit" puts together all the parts of the pdf except the last element
        #the last element needs to be handled specially because the depth inputted
        #might not correspond to the full thickness of the last element
        middlebit = 1 - (d[zlayer] - t[zlayer]/2)/depth[di]
        wtv = np.append(firstbit,middlebit)
        if ((zdim - len(wtv)) < 0):
            wtv_filled = wtv
        else:
            wtv_filled = np.append(wtv, np.zeros(zdim - len(wtv)+1))
        weightedthickvector[di,:] = wtv_filled
    return weightedthickvector
