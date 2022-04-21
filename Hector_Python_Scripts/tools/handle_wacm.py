import numpy as np
from netCDF4 import Dataset
import sys, os
import time as ti
sys.path.append('/nobackup/htorresg/DopplerScat/python/tools/')
sys.path.append('/nobackup/htorresg/DopplerScat/python/tools/IO/')
import IO as io

def read_wacm(file):
    """
    Reading WaCM
    """
    data = io.readNC(file,'r')
    lat = data['latitude'][:,:]
    lon = data['longitude'][:,:]
    time = data['time'][:]
    #rang = data['range'][:]
    wacm = {}
    wacm['lon'] = lon
    wacm['lat'] = lat
    wacm['time'] = time
    #wacm['range'] = rang
    data.close()
    return wacm

def write_preexisting(file,vname,var):
    """
    Writing var on preexisting
    """
    
    data = io.readNC(file,'r+')
    print data.dimensions.keys()
    temp = data.createVariable(vname,'>f4',('time_dim','range_dim'))
    temp[:] = var
    data.close()
    print('NC has been written')

def write_preexisting_L3(file,vname,var):
    data = io.readNC(file,'r+')
    temp = data.createVariable(vname,'>f4',('s_dim','c_dim'))
    temp[:] = var
    print('NC has been written')
    data.close()
