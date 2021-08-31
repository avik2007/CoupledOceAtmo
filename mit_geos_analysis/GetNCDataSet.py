import os, glob, sys
sys.path.append("//nobackup//amondal//Python//Hector_Python_Scripts")
import numpy as np
import xarray as xr
import dask.array as da
import netCDF4

def getNCDataSet(fol, VAR, firstlevel, finallevel):
    firstVAR = VAR+"_"+str(firstlevel)
    vardir = fol+firstVAR+'/'
    vfiles = sorted(os.listdir(vardir))
    vdirfiles =[]
    for subindex in range(0,len(vfiles)):
        vdirfiles.append(vardir+vfiles[subindex])
    vdirset = xr.open_mfdataset(vdirfiles,chunks={'latitude':1, 'longitude':1}, concat_dim='time', parallel=True, combine='nested')
    print('Time combined dataset has been opened.')
    vfullset = vdirset
    print('I concatenated the first Z-layer')
    for index in range(1,finallevel):
        vi = VAR+"_" + str(index)
        vardir = fol + vi + '/'
        vdirfiles = []
        for subindex in range(0,len(vfiles)):
            vdirfiles.append(vardir + vfiles[subindex])
        vdirset = xr.open_mfdataset(vdirfiles,chunks={'latitude':1, 'longitude':1}, concat_dim='time', parallel=True, combine='nested')
        print('Time combined dataset has been opened')
        vfullset = xr.concat([vfullset, vdirset], dim='Zlayers')
        print('I concatenated another Z-layer')
    return vfullset
