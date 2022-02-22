import os, glob, sys
sys.path.append("//nobackup//amondal//Python//Hector_Python_Scripts")
import numpy as np
import xarray as xr
import dask.array as da
import netCDF4
from datetime import datetime, timedelta

def getMITNCDataSet(fol, VAR, firstlevel, finallevel):
    firstVAR = VAR+"_"+str(firstlevel)
    vardir = fol+firstVAR+'/'
    vfiles = sorted(os.listdir(vardir))
    vdirfiles =[]
    new_datetime_list = []
    for subindex in range(0,len(vfiles)):
        vdirfiles.append(vardir+vfiles[subindex])
        datepart = vfiles[subindex].split('_')[1].split('.')[0]
        y = datepart[0:4]
        m = datepart[4:6]
        d = datepart[6:8]
        h = datepart[8:10]
        testdate = datetime(int(y),int(m),int(d),int(h))
        new_datetime_list.append(testdate)
    vdirset = xr.open_mfdataset(vdirfiles,chunks={'latitude':10, 'longitude':10}, concat_dim='time', parallel=True, combine='nested', engine="netcdf4")
    vdirset = vdirset.assign_coords({"time": np.array(new_datetime_list)})
    print('Time combined dataset has been opened.')
    vfullset = vdirset
    print('I concatenated the first Z-layer')
    if (firstlevel < finallevel):
        for index in range(firstlevel+1,finallevel+1):
            vi = VAR+"_" + str(index)
            vardir = fol + vi + '/'
            vdirfiles = []
            new_datetime_list = []
            for subindex in range(0,len(vfiles)):
                vdirfiles.append(vardir + vfiles[subindex])
                datepart = vfiles[subindex].split('_')[1].split('.')[0]
                y = datepart[0:4]
                m = datepart[4:6]
                d = datepart[6:8]
                h = datepart[8:10]
                testdate = datetime(int(y),int(m),int(d),int(h))
                new_datetime_list.append(testdate)
            vdirset = xr.open_mfdataset(vdirfiles,chunks={'latitude':10, 'longitude':10}, concat_dim='time', parallel=True, combine='nested', engine = 'netcdf4')
            vdirset = vdirset.assign_coords({"time": np.array(new_datetime_list)})
            print('Time combined dataset has been opened')
            vfullset = xr.concat([vfullset, vdirset], dim='Zlayers')
            print('I concatenated another Z-layer')
    return vfullset

def getGEOSNCDataSet(fol, VAR):
    firstVAR = VAR
    vardir = fol+firstVAR+'/'
    vfiles = sorted(os.listdir(vardir))
    vdirfiles =[]
    new_datetime_list = []
    for subindex in range(0,len(vfiles)):
        vdirfiles.append(vardir+vfiles[subindex])
        datepart = vfiles[subindex].split('_')[1].split('.')[0]
        y = datepart[0:4]
        m = datepart[4:6]
        d = datepart[6:8]
        h = datepart[8:10]
        if (len(datepart)>10):
            M = datepart[10:12]
            testdate = datetime(int(y),int(m),int(d),int(h), int(M))
        else:
            testdate = datetime(int(y),int(m),int(d),int(h))
        new_datetime_list.append(testdate)
    vdirset = xr.open_mfdataset(vdirfiles,chunks={'latitude':1, 'longitude':1}, concat_dim='time', parallel=True, combine='nested', engine = 'netcdf4')
    vdirset = vdirset.assign_coords({"time": np.array(new_datetime_list)})
    print('Time combined dataset has been opened.')
    vfullset = vdirset
    return vfullset
