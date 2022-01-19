import os, glob, sys
sys.path.append("//nobackup//amondal//Python//Hector_Python_Scripts")
sys.path.append("//nobackup/amondal//Python//mit_geos_analysis")
#from mds_store import openmdsdataset
#playing with netcdf - xarray
import numpy as np
import xarray as xr
import dask.array as da
import matplotlib.pyplot as plt
import netCDF4
import xgcm
from netCDF4 import Dataset
#from xmitgcm import open_mdsdataset
from GetNCDataSet import getMITNCDataSet
from MIT_xr_cwt_dateloc_fol import loadMITData
from MITllc2160Depth import *
sys.path.append("//nobackup/amondal//Python//xmitgcm//xmitgcm")
from xmitgcm.mds_store import open_mdsdataset
import xrft
from spectral_analysis_code import *
from xmovie import Movie

if __name__ == "__main__":
    print('as a script')
    level = 25
    Qfiles = "/nobackup/amondal/NCData/20211116_QTprime_openocean_3month_bigger_region/"
    T = getMITNCDataSet(Qfiles, 'Theta', level,level )
    #Tc = T.compute()
    print('T is loaded')
    W = getMITNCDataSet(Qfiles, 'W', level,level)
    #Wc = W.compute()
    print('W is loaded')
    #Treg = regularizeCoordinates(Tc.Theta,'linear',timeunits = 'hours')
    #Wreg = regularizeCoordinates(Wc.W, 'linear', timeunits = 'hours')
    Qv = T.Theta * W.W
    Qvert =regularizeCoordinates(Qv,'linear', timeunits='hours')
    print('Q is regularized')
    Qvert_winavg_3days = movingWindowAverage(Qvert, 'time', 72).compute()
    print('Qavg has been taken')
    Q = Qvert_winavg_3days[0:-1:12, 0:-1:2, 0:-1:2]#.plot(size=6, aspect=1.4)
    mov = Movie(Q.chunk({'time':48})
    movname = "/nobackup/amondal/NCData/20220118_Qprime_realspace_movies/Qvert_3dayavg_lev_" + int(level) + ".mp4"
    mov.save(moviename,parallel=True)
    print('Movie has been saved')
