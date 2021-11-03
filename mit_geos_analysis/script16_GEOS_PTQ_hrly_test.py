import numpy as np
import xarray as xr
from xmitgcm import open_mdsdataset
import os,glob,sys
sys.path.append("/nobackup/amondal/Python/Hector_Python_Scripts")
import time as tm
from scipy import ndimage, misc
#import dask_ndfilters
from datetime import datetime,timedelta
from scipy import signal
import dask.array as da
from llcmap_bi_split import LLCMap_bi_split
from llcmap_nea_split import LLCMap_nea_split


if __name__ == "__main__":
    # execute only if run as a script
    print('as as script')
    from dask.distributed import Client
    client = Client(memory_limit='75GB', n_workers=30, threads_per_worker=1)
    #CLIENT
    print('Hola')
    
    ##### CHECK THAT HIS WORKS THIS WAS JUST COPIED FROM MIT_xr
    #VAR = 'DTDTDYN' ### <<<<<<<<<<======== Select variable - check holding list for names
    ffilter=0 ### <<<=== ask Hector what this is for at some point 
    fsize=0 ####### <<<===== ask Hector what this means at some point
    y1 = 2020
    m1 = 3
    d1 = 1
    h1 = 9
    M1 = 0
    y2 = 2020
    m2 = 3
    d2 = 3
    h2 = 9
    M2 = 0
    lat1 = 0
    lat2 = 45
    lon1 = 45
    lon2 = 90
    lev1 = 21
    lev2 = 75
    pdirout='/nobackup/amondal/NCData/20211102_GEOS_first_integral/'
    
    from GEOS_coll_date_loc_fol import GEOS_xr_coll_date_location_fol
    GEOS_xr_coll_date_location_fol('TEMP', 'T', ffilter, fsize, y1, m1, d1, h1, M1, y2, m2, d2, h2, M2, lat1, lat2, lon1, lon2,lev1, lev2, pdirout)
    GEOS_xr_coll_date_location_fol('INSTPRESS', 'P', ffilter, fsize, y1, m1, d1, h1, M1, y2, m2, d2, h2, M2, lat1, lat2, lon1, lon2,lev1, lev2, pdirout)
    GEOS_xr_coll_date_location_fol('QV', 'QV', ffilter, fsize, y1, m1, d1, h1, M1, y2, m2, d2, h2, M2, lat1, lat2, lon1, lon2,lev1, lev2, pdirout)
    #MIT_xr needs parameter levels, but GEOS files have levels built into them
