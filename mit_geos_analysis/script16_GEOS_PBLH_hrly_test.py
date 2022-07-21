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
    y1 = 2020 #year 1
    m1 = 3 #month 1
    d1 = 1 #day 1
    h1 = 9 #hour 1
    M1 = 0 #minute 1
    y2 = 2020 #year 2
    m2 = 3 # month 2
    d2 = 4 # day 2
    h2 = 9 # hour 2
    M2 = 0 # minute 2
    lat1 = 34 #lat 1
    lat2 = 35 # lat 2
    lon1 = -65 # lon 1
    lon2 = -64 # lon 2
    lev1 = 21 # level 1 (only for 3D fields)
    lev2 = 72 # level 2 (only for 3D fields)
    pdirout='/nobackup/amondal/NCData/20211102_GEOS_first_integral/' # output directory for lat-lon xarray output
    
    from GEOS_coll_date_loc_fol import GEOS_xr_coll_date_location_fol
    #GEOS_xr_coll_date_location_fol('TAVG15MIN', 'PBLH', ffilter, fsize, y1, m1, d1, h1, M1, y2, m2, d2, h2, M2, lat1, lat2, lon1, lon2,lev1, lev2, pdirout)
    # 3 examples of how GEOS data is loaded
    # the first parameter is the collection in which a field will be stored. The second parameter is the actual field 
    GEOS_xr_coll_date_location_fol('TEMP', 'T', ffilter, fsize, y1, m1, d1, h1, M1, y2, m2, d2, h2, M2, lat1, lat2, lon1, lon2,lev1, lev2, pdirout)
    GEOS_xr_coll_date_location_fol('INSTPRESS', 'P', ffilter, fsize, y1, m1, d1, h1, M1, y2, m2, d2, h2, M2, lat1, lat2, lon1, lon2,lev1, lev2, pdirout)
    GEOS_xr_coll_date_location_fol('QV', 'QV', ffilter, fsize, y1, m1, d1, h1, M1, y2, m2, d2, h2, M2, lat1, lat2, lon1, lon2,lev1, lev2, pdirout)
    #MIT_xr needs parameter levels, but GEOS files have levels built into them

