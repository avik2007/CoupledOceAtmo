import numpy as np
import xarray as xr
import dask.array as da
import dask_ndfilters
import os,glob,sys
sys.path.append("/nobackup/amondal/Python/Hector_Python_Scripts")
sys.path.append("/nobackup/amondal/xmitgcm/xmitgcm")
from xmitgcm import open_mdsdataset
import time as tm
import xgcm
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime,timedelta
from llcmap_bi_split import LLCMap_bi_split
from face_connections import face_connections
from llcmap_nea_split import LLCMap_nea_split
from netCDF4 import Dataset
import pylab as plt


#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
  

if __name__ == "__main__":
    # execute only if run as a script
    print('as a script')
    #from dask_jobqueue import PBSCluster
    #from dask.distributed import Client
    #client = Client(memory_limit='20GB',n_workers = 5, threads_per_worker=1)
    #client = Client('127.0.0.1:8786')
    #client
    print('Hola, dask has been set up!!!')

    from MIT_xr_cwt_dateloc_fol import MIT_xr_date_location_fol
    print('Hector script has been loaded')
    print('Output directory is going to be an input parameter')
    
    # so we really just want to do a balance for a tracer point - and I want to start with not as many hours of time so for oceQnet and the like I'm going to stick to small amounts of data
    ##############################################################################
    # dates don't start until January 19 but you should really not pull until March 
    #set date range from data you want
    y1 = 2020
    y2 = 2020
    # This is a real big view of the map for not too many times just for verification
    m1 = 3
    m2 = 3
    d1 = 1
    d2 = 1
    h1 = 0
    h2 = 1
    M1 = 0
    M2 = 0
    ##########################
    #set location of cell(s)
    """
    Pick out Kuroshio and or Gulf Stream for this once Patrice gets back to you
    """
    lat1 = 35
    lat2 = 38
    latinc =0.04
    lon1 = -126
    lon2 = -123
    loninc = 0.04
    ##############################################################################
    print('Date and location has been set')
    #VAR='KPPhbl' ##### <<<<<<<<<<< ============ Select
    levels=0  #### <<<<<<<====== vertical levels
    ffilter=0 ## <<== don't move
    fsize=0   ## <<==== don't mode
    fol = '/nobackup/amondal/NCData/20220301_grid_vars_for_Ri_CCS/'
    MIT_xr_date_location_fol('dxF',0,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)
    MIT_xr_date_location_fol('dyF',0,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)
    MIT_xr_date_location_fol('rAw',0,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)
    MIT_xr_date_location_fol('rAs',0,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)



    #MIT_xr_date_location_fol('oceQnet',0,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)
    #MIT_xr_date_location_fol('W',15,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)
    #for level in range(28, 44, 4):
     #   MIT_xr_date_location_fol('Theta',level,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)
      #  MIT_xr_date_location_fol('W',level,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)
   
























