import numpy as np
import xarray as xr
import dask.array as da
import dask_ndfilters
from xmitgcm import open_mdsdataset
import os,glob,sys
sys.path.append("/nobackup/amondal/Python/Hector_Python_Scripts")
import time as tm
import xgcm
import xmitgcm
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
    #client
    print('Hola, dask has been set up!!!')
   # cluster = PBSCluster(cores=16, processes=16, memory='150GB', queue='devel',\
   #                      resource_spec='select=16:ncpus=10:model=sky_ele',
    #                     walltime='2:00:00')
    #cluster.scale(1)
                         
    # you can use the schedule_file as suggested in HECC support
    #client = Client(cluster)
    #client = Client(scheduler_file='/sched.json')
    #from MIT_xr_cwt_dateloc_fol import MIT_xr_date_location_fol
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
    m2 = 6
    d1 = 1
    d2 = 1
    h1 = 0
    h2 = 0
    M1 = 0
    M2 = 0
    ##########################
    #set location of cell(s)
    """
    Pick out Kuroshio and or Gulf Stream for this once Patrice gets back to you
    """
    lat1 = 35
    lat2 = 50
    latinc =0.04
    lon1 = -60 
    lon2 = -45
    loninc = 0.04
    ##############################################################################
    print('Date and location has been set')
    #VAR='KPPhbl' ##### <<<<<<<<<<< ============ Select
    levels=0  #### <<<<<<<====== vertical levels
    ffilter=0 ## <<== don't move
    fsize=0   ## <<==== don't mode
    fol = '/nobackup/amondal/NCData/20211103_multicoretest_3month/'
    #MIT_xr_date_location_fol('KPPhbl',levels,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)
    #MIT_xr_date_location_fol('oceQnet',levels,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)
    #MIT_xr_date_location_fol('oceQsw',levels,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)
    #MIT_xr_date_location_fol('Zeta',levels,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)
    for level in range(6, 35):
        MIT_xr_date_location_fol('Theta',level,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)
        MIT_xr_date_location_fol('HAdv',level,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)
        #MIT_xr_date_location_fol('U',level,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)
        #MIT_xr_date_location_fol('V',level,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)
        #MIT_xr_date_location_fol('DxTheta',level,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)
        #MIT_xr_date_location_fol('DyTheta',level,ffilter,fsize,y1,m1,d1,h1,M1,y2,m2,d2,h2,M2,lat1,lat2,latinc,lon1,lon2,loninc,fol)
    

