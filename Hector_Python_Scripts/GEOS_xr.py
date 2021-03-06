print('Hola')
import numpy as np
import xarray as xr
from xmitgcm import open_mdsdataset
import os,glob,sys
import time as tm
from scipy import ndimage, misc
#import dask_ndfilters
from datetime import datetime,timedelta
from scipy import signal
import dask.array as da
from llcmap_bi_split import LLCMap_bi_split


if __name__ == "__main__":
    # execute only if run as a script
    print('as as script')
    from dask.distributed import Client
    client = Client(memory_limit='4GB', n_workers=6, threads_per_worker=1)
    #CLIENT
    print('Hola')
    
    ##### CHECK THAT HIS WORKS THIS WAS JUST COPIED FROM MIT_xr
    VAR='V' ### <<<<<<<<<<======== Select variable - check holding list for names
    ffilter=0 ### <<<=== ask Hector what this is for at some point 
    fsize=0 ####### <<<===== ask Hector what this means at some poitn
    from GEOS_xr import GEOS_xr
    GEOS_xr(VAR, ffilter, fsize=0)
    #MIT_xr needs parameter levels, but GEOS files have levels built into them

def GEOS_xr(VAR,ffilter,fsize=0):


  ##################################
  # Date
  ##################################

  y1 = 2020#2012
  y2 = 2020#2012
  m1 = 1#1 #4
  m2 = 3#3 #7
  d1 = 19#19 #22
  d2 = 23#23 # 6
  h1 = 21#21 # 0
  h2 = 20#20 # 0
  M1 = 30#30 # 0
  M2 = 30#30 # 0


  ##################################
  # filter size
  ##################################

  if ffilter==1:
    fsize_str='_'+str(int((fsize[0]-1)/8))+'x'+str(int((fsize[1]-1)/8))
    filter_str='_filter'
  elif ffilter==2:
    fsize_str=''
    filter_str='_runmean'
  else:
    fsize_str=''
    filter_str=''

  collection='geosgcm_surf'
  # surface quantities collection. Hector mentioned that geosgcm_turb contains
  # fluxes which could also be handy for you as well as geosgcm_tend which apparent
  # contains derivatives of things -- again, could maybe be useful
  expdir='/nobackupp11/dmenemen/DYAMOND/'
  #expdir='/nobackupp2/estrobac/geos5/'
  expid='c1440_llc2160'
  diro = expdir+expid+'/holding/'+collection

  date = np.arange(datetime(y1,m1,d1,h1,M1),datetime(y2,m2,d2,h2,M2),timedelta(hours=1)).astype(datetime)
  t = date.shape

  flist=[datetime.strftime(n,diro+'/DYAMOND_'+expid+'.'+collection+'.' + '%Y%m%d_%H%M' + 'z.nc4') for n in date ]

  nfiles=t[0]

  print('=====================')
  print(flist[0])
  print('=====================')
  ds0 = xr.open_mfdataset(flist[0],parallel=True)

  print(ds0)

  lat_out=np.arange(-90,90+0.0625,0.0625)
  lon_out=np.arange(-180,180,0.0625)

  output=xr.DataArray(np.zeros((lat_out.shape[0],lon_out.shape[0])), \
                      coords=[ lat_out,lon_out], \
                      dims=[ 'lat', 'lon'])

  coords = ds0.coords.to_dataset().reset_coords()
  print((coords.lon).shape)
  print((coords.lat).shape)
  #msk=ds0.FROCEAN[0]
  #msk=np.where(msk>0,1,np.nan) 
  #XC=xr.where(coords.lons>=180,coords.lons*msk-360,coords.lons*msk)
  XC = xr.where(coords.lon>=180,coords.lon-360,coords.lon)
  YC=coords.lat#*msk
  mapper = LLCMap_bi_split(YC.values, XC.values,lat_out,lon_out,radius=15e3)

  for i in range(0,nfiles):

    print('open files '+str(i))

    ds1 = xr.open_mfdataset(flist[i])
    
    print('select variable')
    if VAR=='SPEED':
      TMP=(np.sqrt(ds1.US**2+ds1.VS**2)).rename(VAR)
    elif VAR=='SPEED10': 
      TMP=(np.sqrt(ds1.U10M**2+ds1.V10M**2)).rename(VAR)
    elif VAR=='SPEED2':
      TMP=(np.sqrt(ds1.U2M**2+ds1.V2M**2)).rename(VAR)
    elif VAR=='TAU':
      TMP=(ds1.TAUX**2+ds1.TAUY**2).rename(VAR)
    elif VAR=='LWGNET':
      #LWGNET=SFCEM - LWS : net long wave radiation
      TMP=(ds1.SFCEM+ds1.LWS).rename(VAR)
    elif VAR=='TUFX':
      #TUFX = LHFX + SHFX - turbulent fluxes
      TMP=(ds1.LHFX+ds1.SHFX).rename(VAR)
    elif VAR=='QNET':
      #QNET = RADSRF - LHFX - SHFX : net heat flux
      TMP=(ds1.RADSRF-ds1.LHFX-ds1.SHFX).rename(VAR)
    else:
      TMP=ds1[VAR]
    print('mapping')
    output[:]=mapper(TMP.mean('time').values)


    dirout='/nobackup/htorresg/air_sea/ocean-atmos/NCFILES/geosgcm_surf/'
    print(dirout)
    if not os.path.exists(dirout+VAR):
       print('mkdir')
       os.makedirs(dirout+VAR)

    print('save output')

    output.rename(VAR).to_netcdf(dirout+VAR+'/'+VAR+filter_str+fsize_str+'_'+date[i].strftime("%Y%m%d%H")+'.nc')




