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
from llcmap_nea_split import LLCMap_nea_split


def GEOS_xr_coll_date_location_fol(coll, VAR,ffilter, fsize,y1, m1, d1,h1, M1, y2, m2, d2,h2, M2, lat1, lat2,  lon1, lon2, lev1, lev2, pdirout ):
  # will we have to put level in here eventually?
  # I haven't included latinc and loninc here because the atmosphere is at a fixed distance increment and I think there's no reason in the near future to coarse grain that any further. And frankly, the same is probably true for the ocean.

  ##################################
  # Date
  ##################################

  #y1 = 2020#2012
  #y2 = 2020#2012
  #m1 = 1#1 #4
  #m2 = 3#3 #7
  #d1 = 19#19 #22
  #d2 = 23#23 # 6
  #h1 = 9#21 # 0
  #h2 = 9#20 # 0
  #M1 = 0#30 # 0
  #M2 = 0#30 # 0

  # so the date is tricky because for different  collections you have different time options
  # if collection is SURF, M2 and M1 = 30 no argument. if collection is TEND, the hour and minute are always 0900 no argument. 


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

  collection1='geosgcm_surf'
  # surface quantities collection. Hector mentioned that geosgcm_turb contains
  # fluxes which could also be handy for you as well as geosgcm_tend which apparent
  # contains derivatives of things -- again, could maybe be useful
  collection2='geosgcm_tend'
  # tendency quantities collection. This contains everyhing needed for a potential temperature balance and an analysis of change in specific humidity in time (which will be necessary for virtual temperature) as well as dudt and dvdt
  collection3 = 'const_2d_asm_Mx'
  # constant repository - stores constants like FRLAND, FRLANDICE, FROCEAN, etc
  collection4 = 'inst_15mn_2d_asm_Mx'
  # contains zpbl which is necessary for picking mixed layer height
  collection5 = 'geosgcm_pressure'
    # contains pressure coordinates as a function of time, lat, and lon
  collection6 = 'inst_01hr_3d_T_Mv'
  # contains contains hourly instantaneous values of temperature
  collection7 = 'inst_01hr_3d_QV_Mv'
  # contains hourly specific humidity
  collection8 = 'inst_01hr_3d_P_Mv'
  # contains hourly pressures
  collection9 = 'tavg_15mn_2d_flx_Mx'

  if (coll == 'SURF'):
    collection = collection1
    M1 = 30
    M2 = 30
    deltatime = timedelta(hours=1)
  elif (coll == 'TEND'):
    collection = collection2
    h1 = 9
    h2 = 9
    M1 = 0
    M2 = 0
    deltatime = timedelta(days=1)
  elif (coll == 'PRESSURE'):
    collection = collection5
    h1 = 9
    h2 = 9
    M1 = 0
    M2 = 0
  elif (coll == 'CONST'):
    collection = collection3
    y1 = 2020
    y2 = 2020
    m1 = 1
    m2 = 1
    d1 = 22
    d2 = 22
    M1 = 0
    M2 = 0
    h1 = 0
    h2 = 1
    deltatime = timedelta(hours=1)
  elif (coll == 'INST15'):
    collection = collection4
    deltatime = timedelta(minutes=15)
    # it's harder to code this in, but the options for minutes are at 0, 15,30, or 45.
    # if I have time/ need to, I can try to figure this out!
  elif (coll == 'TEMP'):
    collection = collection6
    M1 = 0
    M2 = 0
    deltatime = timedelta(hours=1)
  elif (coll == 'QV'):
    collection = collection7
    M1 = 0
    M2 = 0
    deltatime = timedelta(hours=1)
  elif (coll == 'INSTPRESS'):
    collection = collection8
    M1 = 0
    M2 = 0
    deltatime = timedelta(hours=1)
  elif (coll == 'TAVG15MIN'):
    collection = collection9
    deltatime = timedelta(minutes=15)
    M1 = 7
    M2 = 7
    # this one is tricky because it's the average of data taken at the hour,  15minute, 30min, and 45 min mark of the hour
    # so this gives options of M = 7, 22, 37, or 52
    

  expdir='/nobackupp11/dmenemen/DYAMOND/'
  #expdir='/nobackupp2/estrobac/geos5/'
  expid='c1440_llc2160'
  diro = expdir+expid+'/holding/'+collection

  date = np.arange(datetime(y1,m1,d1,h1,M1),datetime(y2,m2,d2,h2,M2), deltatime).astype(datetime)
  t = date.shape

  flist=[datetime.strftime(n,diro+'/DYAMOND_'+expid+'.'+collection+'.' + '%Y%m%d_%H%M' + 'z.nc4') for n in date ]

  nfiles=t[0]

  print('=====================')
  print(flist[0])
  print('=====================')
  #ds0 = xr.open_mfdataset(flist[0],parallel=True)
  #print(ds0)

  #I'm testing out the grid code here so that I can ignore interpolation

  #lat_out=np.arange(-90,90+0.0625,0.0625)
  lat_out = np.arange(lat1, lat2 + 0.0625, 0.0625)
  #lon_out=np.arange(-180,180,0.0625)
  lon_out = np.arange(lon1, lon2 + 0.0625, 0.0625)
  
  """
  Here, you need to add a lev_out. Keep in mind that they did a cutoff on the lev's so it only goes from something like 21-72. 
  """
  if (coll == 'TEND' or coll == 'PRESSURE' or coll == 'TEMP' or coll == 'QV' or coll == 'INSTPRESS'):
    lev_out = np.arange(lev1, lev2+1, 1)
    output=xr.DataArray(np.zeros((lat_out.shape[0], lon_out.shape[0], lev_out.shape[0])), coords=[lat_out, lon_out, lev_out], dims = ['lats','lons','levs'])
  else:
    output=xr.DataArray(np.zeros((lat_out.shape[0],lon_out.shape[0])), \
                      coords=[ lat_out,lon_out], \
                      dims=[ 'lat', 'lon'])

  #coords = ds0.coords.to_dataset().reset_coords()
  #print((coords.lon).shape)
  #print((coords.lat).shape)
  #msk=ds0.FROCEAN[0]
  #msk=np.where(msk>0,1,np.nan) 
  #XC=xr.where(coords.lons>=180,coords.lons*msk-360,coords.lons*msk)
  #YC=coords.lat#*msk
  # This is where the grid selection code originally lived 
  
  GEOS_gridfile = "/nobackup/amondal/NCData/geos_c1440_lats_lons_2D.nc"
  gridds = xr.open_dataset(GEOS_gridfile)
  XC = gridds.lons
  XC = xr.where(XC>=180, XC- 360, XC)
  YC = gridds.lats
  print(XC.shape)
  print(YC.shape)
  mapper = LLCMap_nea_split(YC.values, XC.values,lat_out,lon_out,radius=15e3)
  
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
    elif VAR=='DTDT':
      #DTDT: tendency of air temp. Adding contributions due to dynamics (advection), friction, gravity wave drag, moist processes, shortwave radiation, and turbulent processes
      TMP= (ds1.DTDTDYN + ds1.DTDTFRI + ds1.DTDTGWD + ds1.DTDTMST + ds1.DTDTSW + ds1.DTTRB + ds1.DTDTLW).rename(VAR)
    else:
      TMP=ds1[VAR]
    print('mapping')
    
    if 'lev' in str(TMP.dims):
      for levI in range(lev1, lev2+1):  
        output[:, :, levI-lev1]=mapper(TMP.mean('time').sel(lev = levI, method = 'nearest').values)
        print('Level ' + str(levI) + ' completed.')  
    else:
      output[:] = mapper(TMP.mean('time').values)
    # there's something that has to go here that has to do with lev - maybe it's that this needs to be carried out individually for all lev's?
    # I want to make output take in each 'lev' - how can I do that correctly

    #dirout='/nobackup/htorresg/air_sea/ocean-atmos/NCFILES/geosgcm_surf/'
    print(pdirout)
    if not os.path.exists(pdirout+VAR):
       print('mkdir')
       os.makedirs(pdirout+VAR)

    print('save output')

    output.rename(VAR).to_netcdf(pdirout+VAR+'/'+VAR+filter_str+fsize_str+'_'+date[i].strftime("%Y%m%d%H")+'.nc')

def GEOS_xr_no_interp(coll, VAR,ffilter, fsize,y1, m1, d1,h1, M1, y2, m2, d2,h2, M2, lat1, lat2,  lon1, lon2, lev1, lev2, pdirout ):
  # will we have to put level in here eventually?
  # I haven't included latinc and loninc here because the atmosphere is at a fixed distance increment and I think there's no reason in the near future to coarse grain that any further. And frankly, the same is probably true for the ocean.

  ##################################
  # Date
  ##################################

  #y1 = 2020#2012
  #y2 = 2020#2012
  #m1 = 1#1 #4
  #m2 = 3#3 #7
  #d1 = 19#19 #22
  #d2 = 23#23 # 6
  #h1 = 9#21 # 0
  #h2 = 9#20 # 0
  #M1 = 0#30 # 0
  #M2 = 0#30 # 0

  # so the date is tricky because for different  collections you have different time options
  # if collection is SURF, M2 and M1 = 30 no argument. if collection is TEND, the hour and minute are always 0900 no argument. 


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

  collection1='geosgcm_surf'
  # surface quantities collection. Hector mentioned that geosgcm_turb contains
  # fluxes which could also be handy for you as well as geosgcm_tend which apparent
  # contains derivatives of things -- again, could maybe be useful
  collection2='geosgcm_tend'
  # tendency quantities collection. This contains everyhing needed for a potential temperature balance and an analysis of change in specific humidity in time (which will be necessary for virtual temperature) as well as dudt and dvdt
  collection3 = 'const_2d_asm_Mx'
  # constant repository - stores constants like FRLAND, FRLANDICE, FROCEAN, etc
  collection4 = 'inst_15mn_2d_asm_Mx'
  # contains zpbl which is necessary for picking mixed layer height
  collection5 = 'geosgcm_pressure'
  # contains pressure coordinates as a function of time, lat, and lon
  collection6 = 'inst_01hr_3d_T_Mv'
  # contains contains hourly instantaneous values of temperature
  collection7 = 'inst_01hr_3d_QV_Mv'
  # contains hourly instantaneous specific humidity
  collection8 = 'inst_01hr_3d_P_Mv'
  # contains hourly instantaneous pressures

  if (coll == 'SURF'):
    collection = collection1
    M1 = 30
    M2 = 30
    deltatime = timedelta(hours=1)
  elif (coll == 'TEND'):
    collection = collection2
    h1 = 9
    h2 = 9
    M1 = 0
    M2 = 0
    deltatime = timedelta(days=1)
  elif (coll == 'PRESSURE'):
    collection = collection5
    h1 = 9
    h2 = 9
    M1 = 0
    M2 = 0
  elif (coll == 'CONST'):
    collection = collection3
    y1 = 2020
    y2 = 2020
    m1 = 1
    m2 = 1
    d1 = 22
    d2 = 22
    M1 = 0
    M2 = 0
    h1 = 0
    h2 = 1
    deltatime = timedelta(hours=1)
  elif (coll == 'INST15'):
    collection = collection4
    deltatime = timedelta(minutes=15)
    # it's harder to code this in, but the options for minutes are at 0, 15,30, or 45.
    # if I have time/ need to, I can try to figure this out!
  elif (coll == 'TEMP'):
    collection = collection6
    col
    M1 = 0
    M2 = 0
    deltatime = timedelta(hours=1)
  elif (coll == 'QV'):
    collection = collection7
    M1 = 0
    M2 = 0
    deltatime = timedelta(hours=1)
  elif (coll == 'INSTPRESS'):
    collection = collection8
    M1 = 0
    M2 = 0
    deltatime = timedelta(hours=1)
        


  expdir='/nobackupp11/dmenemen/DYAMOND/'
  #expdir='/nobackupp2/estrobac/geos5/'
  expid='c1440_llc2160'
  diro = expdir+expid+'/holding/'+collection

  date = np.arange(datetime(y1,m1,d1,h1,M1),datetime(y2,m2,d2,h2,M2), deltatime).astype(datetime)
  t = date.shape

  flist=[datetime.strftime(n,diro+'/DYAMOND_'+expid+'.'+collection+'.' + '%Y%m%d_%H%M' + 'z.nc4') for n in date ]

  nfiles=t[0]

  print('=====================')
  print(flist[0])
  print('=====================')
  #ds0 = xr.open_mfdataset(flist[0],parallel=True)
  #print(ds0)

  
  
  

  #lat_out=np.arange(-90,90+0.0625,0.0625)
  lat_out = np.arange(lat1, lat2 + 0.0625, 0.0625)
  #lon_out=np.arange(-180,180,0.0625)
  lon_out = np.arange(lon1, lon2 + 0.0625, 0.0625)
  
  """
  Here, you need to add a lev_out. Keep in mind that they did a cutoff on the lev's so it only goes from something like 21-72. 
  """
  if (coll == 'TEND' or coll == 'PRESSURE' or coll == 'TEMP' or coll == 'QV' or coll == 'INSTPRESS'):
    if (lev1 < 21):
      lev1 = 21
    if (lev2 > 72):
      lev2 = 72
    lev_out = np.arange(lev1, lev2+1, 1)
    output=xr.DataArray(np.zeros((lat_out.shape[0], lon_out.shape[0], lev_out.shape[0])), coords=[lat_out, lon_out, lev_out], dims = ['lats','lons','levs'])
  else:
    output=xr.DataArray(np.zeros((lat_out.shape[0],lon_out.shape[0])), \
                      coords=[ lat_out,lon_out], \
                      dims=[ 'lat', 'lon'])

  #coords = ds0.coords.to_dataset().reset_coords()
  #print((coords.lon).shape)
  #print((coords.lat).shape)
  #msk=ds0.FROCEAN[0]
  #msk=np.where(msk>0,1,np.nan) 
  #XC=xr.where(coords.lons>=180,coords.lons*msk-360,coords.lons*msk)
  #YC=coords.lat#*msk
  """
  Below is the interpolation section - I'm commenting this out for this function
  """

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
    elif VAR=='DTDT':
      #DTDT: tendency of air temp. Adding contributions due to dynamics (advection), friction, gravity wave drag, moist processes, shortwave radiation, and turbulent processes
      TMP= (ds1.DTDTDYN + ds1.DTDTFRI + ds1.DTDTGWD + ds1.DTDTMST + ds1.DTDTSW + ds1.DTTRB + ds1.DTDTLW).rename(VAR)
    else:
      TMP=ds1[VAR]
    print('mapping')
    
    """
    # Another interpolating section
    if ((coll == 'TEND' or coll == 'PRESSURE')::
      for levI in range(lev1, lev2+1):  
        output[:, :, levI-lev1]=mapper(TMP.mean('time').sel(lev = levI, method = 'neare3st').values)
        print('Level ' + str(levI) + ' completed.')  
    else:
      output[:] = mapper(TMP.mean('time').values)
    """
    #you need to check if lev is a dim before doing this
    if 'lev' in str(TMP.dims):
      output = TMP.mean('time').sel(lat=slice(lat1,lat2), lon=slice(lon1,lon2), lev=slice(lev1,lev2))
    else:
      output = TMP.mean('time').sel(lat=slice(lat1,lat2), lon=slice(lon1,lon2))
    # there's something that has to go here that has to do with lev - maybe it's that this needs to be carried out individually for all lev's?
    # I want to make output take in each 'lev' - how can I do that correctly

    #dirout='/nobackup/htorresg/air_sea/ocean-atmos/NCFILES/geosgcm_surf/'
    print(pdirout)
    if not os.path.exists(pdirout+VAR):
       print('mkdir')
       os.makedirs(pdirout+VAR)

    print('save output')

    output.rename(VAR).to_netcdf(pdirout+VAR+'/'+VAR+filter_str+fsize_str+'_'+date[i].strftime("%Y%m%d%H")+'.nc')

"""
LLCtoCS - this won't work in it's current form (11/8 11:42)
"""
def LLCtoCS(lat1,lat2,lon1, lon2):
  GEOS_gridfile = "/nobackup/amondal/NCData/geos_c1440_lats_lons_2D.nc"
  gridds = xr.open_dataset(GEOS_gridfile)
  XC = gridds.lons
  XC = xr.where(XC>=180, XC- 360, XC)
  YC = gridds.lats
  npxc_sort = np.sort(XC.values)
  npyc_sort = np.sort(YC.values)
  CSlat1 = np.max(np.nonzero(np.where((npyc_sort > lat1), 0, 1)))
  CSlat2 = np.min(np.nonzero(np.where((npyc_sort < lat2), 0, 1)))
  CSlon1 = np.max(np.nonzero(np.where((npxc_sort > lon1), 0, 1)))
  CSlon2 = np.min(np.nonzero(np.where((npxc_sort < lon2), 0, 1)))
  lat_out = npyc_sort[CSlat1:CSlat2]
  lon_out = npxc_sort[CSlon1:CSlon2]
  return lat_out, lon_out
