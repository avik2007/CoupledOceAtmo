import numpy as np
import xarray as xr
import dask.array as da
import dask_ndfilters
from xmitgcm import open_mdsdataset
import os,glob,sys
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

  
def MIT_xr(VAR,level,ffilter,fsize=0):

  print('Inside MIT_xr')
  ##################################
  # Date
  # initial: 2020-01-19 21:30 --
  # End:     2020-05-10 20:30
  ##################################
  
  y1 = 2020#2012
  y2 = 2020#2012
  m1 = 1#1#1#4 #months
  m2 = 1#3#7  #months
  d1 = 19#22
  d2 = 21#6
  h1 = 21#0
  h2 = 20#0
  M1 = 30#0 #minutes
  M2 = 0#0 #minutes
  date = np.arange(datetime(y1,m1,d1,h1,M1),datetime(y2,m2,d2,h2,M2),
                   timedelta(hours=1)).astype(datetime)
  # timedelta(hours=1) - pretty sure this means that there are 1 hour increments?
  t = date.shape
  print('date')

  ##################################
  # Output grid
  ##################################

  lat_out = np.arange(-90,90+0.04,0.04)
  lon_out = np.arange(-180,180,0.04)

  ##################################
  # MIT iterations
  ##################################

  iter0 = 0#120960
  delta_t=45 #45 second iterations 
  delta = 3600/delta_t # iters
  nfiles = t[0]
  all_iters = iter0 + delta*np.arange(nfiles)
  print(all_iters)
  #### don't change this part
  diro = '/nobackupp11/dmenemen/DYAMOND/c1440_llc2160/mit_output/'+VAR+'/'
  ####
  print('Directory of files')
  print(diro)

  ##################################
  # filter size
  ##################################

  if ffilter==1:
    fsize_str='_'+str(int((fsize[0]-1)/8))+'x'+str(int((fsize[1]-1)/8))
    filter_str='_filter'
  else:
    fsize_str=''
    filter_str=''
  print('filter')


  ##################################
  # MIT grid info
  ##################################

  #### don't chanhge this part
  nx=2160
  GRIDDIR='/nobackupp2/estrobac/geos5/MITGRID/llc2160/'
  ####
  print('before mds')
  ds = open_mdsdataset(diro, grid_dir=GRIDDIR, iters=all_iters[0], 
                       geometry='llc', read_grid=True, 
                       default_dtype=np.dtype('>f4'),
                       delta_t=delta_t, ignore_unknown_vars=True, nx=nx)

  grid = xgcm.Grid(ds, periodic=False, face_connections=face_connections)
  print('grid xgcm.Grid')


  coords = ds.coords.to_dataset().reset_coords()
  msk=coords.hFacC.sel(k=0)
  msk=msk.where(msk>0,np.nan) 
  XC=coords.XC*msk
  YC=coords.YC*msk
  print('initialize mapping to GEOS grid')
  mapper = LLCMap_nea_split(YC.values, XC.values,lat_out,
                            lon_out,radius=10e3)


  ##################################
  # Creating folders
  ################################
  #make directory for the output

  ### change this part
  #prntout = '/nobackup/htorresg/air_sea/ocean-atmos/NCFILES/geosgcm_surf_tides_4km/'
  prntout = '/nobackup/amondal/NCData/20210723_BasicTesting/'
  #####
  dirout=prntout+VAR+"_"+str(level)
  print('==== Folder to be created =======')
  print(dirout)
  if not os.path.exists(dirout):
      os.makedirs(dirout)

    
  #################################
  # MIT iterations
  ################################
  output=xr.DataArray(np.zeros((1,lat_out.shape[0],lon_out.shape[0])),
                      dims=("time","lat","lon"),
                      coords={'lat':lat_out,'lon':lon_out})
  output = output.rename(VAR)


  for i in range(0,(nfiles)):
    start = tm.time()
    print('open files')
    print(all_iters[i])

    ds = open_mdsdataset(diro, grid_dir='/nobackupp2/estrobac/geos5/MITGRID/llc2160/', iters=all_iters[i], geometry='llc', read_grid=True, default_dtype=np.dtype('>f4'), delta_t=delta_t, ignore_unknown_vars=True, nx=nx)
    

    if VAR == 'V':
      #### change this part
      dirou='/nobackupp11/dmenemen/DYAMOND/c1440_llc2160/mit_output/U/'
      ####
      dsu = open_mdsdataset(dirou,
                      grid_dir='/nobackupp2/estrobac/geos5/MITGRID/llc2160/',
                            iters=all_iters[i],geometry='llc',read_grid=True,
                            default_dtype=np.dtype('>f4'),delta_t=delta_t,
                            ignore_unknown_vars=True,nx=nx)
      ds1 = open_mdsdataset(GRIDDIR, iters=1, ignore_unknown_vars=True, geometry='llc', nx=nx)

      AngleCS=ds1['CS'];AngleSN=ds1['SN'];
      print('=== interp 2d ====')
      UV=grid.interp_2d_vector({'X': dsu['U'].isel(k=level), 'Y': ds['V'].isel(k=level)},boundary='fill')
      x=(UV['X']*AngleSN+UV['Y']*AngleCS)
      print('== done interp 2d ===')
      print('=== x interpolated  ==')
      print(x.shape)
    elif VAR == 'U':
      #### change this part
      dirov='/nobackupp11/dmenemen/DYAMOND/c1440_llc2160/mit_output/V/'
      ####
      dsv = open_mdsdataset(dirov,
                      grid_dir='/nobackupp2/estrobac/geos5/MITGRID/llc2160/',
                            iters=all_iters[i],geometry='llc',read_grid=True,
                            default_dtype=np.dtype('>f4'),delta_t=delta_t,
                            ignore_unknown_vars=True,nx=nx)
      ds1 = open_mdsdataset(GRIDDIR, iters=1, ignore_unknown_vars=True, geometry='llc', nx=nx) 
      AngleCS=ds1['CS'];AngleSN=ds1['SN'];
      print('=== interp 2d ====')
      UV=grid.interp_2d_vector({'X': ds['U'].isel(k=level), 'Y': dsv['V'].isel(k=level)},boundary='fill')
      x=(UV['X']*AngleCS-UV['Y']*AngleSN)
      print('== done interp 2d ===')
      print('=== x interpolated  ==')
      print(x)
     
    elif VAR == 'oceTAUX':
      ### change this part
      dirov='/nobackupp11/dmenemen/DYAMOND/c1440_llc2160/mit_output/oceTAUY/'
      ###
      dsv = open_mdsdataset(dirov,
                      grid_dir='/nobackupp2/estrobac/geos5/MITGRID/llc2160/',
                            iters=all_iters[i],geometry='llc',read_grid=True,
                            default_dtype=np.dtype('>f4'),delta_t=delta_t,
                            ignore_unknown_vars=True,nx=nx)
      AngleCS=dsv['CS'];AngleSN=dsv['SN'];
      print('=== interp 2d ====')
      UV=grid.interp_2d_vector({'X': ds['oceTAUX'], 'Y': dsv['oceTAUY']},boundary='fill')
      x=(UV['X']*AngleCS-UV['Y']*AngleSN)
      print('== done interp 2d ===')
      print('=== x interpolated  ==')
      print(np.nanmean(x))
    
    elif VAR == 'oceTAUY':
	###### change this part
      dirou='/nobackupp11/dmenemen/DYAMOND/c1440_llc2160/mit_output/oceTAUX/'
	##### 
      dsu = open_mdsdataset(dirou,
                      grid_dir='/nobackupp2/estrobac/geos5/MITGRID/llc2160/',
                            iters=all_iters[i],geometry='llc',read_grid=True,
                            default_dtype=np.dtype('>f4'),delta_t=delta_t,
                            ignore_unknown_vars=True,nx=nx)
      AngleCS=dsu['CS'];AngleSN=dsu['SN'];
      print('=== interp 2d ====')
      UV=grid.interp_2d_vector({'X': dsu['oceTAUX'], 'Y': ds['oceTAUY']},boundary='fill')
      x=(UV['X']*AngleSN+UV['Y']*AngleCS)
      print('== done interp 2d ===')
      print('=== x interpolated  ==')
      print(np.nanmean(x))
    elif VAR == 'W':
      x=grid.interp(ds['W'],'Z',to='center',boundary='fill').isel(k=level)
    elif VAR == 'Eta':
      x=ds[VAR]
    elif VAR == 'KPPhbl':
      x=ds[VAR]
    elif VAR == 'Theta':
      x=ds[VAR].isel(k=level)
    elif VAR == 'oceQnet':
      x = ds[VAR]
    elif VAR == 'oceQsw':
      x = ds[VAR]
    elif VAR == 'Salt':
      x=ds[VAR].isel(k=level)
    else:
      print('not yet implemented')


    ##################################################
    ##### Mapping X to new grid
    print('map field') 
    TMP=xr.DataArray(mapper(x.values))
    print('mapped')

    if ffilter==1:
      print('apply filter')
      output = TMP - dask_ndfilters.generic_filter(da.from_array(TMP, \
               chunks=[131,960]), \
               function=np.nanmean, \
               size=fsize, \
               mode='wrap', \
               origin=0).compute()
    else:
      output[0] = TMP
    
    del TMP


    ##################################
    # Write output
    ##################################

    print('save output')

    output.rename(VAR).to_netcdf(dirout+'/'+VAR+'_'+date[i].strftime("%Y%m%d%H")+'.nc')

    print('finished day %02d'%i)
    end = tm.time()
    print(end - start)





if __name__ == "__main__":
    # execute only if run as a script
    print('as a script')
    from dask.distributed import Client
    #client = Client(memory_limit='20GB',n_workers = 5, threads_per_worker=1)
    client = Client(memory_limit='4GB',n_workers = 6, threads_per_worker=1)
    #client
    print('Hola')

    from MIT_xr_coas_with_tides import MIT_xr
    VAR='V' ##### <<<<<<<<<<< ============ Select
    nlevels=1  #### <<<<<<<====== vertical levels
    ffilter=0 ## <<== don't move
    fsize=0   ## <<==== don't mode
    MIT_xr(VAR,nlevels,ffilter,fsize=0)
