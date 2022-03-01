import os,glob,sys
import numpy as np
import xarray as xr
import dask.array as da
import dask_ndfilters
sys.path.append("//nobackup//amondal//Python")
sys.path.append("//nobackup//amondal//Python//xmitgcm")
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
from timeline_MITgcm import timeline
def MIT_xr_date_location_fol(VAR, level, ffilter, fsize, y1,m1,d1,h1,M1, y2, m2,d2,h2,M2, lat1, lat2, latinc, lon1,lon2,loninc, pdirout, gridvar=False,):
  print('Entering MIT_xr_date_location')
  ########################################################
  # Same as MIT_xr but date and location are parameters
  # in a future update, this will require some errorhandling, especially when
  # you share this!
  #
  # if you're trying to access one of the grid variables, use 
  ########################################################
  # date = np.arange(datetime(y1,m1,d1,h1,M1),datetime(y2,m2,d2,h2,M2),
  #                 timedelta(hours=1)).astype(datetime)
  # timedelta(hours=1) - pretty sure this means that there are 1 hour increments?
  initial_time = (str(y1) + '%02d' + '%02d' + '%02d' + '%02d') % (m1, d1,h1, M1)
  end_time = (str(y2) + '%02d' + '%02d' + '%02d' + '%02d') % (m2,d2,h2, M2)
  reference_time = '20200119213000'
  timedelta = 1 #hours
  rate = 3600 # sampling rate
  dt = 45 # seconds
  all_iters, date = timeline(initial_time, end_time, reference_time,timedelta,rate,dt)
  t = date.shape
  print('date')
  print(str(date) + '\n')
  print(str(all_iters)+ '\n')
  ##################################
  # Output grid
  ##################################
  print('setting location')
  lat_out = np.arange(lat1, lat2, latinc)
  lon_out = np.arange(lon1, lon2, loninc)
  #lat_out = np.arange(-90,90+0.04,0.04)
  #lon_out = np.arange(-180,180,0.04)
  print('location')
  ##################################
  # MIT iterations
  ##################################

  #iter0 = 0#120960
  delta_t=45 #45 second iterations 
  #delta = 3600/delta_t # iters
  nfiles = t[0]
  #all_iters = iter0 + delta*np.arange(nfiles)
  
  
  #### don't change this part
  if (VAR == 'DxTheta'):
    diro = '/nobackupp11/dmenemen/DYAMOND/c1440_llc2160/mit_output/Theta'
  elif (VAR == 'DyTheta'):
    diro = '/nobackupp11/dmenemen/DYAMOND/c1440_llc2160/mit_output/Theta'
  elif (VAR == 'Zeta'):
    diro = '/nobackupp11/dmenemen/DYAMOND/c1440_llc2160/mit_output/oceTAUX'
  elif (VAR == 'HAdv'):
    diro = '/nobackupp11/dmenemen/DYAMOND/c1440_llc2160/mit_output/Theta'
  elif (VAR == 'dxF' or VAR == 'dyF' or VAR == 'rAw' or VAR == 'rAs'): #basically, the grid variables
    diro = '/nobackupp11/dmenemen/DYAMOND/c1440_llc2160/mit_output/Theta/'
    gridvar = True
  else:
    diro = '/nobackupp11/dmenemen/DYAMOND/c1440_llc2160/mit_output/'+VAR+'/'
    
   
  ####
  print('Directory of files')
  print(diro)
  nx=2160
  GRIDDIR='/nobackupp11/dmenemen/DYAMOND/c1440_llc2160/mit_output/grid/'#'/nobackupp2/estrobac/geos5/MITGRID/llc2160/'
  ####
  print('before mds')
  ds = open_mdsdataset(diro, grid_dir=GRIDDIR, iters=all_iters[0], 
                       geometry='llc', read_grid=True, 
                       default_dtype=np.dtype('>f4'),
                       delta_t=delta_t, ignore_unknown_vars=True, nx=nx,chunks={
                       "i": -1,
                       "j": -1,
                       "time": 1,
                         "face": 1, "k": 1})

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
  prntout = pdirout #'/nobackup/amondal/NCData/20210629_TempHeterogeneity/'
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


  for i in range(0,(nfiles)-1):
    start = tm.time()
    print('open files')
    print(all_iters[i])

    ds = open_mdsdataset(diro, grid_dir=GRIDDIR, iters=all_iters[i], geometry='llc', read_grid=True, default_dtype=np.dtype('>f4'), delta_t=delta_t, ignore_unknown_vars=True, nx=nx,chunks={
                       "i": -1,
                       "j": -1,
                       "time": 1,
                       "face": 1, "k": 1})
    

    if VAR == 'V':
      #### change this part
      dirou='/nobackupp11/dmenemen/DYAMOND/c1440_llc2160/mit_output/U/'
      ####
      dsu = open_mdsdataset(dirou,
                      grid_dir=GRIDDIR,
                            iters=all_iters[i],geometry='llc',read_grid=True,
                            default_dtype=np.dtype('>f4'),delta_t=delta_t,
                            ignore_unknown_vars=True,nx=nx, chunks={
                       "i": -1,
                       "j": -1,
                       "time": 1,
                              "face": 1, "k": 1},)
      ds1 = open_mdsdataset(GRIDDIR, iters=1, ignore_unknown_vars=True, geometry='llc', nx=nx, chunks={
                       "i": -1,
                       "j": -1,
                       "time": 1,
                       "face": 1, "k": 1}, )

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
                      grid_dir=GRIDDIR,
                            iters=all_iters[i],geometry='llc',read_grid=True,
                            default_dtype=np.dtype('>f4'),delta_t=delta_t,
                            ignore_unknown_vars=True,nx=nx, chunks={
                       "i": -1,
                       "j": -1,
                       "time": 1,
                              "face": 1, "k": 1},)
      ds1 = open_mdsdataset(GRIDDIR, iters=1, ignore_unknown_vars=True, geometry='llc', nx=nx,)  
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
                      grid_dir=GRIDDIR,
                            iters=all_iters[i],geometry='llc',read_grid=True,
                            default_dtype=np.dtype('>f4'),delta_t=delta_t,
                            ignore_unknown_vars=True,nx=nx, chunks={
                       "i": -1,
                       "j": -1,
                       "time": 1,
                              "face": 1, "k": 1},)
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
                      grid_dir=GRIDDIR,
                            iters=all_iters[i],geometry='llc',read_grid=True,
                            default_dtype=np.dtype('>f4'),delta_t=delta_t,
                            ignore_unknown_vars=True,nx=nx, chunks={
                       "i": -1,
                       "j": -1,
                       "time": 1,
                              "face": 1, "k": 1},)
      AngleCS=dsu['CS'];AngleSN=dsu['SN'];
      print('=== interp 2d ====')
      UV=grid.interp_2d_vector({'X': dsu['oceTAUX'], 'Y': ds['oceTAUY']},boundary='fill')
      x=(UV['X']*AngleSN+UV['Y']*AngleCS)
      print('== done interp 2d ===')
      print('=== x interpolated  ==')
      print(np.nanmean(x))
    elif VAR == 'Zeta':
      ### change this part
      dirov='/nobackupp11/dmenemen/DYAMOND/c1440_llc2160/mit_output/oceTAUY/'
      ###
      dsv = open_mdsdataset(dirov,
                      grid_dir=GRIDDIR,
                            iters=all_iters[i],geometry='llc',read_grid=True,
                            default_dtype=np.dtype('>f4'),delta_t=delta_t,
                            ignore_unknown_vars=True,nx=nx, chunks={
                       "i": -1,
                       "j": -1,
                       "time": 1,
                              "face": 1, "k": 1},)
      grid = xgcm.Grid(ds)
      zeta = (-grid.diff(ds['oceTAUX']*ds.dxC,'Y') + grid.diff(dsv['oceTAUY']*ds.dyC,'X')) / ds.rAz
      x = grid.interp(grid.interp(zeta,axis='X', to='center'), axis='Y',to='center')
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
    elif VAR == 'Salt':
      x=ds[VAR].isel(k=level)
    elif VAR == 'oceQnet':
      x=ds[VAR]
    elif VAR == 'oceQsw':
      x=ds[VAR]
    elif VAR == 'DyTheta':
      dgrid = xgcm.Grid(ds)
      DxTheta = dgrid.diff(ds['Theta'].isel(k=level),axis='X',boundary='fill') / ds.dxC
      DyTheta = dgrid.diff(ds['Theta'].isel(k=level),axis='Y',boundary='fill') / ds.dyC
      ds1 = open_mdsdataset(GRIDDIR, iters=1, ignore_unknown_vars=True,geometry='llc', nx=nx)
      AngleCS=ds1['CS'];AngleSN=ds1['SN'];
      UV=grid.interp_2d_vector({'X':DxTheta, 'Y':DyTheta},boundary='fill')
      x=(UV['X']*AngleSN+UV['Y']*AngleCS)
    elif VAR =='DxTheta':
      dgrid = xgcm.Grid(ds)
      DxTheta = dgrid.diff(ds['Theta'].isel(k=level),axis='X',boundary='fill') / ds.dxC
      DyTheta = dsdfasdgrid.diff(ds['Theta'].isel(k=level),axis='Y',boundary='fill') / ds.dyC
      ds1 = open_mdsdataset(GRIDDIR, iters=1, ignore_unknown_vars=True,geometry='llc', nx=nx)
      AngleCS=ds1['CS'];AngleSN=ds1['SN'];
      UV=grid.interp_2d_vector({'X':DxTheta, 'Y':DyTheta},boundary='fill')
      x=(UV['X']*AngleCS-UV['Y']*AngleSN)
    elif VAR =='HAdv':
      dirou='/nobackupp11/dmenemen/DYAMOND/c1440_llc2160/mit_output/U/'
      ####
      dsu = open_mdsdataset(dirou,
                      grid_dir=GRIDDIR,
                            iters=all_iters[i],geometry='llc',read_grid=True,
                            default_dtype=np.dtype('>f4'),delta_t=delta_t,
                            ignore_unknown_vars=True,nx=nx, chunks={
                       "i": -1,
                       "j": -1,
                       "time": 1,
                       "face": 1, "k": 1},)
      dirov='/nobackupp11/dmenemen/DYAMOND/c1440_llc2160/mit_output/V/'
      ###
      dsv = open_mdsdataset(dirov,
                      grid_dir=GRIDDIR,
                            iters=all_iters[i],geometry='llc',read_grid=True,
                            default_dtype=np.dtype('>f4'),delta_t=delta_t,
                            ignore_unknown_vars=True,nx=nx, chunks={
                       "i": -1,
                       "j": -1,
                       "time": 1,
                              "face": 1, "k": 1},)
      dgrid = xgcm.Grid(ds)
      DxTheta = dgrid.diff(ds['Theta'].isel(k=level),axis='X',boundary='fill') / ds.dxC
      DyTheta = dgrid.diff(ds['Theta'].isel(k=level),axis='Y',boundary='fill') / ds.dyC
      xadv = dsu['U'].isel(k=level)*DxTheta
      yadv = dsv['V'].isel(k=level)*DyTheta
      xadv_interp = dgrid.interp(xadv,axis='X',to='center')
      yadv_interp = dgrid.interp(yadv,axis='Y',to='center')
      x = xadv_interp + yadv_interp
    elif gridvar:
      x = ds[VAR]
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

    
