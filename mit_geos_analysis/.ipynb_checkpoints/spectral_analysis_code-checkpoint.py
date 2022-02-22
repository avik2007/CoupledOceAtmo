import os, glob, sys
import scipy.io
sys.path.append("//nobackup//amondal//Python//Hector_Python_Scripts")
import numpy as np
import xarray as xr
import dask.array as da
import netCDF4
import math
import xrft

"""
Isotropizes a 2d spectrum (2 spatial dimensions) - effective for use with numpy arrays. k and l are NOT
meshgrids, they are 1d coordinates containing the total range of k and l (wavenumber) coordinates

Serves as basis of isotropize coordinates
"""

def calc_ispec(k,l,E):
    """ calculates isotropic spectrum from 2D spectrum """
    dk,dl = k[1]-k[0],l[1]-l[0]
    l,k = np.meshgrid(l,k)
    wv = np.sqrt(k**2 + l**2)
    if k.max()>l.max():
        kmax = l.max()
    else:
        kmax = k.max()
    # create radial wavenumber

    dkr = np.sqrt(dk**2 + dl**2)

    kr =  np.arange(dkr/2.,kmax+dkr,dkr) #you should ask Hector about this line 

    ispec = np.zeros(kr.size)
    #print(ispec.shape)

    #print(kr.shape)
    for i in range(kr.size):
        fkr =  (wv>=kr[i]-dkr/2) & (wv<=kr[i]+dkr/2)
    #    print(fkr.shape)
        dth = math.pi / (fkr.sum()-1)
        ispec[i] = E[fkr].sum() * kr[i] * dth

    return kr, ispec

"""
Isotropize does what calc_ispec does but for an xarray with some third (usually time or frequency) dimension.
This takes an xarray (spectra) as an input. Idims is a text array of dimensions needing to be isotropized.Ndim is the name of the time/freq dimension. This code can only handle an xarray with 3dims for
now

For the most part, this seems to work just as fast and effectively as xrft's isotropize
"""
def isotropize(spectra, idims, ndim):
    # spectra should just be the xarray containing your fourier transformed data
    # idims should be a list of the dimensions you are reducing (i.e. kx, ky or k,l)
    # ndim is the dimension you use calc_ispec over - usually going to be time
    # I don't know how to support more than one ndim at the moment so I won't bother 
    # additionally, this assumes that the dim order is {ndim, kdstr, lstr}
    kstr = idims[0]
    lstr = idims[1]
    k = spectra[kstr].values
    l = spectra[lstr].values
    t = spectra[ndim].values
    dk,dl = k[1]-k[0],l[1]-l[0]
    l,k = np.meshgrid(l,k)
    wv = np.sqrt(k**2 + l**2)
    if k.max()>l.max():
        kmax = l.max()
    else:
        kmax = k.max()
    # create radial wavenumber
    dkr = np.sqrt(dk**2 + dl**2)
    kr =  np.arange(dkr/2.,kmax+dkr,dkr) #you should ask Hector about this line 
    ispec = np.empty([t.size, kr.size])
    for index in range(0,len(t)):
        E = spectra[index,:,:].values
        #print(kr.shape)
        for i in range(kr.size):
            fkr =  (wv>=kr[i]-dkr/2) & (wv<=kr[i]+dkr/2)
            #    print(fkr.shape)
            dth = math.pi / (fkr.sum()-1)
            check = (E[fkr].sum() * kr[i] * dth)
            #print(len(check))
            ispec[index, i] = check
    isospectra = xr.DataArray(data=ispec, dims = [ndim, 'freq_r'], coords = [t, kr])
    #now you need to put it back together in a convenient xarray type form
    return isospectra


"""
this function takes MIT lat-lon-hour data and converts it to a uniform grid with length (km or m) and day coordinates.
You can choose to not convert from hour to day. 

Performs a spatial interpolation if required; return da_reg (km/m and days/hrs coordinates)
da - array with lat, lon, time (degrees and hours)
"""
def regularizeCoordinates(da,interp=None, timeunits='days', spaceunits ='km'):
    davals = da.values
    time = da.time.values
    lon = da.lon.values
    lat = da.lat.values
    lon_mesh,lat_mesh = np.meshgrid(lat, lon)
    e1,e2 = _e1e2(lon_mesh,lat_mesh)

    if ( spaceunits== 'km'):
        length_factor = 1000
    else:
        length_factor = 1
        
    if ( timeunits == 'days' ):
        time_factor = 24
    else:
        time_factor = 1
    y1d_in = (e1[0,:].cumsum() - e1[0,0] ) / length_factor # convert from m to km

    x1d_in = (e2[:,0].cumsum() - e2[0,0] ) / length_factor # convert from m to km
    
    da_met = xr.DataArray(data = davals, dims = ['time','xdim','ydim'], coords=[time / time_factor, x1d_in, y1d_in])
    
    if interp is not None:
        x1d_new = np.linspace(x1d_in.min(), x1d_in.max(), len(x1d_in))
        y1d_new = np.linspace(y1d_in.min(), y1d_in.max(), len(y1d_in))
        da_reg = da_met.interp(xdim = x1d_new, ydim = y1d_new, method=interp)
    #x2d_in,y2d_in = np.meshgrid(x1d_in,y1d_in)
    else:
        da_reg = da_met
   
    return da_reg #we can work on getting this to interpolate next

"""
this function takes MIT lat-lon-hour data and converts it to a uniform grid with length (km or m) coordinates.
This assumes you have model time units and so you leave those

Performs a spatial interpolation if required; return da_reg (km/m )
da - array with lat, lon, 
"""
def regularizeCoordinatesDateTime(da,interp=None, spaceunits ='km'):
    davals = da.values
    time = da.time.values
    lon = da.lon.values
    lat = da.lat.values
    lon_mesh,lat_mesh = np.meshgrid(lat, lon)
    e1,e2 = _e1e2(lon_mesh,lat_mesh)

    if ( spaceunits== 'km'):
        length_factor = 1000
    else:
        length_factor = 1
        
    y1d_in = (e1[0,:].cumsum() - e1[0,0] ) / length_factor # convert from m to km

    x1d_in = (e2[:,0].cumsum() - e2[0,0] ) / length_factor # convert from m to km
    
    da_met = xr.DataArray(data = davals, dims = ['time','xdim','ydim'], coords=[time, x1d_in, y1d_in])
    
    if interp is not None:
        x1d_new = np.linspace(x1d_in.min(), x1d_in.max(), len(x1d_in))
        y1d_new = np.linspace(y1d_in.min(), y1d_in.max(), len(y1d_in))
        da_reg = da_met.interp(xdim = x1d_new, ydim = y1d_new, method=interp)
    #x2d_in,y2d_in = np.meshgrid(x1d_in,y1d_in)
    else:
        da_reg = da_met
   
    return da_reg #we can work on getting this to interpolate next


"""
Converts lat and lon coordinates to meters - helper function for regularizeCoordinates
"""
def _e1e2(navlon,navlat):
    earthrad = 6371229     # mean earth radius (m)

    deg2rad = np.pi / 180.

    lam = navlon

    phi = navlat

    djlam,dilam = np.gradient(lam)

    djphi,diphi = np.gradient(phi)

    e1 = earthrad * deg2rad * np.sqrt( (dilam * np.cos(deg2rad*phi))**2. + diphi**2.)

    e2 = earthrad * deg2rad * np.sqrt( (djlam * np.cos(deg2rad*phi))**2. + djphi**2.)

    return e1,e2

def movingWindowAverage(xarraydata, dim, windowsize, lengthdim):
    windowsize = int(windowsize)
    if (lengthdim == 'xdim' or lengthdim == 'ydim'):
        chunks = xarraydata.chunk({"xdim": 100, "ydim": 100})
    elif (lengthdim == 'lat' or lengthdim == 'xdim'):
        chunks = xarraydata.chunk({"lat":100,"lon":100})
    xavg = chunks / windowsize
    for index in range(1,windowsize):
        if (dim =='time'):
            xavg += chunks.shift(time=-1*index, fill_value = 0) / windowsize
            # add other potential dimensions. xarray.shift doesn't allow us to pick dimensions in an easier way
        else:
            xavg += xavg
    timecoords = xavg.time
    timecoords_new = timecoords[int(windowsize / 2):-int(windowsize / 2)]
    xavg_new = xavg[0:-1*windowsize]
    xavg_new = xavg_new.assign_coords({"time": timecoords_new })
    return xavg_new

def coriolis(lat):
    omg = 1 / 24.0
    return 2*omg*np.sin((lat*3.14159)/180)