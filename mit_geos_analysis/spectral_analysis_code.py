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
    x1d_in = (e1[0,:].cumsum() - e1[0,0] ) / length_factor # convert from m to km

    y1d_in = (e2[:,0].cumsum() - e2[0,0] ) / length_factor # convert from m to km
    
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
        
    return xavg[0:-1*windowsize]
"""
Calculates isotropic cross spectra of quantities A and B with 3D coordinates (time, lat, lon or time, x, y)
A - input xarray DataArray
B - input xarray DataArray
--A and B should have the exact same coordinates!!!
reg = 'regularize' or NOT - - apply regularize coordinates to a lat lon grid or DON'T
idims = dims to isotropize (two spatial coordinates)
tdim = dim that you do not isotropize over (one time coordinate)
detrendVal = equivalent to "detrend_type" in xrft.detrend function
fftwindow = type of window you want to use on A and B in space and time
segmethod = either welch's method or bartlett's method
"""
#dev notes: segmethod is under development, I hope to do that in the near future
def isotropic_3d_cospectrum( A,B,idims,tdim, reg='regularize',timeunit='hours',lengthunit='km',detrendVal='linear', fftwindow='tukey',segmethod='bartlett',segnumber=5):
    import xrft
    # 1. regularize the coordinates
    if ( reg == 'regularize' ):
        Areg = regularizeCoordinates(A,'linear', timeunits = timeunit, spaceunits = lengthunit)
        Breg = regularizeCoordinates(B,'linear', timeunits = timeunit, spaceunits = lengthunit)
        rdims = ['xdim','ydim']
    else:
        Areg = A
        Breg = B
        rdims = idims
    
    #2. detrend - for now, xarray can't handle 3D detrending
    #2a detrend in space
    Areg_tp = xrft.detrend(Areg, dim = rdims, detrend_type = detrendVal)
    Breg_tp = xrft.detrend(Breg, dim = rdims, detrend_type = detrendVal)
    #2b detrend in time
    Areg_tp_sp = xrft.detrend(Areg_tp, dim = tdim, detrend_type = detrendVal)
    Breg_tp_sp = xrft.detrend(Breg_tp, dim = tdim, detrend_type = detrendVal)
    
    #3. Segmentation - For now, we'll leave it well alone. But it would be good to have Bartlett's or Welch's method here
    #4. Calculate cross spectrum with xrft
    ApBps=xrft.cross_spectrum(Areg_tp_sp, Breg_tp_sp , dim=list(Areg_tp_sp.dims), real_dim = tdim, true_amplitude=True, true_phase=True, window=fftwindow, window_correction=True) * 0.5
    ApBps=xr.apply_ufunc(np.real, ApBps)
    isodims = ["freq_" + d for d in rdims]
    fdim = "freq_" + tdim
    ApBps=ApBps.sortby(isodims)
    # there's a factor of two difference between xrft and hector's code, I'm not sure why - potentially a spurious factor
    # of two in xrft's fft code or something like that
    #5. Isotropize
   
    #ABstar_iso = xrft.xrft.isotropize(ABstar, isodims, truncate = True) 
    #I'm sure this could work, but Hector's matches his own code better, maybe I'll work on this later
    ABstar_iso = isotropize(ApBps, isodims, fdim)
    return ABstar_iso

