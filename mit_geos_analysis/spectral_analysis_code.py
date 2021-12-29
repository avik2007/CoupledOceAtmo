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
    isospectra = xr.DataArray(data=ispec, dims = [ndim, 'kr'], coords = [t, kr])
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
def isotropic_3d_cospectrum(A, B, reg = 'regularize',timeunit = 'hours', lengthunit = 'km', idims, tdim, detrendVal = 'linear', fftwindow = 'tukey', segmethod = 'bartlett', segnumber)
    import xrft
    # 1. regularize the coordinates
    if ( reg == 'regularize' ):
        Areg = regularizeCoordinates(A,'linear', timeunits = timeunit, spaceunits = lengthunit)
        Breg = regularizeCoordinates(B,'linear', timeunits = timeunit, spaceunits = lengthunit)
    else:
        Areg = A
        Breg = B
        
    #2. detrend - for now, xarray can't handle 3D detrending
    #2a detrend in space
    Areg_tp = xrft.detrend(Areg, dim = idims, detrend_type = detrendVal)
    Breg_tp = xrft.detrend(Breg, dim = idims, detrend_type = detrendVal)
    #2b detrend in time
    Areg_tp_sp = xrft.detrend(Areg_tp, dim = tdim, detrend_type = detrendVal)
    Breg_tp_sp = xrft.detrend(Breg_tp, dim = tdim, detrend_type = detrendVal)
    
    #3. calculate fft
    #3a fft in space - assumes real fft and existence of a window and truncation
    Ahat_kl = xrft.xrft.fft(Areg_tp_sp, dim = idims, real_dim = idims[0], window = fftwindow, window_correction = True, true_amplitude = True, truncate = True, )
    Bhat_kl = xrft.xrft.fft(Breg_tp_sp, dim = idims, real_dim = idims[0], window = fftwindow, window_correction = True, true_amplitude = True, truncate = True, )
    
    #3b fft in time
    if ( segnumber < 2 ):
        segmethod = 'bartlett'
        
    if ( segmethod = 'bartlett' ):
        An = Ahat_kl[tdim].size
        Bn = Bhat_kl[tdim].size
        Ahat_kl_om = xrft.xrft.fft(Ahat_kl.chunk({'time':int(An / segnumber)}), dim = tdim, real_dim = tdim, window = fftwindow, window_correction = True, true_amplitude = True, truncate = True, chunks_to_segments=True).compute()
        Bhat_kl_om = xrft.xrft.fft(Bhat_kl.chunk({'time':int(Bn / segnumber)}), dim = tdim, real_dim = tdim, window = fftwindow, window_correction = True, true_amplitude = True, truncate = True, chunks_to_segments=True).compute()
    #else: 
    # segmethod = 'welch' is under development but should become the default segmentation / window of choice
    
    #4. calculate conjugate
    Bhat_kl_om_star = xr.apply_ufunc(np.conjugate, Bhat_kl_om) #np.conjugate(Bhat_kl_om)
    #DEVELOPMENT NOTE: CHECK IF THERE'S A WAY TO CHUNK apply_ufunc or set "parallelize" to ON
    
    #5 calculate cospectrum
    ABstar = Ahat_kl_om * Bhat_kl_om_star
    ABstar = xr.apply_ufunc(np.real, ABstar.mean(dims = 'time_segment') )
    
    #6. Isotropize
    isodims = ["freq_" + d for d in idims]
    ABstar_iso = xrft.xrft.isotropize(ABstar, isodims, truncate = True)