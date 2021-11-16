import os, glob, sys
import scipy.io
sys.path.append("//nobackup//amondal//Python//Hector_Python_Scripts")
import numpy as np
import xarray as xr
import dask.array as da
import netCDF4
import math

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
    isospectra = xr.DataArray(data=ispec, dims = [ndim, 'kr'], coords = [t, kr], name='Isotropic Spectra')
    #now you need to put it back together in a convenient xarray type form
    return isospectra