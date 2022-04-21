from astropy.convolution import (Gaussian2DKernel, CustomKernel,
                                 interpolate_replace_nans,
                                  convolve)

def find_index(xx,yy,ln,lt,dx):
    import numpy as np
    """ Extracting small domain from a bigger one.
       # ilonmin,ilonmax,ilatmin,ilatmax=find_index(xx,yy,ln,lt,dx)
       # var=var[ilatmin[0]:ilatmax[0],ilonmin[0]:ilonmax[0]]
    """
    if ln < 0:
       ln = ln + 360
    else:
       ln= ln
    
    lowlon=xx[1,:]-ln
    uplon =xx[1,:]-(ln+dx)
    lowlat=yy[:,1]-lt
    uplat=yy[:,1]-(lt+dx)
    
    ilonmin=np.where(np.abs(lowlon)==np.abs(lowlon).min())
    ilonmax=np.where(np.abs(uplon)==np.abs(uplon).min())
    ilatmin=np.where(np.abs(lowlat)==np.abs(lowlat).min())
    ilatmax=np.where(np.abs(uplat)==np.abs(uplat).min())
    return ilonmin,ilonmax,ilatmin,ilatmax

def find_corners(y_array, x_array, y_point, x_point):
    import numpy as np
    distance = (y_array-y_point)**2 + (x_array-x_point)**2
    idy,idx = np.where(distance==distance.min())
    return idy[0],idx[0]

def extract_subdomain(lon,lat,a_lon,a_lat):
    cor_sw2,cor_ws2 = find_corners(lon,lat,a_lon[0],a_lat[0])
    cor_se2,cor_es2 = find_corners(lon,lat,a_lon[1],a_lat[0])
    cor_ne2,cor_en2 = find_corners(lon,lat,a_lon[1],a_lat[1])
    cor_nw2,cor_wn2 = find_corners(lon,lat,a_lon[1],a_lat[1])
    return cor_sw2,cor_ne2,cor_ws2,cor_en2

def get_subdomain(var,lon,lat,a_lon,a_lat):
    ilnmn,ilnmx,iltmn,iltmx=extract_subdomain(lon,lat,a_lon,a_lat)
    lon = lon[ilnmn:ilnmx,iltmn:iltmx]
    lat = lat[ilnmn:ilnmx,iltmn:iltmx]
    var = var[ilnm:ilnmx,iltmn:iltmx]
    return lon,lat,var

def winParzen(n):
    from scipy import signal
    window = signal.parzen(n)
    window = window ##/window.sum()
    return window

def Parze2Dfilter(size,dx,var):
    import numpy as np
    """
    dx is m x n array
    var is m x n array
    """
    m,n = var.shape
    var_filt = np.full((var.shape),np.nan)
    var = np.ma.masked_where(var==0.,var)
    for i in range(0,m):
        res = dx[i,:]
        r = res.mean()
        L = np.round(size/r)
        w = winParzen(L)
        var_filt[i,:]=np.convolve(var[i,:],w,'same')
    for i in range(0,n):
        res = dx[:,i]
        r = res.mean()
        L = np.round(size/r)
        w = winParzen(L)
        var_filt[:,i] = np.convolve(var[:,i],w,'same')
    return var

def rms(x,ax=None):
    import numpy as np
    return np.sqrt(np.nanmean(x**2,axis=ax))

def lonlat2xy(lon,lat):
    """convert lat lon to y and x
    x, y = lonlat2xy(lon, lat)
    lon and lat are 1d variables.
    x, y are 2d meshgrids.
    """
    from pylab import meshgrid,cos,pi
    r = 6371.e3
    #lon = lon-lon[0]
    if lon.ndim == 1:
        lon,lat = meshgrid(lon,lat)
    x = 2*pi*r*cos(lat*pi/180.)*lon/360.
    y = 2*pi*r*lat/360.
    return x,y


def fit2Dsurf(x,y,p):
    """
      given y0=f(t0), find the best fit
      p = a + bx + cy + dx**2 + ey**2 + fxy
      and return a,b,c,d,e,f
    """
    from scipy.optimize import leastsq
    import numpy as np
    x,y=abs(x),abs(y)
    def err(c,x0,y0,p):
        a,b,c,d,e,f=c
        return p - (a + b*x0 + c*y0 + d*x0**2 + e*y0**2 + f*x0*y0)
    def surface(c,x0,y0):
        a,b,c,d,e,f=c
        return a + b*x0 + c*y0 + d*x0**2 + e*y0**2 + f*x0*y0
    dpdy = (np.diff(p,axis=0)/np.diff(y,axis=0)).mean()
    dpdx = (np.diff(p,axis=1)/np.diff(x,axis=1)).mean()
    xf=x.flatten()
    yf=y.flatten()
    pf=p.flatten()
    c = [pf.mean(),dpdx,dpdy,1e-22,1e-22,1e-22]
    coef = leastsq(err,c,args=(xf,yf,pf))[0]
    vm = surface(coef,x,y) #mean surface
    va = p - vm #anomaly
    return va,vm



def get_kernel_sigma(delta_in,delta_out):
    """Get the Gaussian filter standard deviation desired
    to get a kernel with a full width 1/2 power
    width of delta_out, when delta_in is the pixel size.
    """
    import numpy as np
    sigma_out = delta_out/np.sqrt(2*np.log(2))/2

    return sigma_out/delta_in # in pixels


def filt_single(u,delta_out=5000.,delta_in=5000.,keep_nan=True):
    """Filter the velocity components to the desired spatial
    resolution, given a data set where the mask has been set."""
    sz = get_kernel_sigma(delta_in,delta_out)
    print(sz)
    kernel = Gaussian2DKernel(sz)

    u = convolve(u,kernel,nan_treatment='interpolate',preserve_nan=keep_nan)

    return u


def savenetcdf4(v, fn = '', ndims=4, lons=[], lats=[], levels=[], records=[]):
        """ save numpy array to netcdf file
        v={vname:vvalue, vname:vvalue}
        """
        import sys, os
        import time as mod_time
        from netCDF4 import Dataset

        # if fn =='':
        #     fn = sys.argv[0].split('.')[0]+'_tmp.nc'
        if os.path.exists(fn):
            rootgrp = Dataset(fn, 'r+', format='NETCDF4')
            for varname in v.keys():
                if ndims == 4:
                    rootgrp.createVariable(varname,'f8',
                                           ('time','depth','lat','lon',), 
                                           fill_value=99999)[:] = v[varname][:]
                elif ndims == 3:
                    rootgrp.createVariable(varname,'f8',
                                           ('depth','lat','lon',), 
                                           fill_value=99999)[:] = v[varname][:]
                else:
                    rootgrp.createVariable(varname,'f8',('lat','lon',), 
                                           fill_value=99999)[:] = v[varname][:]
            rootgrp.close()

        else:
            rootgrp = Dataset(fn, 'w', format='NETCDF4')
            rootgrp.history = 'Created by '+sys.argv[0] + mod_time.ctime(mod_time.time())

            if len(levels) !=0:
                depth = rootgrp.createDimension('depth', len(levels))
                depths = rootgrp.createVariable('depth','f8',('depth',))
                depths[:] = levels
            if len(records)!=0:
                time = rootgrp.createDimension('time', None)
                times = rootgrp.createVariable('time','f8',('time',))
                times.units = 'hours since 0001-01-01 00:00:00.0'
                times.calendar = 'gregorian'
                times[:] = records
            rootgrp.createDimension('lat', len(lats))
            latitude = rootgrp.createVariable('lat','f8',('lat',))
            latitude.units = 'degrees north'
            latitude[:] = lats[:]

            rootgrp.createDimension('lon', len(lons))
            longitude = rootgrp.createVariable('lon','f8',('lon',))
            longitude.units = 'degrees east'
            longitude[:] = lons[:]


            for varname in v.keys():
                if ndims == 4:
                    rootgrp.createVariable(varname,'f8',
                                           ('time','depth','lat','lon',), 
                                           fill_value=99999)[:] = v[varname][:]
                elif ndims == 3:
                    rootgrp.createVariable(varname,'f8',
                                           ('depth','lat','lon',), 
                                           fill_value=99999)[:] = v[varname][:]
                else:
                    rootgrp.createVariable(varname,'f8',
                                           ('lat','lon',),
                                           fill_value=99999)[:] = v[varname][:]
            rootgrp.close()

def add_nc_var(v,fn,ndims=2):
    if os.path.exists(fn):
        rootgrp = Dataset(fn, 'r+', format='NETCDF4')
        for varname in v.keys():
            if ndims == 4:
                rootgrp.createVariable(varname,'f8',
                                       ('time','depth','lat','lon',), 
                                       fill_value=99999)[:] = v[varname][:]
            elif ndims == 3:
                rootgrp.createVariable(varname,'f8',
                                       ('depth','lat','lon',), 
                                       fill_value=99999)[:] = v[varname][:]
            else:
                rootgrp.createVariable(varname,'f8',
                                       ('lat','lon',), 
                                       fill_value=99999)[:] = v[varname][:]
        rootgrp.close()
    else:
        print('File does not exist')
        # break
