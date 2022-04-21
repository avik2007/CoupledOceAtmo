# -*- coding: utf-8 -*-
"""
Created on Sep 2018

@author: htorresg
"""
import sys,os,glob
import numpy as np
from netCDF4 import Dataset,date2num
# ==================

def out4D_nc(namefile,varname,data,x,y,z,dates,description,source,time_units):
    t,l,m,n=data.shape
    dat=Dataset(namefile,'w',format='NETCDF4_CLASSIC')
    # Dimensions
    level=dat.createDimension('level',l)
    lat=dat.createDimension('lat',m)
    lon=dat.createDimension('lon',n)
    time=dat.createDimension('time',t)
    # Create coordinates variable for 4D
    times=dat.createVariable('time','f8',('time',))
    levels=dat.createVariable('level','i4',('level',))
    latitudes=dat.createVariable('lat','f4',('lat',))
    longitudes=dat.createVariable('lon','f4',('lon',))
    # Create the actual 4D variable
    tem=dat.createVariable(varname,'f4',('time','level','lat','lon'))
    # putting in place
    # putting in place
    latitudes[:]=y
    longitudes[:]=x
    levels[:]=z
    tem[:,:,:,:]=data
    # attributes
    dat.description = description
    #dat.history = 'Created ' + time.ctime(time.time())
    dat.source = source
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    times.units = time_units
    times.calendar = 'gregorian'
    times[:]=date2num(dates,units=times.units,calendar=times.calendar)
    dat.close()    
    #dates=[datetime(2012,2,2)+n*timedelta(hours=1) 
    #for n in range(mat.shape[0])]

def outxyz_nc(namefile,varname,data,x,y,z,description,source):
    l,m,n=data.shape
    dat=Dataset(namefile,'w',format='NETCDF4_CLASSIC')
    # Dimensions
    level=dat.createDimension('level',l)
    lat=dat.createDimension('lat',m)
    lon=dat.createDimension('lon',n)
    # Create coordinates variable for 3D(x,y,z)
    levels=dat.createVariable('level','i4',('level',))
    latitudes=dat.createVariable('lat','f4',('lat',))
    longitudes=dat.createVariable('lon','f4',('lon',))
    # Create the actual 4D variable
    tem=dat.createVariable(varname,'f4',('level','lat','lon'))
    latitudes[:]=y
    longitudes[:]=x
    levels[:]=z
    tem[:,:,:,:]=data
    # attributes
    dat.description = description
    #dat.history = 'Created ' + time.ctime(time.time())
    dat.source = source
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    dat.close()

def outxyt_nc(namefile,varname,data,x,y,dates,description,source,time_units):
    t,m,n=data.shape
    dat=Dataset(namefile,'w',format='NETCDF4_CLASSIC')
    # Dimensions
    lat=dat.createDimension('lat',m)
    lon=dat.createDimension('lon',n)
    time=dat.createDimension('time',t)
    # Create coordinates variable for 4D
    times=dat.createVariable('time','f8',('time',))
    latitudes=dat.createVariable('lat','f4',('lat',))
    longitudes=dat.createVariable('lon','f4',('lon',))
    # Create the actual 4D variable
    tem=dat.createVariable(varname,'f4',('time','lat','lon'))
    # putting in place
    # putting in place
    latitudes[:]=y
    longitudes[:]=x
    tem[:,:,:]=data
    # attributes
    dat.description = description
    #dat.history = 'Created ' + time.ctime(time.time())
    dat.source = source
    latitudes.units = 'degrees north'
    longitudes.units = 'degrees east'
    times.units = time_units
    times.calendar = 'gregorian'
    times[:]=date2num(dates,units=times.units,calendar=times.calendar)
    dat.close()    
    #dates=[datetime(2012,2,2)+n*timedelta(hours=1) 
    #for n in range(mat.shape[0])]
