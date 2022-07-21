import os, glob, sys
sys.path.append("//nobackup//amondal//Python//mit_geos_analysis")
import numpy as np
import xarray as xr
import xrft
import netCDF4

#calculates level values from virtual temperature and pressure 
def getLevelThicknesses(Tv,P):
    epsilon=0.622
    g0 = 9.8#m/s^2
    Rd=287.058#J/(kg*K)
    Tvs = Tv.shift(levs=-1,fill_value=0)
    Pcs = Pc.shift(levs=-1,fill_value=0)
    levels = -Rd/g0*((Tv+Tvs)/2 * (xr.apply_ufunc(np.log,Pc) - xr.apply_ufunc(np.long,Pcs))
    levels = levels[:,:,:,0:levels.levs.size-1]
    return levels

#calculates heights of the levels
def getLevelHeights(levels);
    height = levels
    for index in range(1,levels.levs.size):
        height += levels.shift(levs=-1*index,fill_value=0)
    zeroslayer = levels[:,:,:,0:1] - Tv[:,:,:,0:1]
    height = xr.concat([height,zeroslayer],dim='levs')
    return height

#get the thickness of a level at a certain lat and lon
def getLevelThickness(levels,timesel,latsel,lonsel,levsel):
    return levels.sel(time=timesel, lats=latsel,lons=lonsel, levs=levsel)

#get the thickness of a level over a range of lats and lons
def getLevelThicknesses(levels,timesel, levsel):
    return levels.sel(time=timesel, levs=levsel)

#get the height of a level
def getLevelHeight(heights,timesel, latsel,lonsel,levsel):
    return heights.sel(time=teimsel,lats=latsel,lons=lonsel,levs=levsel)

#get the height of a level over a range of lats and lons
def getLevelHeights(heights,timsel,levsel):
    return heights.sel(time=timesel, levs=levsel) 

#given a height, the function will tell you what level that height is in
def getLevel(heights, timesel,latsel,lonsel,height):
    hs = heights.sel(time=timesel, lats = latsel, lons=lonsel)
    zdim = xr.where(hs < height, 1, 0).sum(dim='levs')
    return zdim

# return thick weighted vector 
