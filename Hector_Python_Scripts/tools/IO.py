# -*- coding: utf-8 -*-
import sys,os,glob
import numpy as np
from netCDF4 import Dataset,date2num
# =====================

def readNC(path_name,mode):
    dat = Dataset(path_name,mode,format='NETCDF4_CLASSIC')
    return dat
