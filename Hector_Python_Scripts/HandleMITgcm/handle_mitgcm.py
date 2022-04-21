"""
  This python script is to manipulate llc_hires_simulation
"""

import pylab as plt
import numpy as np
import sys
import datetime
import math
import scipy.io as sci

class LLChires:
    """ A class created in order to handle the MITgcm simulation """

    def __init__(self,
        	 grid_dir = None,
		 data_dir = None,
                 Nlon = None,
      		 Nlat = None,
                 Nz = None,
		 tini = None,
                 tref = None,
                 tend = None,
                 dt    = None, # time step in second
                 dtype = np.dtype('>f4'),
                 ):
        self.grid_dir = grid_dir
        self.data_dir = data_dir
        self.Nlon = Nlon
        self.Nlat = Nlat
        self.Nz = Nz
        self.tini = tini
        self.tend = tend
        self.tref = tref
        self.dt = dt#datetime.timedelta(min=dt)
        self.dtype = dtype
        self.grid_size = str(Nlon)+'x'+str(Nlat)

    def load_grid(self):
        self.lon = np.memmap(self.grid_dir+'XC.data',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')
        self.lat = np.memmap(self.grid_dir+'YC.data',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')
        data = sci.loadmat('/u/dmenemen/llc_4320/grid/thk90.mat')
        self.dpt = data['dpt90'][0:self.Nz].squeeze()
        self.thk = data['thk90'][0:self.Nz].squeeze()

    def load_depth(self):
        self.depth = np.memmap(self.grid_dir+'Depth.data',
                               dtype=self.dtype,shape=(self.Nlat,self.Nlon),                                   mode='r')

    def load_dxc(self):
        self.dxc = np.memmap(self.grid_dir+'DXC.bin',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')
        self.dyc = np.memmap(self.grid_dir+'DYC.bin',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')

    def load_dxf(self):
        self.dxf = np.memmap(self.grid_dir+'DXF.bin',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')
        self.dyf = np.memmap(self.grid_dir+'DYF.bin',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')

    def load_dxg(self):
        self.dxg = np.memmap(self.grid_dir+'DXG.bin',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')
        self.dyg = np.memmap(self.grid_dir+'DYG.bin',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')

    def load_duv(self):
        self.dxv = np.memmap(self.grid_dir+'DXV.bin',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')
        self.dyu = np.memmap(self.grid_dir+'DYU.bin',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')

    def load_mask(self,layer):
        self.maskc = np.memmap(self.grid_dir+'hFacC.data',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r',offset=(self.Nlat*self.Nlon*4)*layer)
        self.masks = np.memmap(self.grid_dir+'hFacS.data',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r',offset=(self.Nlat*self.Nlon*4)*layer)
        self.maskw = np.memmap(self.grid_dir+'hFacW.data',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r',offset=(self.Nlat*self.Nlon*4)*layer)

    def load_rac(self):
        self.rac = np.memmap(self.grid_dir+'RAC.data',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')
    def load_rf(self):
        self.rf = np.memmap(self.grid_dir+'RF.data',
                             dtype=self.dtype,shape=(self.Nz),mode='r')
    
    def load_drf(self):
         self.drf = np.memmap(self.grid_dir+'DRF.data',
                              dtype=self.dtype,shape=(self.Nz),mode='r')
 
    def load_2d_data(self,fni):
        """ Reading 2D files """
        return np.memmap(fni,dtype=self.dtype,
                         shape=(self.Nlat,self.Nlon),mode='r')

    def load_3d_data(self,fni,layer):
        """ Reading 3D files and returning a 2D array
             corresponding to the layer given """
        return np.memmap(fni,dtype=self.dtype,
                         shape=(self.Nlat,self.Nlon),mode='r',
                         offset=(((self.Nlat)*(self.Nlon))*4)*layer)

    def coriolis(self):
        """ Coriolis parameter f_{o}"""
        self.f = 2*(7.2921*10**-5)*np.sin((self.lat*np.pi)/180)
 
    def timeline(self):
        """ Return timesteps corresponding to 
            the period selected """
        from dateutil.parser import parse
        from datetime import timedelta
        tim0 = parse(self.tini)
        tim1 = parse(self.tend)
        tref = parse(self.tref)
        self.date=np.arange(tim0,tim1,
                  timedelta(hours=1)).astype(datetime.datetime)
        steps = int((tim0-tref).total_seconds()/3600)
        i0 = steps*self.dt
        del steps
        steps = int((tim1-tim0).total_seconds()/3600)
        i1 = i0+steps*self.dt
        self.timesteps = np.arange(i0,i1,self.dt)
