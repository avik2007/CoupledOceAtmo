"""
  This python script is to manipulate llc_hires_simulation
"""

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
                 rate = None,
                 timedelta = None,
                 dtype = '>f4',
                 ):
        self.grid_dir = grid_dir
        self.data_dir = data_dir
        self.Nlon = Nlon
        self.Nlat = Nlat
        self.Nz = Nz
        self.tini = tini
        self.tend = tend
        self.tref = tref
        self.timedelta=timedelta
        self.dt = dt  #datetime.timedelta(min=dt)
        self.rate = rate
        self.dtype = '>f4'
        self.grid_size = str(Nlon)+'x'+str(Nlat)
        
    def load_grid(self):
        self.lon = np.memmap(self.grid_dir+'LONC.bin',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')
        self.lat = np.memmap(self.grid_dir+'LATC.bin',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')
        self.long = np.memmap(self.grid_dir+'LONG.bin',
				dtype=self.dtype,shape=(self.Nlat,self.Nlon),
				mode='r')
        self.latg = np.memmap(self.grid_dir+'LATG.bin',
				dtype=self.dtype,shape=(self.Nlat,self.Nlon),
				mode='r')
        data = sci.loadmat('/nobackup/htorresg/DopplerScat/modelling/GS/programs/tools/thk90.mat')
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
        self.maskv = np.memmap(self.grid_dir+'hFacS.data',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r',offset=(self.Nlat*self.Nlon*4)*layer)
        self.masku = np.memmap(self.grid_dir+'hFacW.data',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r',offset=(self.Nlat*self.Nlon*4)*layer)

    def load_rac(self):
        self.rac = np.memmap(self.grid_dir+'RAC.data',
                             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                             mode='r')
    def load_raz(self):
        self.raz = np.memmap(self.grid_dir+'RAZ.bin',
                              dtype=self.dtype,shape=(self.Nlat,self.Nlon),
                              mode='r')

    def load_ras(self):
        self.ras = np.memmap(self.grid_dir+'RAS.bin',
		             dtype=self.dtype,shape=(self.Nlat,self.Nlon),
				mode='r')

    def load_raw(self):
        self.raw = np.memmap(self.grid_dir+'RAW.bin',
				dtype=self.dtype,shape=(self.Nlat,self.Nlon),
				mode='r')

    def load_rf(self):
        self.rf = np.memmap(self.grid_dir+'RF.data',
                             dtype=self.dtype,shape=(self.Nz),mode='r')
    
    def load_drf(self):
         self.drf = np.memmap(self.grid_dir+'DRF.data',
                              dtype=self.dtype,shape=(self.Nz),mode='r')
 
    def loadding_3D_data(self,fni,maxlevel,type):
        """ Reading 3D files and returning a 3D array
             from surface to maxlevel """
        data = np.empty((maxlevel,self.Nlat,self.Nlon))
        for i in range(0,maxlevel):
            var = np.memmap(fni,dtype=self.dtype,
                         shape=(self.Nlat,self.Nlon),mode='r+',
                            offset=(((self.Nlat)*(self.Nlon))*4)*i)
            if type == 'tracer':
                maskc = np.memmap(self.grid_dir+'hFacC.data',
                                  dtype=self.dtype,
                         shape=(self.Nlat,self.Nlon),mode='r',
                            offset=(((self.Nlat)*(self.Nlon))*4)*i)
                var[maskc==0] = np.nan
                #var = var*maskc
                var = np.ma.masked_invalid(var)
            elif type == 'uvel':
                masku = np.memmap(self.grid_dir+'hFacW.data',
                                  dtype=self.dtype,
                         shape=(self.Nlat,self.Nlon),mode='r',
                            offset=(((self.Nlat)*(self.Nlon))*4)*i)
                var[masku==0] = np.nan
                #var = var*masku
                var = np.ma.masked_invalid(var)
            elif type == 'vvel':
                maskv = np.memmap(self.grid_dir+'hFacS.data',
                                  dtype=self.dtype,
                         shape=(self.Nlat,self.Nlon),mode='r',
                            offset=(((self.Nlat)*(self.Nlon))*4)*i)
                var[maskv==0]=np.nan
                var = var*maskv
                var = np.ma.masked_invalid(var)
            data[i,:,:] = var
        return data

    def loadding_3D_masks(self,fni,maxlevel):
        """
        Loadding 3D masks
        """
        data = np.empty((maxlevel,self.Nlat,self.Nlon))
        for i in range(0,maxlevel):
            data[i,:,:]=np.memmap(self.grid_dir+fni,
                                  dtype=self.dtype,
                                  shape=(self.Nlat,self.Nlon),mode='r',
                                  offset=(((self.Nlat)*(self.Nlon))*4)*i)
        return data

    def load_2d_data(self,fni,type):
        """ Reading 2D files """
        var = np.memmap(fni,dtype=self.dtype,
                         shape=(self.Nlat,self.Nlon),mode='r+')
        if type == 'tracer':
            maskc = np.memmap(self.grid_dir+'hFacC.data',
                                  dtype=self.dtype,
                         shape=(self.Nlat,self.Nlon),mode='r',
                              offset=(((self.Nlat)*(self.Nlon))*4)*0)
            var[maskc==0] = np.nan
        elif type == 'uvel':
            masku = np.memmap(self.grid_dir+'hFacW.data',
                                  dtype=self.dtype,
                         shape=(self.Nlat,self.Nlon),mode='r',
                              offset=(((self.Nlat)*(self.Nlon))*4)*0)
            var[masku==0] = np.nan
        elif type == 'vvel':
            maskv = np.memmap(self.grid_dir+'hFacS.data',
                                  dtype=self.dtype,
                         shape=(self.Nlat,self.Nlon),mode='r',
                              offset=(((self.Nlat)*(self.Nlon))*4)*0)
            var[maskv==0] = np.nan
        return var

    def load_3d_data(self,fni,layer,type):
        """ Reading 3D files and returning a 2D array
             corresponding to the layer given """
        var =  np.memmap(fni,dtype=self.dtype,
                         shape=(self.Nlat,self.Nlon),mode='r+',
                         offset=(((self.Nlat)*(self.Nlon))*4)*layer)
        if type == 'tracer':
            var[self.maskc==0] = np.nan
        elif type == 'uvel':
            var[self.masku==0] = np.nan
        elif type == 'vvel':
            var[self.maskv==0] = np.nan
        return var

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
        self.date=np.arange(tim0,tim1+timedelta(hours=self.timedelta),
                            timedelta(hours=self.timedelta)).astype(datetime.datetime)
        steps = int((tim0-tref).total_seconds()/self.rate)##self.rate
        i0 = steps*self.dt
        del steps
        steps = int((tim1-tim0).total_seconds()/self.rate) ##self.rate
        i1 = i0+steps*self.dt
        self.timesteps = np.arange(i0,i1+self.dt,self.dt)
    
    def depth2level(self,depth):
        """Return the z-level corresponding to a particular depth [m] """
        izp = (np.abs(self.dpt-depth)).argmin()
        return izp

        
    def from_2D_to_3D_array(self,var,nz):
        """
        var is a 2D variable [x,y] and is extended to
        a 3D var [z,x,y]
        """
        import numpy as np
        var = var[None,:,:]
        return  np.tile(var,(nz,1,1))

    def compute_pressure(self,gsw_path,m,n,nz,depth,lat):
        """
        Compute pressure from depth [m] and latitude.
        The seawater packages is used
        Input parameters:
        m,n = size of the output
        nz = vertical levels
        """
        import sys 
        import numpy as np
        sys.path.append(gsw_path)
        import gsw as sw
        pp = np.zeros((nz,m))
        print('m,n: ',[m,n])
        print('Depth.shape: ',depth.shape)
        print('lat.shape: ',lat.shape)
        ###
        if nz == 0:
           for j in range(m):
              prc = sw.p_from_z(-depth,lat[j,0])	
        else:
           for j in range(m):
              pp[:,j]=sw.p_from_z(-depth[0:nz],lat[j,0])
           prc=np.ones((nz,m,n))*pp.reshape(nz,m,1)
        return prc
        
    def compute_dens(self,salt,theta,pressure):
        """
        Input parameters:
        salt := salinity 
        theta := potential temperature [deg C]
        depth := depth [m]
        """
        import dens
        if salt.ndim == 3:
        	self.dens = np.empty((salt.shape))
        	for i in range(salt.shape[0]):
            		self.dens[i,:,:] = dens.densjmd95(salt[i,:,:],
                               	theta[i,:,:],pressure[i,:,:])
        elif salt.ndim == 2:
                self.dens = dens.densjmd95(salt,theta,pressure)

    
    def compute_buoyancy(self,rho):
        g = 9.81 # acceleration of gravity
        rho_ref = np.nanmean(rho)
        rho_prime = rho
        print('++++ rho-ref +++')
        print(rho_ref)
        b = -g*(rho_prime/rho_ref)
        return b,rho_ref

    def compute_vorticity(self,u,v,dxc,dyc,raz):
        """
        Compute relative vorticity in flux-form 
        """
        import numpy as np
        vort = np.zeros(u.shape)
        if u.ndim == 3:
            vix = v[:,:,:-1]*dyc[None,:,:-1]
            vox = v[:,:,1:] *dyc[None,:,1:]
            vx = vix - vox
            ###
            uiy = u[:,:-1,:]*dxc[None,:-1,:]
            uoy = u[:,1:,:] *dxc[None,1:,:]
            uy = uiy - uoy
            vort = -(vx[:,1:,:] - uy[:,:,1:])/(raz[None,:-1,:-1])
        elif u.ndim == 2:
            vix = v[:,:-1]*dyc[:,:-1]
            vox = v[:,1:] *dyc[:,1:]
            vx = vix - vox
            ###
            uiy = u[:-1,:]*dxc[:-1,:]
            uoy = u[1:,:] *dxc[1:,:]
            uy = uiy - uoy
            vort = -(vx[1:,:] - uy[:,1:])/(raz[:-1,:-1])
        return vort   
        
    def calculate_uv_gradients_xy(self,uvel,vvel,dxg,
                                  dyg,dxc,dyc,dxf,dyf,dxv,dyu):
        import numpy as np
        """
        Calculates du/dx, du/dy, dv/dx, dv/dy
        at T points if u,v are given at U,V points in Arakawa C-grid
        """

        dudx_xy = np.zeros(uvel.shape)
        dvdx_xy = np.zeros(uvel.shape)
        dudy_xy = np.zeros(uvel.shape)
        dvdy_xy = np.zeros(uvel.shape)
        
        if uvel.ndim == 3:
            for k in range(0,uvel.shape[0]):
                dudx_xy[k,1:-1,1:-1] = (uvel[k,1:-1,1:-1] - uvel[k,1:-1,0:-2]) / dxg[1:-1,1:-1]
                dudy_xy[k,1:-1,1:-1] = 0.5 * ( (uvel[k,2:,1:-1] - uvel[k,0:-2,1:-1]) / (2*dyu[1:-1,1:-1]) + (uvel[k,2:,0:-2] - uvel[k,0:-2,0:-2]) / (2*dyu[1:-1,1:-1]) )
                dvdx_xy[k,1:-1,1:-1] = 0.5 * ( (vvel[k,1:-1,2:] - vvel[k,1:-1,0:-2]) / (2*dxc[1:-1,1:-1]) + (vvel[k,0:-2,2:] - vvel[k,0:-2,0:-2]) / (2*dxc[1:-1,1:-1]) )
                dvdy_xy[k,1:-1,1:-1] = (vvel[k,1:-1,1:-1] - vvel[k,0:-2,1:-1]) / dyf[1:-1,1:-1]

        data = {}
        data['dudx_xy'] = dudx_xy
        data['dvdx_xy'] = dvdx_xy
        data['dudy_xy'] = dudy_xy
        data['dvdy_xy'] = dvdy_xy
        vort = dvdx_xy - dudy_xy
        delta  = dudx_xy + dvdy_xy
        sigma_n = dudx_xy - dvdy_xy
        sigma_s = dvdx_xy + dudy_xy
        strain = np.sqrt(sigma_s**2 + sigma_n**2)
        return data,vort,delta,strain

    def gradT(self,T,dxc,dyc):
        """"
        Calculate dt/dx and dt/dy at T points
        using centered difference scheme
        """
        dtdx = np.zeros(T.shape)
        dtdy = np.zeros(T.shape)
        if T.ndim == 3:
            dtdx[:,1:-1,1:-1] = (T[:,1:-1,2:]-T[:,1:-1,0:-2])/(2*dxc[None,1:-1,1:-1])
            dtdy[:,1:-1,1:-1] = (T[:,2:,1:-1]-T[:,0:-2,1:-1])/(2*dyc[None,1:-1,1:-1])
            # Force to zero at boundaries
            dtdy[:,0,:] = 0.;dtdy[:,:,0] = 0.
            dtdy[:,-1,:]= 0.;dtdy[:,:,-1] = 0.
            dtdx[:,0,:]=0.;dtdx[:,:,0]=0.
            dtdx[:,:,-1]=0.;dtdx[:,-1,:]=0.
        data = {}
        data['dtdx'] = dtdx
        data['dtdy'] = dtdy
        data['mag_grad'] = np.sqrt(dtdx**2 + dtdy**2)
        return data
