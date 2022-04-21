import numpy as np
import sys
import tools
import handle_mitgcm as model
from netCDF4 import Dataset
import basic_calculations_MITgcm as basics
import EPV_FluxForm as epv

def v2rho(v):
    out = np.empty(v.shape)
    if v.ndim == 3:
        out[:,1:,:] = 0.5*(v[:,1:,:]+v[:,:-1,:])
        out[:,0,:] = out[:,1,:]
    elif v.ndim == 2:
        out[1:,:] = 0.5*(v[1:,:]+v[:-1,:])
        out[0,:] = out[1,:]       
    return out

def u2rho(u):
    out = np.empty(u.shape)
    if u.ndim == 3:
        out[:,:,1:] = 0.5*(u[:,:,1:]+u[:,:,:-1])
        out[:,:,0] = out[:,:,1]
    elif u.ndim ==2:
        out[:,1:] = 0.5*(u[:,1:]+u[:,:-1])
        out[:,0] = out[:,1]        
    return out


class fronto(object):
    """
    Frontogenesis analysis
    """
    def __init__(self,c,s,t,u,v,w,tx,ty,Qnet,KppviscA,KdiffT,Kpphbl,
                 grid,maxlevel):
        self.s = s
        self.t = t
        self.u = u
        self.v = v
        self.w = w
        self.tx = tx
        self.ty = ty
        self.Qnet = Qnet
        #self.KppviscA = KppviscA
        self.Kpphbl = Kpphbl
        self.grid = grid
      	


        ####### Interpolating KppviscA ###### 
	self.KppviscA = np.empty(KppviscA.shape)
        self.KppviscA[1:-1,:,:] = (KppviscA[2:,:,:]+KppviscA[:-2,:,:])*0.5
        self.KppviscA[0,:,:] = KppviscA[1,:,:]
        self.KppviscA[-1,:,:] = KppviscA[-2,:,:]


        ####### Interpolating Kdifft (vertical diffusion coefficient) ###### 
	self.KdiffT = np.empty(KdiffT.shape)
        self.KdiffT[1:-1,:,:] = (KdiffT[2:,:,:]+KdiffT[:-2,:,:])*0.5
        self.KdiffT[0,:,:] = KdiffT[1,:,:]
        self.KdiffT[-1,:,:] = KdiffT[-2,:,:]


        ##### Coriolis ####
        c.coriolis() 
	self.fo = c.f[self.grid['ilnc'][0]:self.grid['ilnc'][1],
	           self.grid['iltc'][0]:self.grid['iltc'][1]]

	### Making 3D arrays ###
	self.D3()

	### Pressure #####
	self.get_pressure(c)

	### get buoyancy ####
	self.get_buoyancy_and_N2(c)


        ####### vertical derivative of T ####
        self.dTdz()

        #### wind stress at rho-point 
        self.tx = u2rho(self.tx)
        self.ty = v2rho(self.ty)

        ##### Gradient of velocity ####
        self.ux,self.uy,self.vx,self.vy = basics.gradvel(u,v,
                       self.grid['dxc'],self.grid['dyc'],
                       self.grid['dxg'],self.grid['dyg'],
                       self.grid['hFacW'],self.grid['hFacS'],self.grid['drf'],
	               self.grid['rac'],self.grid['ras'])
	
	#### Gradient of Temperature ####
	self.Tx,self.Ty = basics.gradT(self.t,
	                   self.grid['dxc'],self.grid['dyc'])

	### Gradient of buoyancy #####
	self.bx,self.by = basics.gradT(self.b,self.grid['dxc'],self.grid['dyc']) 

	##################################################
	#####
	##### Q-vector = -b_x * nabla(u) - b_y * nabla(v)
	####
	##################################################

	##### Frontogenetic vector Q #######
        self.Q1 = -self.ux*self.bx - self.vx*self.by
	self.Q2 = -self.uy*self.bx - self.vy*self.by


	###############################################@#
	######
	##### Fs = vec(Q) dot nabla(b)
	######
	################################################
	self.Fs = self.Q1*self.bx + self.Q2*self.by


	##################################################
	######
	#####
	##### Straining deformation by vertical velocity
	#####
	#### vec(Q)_w = N2*nabla(w)
	#####
	####  w:= vertical velocity
	################################################
	self.wx,self.wy = basics.gradT(self.w,self.grid['dxc'],
		                       self.grid['dyc'])
	self.Qw1 = self.N2*(self.wx)
	self.Qw2 = self.N2*(self.wy)	
	self.Fw = self.Qw1*self.bx + self.Qw2*self.by
	
	######### Vertical velocity shear ######
	self.vertical_shear_vel()

	################################################
	######
	###### Ertel-PV
	######
	###############################################
	self.zeta = self.vx - self.uy
	self.strain = np.sqrt((self.ux -self.vy)**2 + (self.vx**2 + self.uy**2))
	self.divergence = self.ux + self.vy
	self.epv_k = (self.fo3d + self.zeta)*self.N2
	self.epv_ij = -self.dvdz*self.bx + self.dudz*self.by
	self.epv = self.epv_k + self.epv_ij	



    def D3(self):
        ### Depth
        dpt = -self.grid['depth'][:].squeeze()
        dpt = dpt[:,None,None]
        dpt = np.tile(dpt,(1,self.grid['m'],self.grid['n']))
        dptmd=(dpt[:-2,:,:]+dpt[2:,:,:])*0.5
        self.depthmd = dptmd[:,:,:] ## << ==
	self.depth3d = dpt

        ### Thickness
        thk = self.grid['thick'][:].squeeze()
        thk = thk[:,None,None]
        thk = np.tile(thk,(1,self.grid['m'],self.grid['n']))
        thkmd = (thk[:-2,:,:]+thk[2:,:,:])*0.5
        self.thkmd=thkmd
	self.thk3d = thk

        ### Lon and Lat 3D
        lon3d = np.tile(self.grid['lonc'],((self.depth3d).shape[0],1,1))
        self.lon3d = lon3d[:,:,:] # <<==
        lat3d = np.tile(self.grid['latc'],((self.depth3d).shape[0],1,1))
        self.lat3d = lat3d[:,:,:] # <<==

        ### Coriolis, f
        fo3d = np.tile(self.fo,((self.depth3d).shape[0],1,1))
        self.fo3d = fo3d[:,:,:]
    ##

    def get_pressure(self,c): 
        """
        Pressure
        """
        self.pressure=c.compute_pressure('/u/htorresg/gsw-3.0.3/',
                                         self.grid['m'],self.grid['n'],
                                         self.grid['k'],self.grid['depth'],
                                         self.grid['latc'])

    def get_buoyancy_and_N2(self,c):
	"""
	Buoyancy 
	"""
	c.compute_dens(self.s,self.t,self.pressure)
	self.dens = c.dens
	self.b,self.rho_ref = c.compute_buoyancy(c.dens)

	###
	#::::: N2 Brunt-Vaissalla :::::
        alpha = 2e-4
	N2 = np.empty((self.b).shape)
	drho = self.b[:-2,:,:]-self.b[2:,:,:] ## centered scheme
	N2[1:-1,:,:] = (drho/(self.thk3d[2:,:,:]+self.thk3d[:-2,:,:]))
        #N2 = alpha*9.81*N2
	N2[0,:,:] =  N2[1,:,:]
	N2[-1,:,:] = N2[-2,:,:]
	self.N2 = N2


    def dTdz(self):
        dT_z = self.t[:-2,:,:] - self.t[2:,:,:]
        dTdz = np.empty((self.t).shape)
        print('dTdz shape')
        print(dTdz.shape)
        dTdz[1:-1,:,:] = dT_z/(self.thk3d[2:,:,:]+self.thk3d[:-2,:,:])
        dTdz[0,:,:] = dTdz[1,:,:]
        dTdz[-1,:,:] = dTdz[-2,:,:]
        self.dTdz = dTdz

 

    def vertical_shear_vel(self):
	######## du/dz #######
	du = self.u[:-2,:,:] - self.u[2:,:,:] ### centered scheme
	print('du shape ',du.shape)
	dudz = np.empty((self.u).shape)
	dudz[1:-1,:,:] = du/(self.thk3d[:-2,:,:]+self.thk3d[2:,:,:])
  	dudz[0,:,:] = dudz[1,:,:]
	dudz[-1,:,:] = dudz[-2,:,:]
	dudz = u2rho(dudz)
	self.dudz = dudz
 	del du,dudz
	###### dv/dz #########
	du = self.v[:-2,:,:] - self.v[2:,:,:] ### centered
	dvdz = np.empty((self.v).shape)
	dvdz[1:-1,:,:] = du/(self.thk3d[:-2,:,:]+self.thk3d[2:,:,:])
	dvdz[0,:,:] = dvdz[1,:,:]
	dvdz[-1,:,:] = dvdz[-2,:,:]
	dvdz = v2rho(dvdz)
	self.dvdz = dvdz
	del du,dvdz


	 
