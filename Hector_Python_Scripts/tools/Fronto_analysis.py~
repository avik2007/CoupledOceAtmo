import numpy as np
import sys
sys.path.append('/u/htorresg/Experiment_CCS/programas/tools/')
import handle_mitgcm as model
sys.path.append('/u/htorresg/Experiment_CCS/programas/')
import tools
import EPV_FluxForm as epv
import scipy.io as io
from numpy import r_,diff,diag,matrix,identity
import scipy.fftpack as fft
from numpy import  pi
from netCDF4 import Dataset
 
class fronto(object):
    """
    Frontogenesis analysis
    """

    ### model config  ###### Check params.py before run this code      
    import params as p
   
    ### common params 
    rho_ref = 1024.

    global c,p
    ####### model handler ########
    c = model.LLChires(p.dirc,p.dirc,
                       p.nx,p.ny,p.nz,
                       p.tini,p.tref,p.tend,p.steps)
    ###
    ## call grid info
    c.timeline()
    c.load_dxc()
    c.load_grid()
    c.load_duv()
    c.load_dxf()
    c.load_dxg()
    c.coriolis()
    c.load_raz()
    c.load_ras()
    c.load_raw()
    c.load_rac()

    def __init__(self,s,t,u,v,w,iln,ilt,maxlevel,netcdfPath):
        
        self.s = s
        self.t = t
        self.u = u
        self.v = v
        self.w = w

        ##### subdomain based on iln ilt
        self.ilnmn,self.ilnmx,self.iltmn,self.iltmx = \
               tools.extract_subdomain(c.lon,c.lat,iln,ilt)
        self.ilnmng,self.ilnmxg,self.iltmng,self.iltmxg = \
               tools.extract_subdomain(c.long,c.latg,iln,ilt)

        self.lon = c.lon[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx]
        self.lat = c.lat[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx]
        self.long = c.long[self.ilnmng:self.ilnmxg,self.iltmng:self.iltmxg]
        self.latg = c.latg[self.ilnmng:self.ilnmxg,self.iltmng:self.iltmxg]
        self.fo  = c.f[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx]
        self.m,self.n = self.lon.shape
        self.depth = c.dpt[:maxlevel]
        self.thick = c.thk[:maxlevel]
        self.nz = len(self.depth)

        ##### construct 3D matrix
        self.D3()

        ##### compute pressure
        self.get_pressure()

        #### compute buoyancy
        self.get_buoyancy()

        #### Laplacian of density
        self.get_LapDens(c.dxf[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                     c.dyf[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                     c.dxg[self.ilnmng:self.ilnmxg,self.iltmng:self.iltmxg],
                     c.dyg[self.ilnmng:self.ilnmxg,self.iltmng:self.iltmxg],
                     c.ras[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                     c.raw[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                     c.rac[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx])
        print('laplacian: ',self.d2bdx2.shape,self.d2bdy2.shape)


        self.Lapb = dT2Dx2(self.b,
                    c.dxc[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                    c.dyc[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx])
        print(' New Lapb: ',(self.Lapb).shape)


        #### Grad buoyancy
        self.gradb(self.b,c.dxf[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                      c.dyf[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                      c.ras[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                      c.raw[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx])

        ### Gradient of w (total vertical velocity)
        self.Wx,self.Wy = gradT(self.w,
                        c.dxf[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                      c.dyf[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                      c.ras[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                      c.raw[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx])


        #### Grad velocity
        self.gradvel(self.u,self.v,c.dxc[self.ilnmn:self.ilnmx,
                                         self.iltmn:self.iltmx],
                    c.dyc[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                    c.dxg[self.ilnmng:self.ilnmxg,self.iltmng:self.iltmxg],
                    c.dyg[self.ilnmng:self.ilnmxg,self.iltmng:self.iltmxg],
                    c.raz[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                    c.rac[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx])
        self.vorticity = self.vx - self.uy
        self.divergence = self.ux + self.vy
        self.strain = np.sqrt((self.ux - self.vy)**2 + (self.vx + self.uy)**2)
        self.absolute = self.vorticity - self.fo[1:-1,1:-1]
        self.condition = self.absolute - self.strain



        ######################################################
        ######
        ##### Q-vector = -b_x * nabla(u) - b_y * nabla(v)
        #####
        ######################################################
        self.Q1 = -self.ux*self.dbdx - self.vx*self.dbdy
        self.Q2 = -self.uy*self.dbdx - self.vy*self.dbdy
        self.magQ = np.sqrt(self.Q1**2 + self.Q2**2)
        self.magb2 = 0.5*(self.magb)**2



        #######################################################
        #####
        ##### Divergence of Q := dot(nabla,Q-vector) at rho-point
        ####
        #######################################################
        dyf = c.dyf[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx]
        dxf = c.dxf[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx]
        ras = c.ras[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx]
        raw = c.raw[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx]
        self.divQ = div_at_rho(self.Q1,self.Q2,dxf[1:-1,1:-1],
                      dyf[1:-1,1:-1],
                      ras[1:-1,1:-1],
                      raw[1:-1,1:-1])


        print('divQ shape: ',self.divQ.shape)

	######################################################
	##### 
	#####
	##### Fs = Q dot nabla(b)
	#####
	######################################################
	self.Fs = self.Q1*self.dbdx + self.Q2*self.dbdy


        #######################################################
        #####
        ##### Ertel Potential Vorticity
        ####
        #######################################################
        _,_,_,self.dbdz,self.dudz,self.dvdz,self.pv_k, \
            self.pv_i,self.pv_j,self.ertel = epv.epv(self.u,self.v,self.b,
                            c.dxc[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                            c.dyc[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                            c.dxf[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                            c.dyf[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                            c.raz[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                            c.raw[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                            c.ras[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx],
                                        self.thick.squeeze(),self.fo)


        ################################
        ####
        #### Adjust size of matrix
        #### Since all variables are
        ### located at rho-points, 
        ### it is convinient adjust the size of all matrix 
        ### so we don't have to deal with it later.
        ###
        ### Because divQ is N-4 and M-4 final size, we
        ### adjust the rest to this size.
        ################################
        self.adjust_arrays()

        print('Wx, Wy shape: ',(self.Wx).shape,(self.Wy).shape)

        ##########################################
        ###
        ### f2 * d2W/dz + nabla(N2*nabla(W)) = nabla dot Q 
        ###
        #########################################
        self.Ws = from_divQ_to_w(self.n-4,self.m-4,self.N2.shape[0],p.dx,p.dy,
                                 self.lon3d[0,:,:],self.lat3d[0,:,:],
                                 self.depthmd,
                                 np.nanmean(self.fo3d[0,:,:]),
                                 self.N2,self.divQ)

        #### Gradient of Ws
        dxf = c.dxf[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx]
        dyf = c.dyf[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx]
        ras = c.ras[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx]
        raw = c.raw[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx]
        Wsx,Wsy = gradT(self.Ws,
                        dxf[2:-2,2:-2],
                      dyf[2:-2,2:-2],
                      ras[2:-2,2:-2],
                      raw[2:-2,2:-2] )
	WSx = np.zeros((self.Ws.shape))
	WSy = np.zeros((self.Ws.shape))
	WSx[:,1:-1,1:-1] = Wsx
	WSy[:,1:-1,1:-1] = Wsy
        self.Wsx = WSx
        self.Wsy = WSy

        

        ##########################################
        ###
        ### Garret and Loder (1981) 
        ###
        ### w = (g/(f2*rho_ref))*Av* Lap(rho)
        ###
        #########################################
        self.wvel_GL = wvel_Garret(self.Lapb,
                                   self.fo3d,self.rho_ref)

        ####################################
        ###################################
        ###
        ###     Save
        ###
        ###################################
        self.dump_to_netcdf(netcdfPath)

        ############# END #################

    def D3(self):
        ####$ Depth
        dpt = -self.depth.squeeze()
        dpt = dpt[:,None,None]
        dpt = np.tile(dpt,(1,self.m,self.n))
        dptmd = (dpt[:-2,:,:] + dpt[2:,:,:])*0.5 
        self.depthmd = dptmd[:,1:-1,1:-1]

        #### Thickness
        thk = self.thick.squeeze()
        thk = thk[:,None,None]
        thk = np.tile(thk,(1,self.m,self.n))
        thkmd = (thk[:-2,:,:]+thk[2:,:,:])*0.5
        self.thkmd = thkmd

        ### Lon3D & Lat3D
        lon3d = np.tile(self.lon,((self.depthmd).shape[0],1,1))
        self.lon3d = lon3d[:,1:-1,1:-1]
        lat3d = np.tile(self.lat,((self.depthmd).shape[0],1,1))
        self.lat3d = lat3d[:,1:-1,1:-1]
        
        ### f
        fo3d = np.tile(self.fo,((self.depthmd).shape[0],1,1))
        self.fo3d = fo3d[:,1:-1,1:-1]
        
    def get_pressure(self):
        self.pressure = c.compute_pressure('/u/htorresg/gsw-3.0.3/',
                                  self.m,self.n,self.nz,self.depth,self.lat)

    def get_buoyancy(self):
        c.compute_dens(self.s,self.t,self.pressure)
        self.b,self.rho_ref = c.compute_buoyancy(c.dens)
        #
        #### ::: N2 Brunt-Vaissalla ::::::
        drho = self.b[:-2,:,:] - self.b[2:,:,:]
        self.N2 = (drho/self.thkmd)

    def get_LapDens(self,dxf,dyf,dxg,dyg,rau,rav,rac):
        c.compute_dens(self.s,self.t,self.pressure)
        print(c.dens.shape)
        d2bdx2,d2bdy2 = lapb(c.dens,dxf,dyf,dxg,dyg,
                                       rau,rav,rac)
        self.d2bdx2 = d2bdx2[:,1:-1,:]
        self.d2bdy2 = d2bdy2[:,:,1:-1]
        print('lap(rho):',self.d2bdx2.shape)

    def gradb(self,b,dxf,dyf,rau,rav):
        """
        Grad(b) at rho-points
        """
        if b.ndim == 3:
            dbdx=(b[:,:,2:]*dyf[None,:,2:]-b[:,:,:-2]*dyf[None,:,:-2])/ \
                (rau[None,:,1:-1]+rau[None,:,:-2])
            dbdy=(b[:,2:,:]*dxf[None,2:,:] - \
                    b[:,:-2,:]*dxf[None,:-2,:])/(rav[None,1:-1,:]+\
                                             rav[None,:-2,:])
            self.dbdx = (dbdx[:-2,1:-1,:]+dbdx[2:,1:-1,:])*0.5
            self.dbdy = (dbdy[:-2,:,1:-1]+dbdy[2:,:,1:-1])*0.5
            self.magb = np.sqrt(self.dbdx**2 + self.dbdy**2)



    def adjust_arrays(self):
        self.N2 = self.N2[:,2:-2,2:-2] 
        self.ertel = self.ertel[:,1:-1,1:-1]
        self.dbdz = self.dbdz[:,1:-1,1:-1]
        self.dudz = self.dudz[:,1:-1,1:-1]
        self.dvdz = self.dvdz[:,1:-1,1:-1]
        self.vx = self.vx[:,1:-1,1:-1]
        self.vy = self.vy[:,1:-1,1:-1]
        self.ux = self.ux[:,1:-1,1:-1]
        self.uy = self.uy[:,1:-1,1:-1]
        self.pv_k = self.pv_k[:,1:-1,1:-1]
        self.pv_i = self.pv_i[:,1:-1,1:-1]
        self.pv_j = self.pv_j[:,1:-1,1:-1]
        self.Q1   = self.Q1[:,1:-1,1:-1]
        self.Q2   = self.Q2[:,1:-1,1:-1]
        self.magQ = self.magQ[:,1:-1,1:-1]
        self.magb2 = self.magb2[:,1:-1,1:-1]
        self.vorticity = self.vorticity[:,1:-1,1:-1]
        self.divergence = self.divergence[:,1:-1,1:-1]
        self.strain = self.strain[:,1:-1,1:-1]
        self.condition = self.condition[:,1:-1,1:-1]
        self.absolute = self.absolute[:,1:-1,1:-1]
        self.depthmd = self.depthmd[:,1:-1,1:-1]
        self.lon3d = self.lon3d[:,1:-1,1:-1]
        self.lat3d = self.lat3d[:,1:-1,1:-1]
        self.fo3d  = self.fo3d[:,1:-1,1:-1]
        self.b = (self.b[2:,2:-2,2:-2]+self.b[:-2,2:-2,2:-2])*0.5
        self.t = (self.t[:-2,2:-2,2:-2]+self.t[2:,2:-2,2:-2])*0.5
        self.s = (self.s[:-2,2:-2,2:-2]+self.s[2:,2:-2,2:-2])*0.5
        self.w = (self.w[:-2,2:-2,2:-2]+self.w[2:,2:-2,2:-2])*0.5
        self.u = (self.u[:-2,2:-2,2:-2]+self.u[2:,2:-2,2:-2])*0.5
        self.v = (self.v[:-2,2:-2,2:-2]+self.v[2:,2:-2,2:-2])*0.5
        self.Wx = (self.Wx[2:,1:-1,1:-1]+self.Wx[:-2,1:-1,1:-1])*0.5
        self.Wy = (self.Wy[2:,1:-1,1:-1]+self.Wy[:-2,1:-1,1:-1])*0.5
        self.d2bdx2 = (self.d2bdx2[:-2,1:-1,1:-1]+self.d2bdx2[2:,1:-1,1:-1])*0.5
        self.d2bdy2 = (self.d2bdy2[:-2,1:-1,1:-1]+self.d2bdy2[2:,1:-1,1:-1])*0.5
        self.dbdx = self.dbdx[:,1:-1,1:-1]
        self.dbdy = self.dbdy[:,1:-1,1:-1]
        self.Lapb = self.Lapb[:,1:-1,1:-1]
        self.Fs = self.Fs[:,1:-1,1:-1]
        self.nz,self.ny,self.nx = self.Fs.shape

    def gradvel(self,u,v,dxc,dyc,dxg,dyg,raz,rac):
        """
        Grad(vel) in Flux-Form as MITgcm 
        """
        if u.ndim == 3:
            vx_out = v[:,:,:-1]*dyc[None,:,:-1]
            vx_in = v[:,:,1:]*dyc[None,:,1:]
            vx = (vx_in - vx_out)/raz[None,:,1:]
            #
            uy_out = u[:,:-1,:]*dxc[None,:-1,:]
            uy_in = u[:,1:,:]*dxc[None,1:,:]
            uy = (uy_in - uy_out)/raz[None,1:,:]
            ####
            ### At rho-points
            vx = vx[:,1:,:]
            uy = uy[:,:,1:]
            ##
            vx_cent = (vx[:,:-1,:]+vx[:,1:,:])*0.5
            vx_rho = (vx_cent[:,:,:-1]+vx_cent[:,:,1:])*0.5
            vx_rho = (vx_rho[:-2,:,:]+vx_rho[2:,:,:])*0.5
            ##
            uy_cent = (uy[:,:-1,:]+uy[:,1:,:])*0.5
            uy_rho = (uy_cent[:,:,:-1]+uy_cent[:,:,1:])*0.5
            uy_rho = (uy_rho[:-2,:,:] + uy_rho[2:,:,:])*0.5
            ###
            #####################################
            
            #####################################
            ux_out = u[:,:,:-1]*dyg[None,:,:-1]
            ux_in = u[:,:,1:]*dyg[None,:,1:]
            ux = (ux_in - ux_out)/rac[None,:,1:]
            ##############
            vy_out = v[:,:-1,:]*dxg[None,:-1,:]
            vy_in = v[:,1:,:]*dxg[None,1:,:]
            vy = (vy_in - vy_out)/rac[None,1:,:]
            ####
            #### AT rho-point
            ux = ux[:,1:,:]
            vy = vy[:,:,1:]
            ux = (ux[:,:-1,:]+ux[:,1:,:])*0.5
            ux = (ux[:,:,:-1]+ux[:,:,1:])*0.5
            ux = (ux[2:,:,:]+ux[:-2,:,:])*0.5
            ##
            vy = (vy[:,:-1,:]+vy[:,1:,:])*0.5
            vy = (vy[:,:,:-1]+vy[:,:,1:])*0.5
            vy = (vy[2:,:,:]+vy[:-2,:,:])*0.5
            ###
            self.uy = uy_rho
            self.vx = vx_rho
            self.ux = ux
            self.vy = vy

    def dump_to_netcdf(self,filepath):
        """
        Dumps all the data in netcdf
        
        Parameters
        ----------
        filepath:
               Full path to the necdf file to be created. 
        If exists, it will overwritten.
        
        """
        self.ncFile = Dataset(filepath,'w',format='NETCDF4_CLASSIC')
        
        #### netcdf dimensions
        self.ncFile.createDimension('depth',self.nz)
        self.ncFile.createDimension('lat',self.ny)
        self.ncFile.createDimension('lon',self.nx)
        
        #### netcdf variables
        add_nc_var(self.ncFile,'Depth3d',self.depthmd,'m',
                   'f8',('depth','lat','lon'))
        add_nc_var(self.ncFile,'N2',self.N2,'s-2','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'lon3d',self.lon3d,'degrees east','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'lat3d',self.lat3d,'degrees north','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'fo',self.fo3d,'Coriolis s-1','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'Ertel',self.ertel,'Ertel-PV','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'dbdz',self.dbdz,'z-grad of b','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'dudz',self.dudz,'z-vertical shear u','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'dvdz',self.dvdz,'z-vertical shear v','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'dudx',self.ux,'dudx','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'dudy',self.uy,'dudy','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'dvdx',self.vx,'dvdx','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'dvdy',self.vy,'dvdy','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'pv_k',self.pv_k,'EPV_k','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'pv_i',self.pv_i,'EPV_i','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'pv_j',self.pv_j,'EPV_j','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'Q1',self.Q1,'Frontogenetic vector-x','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'Q2',self.Q2,'Frontogenetic vector-y','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'divQ',self.divQ,'div(Q)','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'magQ',self.magQ,'mag(Q)','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'magb2',self.magb2,'mag(b)2','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'vorticity',self.vorticity,'s-1','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'divergence',self.divergence,'s-1','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'strain',self.strain,'s-1','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'condition',self.condition,
                   'abs(vort) minus f','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'absolute',self.absolute,'(vort plus f)','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'b',self.b,'buoyancy','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'t',self.t,'temperature','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'s',self.s,'salinity','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'w',self.w,'vertical velocity','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'u',self.u,'u-component','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'v',self.v,'v-component','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'dbdx',self.dbdx,'grad(b)-x','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'dbdy',self.dbdy,'grad(b)-y','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'dwdx',self.Wx,'grad(w)-x','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'dwdy',self.Wy,'grad(w)-y','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'Fs',self.Fs,'Fs','f8',
                   ('depth','lat','lon'))
        add_nc_var(self.ncFile,'w-omega',self.Ws,
                   'vertical velocity from omega','f8',
                    ('depth','lat','lon'))
        add_nc_var(self.ncFile,'Wsx',self.Wsx,
                   'derivate omega-x','f8',
                    ('depth','lat','lon'))
        add_nc_var(self.ncFile,'Wsy',self.Wsy,
                   'derivate omega-y','f8',
                    ('depth','lat','lon'))
        add_nc_var(self.ncFile,'d2bdx2',self.d2bdx2,
                   'Lap(rho)-x','f8',
                    ('depth','lat','lon'))
        add_nc_var(self.ncFile,'d2bdy2',self.d2bdy2,
                   'Lap(rho)-y','f8',
                    ('depth','lat','lon'))
        add_nc_var(self.ncFile,'wvel_GL',self.wvel_GL,
                    'Wvel-GarretLoder','f8',
                    ('depth','lat','lon'))
        add_nc_var(self.ncFile,'Lapb',self.Lapb,
                    'Laplacian(b)','f8',
                    ('depth','lat','lon'))

        self.ncFile.close()

def add_nc_var(ncFile,varname,varval,varunit,vartype,vardim):
    ncvar = ncFile.createVariable(varname,vartype,vardim,fill_value=99999)
    ncvar[:] = varval[:]

def gradT(T,dxf,dyf,rau,rav):
        """
        Grad(b) at rho-points
        """
        if T.ndim == 3:
            dbdx=(T[:,:,2:]*dyf[None,:,2:]-T[:,:,:-2]*dyf[None,:,:-2])/ \
                (rau[None,:,1:-1]+rau[None,:,:-2])
            dbdy=(T[:,2:,:]*dxf[None,2:,:] - \
                    T[:,:-2,:]*dxf[None,:-2,:])/(rav[None,1:-1,:]+\
                                             rav[None,:-2,:])
            dbdx = (dbdx[:,1:-1,:]+dbdx[:,1:-1,:])*0.5
            dbdy = (dbdy[:,:,1:-1]+dbdy[:,:,1:-1])*0.5
            return dbdx,dbdy

def div_at_rho(u,v,dxf,dyf,rau,rav):
    """
    div(vel) at rho points
    """
    if u.ndim == 3:
        dbdx = (u[:,:,2:]*dyf[None,:,2:]-u[:,:,:-2]*dyf[None,:,:-2])/ \
                   (rau[None,:,1:-1]+rau[None,:,:-2])
        dbdy = (v[:,2:,:]*dxf[None,2:,:] - \
                    v[:,:-2,:]*dxf[None,:-2,:])/(rav[None,1:-1,:]+\
                                             rav[None,:-2,:])
        dbdx = (dbdx[:,1:-1,:]+dbdx[:,1:-1,:])*0.5
        dbdy = (dbdy[:,:,1:-1]+dbdy[:,:,1:-1])*0.5
        return dbdx + dbdy
    
        
def lonlat2xy(lon,lat):
    """convert lat lon to y and x
    x, y = lonlat2xy(lon, lat)
    lon and lat are 1d variables.
    x, y are 2d meshgrids.
    """
    import numpy as np 
    # meshgrid,cos,pi
    r = 6371.e3
    #lon = lon-lon[0]
    if lon.ndim == 1:
        lon,lat = np.meshgrid(lon,lat)
    x = 2*pi*r*np.cos(lat*np.pi/180.)*lon/360.
    y = 2*pi*r*lat/360.
    return x,y



def from_divQ_to_w(nx,ny,nz,dx,dy,lon,lat,depth,fo,N2,divQ):


    ######## Vector of depth
    z = depth[:,100,100].squeeze()

    ######### Window
    wx = np.matrix(np.hanning(nx))
    wy = np.matrix(np.hanning(ny))
    window = np.array(wy.T*wx) # hanning window # periodic domain
    Wss = np.sum(window)
    w_rms = np.sqrt(Wss/nx/ny)
    print('windowing: ',window.shape)


    ## physical space
    xx, yy = lonlat2xy(lon,lat)
    # dx and dy
    #dx = (xx[:, 1] - xx[:, 0]).reshape(-1, 1)
    #dy = (yy[1, :] - yy[0, :]).reshape(1, -1)
    #ny, nx = xx.shape
    #print('xx:',xx.shape,yy.shape)


    ####### SPectral space
    k = fft.fftfreq(nx) # 2*pi
    l = fft.fftfreq(ny) # 2*pi
    print('freq:',k.shape,l.shape)


    k = k/(dx)
    l = l/(dy)
    [kx0,ky0] = np.meshgrid(k,l)
    ks = np.sqrt(kx0**2 + ky0**2)
    divQhat = np.zeros((nz,ny,nx))*complex(0, 0)

    print('Nz: ',nz)
    for i in range(0,nz):
        # removing the mean
        # Qanom,Qmn = fit2Dsurf(xx,yy,1*divQ[i,:,:])
        # plt.imshow(Qanom,origin='lower')
        # plt.show()
        #print('Qanom:',Qanom.shape)
        #print(str(i))
        divQhat[i,:,:] = fft.fft2(divQ[i,:,:]*window/(w_rms+1e-15))/nx/ny ##
    ##
    What = np.zeros((nz,ny,nx))*complex(0, 0)
    a = -r_[0,z,z[-1]]
    del1 = a[:-2]-a[1:-1]
    del2 = a[1:-1]-a[2:]



    for jj in range(0,ny):
        for ii in range(0,nx):
            R = divQhat[:,jj,ii].squeeze()
            ################
            ###   BC
            # w|z=0 = 0
            ##
            A1 = 2*(fo**2)/del1/(del1+del2+1e-15)
            A1 = A1[1:]
            A2 =-(ks[jj,ii]**2)*N2[0,jj,ii]-2*(fo**2)/(del1+1e-15)/(del2+1e-15)

            #################
            #### BC
            # dw/dz=0 at bottom
            ##
            A2[-1]=-(ks[jj,ii]**2)*N2[-1,jj,ii]-2*(fo**2)/(del1[-1]+1e-15)/(del1[-1]+del2[-1]+1e-15)

            A3=2*(fo**2)/(del2+1e-15)/(del1+del2+1e-15) 
                # The 1e-15 is to avoid division by zero
            A3=A3[:-1]

            for kk in range(1,nz):
                A2[kk]=A2[kk]-A3[kk-1]*A1[kk-1]/A2[kk-1]
                R[kk]=R[kk]-R[kk-1]*A1[kk-1]/A2[kk-1]
            ##    

            What[-1,jj,ii]=R[-1]/A2[-1]

            for kk in reversed(range(0,nz-1)):
                What[kk,jj,ii]=(R[kk]-A3[kk]*What[kk+1,jj,ii])/A2[kk]
            ###

    ww0 = np.ones((nz,ny,nx))
    for kk in range(0,nz):
        ww0[kk,:,:] = np.real(fft.ifft2(What[kk,:,:])*nx*ny)
    print('ww0 shape: ',ww0.shape)
    return ww0    

def lapb(b,dxf,dyf,dxg,dyg,rau,rav,rac):
    """
    Lap(rho) at rho points
    """
    dbdx = (b[:,:,1:]*dyf[None,:,1:] - b[:,:,:-1]*dyf[None,:,:-1])/ \
               (rau[None,:,1:])
    d2bdx2 = (dbdx[:,:,1:]*dyg[None,:,1:-1]-dbdx[:,:,:-1]*dyg[None,:,1:-1])/ \
                (rac[None,:,1:-1])
    #
    dbdy = (b[:,1:,:]*dxf[None,1:,:]-b[:,:-1,:]*dxf[None,:-1,:])/ \
               (rav[None,1:,:])
    d2bdy2 = (dbdy[:,1:,:]*dxf[None,1:-1,:]-dbdy[:,:-1,:]*dxf[None,1:-1,:])/ \
                (rac[None,1:-1,:])
    return d2bdx2,d2bdy2


def dT2Dx2(var,dx,dy):
    import numpy as np
    # Second-order derivate
    dxmd = (0.5*(dx[:,2:]+dx[:,:-2]))
    dymd = (0.5*(dy[2:,:]+dy[:-2,:]))
    #dxmd = (0.5*(dx[2:,:]+dx[:-2,:]))
    #dymd = (0.5*(dy[:,2:]+dy[:,:-2]))
    dvar2dx2 = np.diff(var,n=2,axis=2)/dxmd[None,:,:]**2
    dvar2dy2 = np.diff(var,n=2,axis=1)/dymd[None,:,:]**2
    print('dvar2dx2: ',dxmd.shape)
    print('dvar2dy2: ',dymd.shape)
    Lap = 0.5*(dvar2dx2[:,2:,:]+dvar2dx2[:,:-2,:])+ \
          0.5*(dvar2dy2[:,:,2:]+dvar2dy2[:,:,:-2])
    #Lap = 0.5*(dvar2dx2[:,:,2:]+dvar2dx2[:,:,:-2])+ \
    #      0.5*(dvar2dy2[:,2:,:]+dvar2dy2[:,:-2,:])
    Lap = Lap[1:-1,:,:]
    return Lap

def wvel_Garret(Lapb,f,rho_ref):
    """
    Diabatic contribution to the W field
    (Garret and Loder 981)
    """
    g = 9.81 # Gravity acceleration
    f2 = f*f # f^{2}
    Av = 2e-2 # m^2 s^-1
    factor = Av/(f2)
    wvel = factor*(Lapb)
    return wvel
