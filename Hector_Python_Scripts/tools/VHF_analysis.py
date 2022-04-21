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

class vhf(object):
    """
    Vertical heat fluxes analysis
    
    Jw = Rho_ref*Cp*w*T' 

    Jk = Rho_ref*Cp*kappa*\frac{\partial{T'}}{\partial{z}
    """

    #### model configuration  
    ## Check params.py before run this code
    import params as p


    global c,p

    #### Model handler #####
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


    def __init__(self,t,w,kappa,iln,ilt,maxlevel):
        
        self.t = t
        self.w = w
        factor = 4.2e6 ###(rho*Cp)

        
        ##### subdomain based on iln ilt
        self.ilnmn,self.ilnmx,self.iltmn,self.iltmx = \
               tools.extract_subdomain(c.lon,c.lat,iln,ilt)

        self.lon = c.lon[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx]
        self.lat = c.lat[self.ilnmn:self.ilnmx,self.iltmn:self.iltmx]
        self.m,self.n = self.lon.shape
        self.depth = c.dpt[:maxlevel]
        self.thick = c.thk[:maxlevel]
        self.nz = len(self.depth)

        #### vertical gradients of T
        self.gradT()
        
        ### Vertical heat flux due to vertical advection
        self.Jw = factor*self.w*self.t
        self.Jw = 0.5*(self.Jw[:-2,:,:]+self.Jw[2:,:,:])

        ### Vertical heat flux due to vertical mixing
        self.Jk = factor*(0.5*(kappa[:-2,:,:]+kappa[2:,:,:]))*self.dtdz


    def gradT(self):
        ### dbdz
        self.dtdz = (self.t[:-2,:,:]-self.t[2:,:,:])/ \
               (self.thick[:-2,None,None]+self.thick[2:,None,None])
    


   
