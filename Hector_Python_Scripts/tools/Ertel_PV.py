import numpy as np
import sys
sys.path.append('/u/htorresg/Experiment_CCS/programas/tools/')
import handle_mitgcm as model
sys.path.append('/u/htorresg/Experiment_CCS/programas/')
### Load config file
####import params as p



class EPV(object):
     """ Ertel Potential Vorticity """
     
     import params as p ##### config file

     ### ::::::::: MITgcm handler ::::::::::
     c = model.LLChires(p.dirc,p.dirc,p.nx,p.ny,
			p.nz,p.tini,p.tref,p.tend,p.steps)
     c.load_raz()
     c.load_dxc()
     c.load_dxg()
     c.load_duv()
     c.load_dxf()
     ###

     def __init__(self,
		  u = None,
                  v = None,
	          s = None,
	          t = None,
		  ):
	self.u = u
	self.v = v
	self.s = s
	self.t = t
     
	print('p.dirc')
       


