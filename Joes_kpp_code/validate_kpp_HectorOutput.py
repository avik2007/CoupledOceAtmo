import os
import sys
sys.path.append("12_16_21")
import numpy as np
import fld_tools as ft
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from repmat import repmat
from flds import FldsU
from flds import FldsNU
#sys.path.append('/nobackup/htorresg/DopplerScat/modelling/GS/programs/tools')
sys.path.append('/u/htorresg/Experiment_CCS/programas/tools')
sys.path.append('/nobackup/amondal/Python/CCS_Analysis')
import handle_mitgcm as model
import params_500mSNAPS_joecode as p


dims = [87, 2400,2400]; #Hector's code has 87 layers # I think switching the height might fix the switch_column_row function (7/7/22)
truncdims = [0, 100, 0, 100, 0, 87];
data_dir = '/nobackup/amondal/Python/Joes_kpp_code/Hector_Sub_Data/'#'/scratch/p/peltier/jskitka/runs/2km_109l/kpp_all_on/run/';
aux_data_dir = '/nobackup/amondal/Python/Joes_kpp_code/Hector_Sub_Data/';#'/home/p/peltier/jskitka/aux_data/low_res/';
home_dir ='/nobackup/amondal/Python/Joes_kpp_code/'#'/scratch/p/peltier/jskitka/diagnostics/kpp_all_on/';
save_dir = '/nobackup/amondal/Python/Joes_kpp_code/validate_Hector_kpp_save_dir/'#'/scratch/p/peltier/jskitka/diagnostics/kpp_all_on/figures/';
os.chdir(home_dir)



## Build File List

lw = 3; fsize=16;
iters = [798408];
cmax = 1e-6;

c = model.LLChires(p.dirc, p.dirc, p.nx, p.ny, p.nz, p.tini, p.tref, p.tend, p.steps,p.rate, p.timedelta);

## Read in the Data

trim_ml = 30; # in meters
post_hann = [1,1,0]; # this is passed in but not actually used... 

f = FldsNU();
f.load_mitgcm_data_trunc(data_dir,aux_data_dir,iters,dims, truncdims);
KPPviscA = f.load_mitgcm_field_trunc('KPPviscA', data_dir, iters,dims,truncdims);#f.load_mitgcm_field('KPPviscA',data_dir,iters,dims);
print('KPPviscA loaded');
KPPhbl = f.load_mitgcm_field_trunc('KPPhbl',data_dir,iters,dims[1:3], truncdims[0:4]);
print('KPPhbl loaded');
#KPPviscA = c.loadding_3D_data(p.dirc+'KPPviscA.%010i.data'%iters[0], maxlevel, 'tracer')
# Can we set f's KPPvisc equal to c's KPPviscA?




## - Compute KPP Diagnostic Parameters
(dU,dV,viscAz,dbloc) = f.compute_kpp_diag(post_hann,trim_ml);


## - Mask ML
Z = np.cumsum(f.thknss,2) - 2*f.thknss;
MLmask = (Z < repmat(KPPhbl,(1,1,f.nz,1)));
viscAz[MLmask] = 0; KPPviscA[MLmask] = 0;
#dbloc[MLmask] = 0; KPPdbloc[MLmask] = 0;
#dU[MLmask] = 0; KPPdU[MLmask] = 0;
#dV[MLmask] = 0; KPPdV[MLmask] = 0;
viscAzError = (KPPviscA-viscAz)/(np.mean(KPPviscA**2)**0.5);
#dblocError = (KPPdbloc-dbloc)/(np.mean(KPPdbloc**2)**0.5);
#dblocError2 = (KPPdbloc-dbloc)/(np.mean(KPPdbloc[:,:,25,0]**2)**0.5);
#dUError = (KPPdU-dU)/(np.mean(KPPdU**2)**0.5);
#dVError = (KPPdV-dV)/(np.mean(KPPdV**2)**0.5);



## Plot Colormaps

plt.close('all')

titl = 'viscAz Error at Z=' + str(Z[100,100,0]); 
savestr = 'KPP_viscAz_Error_horz';
fig = plt.figure(); ax = plt.gca(); ax.set_title(titl,size=fsize);
ax.set_ylabel('Y-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(viscAzError[:,:,25,0]);
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)

titl = 'viscAz Error at X = 53km';
savestr = 'KPP_viscAz_Error_vert';
fig = plt.figure(); ax = plt.gca(); ax.set_title(titl,size=fsize);
ax.set_ylabel('Z-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(np.transpose(np.flip(viscAzError[25,:,:,0],1)))
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)
"""
titl = 'dbloc Error at Z=' + str(Z[100,100,25]); 
savestr = 'KPP_dbloc_Error_horz';
fig = plt.figure(); ax = plt.gca(); ax.set_title(titl,size=fsize);
ax.set_ylabel('Y-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(dblocError[:,:,25,0]);
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)
"""
"""
titl = 'dbloc Error at X = 53km';
savestr = 'KPP_dbloc_Error_vert';
fig = plt.figure(); ax = plt.gca(); ax.set_title(titl,size=fsize);
ax.set_ylabel('Z-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(np.transpose(np.flip(dblocError[25,:,:,0],1)))
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)
"""
"""
titl = 'dbloc Error rescaled at Z=' + str(Z[100,100,25]); 
savestr = 'KPP_dbloc_Error_rescale_horz';
fig = plt.figure(); ax = plt.gca(); ax.set_title(titl,size=fsize);
ax.set_ylabel('Y-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(dblocError2[:,:,25,0],norm=colors.LogNorm(vmin=np.min(np.abs(dblocError2[dblocError2!=0])), vmax=np.max(np.abs(dblocError2[:,:,25,0]))));
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)
"""
"""
titl = 'dbloc Error rescaled at X = 53km';
savestr = 'KPP_dbloc_Error_rescale_vert';
fig = plt.figure(); ax = plt.gca(); ax.set_title(titl,size=fsize);
ax.set_ylabel('Z-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(np.transpose(np.flip(dblocError2[25,:,:,0],1)),norm=colors.LogNorm(vmin=np.min(np.abs(dblocError2[dblocError2!=0])), vmax=np.max(np.abs(dblocError2[25,:,:,0]))));
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)
"""
"""
titl = 'dU Error at Z=' + str(Z[100,100,25]); 
savestr = 'KPP_dU_Error_horz';
fig = plt.figure(); ax = plt.gca(); ax.set_title(titl,size=fsize);
ax.set_ylabel('Y-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(dUError[:,:,25,0]);
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)
"""
"""
titl = 'dU Error at X = 53km';
savestr = 'KPP_dU_Error_vert';
fig = plt.figure(); ax = plt.gca(); ax.set_title(titl,size=fsize);
ax.set_ylabel('Z-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(np.transpose(np.flip(dUError[25,:,:,0],1)))
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)
"""
"""
titl = 'dV Error at Z=' + str(Z[100,100,25]); 
savestr = 'KPP_dV_Error_horz';
fig = plt.figure(); ax = plt.gca(); ax.set_title(titl,size=fsize);
ax.set_ylabel('Y-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(dVError[:,:,25,0]);
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)
"""
"""
titl = 'dV Error at X = 53km';
savestr = 'KPP_dV_Error_vert';
fig = plt.figure(); ax = plt.gca(); ax.set_title(titl,size=fsize);
ax.set_ylabel('Z-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(np.transpose(np.flip(dVError[25,:,:,0],1)))
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)
"""

titl = 'KPPviscA at X = 53km';
savestr = 'KPPviscA_vert';
fig = plt.figure(); ax = plt.gca(); ax.set_title(titl,size=fsize);
ax.set_ylabel('Z-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(np.transpose(np.flip(KPPviscA[25,:,:,0],1)))
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)

titl = 'viscA at X = 53km';
savestr = 'viscA_vert';
fig = plt.figure(); ax = plt.gca(); ax.set_title(titl,size=fsize);
ax.set_ylabel('Z-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(np.transpose(np.flip(viscAz[25,:,:,0],1)))
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)
"""
titl = 'KPPdbloc Error at X = 53km';
savestr = 'KPPdbloc_vert';
fig = plt.figure(); ax = plt.gca(); ax.set_title(titl,size=fsize);
ax.set_ylabel('Z-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(np.transpose(np.flip(KPPdbloc[25,:,:,0],1)))
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)
"""
"""
titl = 'dbloc at X = 53km';
savestr = 'dbloc_vert';
fig = plt.figure(); ax = plt.gca();
ax.set_title(titl,size=fsize);
ax.set_ylabel('Z-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(np.transpose(np.flip(dbloc[25,:,:,0],1)))
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)
"""
"""
titl = 'dU at X = 53km';
savestr = 'dU_vert';
fig = plt.figure(); ax = plt.gca(); ax.set_title(titl,size=fsize);
ax.set_ylabel('Z-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(np.transpose(np.flip(dU[25,:,:,0],1)), vmin=-cmax, vmax=cmax)
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)
"""
"""
titl = 'KPPdU at X = 53km';
savestr = 'KPPdU_vert';
fig = plt.figure(); ax = plt.gca(); ax.set_title(titl,size=fsize);
ax.set_ylabel('Z-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(np.transpose(np.flip(KPPdU[25,:,:,0],1)), vmin=-cmax, vmax=cmax)
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)
"""
"""
titl = 'dV at X = 53km';
savestr = 'dV_vert';
fig = plt.figure(); ax = plt.gca(); ax.set_title(titl,size=fsize);
ax.set_ylabel('Z-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(np.transpose(np.flip(dV[25,:,:,0],1)), vmin=-cmax, vmax=cmax)
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)
"""
"""
titl = 'KPPdV at X = 53km';
savestr = 'KPPdV_vert';
fig = plt.figure(); ax = plt.gca(); ax.set_title(titl,size=fsize);
ax.set_ylabel('Z-Position (cells)',fontsize=fsize-4)
ax.set_xlabel('X-Position (cells)',fontsize=fsize-4)
c = ax.pcolor(np.transpose(np.flip(KPPdV[25,:,:,0],1)), vmin=-cmax, vmax=cmax)
fig.colorbar(c, ax=ax)
fig.savefig(save_dir + savestr)
"""







































