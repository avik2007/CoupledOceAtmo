import numpy as np
import pylab as plt
import sys
sys.path.append('/nobackup/htorresg/cmocean-master/')
import cmocean
from matplotlib.colors import LogNorm
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib as  mpl


mpl.rcParams['axes.linewidth'] =2
mpl.rc('xtick',labelsize=15)
mpl.rc('ytick',labelsize=15)


######################
prnt = '/nobackup/htorresg/air_sea/reanalysis/'
nm_era = 'KEwinds_wf_ERA5_KEx.npz'
nm_ecm = 'KEwinds_wf_ECMWF_KEx.npz'

prnt_coas = '/nobackup/htorresg/air_sea/proc/'
nm_coas = 'KEwinds_WF_currents_45days_wf_GEOS5_2020011921_2020050923_KEx_surf.npz'


era = np.load(prnt+nm_era)
ecm = np.load(prnt+nm_ecm)

kiso_era = era['kiso'][1:]
om_era = era['om'][1:]
Eiso_era = era['Eiso'][1:,1:]*2*30

kiso_ecm = ecm['kiso'][1:]
om_ecm = ecm['om'][1:]
Eiso_ecm = ecm['Eiso'][1:,1:]*2*30

coas = np.load(prnt_coas+nm_coas)
kiso_coas = coas['kiso'][1:]
om_coas = coas['om'][1:]
Eiso_coas = coas['Eiso'][1:,1:]*2*30


#########################
fig = plt.figure(figsize=(12,4))
plt.subplots_adjust(wspace=0.4)


ax1=fig.add_subplot(131)
cs=plt.pcolormesh(kiso_ecm,om_ecm,
                  (Eiso_ecm.T*kiso_ecm[None,...]*om_ecm[...,None]),
                  vmin=1e-2,vmax=5e0,cmap='rainbow',norm=LogNorm())
ax1.set_yscale('log')
ax1.set_ylim(1/(24.*40),1/2.)
ax1.set_xscale('log')
ax1.set_xlim(1./1000.,1/10.)
plt.ylabel(r'Frequency [cph]',size=15,fontweight='normal')
plt.xlabel('Wavenumber [cpkm]',size=15)
ax1a = ax1.twiny()
ax1a.set_yscale('log')
ax1a.set_xscale('log')
ax1a.set_xlim(1./1000.,1/10.)
ax1a.set_xticks([1/1000.,1/100.,1/10.])
ax1a.set_xticklabels(['1000','100','10'])
ax1a.set_xlabel('Wavelength [km]',size=15)
ax1a = ax1.twinx()
ax1a.set_yscale('log')
ax1a.set_xscale('log')
ax1a.set_ylim(1/(24.*40.),1/2.)
ax1a.set_yticks([1/24.,1/(24.*7.),1/(24*30.)])
ax1a.set_yticklabels(['1','7','30'])
plt.text(1/50.,1/(5.),r'ECMWF',
         size=12,fontweight='bold',ha="center",va="center",
         bbox=dict(boxstyle="round",
                   ec='k',
                   fc='w',
                   ))




ax1=fig.add_subplot(132)
cs=plt.pcolormesh(kiso_era,om_era,
                  (Eiso_era.T*kiso_era[None,...]*om_era[...,None]),
                  vmin=1e-2,vmax=5e0,cmap='rainbow',norm=LogNorm())
ax1.set_yscale('log')
ax1.set_ylim(1/(24.*40),1/2.)
ax1.set_xscale('log')
ax1.set_xlim(1./1000.,1/10.)
plt.xlabel('Wavenumber [cpkm]',size=15)
ax1a = ax1.twiny()
ax1a.set_yscale('log')
ax1a.set_xscale('log')
ax1a.set_xlim(1./1000.,1/10.)
ax1a.set_xticks([1/1000.,1/100.,1/10.])
ax1a.set_xticklabels(['1000','100','10'])
ax1a.set_xlabel('Wavelength [km]',size=15)
ax1a = ax1.twinx()
ax1a.set_yscale('log')
ax1a.set_xscale('log')
ax1a.set_ylim(1/(24.*40.),1/2.)
ax1a.set_yticks([1/24.,1/(24.*7.),1/(24*30.)])
ax1a.set_yticklabels(['1','7','30'])
#ax1a.set_ylabel('Period [days]',size=15)
plt.text(1/40.,1/(5.),r'ERA5',
         size=12,fontweight='bold',ha="center",va="center",
         bbox=dict(boxstyle="round",
                   ec='k',
                   fc='w',
                   ))


ax1=fig.add_subplot(133)
cs=plt.pcolormesh(kiso_coas,om_coas,
                  (Eiso_coas.T*kiso_coas[None,...]*om_coas[...,None]),
                  vmin=1e-2,vmax=5e0,cmap='rainbow',norm=LogNorm())
ax1.set_yscale('log')
ax1.set_ylim(1/(24.*40),1/2.)
ax1.set_xscale('log')
ax1.set_xlim(1./1000.,1/10.)
plt.xlabel('Wavenumber [cpkm]',size=15)
ax1a = ax1.twiny()
ax1a.set_yscale('log')
ax1a.set_xscale('log')
ax1a.set_xlim(1./1000.,1/10.)
ax1a.set_xticks([1/1000.,1/100.,1/10.])
ax1a.set_xticklabels(['1000','100','10'])
ax1a.set_xlabel('Wavelength [km]',size=15)
ax1a = ax1.twinx()
ax1a.set_yscale('log')
ax1a.set_xscale('log')
ax1a.set_ylim(1/(24.*40.),1/2.)
ax1a.set_yticks([1/24.,1/(24.*7.),1/(24*30.)])
ax1a.set_yticklabels(['1','7','30'])
ax1a.set_ylabel('Period [days]',size=15)
plt.text(1/40.,1/(5.),r'COAS',
         size=12,fontweight='bold',ha="center",va="center",
         bbox=dict(boxstyle="round",
                   ec='k',
                   fc='w',
                   ))



cbar_ax=fig.add_axes([0.99,0.2,0.02,0.6])
fig.colorbar(cs,cax=cbar_ax).set_label(r'$\kappa$ $\times$ $\omega$ $\times$ $\widehat{KE}_{winds}$ [m$^{2}$/s$^{2}$]',size=12)


prntout = '/nobackup/htorresg/air_sea/figures/'
nmout='WF_KEwinds_KEx_ECMWF_ERA5.png'
plt.savefig(prntout+nmout,dpi=500,format='png',bbox_inches='tight')

plt.show()
