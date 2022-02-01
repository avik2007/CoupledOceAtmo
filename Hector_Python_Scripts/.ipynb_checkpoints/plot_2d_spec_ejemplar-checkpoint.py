# -*- Coding: utf-8 -*-
"""
Created on Jan 30 21:59 2017

@author: htorresg
"""

""" plot_wf hires simulations """
import numpy as np
from numpy import  pi
import os
import warnings
import glob
import scipy.io as sci
import matplotlib
import matplotlib as mpl
#mpl.use('Agg')
#from pylab import *
from matplotlib.patches import Patch
#import matplotlib
#matplotlib.rc('text',usetex=True)
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
from matplotlib import colors, ticker, cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.mathtext as mathtext
#import wf_spectrum
from scipy.interpolate import griddata
from scipy.interpolate import interp2d
import matplotlib.mathtext as mathtext
from matplotlib.colors import LogNorm
from scipy import signal
import h5py
#from mpl_toolkits.mplot3d import Axes3D
#:::::::::::::::::::::::::::::::::::::::::::::::::::

matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rc('xtick',labelsize=14)
matplotlib.rc('ytick',labelsize=14)


def coriolis(lat):
    omg = (1.)/24
    return 2*(omg)*np.sin((lat*3.141519)/180)

#========== Load ========
directory = '/nobackup/htorresg/air_sea/ocean-atmos/boxes/'
PrntOut='/nobackup/htorresg/DopplerScat/figures/' # <
#filename = 'KE_wf_California_Current_z1m_r250m_October_20121002000000_20121014000000.npz'  # <=== change
#filename='WVEL_CCS_R500m_z40m_LargeBox_20120328000000_20120510010000.npz'
#filename='Wvel_total_wf_transition_z40m.npz'
#filename = 'RV_tot_wf_transition_z40m_zref100m.npz'
#filename = 'DV_wf_Residual_from_SpatialFilter_res_upwelling_4000.0_layer14.npz'
#filename = 'Wvel_wf_tot_offshore_z40m_zref200m_20120401000000_20120420000000_Jun02.npz'
#filename = 'WT_wf_tot_transition_zref100m_20120401000000_20120420000000.npz'
filename='cospectrum_WT_GS_z45m.npz'
#figurename='Wvel_residual_wf_offshore_z10m.png'
#figurename = 'KE_total_motions_10days_LargeBox_z40m' # <=========== change
#output_format = '.png'   # <============ change
#mat=np.load(directory+'Deformation_Rd_offshore_April.npz','r')
#mat = sci.loadmat(directory+'radius_from_modver.mat')
#radii = (mat['raddi'][...]) ## km
#print(radii[0:11]) 

data = np.load(directory+filename)
kiso = data['kiso']
Eiso = data['Eiso']*4.2e6
om = data['om']


#====================================================
#===================================================
kk = kiso[1:]
omm = om[1:]
ff = coriolis(38)
f0 =  0.89e-4
print(f0)
kiso = np.logspace(-3,0.,100)

#################
fig = plt.figure(figsize=(6,5))

#fig = plt.figure(figsize=(10,6))
#ax1 = plt.subplot2grid((3,6),(0,1),colspan=2)
#ax2 = plt.subplot2grid((3,6),(1,0),rowspan=2)
#ax3 = plt.subplot2grid((3,6),(1,1),rowspan=2, colspan=2)


ax1 = fig.add_subplot(111)
cs=plt.pcolormesh(kk,omm,1*Eiso.T[1:,1:]*kk[None,...]*omm[...,None],
                   shading='flat',cmap='bwr')
ax1.set_yscale('log')
ax1.set_xscale('log')
plt.xlabel(r'$\kappa$ [cpkm]',size=16)
plt.ylabel(r'Frequency [cph]',size=16)
plt.clim([0.1,-0.1])
plt.colorbar()
ax1.set_ylim([1./(9*24.),1])
ax1.set_xlim([1./(500.),1/8.])
#ax3.set_yticks([])
#plt.title('Mode-1',fontweight='bold',size=20)
#

#plt.savefig(PrntOut+figurename+output_format,format='png',
#            dpi=500,bbox_inches='tight')

plt.show()


