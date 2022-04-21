#import matplotlib
import math
import numpy as np
import strain
import shear
import full_strain
import scipy.io
import h5py
#import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
#import matplotlib.cm as cm
#from matplotlib import colors, ticker, cm
import os
import warnings
import dif
import rms

nx = 480
ny = 877
nz = 88

# format
dtype = np.dtype('>f4')
# load data
prnt = '/data25/llc/llc_2160/regions/CalCoast/grid/'
ud='/data25/llc/llc_2160/regions/CalCoast/U/'
vd = '/data25/llc/llc_2160/regions/CalCoast/V/'


# ====== load grid ==========
dxc = np.memmap(os.path.join(prnt,'DXC_480x877'),dtype,shape=(ny,nx),mode='r')
dyc = np.memmap(os.path.join(prnt,'DYC_480x877'),dtype,shape=(ny,nx),mode='r')
rac = np.memmap(os.path.join(prnt,'RAC_480x877'),dtype,shape=(ny,nx),mode='r')
dxg = np.memmap(os.path.join(prnt,'DXG_480x877'),dtype,shape=(ny,nx),mode='r')
dyg = np.memmap(os.path.join(prnt,'DYG_480x877'),dtype,shape=(ny,nx),mode='r')
raz = np.memmap(os.path.join(prnt,'RAZ_480x877'),dtype,shape=(ny,nx),mode='r')
# =========================

# =========================

# ===== load matfiles =====
mu = h5py.File('/data4/HT_output/llc_2160/daily_avg/temp_2day_averagedu.mat')
u = mu['bajas'][:,:,:]

mv = h5py.File('/data4/HT_output/llc_2160/daily_avg/temp_2day_averagedv.mat')
v = mv['bajas'][:,:,:]

t,n,m = np.shape(u)

print n,m,t
RMS = np.empty(t)
f = 0.000037
#=====Loop for  ====
for ii in range(0,t):
    # handling the nan's
    s1 = strain.st(u.T[:,:,ii],v.T[:,:,ii],dxg[200:350,200:350],dyg[200:350,200:350],rac[200:350,200:350],150,150)
    s2 = shear.sh(u.T[:,:,ii],v.T[:,:,ii],dxc[200:350,200:350],dyc[200:350,200:350],raz[200:350,200:350],150,150)
    s = full_strain.s(s1**2,s2**2,150,150)
    s = np.sqrt(s)/f
    RMS[ii] = rms.var(s[~np.isnan(s)])
    #print np.shape(xi)

# Coriolis freq
#f = 0.000037
#levels = np.arange(-1.5,1.5,0.1)
#cs = plt.contourf(xi,levels,cmap=cm.jet)
#cb = plt.colorbar(cs,extend='both',pad=0.05)
#plt.show()
scipy.io.savemat('RMS_str_2daily_avg.mat',dict(RMS=RMS))
