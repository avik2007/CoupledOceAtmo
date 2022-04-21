import matplotlib
import math
import numpy as np
import KE
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
from matplotlib import colors, ticker, cm
import os
import warnings
import rms

ny = 877
nx = 480
nz = 88

# format
dtype = np.dtype('>f4')
# load data
prnt = '/data25/llc/llc_2160/regions/CalCoast/grid/'
ud='/data25/llc/llc_2160/regions/CalCoast/U/'
vd = '/data25/llc/llc_2160/regions/CalCoast/V/'
prnt_out = '/data25/home/hectorg/ecco/ecco4km/proces/vorticity/'


# ====== load grid ==========
#dx = np.memmap(os.path.join(prnt,'DXC_480x877'),dtype,shape=(ny,nx),mode='r')
#dy = np.memmap(os.path.join(prnt,'DYC_480x877'),dtype,shape=(ny,nx),mode='r')
#raz = np.memmap(os.path.join(prnt,'RAZ_480x877'),dtype,shape=(ny,nx),mode='r')
# =========================

# ===== dir =====
list1 = os.listdir(ud)
list1.sort()
list2 = os.listdir(vd)
list2.sort()
ku = len(list1)
kv = len(list2)
print ku,kv
RMS = np.empty(ku)

#=====Loop for  ====
for ii in range(ku-1):
    # read binary
    u=np.memmap(os.path.join(ud,list1[ii]),dtype,shape=(nz,ny,nx),mode='r')
    v=np.memmap(os.path.join(vd,list2[ii]),dtype,shape=(nz,ny,nx),mode='r')
    # shape matrix
    l,m,n = np.shape(u)
    # output file
    name = list1[ii]
    ke = KE.ke(u[7,0:350,0:350],v[7,0:350,0:350],350,350)
    # handling the nan's
    #print np.shape(ke)
     # RMS:
    RMS[ii] = rms.var(ke[~np.isnan(ke)])

# Coriolis freq
#f = 0.000037
#levels = np.arange(0,0.2,0.01)
#cs = plt.contourf(ke,levels,cmap=cm.Blues)
#cb = plt.colorbar(cs,extend='both',pad=0.05)
#plt.show()
# === save ===
fileout ='/data17/home/hectorg/ecco/ecco4km/proces/ke_rms_4km.mat'
scipy.io.savemat('vort.mat',dict(RMS=RMS))
