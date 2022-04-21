import matplotlib
import math
import numpy as np
import strain
import shear
import full_strain
import divergence
import vorticity
import ow
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
from matplotlib import colors, ticker, cm
import os
import warnings
import rms

nx = 8640
ny = 4320
nz = 1

# format
dtype = np.dtype('>f4')
# load data
prnt = '/data24/llc/llc_2160/regions/latlon/grid/'
ud='/data24/llc/llc_2160/regions/latlon/U/'
vd = '/data24/llc/llc_2160/regions/latlon/V/'
# ----

# ====== load grid ==========
dxc = np.memmap(os.path.join(prnt,'DXC_8640x4320'),dtype,shape=(ny,nx),mode='r')
dyc = np.memmap(os.path.join(prnt,'DYC_8640x4320'),dtype,shape=(ny,nx),mode='r')
rac = np.memmap(os.path.join(prnt,'RAC_8640x4320'),dtype,shape=(ny,nx),mode='r')
dxg = np.memmap(os.path.join(prnt,'DXG_8640x4320'),dtype,shape=(ny,nx),mode='r')
dyg = np.memmap(os.path.join(prnt,'DYG_8640x4320'),dtype,shape=(ny,nx),mode='r')
raz = np.memmap(os.path.join(prnt,'RAZ_8640x4320'),dtype,shape=(ny,nx),mode='r')
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
# ====================

layer =7
#=====Loop for  ====
#for ii in range(0,1):
for ii in range(4000,4001):
    # read binary
    u=np.memmap(os.path.join(ud,list1[ii]),dtype,shape=(nz,ny,nx),mode='r')
    v=np.memmap(os.path.join(vd,list2[ii]),dtype,shape=(nz,ny,nx),mode='r')
    
    # Calculat strain at xi points -----
    s1 = strain.st(u,v,dxg,dyg,rac,ny,nx)
    s2 = shear.sh(u,v,dxc,dyc,raz,ny,nx)
    s = full_strain.s(s1**2,s2**2,ny,nx)
    # ---------------
    #
    #----- Divergence at tracer points-------
    # after the calculation we needs to interpolate 
    # the results at the xi points
    #dv = divergence.d(u[layer,:,:],v[layer,:,:],dxg,dyg,rac,ny,nx)
    # -------------------------------
    #
    # ----- Relative vorticity -----------
    xi = vorticity.xi(u,v,dxc,dyc,raz,ny,nx)
    # -----------------------------------
    #
    # ------- OW --------
    # remember that full-strain gives us
    # s**2
    #print np.shape(xi)
    #print np.shape(s)
    ooww = ow.okubo(xi**2,s,ny,nx)
    #--------------------
    #RMS[ii] = rms.var(ooww[~np.isnan(ooww)])

#f = 0.000037
#f2 = f**2
#levels = np.arange(-1,1,0.1)
#cs = plt.contourf(ow/f2,levels,cmap=cm.jet)
#cb = plt.colorbar(cs,extend='both',pad=0.05)
#plt.show()
scipy.io.savemat('ow_index_4000.mat',dict(ooww=ooww))
