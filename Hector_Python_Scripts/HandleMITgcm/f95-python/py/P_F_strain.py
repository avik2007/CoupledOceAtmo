import matplotlib
import math
import numpy as np
import strain
import shear
import full_strain
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
from matplotlib import colors, ticker, cm
import os
import warnings
import rms
import find

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
x = np.memmap(os.path.join(prnt,'XG_480x877'),dtype,shape=(ny,nx),mode='r')
y = np.memmap(os.path.join(prnt,'YG_480x877'),dtype,shape=(ny,nx),mode='r')
# =========================

# find coordinates of the regio
# A:
a_lon = [-130,-120]
a_lat = [20,25]
cor_sw,cor_ws = find.find_index(x,y,a_lon[0],a_lat[0])
cor_se,cor_es = find.find_index(x,y,a_lon[1],a_lat[0])
cor_ne,cor_en = find.find_index(x,y,a_lon[1],a_lat[1])
cor_nw,cor_wn = find.find_index(x,y,a_lon[1],a_lat[1])
print a_lat[0],a_lon[0],a_lat[1],a_lon[1]
print y[cor_sw,cor_ws], x[cor_sw,cor_ws]
print y[cor_ne,cor_en], x[cor_ne,cor_en]
print y[cor_sw,cor_ws],y[cor_ne,cor_en]
#print x[cor_sw,cor_ws],x[cor_ne,cor_en]
print cor_sw,cor_ne
print cor_ws,cor_en


# ===== dir =====
list1 = os.listdir(ud)
list1.sort()
list2 = os.listdir(vd)
list2.sort()
ku = len(list1)
kv = len(list2)
print ku,kv
RMS = np.empty(ku)
# ================

f = 0.000037
layer = 7
#=====Loop for  ====
for ii in range(0,ku-1):
    # read binary
    u=np.memmap(os.path.join(ud,list1[ii]),dtype,shape=(nz,ny,nx),mode='r')
    v=np.memmap(os.path.join(vd,list2[ii]),dtype,shape=(nz,ny,nx),mode='r')
    # shape matrix`
    m,n = np.shape(u[layer,cor_sw:cor_ne,cor_ws:cor_en])
    u = u[layer,cor_sw:cor_ne,cor_ws:cor_en]
    v = v[layer,cor_sw:cor_ne,cor_ws:cor_en]
    # Calculat strain at traicer points
    s1 = strain.st(u,v,dxg[cor_sw:cor_ne,cor_ws:cor_en],dyg[cor_sw:cor_ne,cor_ws:cor_en],rac[cor_sw:cor_ne,cor_ws:cor_en],m,n)
    s2 = shear.sh(u,v,dxc[cor_sw:cor_ne,cor_ws:cor_en],dyc[cor_sw:cor_ne,cor_ws:cor_en],raz[cor_sw:cor_ne,cor_ws:cor_en],m,n)
    s = full_strain.s(s1**2,s2**2,m,n)
    s = np.sqrt(s)/f
    # RMS =======
    RMS[ii] = rms.var(s[~np.isnan(s)])
    #print np.shape(s1)
    #print np.shape(s2)
    #print np.shape(s)

# Coriolis freq
#f = 0.000037
#levels = np.arange(-1.5,1.5,0.1)
#cs = plt.contourf(s/f,levels,cmap=cm.jet)
#cb = plt.colorbar(cs,extend='both',pad=0.05)
#plt.show()
#scipy.io.savemat('s1.mat',dict(s1=s1))
#scipy.io.savemat('s2.mat',dict(s2=s2))
scipy.io.savemat('s_reg_c.mat',dict(RMS=RMS))



