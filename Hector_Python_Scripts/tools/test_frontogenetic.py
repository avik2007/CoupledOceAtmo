"""
 Frontogenetic vector test
"""
import numpy as np
from numpy import  pi
import os
import warnings
import glob
import scipy.io as sci
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import matplotlib.cm as cm
import Frontogenetic_vector as fronto
#
matplotlib.rcParams['axes.linewidth'] = 2
matplotlib.rc('xtick',labelsize=19)
matplotlib.rc('ytick',labelsize=19)
matplotlib.rc('text',usetex=True)
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
matplotlib.rcParams['font.family']='Times New Roman'
#
#dataLat=[20:0.01544:50] -dataLong=[-135:0.02084:-104]
lat = np.arange(20.0,50,0.01544)
lon = np.arange(-135,-104,0.02084)
print('Coordinates size')
print('Lon: ',lon.shape)
print('Lat: ',lat.shape)
x,y = np.meshgrid(lon,lat)
#
prnt = '/Users/hectorg/Documents/JPL_paper/yackar/matfiles/'
rhoname = 'Rho_0715_2012_3D.mat'
uname   = 'U3D_0715_2012.mat'
vname   = 'V3D_0715_2012.mat'
wname   = 'W3D_0715_2012.mat'
zname= '/Users/hectorg/Documents/JPL_paper/Ultra_high_res/KuroshioExt/files/raw/grid/'
Z = sci.loadmat(zname+'thk90.mat')
thk = Z['thk90'][:,0:23]

# Reading
dat = h5py.File(prnt+rhoname,'r')
rho = dat['Rho'][:,:,:]
rho = np.swapaxes(rho,0,2)
rho = np.swapaxes(rho,0,1)
dat = h5py.File(prnt+uname,'r')
u = dat['Data'][:,:,:]
u = np.swapaxes(u,0,2)
u = np.swapaxes(u,0,1)
dat = h5py.File(prnt+vname,'r')
v = dat['Data'][:,:,:]
v = np.swapaxes(v,0,2)
v = np.swapaxes(v,0,1)
dat = h5py.File(prnt+wname,'r')
w = dat['Data'][:,:,:]
w = np.swapaxes(w,0,2)
w = np.swapaxes(w,0,1)
print('var size')
print('rho: ',rho.shape)
print('u: ',u.shape)
print('v: ',v.shape)
print('w: ',w.shape)

# nan's
rho[rho == 999.8444641620168] = np.nan
u[u == 0.] = np.nan
v[v == 0.] = np.nan

# grid
dx = np.ones((u.shape))*2000

# :::::::::::::::::::::::::::::::::::::::::::::::
f = fronto.coriolis(y)

# Frontogenetic vector
Qx,Qy,rhox,rhoy,twi_x,twi_y,dUadz,dVadz=fronto.Q(u[:,:,:],v[:,:,:],
                                     rho[:,:,:],dx[:,:,:],
                                     dx[:,:,:],thk,f)
#
# |drho/dx|^2
mag = np.sqrt(rhox**2 + rhoy**2)

l,m,n = Qx.shape
# Laplacian of Rho
LapRho = np.empty((l,m,n))
for i in range(n):
    LapRho[:,:,i] = fronto.dT2Dx2(rho[:,:,i],dx[:,:,i],dx[:,:,i])
#

magQ = np.sqrt(Qx**2 + Qy**2)
divQ = np.empty((l,m,n))
divTW = np.empty((l,m,n))
for i in range(n):
    # Divergence of Q
    QXx,QXy = np.gradient(Qx[:,:,i],2000,2000) ## data, dx,dy
    QYx,QYy = np.gradient(Qy[:,:,i],2000,2000)
    divQ[:,:,i] = QXx + QYy
    TWXx,TWXy = np.gradient(twi_x[:,:,i],2000,2000)
    TWYx,TWYy = np.gradient(twi_y[:,:,i],2000,2000)
    divTW[:,:,i] = TWXx + TWYy
#
# Qxx,Qxy = np.gradient(Qx[:,:,12]+twi_x[:,:,12],2000,2000)
# Qyx,Qyy = np.gradient(Qy[:,:,12]+twi_y[:,:,12],2000,2000)
# divQtot = Qxx + Qyy
layer = 14
uy,ux = np.gradient(u[:,:,layer],2000,2000)
vy,vx = np.gradient(v[:,:,layer],2000,2000)
rv = vx - uy
strain = ((ux-vy)**2 + (vx+uy)**2)**.5
print('strain:',strain.shape)
# ::::::::::::::::::::::::::::::::::::::::::::::::

# # ::::::::: Vertical velocity due to vertical mixing :::::
Av = 5e-5 # m2 s-1
g = 9.81 # m s-2
rho_ref = 1024
Wav = g/((f[1:-1,1:-1]**2)*rho_ref)*Av*LapRho[:,:,layer]
# # :::::::::::::::::::::::::

# ::::::: Saving the outcomes :::::::
sci.savemat('Frontogenetic_Qtw_0715_layer'+str(layer)+'.mat',
            dict(divQ=divQ,x=x[1:-1,1:-1],y=y[1:-1,1:-1]))
sci.savemat('Frontogenetic_Qtwi_0715_layer'+str(layer)+'.mat',
            dict(divTW=divTW,x=x[1:-1,1:-1],y=y[1:-1,1:-1]))
sci.savemat('RelativeVorticity_and_Strain_0715_layer'+str(layer)+'.mat',dict(Rv=rv,strain=strain))
sci.savemat('W_glr86_0715_layer'+str(layer)+'.mat',
            dict(Wav=Wav,x=x[1:-1,1:-1],y=y[1:-1,1:-1]))
sci.savemat('magGradRho_0715_layer'+str(layer)+'.mat',
            dict(gradRho=mag,x=x[1:-1,1:-1],y=y[1:-1,1:-1]))

# print('x: ',x.shape)
# print('y: ',y.shape)
# print('rv: ',rv.shape)
# print('strain: ',strain.shape)
# print('magQ: ',magQ.shape)
# print('dvQ: ',divQ.shape)
#
# fast inspection
plt.figure(figsize=(12,7))
plt.subplot(221)
plt.pcolormesh(mag[30:730,30:730,layer],vmin=0,vmax=1e-4,cmap=cm.rainbow)
plt.colorbar()
plt.subplot(222)
plt.pcolormesh(w[30:730,30:730,layer]*86400,vmin=-15,vmax=15,cmap=cm.jet)
plt.colorbar(shrink=0.5)
# plt.ylim([20,32])
# plt.xlim([-130,-111])
plt.title('W-vel')
plt.subplot(223)
plt.pcolormesh(strain[30:730,30:730],vmin=0,vmax=4e-5,cmap=cm.Reds)
plt.colorbar(shrink=0.5)
# plt.ylim([20,32])
# plt.xlim([-130,-111])
plt.subplot(224)#x[1:-1,1:-1],y[1:-1,1:-1]
plt.pcolormesh(divQ[30:730,30:730,layer],vmin=-5e-16,vmax=5e-16,cmap=cm.bwr)
plt.colorbar(shrink=0.5)

# plt.ylim([20,32])
# plt.xlim([-130,-111])
# plt.subplot(224)
# plt.pcolormesh(x[1:-1,1:-1],y[1:-1,1:-1],divTW,vmin=-5e-16,vmax=5e-16,cmap=cm.bwr)
# plt.colorbar(shrink=0.5)
# plt.ylim([20,32])
# plt.xlim([-130,-111])
plt.show()
