import numpy as np
import scipy.io
import os
import matplotlib
#matplotlib.use('GTK')
#from pylab import *
from matplotlib.patches import Patch
#import matplotlib
#matplotlib.rc('text',usetex=True)
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
from matplotlib import colors, ticker, cm




# parameters ====
ny = 877
nx = 480
nz = 88

dtype = np.dtype('>f4')
prnt = '/data25/llc/llc_2160/regions/CalCoast/grid/'
#=================

# ====== load grid ==========
dxc = np.memmap(os.path.join(prnt,'XC_480x877'),dtype,shape=(ny,nx),mode='r')
dyc = np.memmap(os.path.join(prnt,'YC_480x877'),dtype,shape=(ny,nx),mode='r')
#===========================

# find index
def find_index(y_array, x_array, y_point, x_point):
    distance = (y_array-y_point)**2 + (x_array-x_point)**2
    idy,idx = np.where(distance==distance.min())
    return idy[0],idx[0]

#def do_all(y_array, x_array, points):
#    store = []
#    for i in xrange(points.shape[1]):
#        store.append(find_index_of_nearest_xy(y_array,x_array,points[0,i],points[1,i]))
#    return store

plt.plot(dxc,dyc,'.k')
plt.show()
idy,idx = find_index(dyc,dxc,20,-110)
plt.scatter(dxc[idy,idx],dyc[idy,idx],'o',color='r')
print dxc[idy,idx],dyc[idy,idx]
