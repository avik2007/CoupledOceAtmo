import pylab as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.cm as cm
import matplotlib
import numpy as np

def CCS(lon,lat,var,vmin,vmax,cmap,label,title,nameout):
    matplotlib.rc('xtick',labelsize=19)
    matplotlib.rc('ytick',labelsize=19)
    fig=plt.figure(figsize=(6,6))
    #
    ax = fig.add_subplot(111)
    # :::: Plotting ::::::::::
    #m = Basemap(width=4000000,height=1900000,projection='aea',resolution='l',
    #            lat_1=31,lat_2=41,lat_0=35,lon_0=-120.)
    ### GS: lat_1=20,lat_2=60,lat_0=35,lon_0=-60.
    m=Basemap(projection='mill',lat_ts=30,llcrnrlon=lon.min(), \
    urcrnrlon=lon.max(),llcrnrlat=lat.min(),urcrnrlat=lat.max(), \
    resolution='l')

    im = m.pcolormesh(lon,lat,var,vmin=vmin,vmax=vmax,
                       cmap=cmap,latlon=True)
     
    cbar = m.colorbar(im,'right',shrink=0.5,extend='both',size='4%',pad='5%')
    cbar.set_label(label,size=15)
    plt.title(title,size=16)
    
    m.shadedrelief()
    
    # Built-up map boundaries
    m.drawcoastlines()
    #m.drawlsmask(land_color='white',ocean_color='0.9',lakes=False)
    m.drawcountries()
    m.drawstates()
    #
    m.drawmeridians(np.arange(0,360,5),labels=[False,False,False,True])
    m.drawparallels(np.arange(-90,90,5),labels=[True,False,False,False])
    plt.savefig(nameout,
                dpi=650,format='png',bbox_inches='tight')
    plt.show()
