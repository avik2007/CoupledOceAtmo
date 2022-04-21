def lonlat2xy(lon,lat):
    """
    convert lat lon to x and x
    x,y = lonlat2xy(lon,lat)
    x,y, are 2d meshgrids
    """
    from pylab import meshgrid,cos,pi
    r = 6371.e3
    if lon.ndim == 1:
       lon,lat=meshgrid(lon,lat)
    x = 2*pi*r*cos(lat*pi/180.)*lon/360.
    y = 2*pi*r*lat/360.
    return x,y
