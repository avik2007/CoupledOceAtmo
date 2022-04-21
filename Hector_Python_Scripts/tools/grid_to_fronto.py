def prepare(c,maxlevel,iln,ilt):
    import tools
    
    """
    c is the structure 
    that contains grid info and model settings from handle_mitgcm
    """

    ### call grid info
    c.load_grid()
    c.load_dxc()
    c.load_duv()
    c.load_dxf()
    c.load_dxg()
    c.coriolis()
    c.load_raz()
    c.load_ras()
    c.load_raw()
    c.load_rac()
    hFacS = c.loadding_3D_masks('hFacS.data',maxlevel)
    hFacW = c.loadding_3D_masks('hFacW.data',maxlevel)
    hFacC = c.loadding_3D_masks('hFacC.data',maxlevel)
    c.load_drf()

    #### subdomain
    ilnmnc,ilnmxc,iltmnc,iltmxc=tools.extract_subdomain(c.lon,c.lat,iln,ilt)
    ilnmng,ilnmxg,iltmng,iltmxg=tools.extract_subdomain(c.long,c.latg,iln,ilt)#

    #### grid structure ###
    grid = {}
    grid['Time'] = c.timeline()
    grid['ilnc']=[ilnmnc,ilnmxc]
    grid['iltc']=[iltmnc,iltmxc]
    grid['lonc'] = c.lon[ilnmnc:ilnmxc,iltmnc:iltmxc]
    grid['latc'] = c.lat[ilnmnc:ilnmxc,iltmnc:iltmxc]
    grid['dxc'] = c.dxc[ilnmnc:ilnmxc,iltmnc:iltmxc]
    grid['dyc'] = c.dyc[ilnmnc:ilnmxc,iltmnc:iltmxc]
    grid['dxf']=c.dxf[ilnmnc:ilnmxc,iltmnc:iltmxc]
    grid['dyf']=c.dyf[ilnmnc:ilnmxc,iltmnc:iltmxc]
    grid['dxg']=c.dxg[ilnmnc:ilnmxc,iltmnc:iltmxc]
    grid['dyg']=c.dyg[ilnmnc:ilnmxc,iltmnc:iltmxc]
    grid['ras']=c.ras[ilnmnc:ilnmxc,iltmnc:iltmxc]
    grid['raw']=c.raw[ilnmnc:ilnmxc,iltmnc:iltmxc]
    grid['rac']=c.rac[ilnmnc:ilnmxc,iltmnc:iltmxc]
    grid['depth'] = c.dpt[:maxlevel]
    grid['thick'] = c.thk[:maxlevel] 
    print(grid['thick'][0:2])
    grid['drf'] = c.drf[:maxlevel]
    print(grid['drf'][0:2])
    grid['hFacC'] = hFacC[:,ilnmnc:ilnmxc,iltmnc:iltmxc]
    grid['hFacW'] = hFacW[:,ilnmnc:ilnmxc,iltmnc:iltmxc]
    grid['hFacS'] = hFacS[:,ilnmnc:ilnmxc,iltmnc:iltmxc]
    grid['m'] = grid['lonc'].shape[0]
    grid['n'] = grid['lonc'].shape[1]
    grid['k'] = len(c.dpt[:maxlevel])

    return grid
