import os
import numpy as np
#from matplotlib import pyplot as plt
#os.environ['PROJ_LIB'] = '/nobackupp2/estrobac/miniconda/envs/xmitgcm/share/proj'
#from mpl_toolkits.basemap import Basemap
import pyresample


class LLCMap_nea_split:
    
    def __init__(self, lat_in, lon_in, lat, lon, projection='longlat', lat_0=0, lon_0=0,radius=10e3):
        
        # Extract LLC 2D coordinates
        lons_1d = lon_in.ravel()
        lats_1d = lat_in.ravel()

        # Define original grid
        self.source_def = pyresample.geometry.SwathDefinition(lons=lons_1d, lats=lats_1d)
        self.new_grid_lon, self.new_grid_lat = np.meshgrid(lon, lat)
        self.target_def = pyresample.geometry.AreaDefinition('Global',
                                             'latlon',
                                             'areaD',
                                             {'a': '6378144.0', 'b': '6356759.0',
                                             'lat_0': str(lat_0),
                                            'lon_0': str(lon_0), 'proj': projection},
                                             lon.shape[0], lat.shape[0],
                                             [-180, -90, 180, 90],
                                             lons=self.new_grid_lon, lats=self.new_grid_lat)

        self.t_params, self.s_params, self.input_idxs, self.idx_ref = pyresample.kd_tree.get_neighbour_info(self.source_def, 
                                                                                            self.target_def, neighbours=1,
                                                                                            nprocs=1,radius_of_influence=radius,
                                                                                            reduce_data=True, segments=None,
                                                                                            epsilon=0)

    def __call__(self, da):

        field = pyresample.kd_tree.get_sample_from_neighbour_info('nn',self.target_def.shape,da.ravel(), self.t_params, self.s_params,
                                                  self.input_idxs,fill_value=np.nan)
       
        return field



