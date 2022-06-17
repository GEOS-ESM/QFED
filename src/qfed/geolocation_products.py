'''
VIIRS and MODIS geolocation products.
'''


import sys
import abc

import numpy as np
import netCDF4 as nc
from pyhdf import SD

if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})



class GeolocationProduct(ABC):
    '''
    Abstract class for accessing Level 1 geolocation data.
    '''
     
    def __init__(self, verbosity=0):
        self.verbosity = verbosity

    def get_coordinates(self, file):
        '''
        Template method to read and return geolocation data.
        '''

        lon, lon_range = self.get_longitude(file)
        lat, lat_range = self.get_latitude(file)

        valid = np.logical_and(np.logical_and(lon >= lon_range[0], lon <= lon_range[1]), 
                               np.logical_and(lat >= lat_range[0], lat <= lat_range[1]))     

        return lon, lat, valid, lon_range, lat_range

    def message_on_file_error(self, file):
        if self.verbosity >= 1:
            print('[w] cannot open geo-location file <{0:s}>, ignoring granule'.format(file))

    @abc.abstractmethod
    def get_longitude(self):
        return

    @abc.abstractmethod
    def get_latitude(self):
        return



class MODIS(GeolocationProduct):
    '''
    MODIS geolocation product (MOD03, MYD03) reader.
    '''

    # proper bounds can be obtained from the metadata:
    # [EAST|WEST|SOUTH|NORTH]BOUNDINGCOORDINATE

    def __read(self, file, variable):
        try:
            mxd03 = SD.SD(file)
        except SD.HDF4Error:
            self.message_on_file_error(file)
            return

        data = mxd03.select(variable).get()
        min_val, max_val = mxd03.select(variable).getrange()
        return data, (min_val, max_val)

    def get_longitude(self, file, ):
        return self.__read(file, 'Longitude')

    def get_latitude(self, file):
        return self.__read(file, 'Latitude')



class VIIRS_NPP(GeolocationProduct):
    '''
    VIIRS S-NPP geolocation product (NPP_IMFTS_L1) reader. 
    '''

    lon_valid_range = (-180.0, 180.0)
    lat_valid_range = ( -90.0,  90.0)

    def __read(self, file, variable):
        try:
            vnp03 = SD.SD(file)
        except SD.HDF4Error:
            self.message_on_file_error(file)
            return

        data = vnp03.select(variable).get()
        return data

    def get_longitude(self, file):
        return self.__read(file, 'Longitude'), VIIRS_NPP.lon_valid_range

    def get_latitude(self, file):
        return self.__read(file, 'Latitude'), VIIRS_NPP.lat_valid_range



class VIIRS_JPSS(GeolocationProduct):
    '''
    VIIRS JPSS1 geolocation product (VJ103IMG) reader. 
    '''

    def __read(self, file, variable):
        try:
           vjs03 = nc.Dataset(file)
        except IOError:
           self.message_on_file_error(file)
           return

        data = vjs03.groups['geolocation_data'].variables[variable]
        
        if variable == 'longitude':
            min_val, max_val = vjs03.geospatial_lon_min, vjs03.geospatial_lon_max
        elif variable == 'latitude':
            min_val, max_val = vjs03.geospatial_lat_min, vjs03.geospatial_lat_max
        else:
            min_val = max_val = None
            assert min_val is not None

        return data[...], (min_val, max_val)

    def get_longitude(self, file):
        return self.__read(file, 'longitude')

    def get_latitude(self, file):
        return self.__read(file, 'latitude')



def create(instrument, satellite, verbosity=0):
    '''
    Geolocation product reader factory.
    '''

    name = instrument.lower() + '/' + satellite.lower()
    if name in ('modis/terra', 'modis/aqua', 'modis/'):
        return MODIS(verbosity)

    elif name in ('viirs/jpss-1', 'viirs/noaa-20'):
        return VIIRS_JPSS(verbosity)

    elif name in ('viirs/npp', 'viirs/s-npp', 'viirs/suomi-npp'):
        return VIIRS_NPP(verbosity)

    else:
        msg = "Unrecognized instrument '{0:s}' and/or satellite '{1:s}'.".format(instrument, platform)
        raise ValueError(msg)



def __test__():

    '''
    Test basic functionality of GeolocationProduct instances. 
    '''

    import os
    import numpy as np

    modis_dir = '/discover/nobackup/dao_ops/intermediate/flk/modis/061'
    viirs_dir = '/discover/nobackup/projects/eis_fire/data/VIIRS/Level1'

    test = {'modis/terra' : os.path.join(modis_dir, 'MOD03',        '2020', '300', 'MOD03.A2020300.1215.061.NRT.hdf'),
            'modis/aqua'  : os.path.join(modis_dir, 'MYD03',        '2020', '300', 'MYD03.A2020300.1215.061.NRT.hdf'),
            'viirs/npp'   : os.path.join(viirs_dir, 'NPP_IMFTS_L1', '2020', '300', 'NPP_IMFTS_L1.A2020300.1218.001.2020300185829.hdf'),
            'viirs/jpss-1': os.path.join(viirs_dir, 'VJ103IMG',     '2020', '300', 'VJ103IMG.A2020300.1218.002.2020300180500.nc')}

    for id, file in test.items():
        instrument, satellite = id.split('/')
        print('instrument: {0:s}'.format(id))
        print('file: {0:s}'.format(os.path.basename(file)))

        reader = create(instrument, satellite, verbosity=10)
        lon, lat, valid, lon_range, lat_range = reader.get_coordinates(file)
        print(' lon: min={0:f}, max={1:f}, range={2:s}'.format(np.min(lon), np.max(lon), str(lon_range)))
        print(' lat: min={0:f}, max={1:f}, range={2:s}'.format(np.min(lat), np.max(lat), str(lat_range)))
        print('')


if __name__ == '__main__':

    __test__()
