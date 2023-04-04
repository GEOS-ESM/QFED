'''
VIIRS and MODIS geolocation products.
'''


import sys
import os
import abc

import numpy as np
import netCDF4 as nc
from pyhdf import SD

from qfed.instruments import Instrument, Satellite


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
        if self.verbosity > 0:
            print('[w]    cannot open geo-location file <{0:s}>, ignoring granule'.format(file))

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

    def __read_hdf(self, file, variable):
        try:
            vnp03 = SD.SD(file)
        except SD.HDF4Error:
            self.message_on_file_error(file)
            return

        data = vnp03.select(variable).get()
        return data

    def __read_nc(self, file, variable):
        try:
           vnp03 = nc.Dataset(file)
        except IOError:
           self.message_on_file_error(file)
           return

        data = vnp03.variables[variable]
        return data[...]

    def __read(self, file, variable, respect_file_extension=False):
        reader = {'.hdf': self.__read_hdf,
                  '.nc' : self.__read_nc,
                  '.nc4': self.__read_nc}

        if respect_file_extension:
            _, file_extension = os.path.splitext(file)
            read = reader[file_extension]
        else:
            read = self.__read_nc

        return read(file, variable)

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

    if (instrument == Instrument.MODIS) \
       and (satellite in (Satellite.AQUA, 
                          Satellite.TERRA)):
        return MODIS(verbosity)

    elif (instrument == Instrument.VIIRS) \
         and (satellite in (Satellite.JPSS1, 
                            Satellite.NOAA20)):
        return VIIRS_JPSS(verbosity)

    elif (instrument == Instrument.VIIRS) \
         and (satellite in (Satellite.NPP, 
                            Satellite.SNPP, 
                            Satellite.SuomiNPP)):
        return VIIRS_NPP(verbosity)

    else:
        msg = "Unrecognized instrument '{0:s}' and/or satellite '{1:s}'.".format(instrument, satellite)
        raise ValueError(msg)


