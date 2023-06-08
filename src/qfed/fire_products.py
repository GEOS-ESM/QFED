'''
VIIRS and MODIS fire products.
'''

import sys
import os
import logging
import re
import abc

import numpy as np

from qfed.utils import DatasetAccessEngine_HDF4, DatasetAccessEngine_NetCDF4 
from qfed.instruments import Instrument, Satellite
from qfed.instruments import modis_pixel_area, viirs_pixel_area


if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class ActiveFireProduct(ABC):
    '''
    Handles only fire pixels.
    '''

    def __init__(self, engine):
        self._engine = engine

    @abc.abstractmethod
    def get_geolocation_file(self, file):
        '''
        The geolocation filename included in the metadata.
        '''

    def get_fire_coordinates(self, file):
        lon = self.get_fire_longitude(file)
        lat = self.get_fire_latitude(file)
        return lon, lat

    @abc.abstractmethod
    def get_fire_longitude(self, file):
        '''
        Longitude at center of fire pixel.
        '''

    @abc.abstractmethod
    def get_fire_latitude(self, file):
        '''
        Latitude at center of fire pixel.
        '''

    @abc.abstractmethod
    def get_num_fire_pixels(self, file):
        '''
        Number of fire pixels detected in granule.
        '''

    @abc.abstractmethod
    def get_fire_pixel_area(self, file):
        '''
        Area of fire pixel.
        '''

    @abc.abstractmethod
    def get_fire_frp(self, file):
        '''
        Fire radiative power.
        '''

    @abc.abstractmethod
    def get_fire_line(self, file):
        '''
        Granule line of fire pixel.
        '''

    @abc.abstractmethod
    def get_fire_sample(self, file):
        '''
        Granule sample of fire pixel.
        '''



class MODIS(ActiveFireProduct):
    '''
    Implements MODIS/MxD14 reader
    '''

    def get_geolocation_file(self, file):
        # NOTE: the attribute name 'MOD03 input file' is not a typo!!!  
        # Both MOD14 and MYD14 have this global attribute
        tmp_path = self._engine.get_attribute(file, 'MOD03 input file')
        result = os.path.basename(tmp_path)

        # unpack info from the geolocation filename
        _p = re.compile(('^(?P<product>M(O|Y)D03)'
                         '.'
                         'A(?P<year>\d\d\d\d)(?P<doy>\d\d\d)'
                         '.'
                         '(?P<hh>\d\d)(?P<mm>\d\d)'
                         '.'
                         '(?P<version>\d\d\d)'
                         '.*.'
                         '(?P<extension>(nc|hdf))'))
        _m = _p.match(result)

        product = _m.group('product')
        year = _m.group('year')
        doy  = _m.group('doy')
        hhmm = _m.group('hh') + _m.group('mm')
        version = _m.group('version')
        extension = _m.group('extension')
        base = '{product:s}.A{year:s}{doy:s}.{hhmm:s}.{version:s}.*.{extension:s}'

        result = base.format(product=product, year=year, doy=doy, hhmm=hhmm, 
                             version=version, extension=extension)
        return result
 
    def get_num_fire_pixels(self, file):
        return self._engine.get_attribute(file, 'FirePix')

    def get_fire_longitude(self, file):
        return self._engine.get_variable(file, 'FP_longitude')

    def get_fire_latitude(self, file):
        return self._engine.get_variable(file, 'FP_latitude')

    def get_fire_frp(self, file):
        return self._engine.get_variable(file, 'FP_power')

    def get_fire_line(self, file):
        return self._engine.get_variable(file, 'FP_line')

    def get_fire_sample(self, file):
        return self._engine.get_variable(file, 'FP_sample')

    def get_fire_pixel_area(self, file):
        sample = self.get_fire_sample(file)
        return modis_pixel_area(sample)



class VIIRS(ActiveFireProduct):
    '''
    Implements base VIIRS reader
    '''

    def get_num_fire_pixels(self, file):
        return self._engine.get_attribute(file, 'FirePix')

    def get_fire_longitude(self, file):
        return self._engine.get_variable(file, 'FP_longitude')

    def get_fire_latitude(self, file):
        return self._engine.get_variable(file, 'FP_latitude')

    def get_fire_frp(self, file):
        return self._engine.get_variable(file, 'FP_power')

    def get_fire_line(self, file):
        return self._engine.get_variable(file, 'FP_line')

    def get_fire_sample(self, file):
        return self._engine.get_variable(file, 'FP_sample')

    def get_fire_pixel_area(self, file):
        sample = self.get_fire_sample(file)
        return viirs_pixel_area(sample)
    

class VIIRS_NPP(VIIRS):
    '''
    Implements VIIRS/VNP14IMG reader
    '''

    def get_geolocation_file(self, file):
        result = self._engine.get_attribute(file, 'IMFTS')
        # workaround messy IMFS strings
        tmp = result.split('.hdf')
        result = '{0:s}.hdf'.format(str(tmp[0]))
        
        return result
    

class VIIRS_JPSS(VIIRS):
    '''
    Implements VIIRS/VJ114IMG reader
    '''

    def get_geolocation_file(self, file):
        result = self._engine.get_attribute(file, 'VNP03IMG')
        # workaround messy IMFS strings
        tmp = result.split('.nc')
        result = '{0:s}.nc'.format(str(tmp[0]))
        
        return result



def create(instrument, satellite):
    '''
    Active fire product factory.
    '''

    if instrument == Instrument.MODIS and \
       satellite in (Satellite.AQUA, Satellite.TERRA):
        engine = DatasetAccessEngine_HDF4()
        return MODIS(engine)

    if instrument == Instrument.VIIRS and \
       satellite in (Satellite.JPSS1, Satellite.NOAA20):
        engine = DatasetAccessEngine_NetCDF4()
        return VIIRS_JPSS(engine)

    if instrument == Instrument.VIIRS and \
       satellite in (Satellite.NPP, Satellite.SNPP, Satellite.SuomiNPP):
        engine = DatasetAccessEngine_NetCDF4()
        return VIIRS_NPP(engine)

    msg = ("Unrecognized satellite observing system platform: "
           "{0:s} on board of {1:s}.".format(instrument, satellite))
    raise ValueError(msg)


