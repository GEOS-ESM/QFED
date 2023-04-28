'''
VIIRS and MODIS fire products.
'''


import sys
import os
import re
import abc
from enum import Enum

import numpy as np
from pyhdf import SD
import netCDF4 as nc

from qfed.instruments import Instrument, Satellite
from qfed.instruments import modis_pixel_area, viirs_pixel_area


if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class MODIS_FireMaskClass(Enum):
    '''
    ...
    '''
    WATER  = 3     # non-fire water pixel
    CLOUD  = 4     # cloud (land or water)
    NOFIRE = 5     # non-fire land pixel

class MODIS_QA(Enum):
    ''' 
    MxD14 collection 6 algorithm quality assessment bits: 
    land/water state (bits 0-1)
    '''
    WATER = 0
    COAST = 1
    LAND  = 2



class DatasetAccessEngine(ABC):

    def __init__(self, verbosity=0):
        self.verbosity = verbosity

    def message_on_file_error(self, file):
        if self.verbosity > 0:
            print('[w]    cannot open fire product file <{0:s}>, ignoring granule'.format(file))

    @abc.abstractmethod
    def get_handle(self, file):
        return

    @abc.abstractmethod
    def get_variable(self, file, variable):
        return

    @abc.abstractmethod
    def get_attribute(self, file, attribute):
        return


class DatasetAccessEngine_HDF4(DatasetAccessEngine):

    def get_handle(self, file):
        try:
            f = SD.SD(file)
        except SD.HDF4Error:
            self.message_on_file_error(file)
            f = None

        return f

    def get_variable(self, file, variable):
        f = self.get_handle(file)
        
        if f is not None:
            sds = f.select(variable)
            if sds.checkempty():
                data = []
            else:    
                data = f.select(variable).get()
        else: 
            data = None

        return data

    def get_attribute(self, file, attribute):
        f = self.get_handle(file)
        
        if f is not None:
            attr = f.attributes()[attribute]
        else:
            attr = None

        return attr


class DatasetAccessEngine_NetCDF4(DatasetAccessEngine):

    def get_handle(self, file):
        try:
           f = nc.Dataset(file)
        except IOError:
           self.message_on_file_error(file)
           f = None

        return f

    def get_variable(self, file, variable):
        f = self.get_handle(file)

        if f is not None:
            data = f.variables[variable][...]
        else:
            data = None

        return data

    def get_attribute(self, file, attribute):
        f = self.get_handle(file)

        if f is not None:
            attr = f.__dict__[attribute]
        else:
            attr = None
        
        return attr



class ActiveFireProduct(ABC):

    def __init__(self, engine, verbosity=0):
        self.dataset = engine
        self.verbosity = verbosity

    @abc.abstractmethod
    def get_geolocation_file(self, file):
        return  

    def get_fire_coordinates(self, file):
        lon = self.get_fire_longitude(file)
        lat = self.get_fire_latitude(file)
        return lon, lat

    @abc.abstractmethod
    def get_fire_longitude(self, file):
        return

    @abc.abstractmethod
    def get_fire_latitude(self, file):
        return

    @abc.abstractmethod
    def get_num_fire_pixels(self, file):
        return

    @abc.abstractmethod
    def get_fire_pixel_area(self, file):
        return

    @abc.abstractmethod
    def get_fire_frp(self, file):
        return

    @abc.abstractmethod
    def get_fire_line(self, file):
        return

    @abc.abstractmethod
    def get_fire_sample(self, file):
        return

    @abc.abstractmethod
    def get_fire_mask(self, file):
        return

    @abc.abstractmethod
    def get_algorithm_qa(self, file):
        return

    @abc.abstractmethod
    def get_land_water_mask(self, file, land=0, water=1):
        return

    @abc.abstractmethod
    def get_pixel_classes(self, file):
        return

    @abc.abstractmethod
    def get_pixel_area(self, sample):
        return



class MODIS(ActiveFireProduct):
    '''
    Implements MODIS/MxD14 reader
    '''

    # fire mask pixel classes
    NOT_PROCESSED_MISSING   = 0  # not processed (missing input data)
    NOT_PROCESSED_OBSOLETE  = 1  # not processed (obsolete; not used since Collection 1)
    NOT_PROCESSED_OTHER     = 2  # not processed (other reason)
    WATER                   = 3  # non-fire water pixel
    CLOUD                   = 4  # cloud (land or water)
    NON_FIRE                = 5  # non-fire land pixel
    UNKNOWN                 = 6  # unknown (land or water)
    FIRE_LOW_CONFIDENCE     = 7  # fire (low confidence, land or water) 
    FIRE_NOMINAL_CONFIDENCE = 8  # fire (nominal confidence, land or water) 
    FIRE_HIGH_CONFIDENCE    = 9  # fire (high confidence, land or water) 

    # algorithm quality assessment bits: land/water state (bits 0-1)
    QA_WATER  = 0b00   # water
    QA_COAST  = 0b01   # coast
    QA_LAND   = 0b10   # land
    QA_UNUSED = 0b11   # unused


    def get_geolocation_file(self, file):
        # NOTE: the attribute name 'MOD03 input file' is not a typo!!!  
        # Both MOD14 and MYD14 have this global attribute
        tmp_path = self.dataset.get_attribute(file, 'MOD03 input file')
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
        return self.dataset.get_attribute(file, 'FirePix')

    def get_fire_longitude(self, file):
        return self.dataset.get_variable(file, 'FP_longitude')

    def get_fire_latitude(self, file):
        return self.dataset.get_variable(file, 'FP_latitude')

    def get_fire_frp(self, file):
        return self.dataset.get_variable(file, 'FP_power')

    def get_fire_line(self, file):
        return self.dataset.get_variable(file, 'FP_line')

    def get_fire_sample(self, file):
        return self.dataset.get_variable(file, 'FP_sample')

    def get_fire_pixel_area(self, file):
        sample = self.get_fire_sample(file)
        return modis_pixel_area(sample)

    def get_pixel_area(self, sample):
        # TODO: is this correct for MODIS?
        return modis_pixel_area(1+sample)

    def get_fire_mask(self, file):
        return self.dataset.get_variable(file, 'fire mask')

    def get_algorithm_qa(self, file):
        return self.dataset.get_variable(file, 'algorithm QA')

    def get_land_water_mask(self, file, land=0, water=1):
        qa = self.get_algorithm_qa(file)

        # land/water state is stored in bits 0-1
        lws = np.bitwise_and(qa, 0b11)
        assert np.all(lws != MODIS.QA_UNUSED)

        return np.where(lws == MODIS.QA_WATER, water, land)
 
    def get_pixel_classes(self, file, land=0, water=1):
        lws = self.get_land_water_mask(file, land=land, water=water)
        fire_mask = self.get_fire_mask(file)

        i_water = (lws == water)
        i_land = (lws == land)
        i_land_nofire = (fire_mask == MODIS.NON_FIRE)
        i_land_cloud = np.logical_and(fire_mask==MODIS.CLOUD, i_land)
        i_water_nofire = (fire_mask == MODIS.WATER)
        i_water_cloud = np.logical_and(fire_mask==MODIS.CLOUD, i_water)

        return i_water, i_land_nofire, i_land_cloud, i_water_cloud


class VIIRS(ActiveFireProduct):
    '''
    Implements base VIIRS reader
    '''

    # VNP14IMG fire mask pixel classes
    NOT_PROCESSED           = 0  # not processed (poor or missing input data)
    BOW_TIE_DELETION        = 1  # not processed (redundant data elements towards the edge of the swath that are deleted prior to relay to the ground station)
    SUN_GLINT               = 2  # processed (potentially affected by Sun glint where pixels are processed although algorithm performance is normally reduced)
    WATER                   = 3  # non-fire water pixel
    CLOUD                   = 4  # cloud (land or water)
    LAND                    = 5  # non-fire land pixel
    UNCLASSIFIED            = 6  # unknown (land or water)
    FIRE_LOW_CONFIDENCE     = 7  # fire (low confidence, land or water) 
    FIRE_NOMINAL_CONFIDENCE = 8  # fire (nominal confidence, land or water) 
    FIRE_HIGH_CONFIDENCE    = 9  # fire (high confidence, land or water) 

    # algorithm quality assessment bits [0--31]: stored in 32-bit integer format
    QA_FIRE_OVER_WATER  = 1<<19  # bit 19 (0 = false, 1 = true)
    QA_RESIDUAL_BOW_TIE = 1<<22  # bit 22 (0 = false, 1 = true)

    @abc.abstractmethod
    def get_geolocation_file(self, file):
        return

    def get_num_fire_pixels(self, file):
        return self.dataset.get_attribute(file, 'FirePix')

    def get_fire_longitude(self, file):
        return self.dataset.get_variable(file, 'FP_longitude')

    def get_fire_latitude(self, file):
        return self.dataset.get_variable(file, 'FP_latitude')

    def get_fire_frp(self, file):
        return self.dataset.get_variable(file, 'FP_power')

    def get_fire_line(self, file):
        return self.dataset.get_variable(file, 'FP_line')

    def get_fire_sample(self, file):
        return self.dataset.get_variable(file, 'FP_sample')

    def get_fire_pixel_area(self, file):
        sample = self.get_fire_sample(file)
        return viirs_pixel_area(sample)
    
    def get_pixel_area(self, sample):
        return viirs_pixel_area(sample)

    def get_fire_mask(self, file):
        return self.dataset.get_variable(file, 'fire mask')

    def get_algorithm_qa(self, file):
        return self.dataset.get_variable(file, 'algorithm QA')

    def get_land_water_mask(self, file, land=0, water=1):
        '''
        Not needed
        '''
        NotImplementedError

    def get_pixel_classes(self, file, land=0, water=1):

        '''
        # non-fire land pixel
        i = np.logical_and(i_land_nofire, valid)


        # non-fire water or cloud over water
        i = np.logical_and(np.logical_or(i_water_cloud, i_water), valid)

        # cloud over land only
        i = np.logical_and(i_land_cloud, valid)
        '''

        fire_mask = self.get_fire_mask(file)

        i_water = (fire_mask == VIIRS.WATER) 
        i_land = (fire_mask == VIIRS.LAND)
        i_land_nofire = (fire_mask == VIIRS.LAND)
        i_land_cloud = np.logical_and(fire_mask==VIIRS.CLOUD, False)
        i_water_nofire = (fire_mask == VIIRS.WATER)
        i_water_cloud = np.logical_and(fire_mask==VIIRS.CLOUD, False)

        return i_water, i_land_nofire, i_land_cloud, i_water_cloud



class VIIRS_NPP(VIIRS):
    '''
    Implements VIIRS/VNP14IMG reader
    '''

    def get_geolocation_file(self, file):
        result = self.dataset.get_attribute(file, 'IMFTS')
        # workaround messy IMFS strings
        tmp = result.split('.hdf')
        result = '{0:s}.hdf'.format(str(tmp[0]))
        
        # extend the list of possible file extensions to handle: 
        #    1) original HDF files and 
        #    2) original HDF files that were compressed and 
        #       as a result have different file extension, i.e. 'nc4'
        file_extensions = ('.hdf', '.nc4', '.nc')
        tmp = os.path.splitext(result)[0]
        result = [tmp + extension for extension in file_extensions]

        return result
    

class VIIRS_JPSS(VIIRS):
    '''
    Implements VIIRS/VJ114IMG reader
    '''

    def get_geolocation_file(self, file):
        result = self.dataset.get_attribute(file, 'VNP03IMG')
        # workaround messy IMFS strings
        tmp = result.split('.nc')
        result = '{0:s}.nc'.format(str(tmp[0]))
        
        return result



def create(instrument, satellite, verbosity=0):
    '''
    Active fire product factory.
    '''

    if instrument == Instrument.MODIS and \
       satellite in (Satellite.AQUA, Satellite.TERRA):
        dataset_engine = DatasetAccessEngine_HDF4(verbosity)
        return MODIS(dataset_engine, verbosity)   

    if instrument == Instrument.VIIRS and \
       satellite in (Satellite.JPSS1, Satellite.NOAA20):
        dataset_engine = DatasetAccessEngine_NetCDF4(verbosity)
        return VIIRS_JPSS(dataset_engine, verbosity)

    if instrument == Instrument.VIIRS and \
       satellite in (Satellite.NPP, Satellite.SNPP, Satellite.SuomiNPP):
        dataset_engine = DatasetAccessEngine_NetCDF4(verbosity)
        return VIIRS_NPP(dataset_engine, verbosity)

    msg = ("Unrecognized satellite observing system platform: "
           "{0:s} on board of {1:s}.".format(instrument, satellite))
    raise ValueError(msg)


