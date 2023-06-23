'''
Retrieve pixel information.
'''

import sys
import logging
import abc

import numpy as np


from qfed.utils import DatasetAccessEngine_HDF4, DatasetAccessEngine_NetCDF4 
from qfed.instruments import Instrument, Satellite
from qfed.instruments import modis_pixel_area, viirs_pixel_area


if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


def _select(x, mask=None):
    if mask is None:
        return np.array(x, dtype=bool)
    else:
        return np.logical_and(x, mask)


def _get_bit(x, bit=0):
    '''
    Extracts the value of a single bit.
    '''
    return (x >> bit) & 1


class PixelClassifier(ABC):
    '''
    Handles all pixels in a granule.
    '''

    def __init__(self, engine):
        self._engine = engine

    @abc.abstractmethod
    def get_not_processed(self):
        '''
        Pixels that could not be processed due to 
        missing data or bad quality input data.
        '''

    @abc.abstractmethod
    def get_unclassified(self):
        '''
        Pixels that could not be definitely classified. 
        '''

    @abc.abstractmethod
    def get_cloud(self):
        '''
        Pixels classified as cloud. 
        '''

    @abc.abstractmethod
    def get_clear_sky(self):
        '''
        Non-fire clear sky (cloud free) pixels.
        '''

    @abc.abstractmethod
    def get_fire(self, confidence='all'):
        '''
        Fire pixels.
        '''

    @abc.abstractmethod
    def get_area(self):
        '''
        Pixel area.
        '''

class MODIS(PixelClassifier):

    # fire mask pixel classes
    NOT_PROCESSED_MISSING   = 0  # not processed (missing input data)
    NOT_PROCESSED_OBSOLETE  = 1  # not processed (obsolete; not used since Collection 1)
    NOT_PROCESSED_OTHER     = 2  # not processed (other reason)
    NON_FIRE_WATER          = 3  # non-fire water pixel
    CLOUD                   = 4  # cloud (land or water)
    NON_FIRE_LAND           = 5  # non-fire land pixel
    UNKNOWN                 = 6  # unknown (land or water)
    FIRE_LOW_CONFIDENCE     = 7  # fire (low confidence, land or water) 
    FIRE_NOMINAL_CONFIDENCE = 8  # fire (nominal confidence, land or water) 
    FIRE_HIGH_CONFIDENCE    = 9  # fire (high confidence, land or water) 

    # algorithm quality assessment bits
    AQA_BITMASK_LWS = 0b11 # land-water state, bits 0-1
    AQA_WATER  = 0b00   # 0=water 
    AQA_COAST  = 0b01   # 1=coast
    AQA_LAND   = 0b10   # 20=land
    AQA_UNUSED = 0b11   # 3=unused
    
    AQA_BITS_POTENTIAL_FIRE = 1 << 5 # 0b100000, bit 5
    AQA_POTENTIAL_FIRE_TRUE = 1
    AQA_POTENTIAL_FIRE_FALSE = 0


    def __init__(self, engine, mask=None):
        self._engine = engine
        #TODO: implement selective processing using a mask
        self._mask = mask

    def read(self, file):
        self._file = file
        self._get_fire_mask()
        self._get_algorithm_qa()

    def _get_fire_mask(self):
        self._fire_mask = self._engine.get_variable(self._file, 'fire mask')

    def _get_algorithm_qa(self):
        self._algorithm_qa = self._engine.get_variable(self._file, 'algorithm QA')

    def _get_land_water_state(self):
        '''
        Extract the land/water state.
        '''
        return (self._algorithm_qa & MODIS.AQA_BITMASK_LWS)

    def _is_over_land(self):
        '''
        Land pixels.
        '''
        lws = self._get_land_water_state()
        return (lws == MODIS.AQA_LAND)

    def _is_over_coast(self):
        '''
        Coast pixels.
        '''
        lws = self._get_land_water_state()
        return (lws == MODIS.AQA_COAST)

    def _is_over_water(self):
        '''
        Non-fire water pixels.
        '''
        lws = self._get_land_water_state()
        return (lws == MODIS.AQA_WATER)

    def _place(self, pixel):
        result = {}
        result['land' ] = pixel & self._is_over_land()
        result['coast'] = pixel & self._is_over_coast()
        result['water'] = pixel & self._is_over_water()
        return result      

    def _info(self, class_name, result):
        '''
        Helper method that shows debug info.
        '''
        for surface in ('land', 'coast', 'water'):
            label = f"{class_name:>13}({surface:<5})"
            count = np.sum(result[surface])
            logging.debug(f"{label:>18} : {count = }")

    def get_not_processed(self):
        '''
        Pixels that could not be processed due to missing input data
        or other reasons'.
        '''
        pixel = (self._fire_mask == MODIS.NOT_PROCESSED_MISSING ) | \
                (self._fire_mask == MODIS.NOT_PROCESSED_OBSOLETE) | \
                (self._fire_mask == MODIS.NOT_PROCESSED_OTHER)

        #potential_fire = np.bitwise_and(self._fire_mask, MODIS.AQA_BITS_POTENTIAL_FIRE)
        #assert np.all(pixel[potential_fire==MODIS.AQA_POTENTIAL_FIRE_NO])

        result = self._place(pixel)
        self._info('not processed', result)
        return result

    def get_unclassified(self):
        '''
        Pixels (land or water) that could not be definitively classified.
        '''
        pixel = (self._fire_mask == MODIS.UNKNOWN)

        result = self._place(pixel)
        self._info('unclassified', result)
        return result

    def get_cloud(self):
        '''
        Cloud pixels. Can occur either over land, coast or water.
        '''
        pixel = (self._fire_mask == MODIS.CLOUD)

        result = self._place(pixel)
        self._info('cloud', result)
        return result

    def get_clear_sky(self):
        '''
        Non-fire clear sky pixels. Can occur either over land, coast or water.
        '''
        pixel = (self._fire_mask == MODIS.NON_FIRE_WATER) | \
                (self._fire_mask == MODIS.NON_FIRE_LAND)

        result = self._place(pixel)
        self._info('clear sky', result)
        return result

    def _get_fire_confidence_low(self):
        '''
        Fire pixels - low confidence.
        Can occur either over land, coast or water.
        '''
        return (self._fire_mask == MODIS.FIRE_LOW_CONFIDENCE)

    def _get_fire_confidence_nominal(self):
        '''
        Fire pixels - nominal confidence.
        Can occur either over land, coast or water.
        ''' 
        return (self._fire_mask == MODIS.FIRE_NOMINAL_CONFIDENCE)

    def _get_fire_confidence_high(self):
        '''
        Fire pixels - high confidence.
        Can occur either over land, coast or water.
        ''' 
        return (self._fire_mask == MODIS.FIRE_HIGH_CONFIDENCE)

    def _get_fire_all(self):
        '''
        Fire pixels - all fires regardless of the detection 
        confidence.
        Can occur either over land, coast or water.
        ''' 
        return self._get_fire_confidence_low()     | \
               self._get_fire_confidence_nominal() | \
               self._get_fire_confidence_high()

    def get_fire(self, confidence=''):
        '''
        Fire pixels. Use the optional argument to select 
        fires with either low|nominal|high or any confidence.
        Can occur either over land, coast or water.
        '''
        select = {'low'    : self._get_fire_confidence_low,
                  'nominal': self._get_fire_confidence_nominal,
                  'high'   : self._get_fire_confidence_high,
                  ''       : self._get_fire_all,
                  'all'    : self._get_fire_all,
                  'any'    : self._get_fire_all,} 
       
        pixel = select[confidence]()
        logging.debug(f"fires({confidence} confidence) = {np.sum(pixel)}") 

        result = self._place(pixel)
        self._info('fire', result)
        return result

    def get_area(self):
        '''
        Calculate pixel area.
        '''
        area = np.zeros_like(self._fire_mask)
        n_lines, n_samples = area.shape
        
        # TODO: this follows the original MxD14 code
        sample = 1 + np.arange(n_samples)
        area[:] = modis_pixel_area(sample)
        return area


class VIIRS(PixelClassifier):

    # VNP14IMG fire mask pixel classes
    NOT_PROCESSED           = 0  # not processed (poor or missing input data)
    BOW_TIE_DELETION        = 1  # not processed (redundant data elements towards the edge of the swath that are deleted prior to relay to the ground station)
    SUN_GLINT               = 2  # processed (potentially affected by Sun glint where pixels are processed although algorithm performance is normally reduced)
    NON_FIRE_WATER          = 3  # non-fire water pixel
    CLOUD                   = 4  # cloud (land or water)
    NON_FIRE_LAND           = 5  # non-fire land pixel
    UNCLASSIFIED            = 6  # unknown (land or water)
    FIRE_LOW_CONFIDENCE     = 7  # fire (low confidence, land or water) 
    FIRE_NOMINAL_CONFIDENCE = 8  # fire (nominal confidence, land or water) 
    FIRE_HIGH_CONFIDENCE    = 9  # fire (high confidence, land or water) 

    # algorithm quality assessment bits [0--31]: stored in 32-bit integer format
    AQA_BIT_FIRE_PIXEL_OVER_WATER   = 19  # bit 19 (0 = false, 1 = true)
    AQA_FIRE_PIXEL_OVER_WATER_TRUE  = 1
    AQA_FIRE_PIXEL_OVER_WATER_FALSE = 0

    AQA_BIT_RESIDUAL_BOWTIE_PIXEL   = 22  # bit 22 (0 = false, 1 = true)
    AQA_RESIDUAL_BOWTIE_PIXEL_TRUE  = 1
    AQA_RESIDUAL_BOWTIE_PIXEL_FALSE = 0


    def __init__(self, engine, mask=None):
        self._engine = engine
        #TODO: implement selective processing using a mask
        self._mask = mask

    def read(self, file):
        self._file = file
        self._get_fire_mask()
        self._get_algorithm_qa()

    def _get_fire_mask(self):
        self._fire_mask = self._engine.get_variable(self._file, 'fire mask')

    def _get_algorithm_qa(self):
        self._algorithm_qa = self._engine.get_variable(self._file, 'algorithm QA')


    def _no_such_classification(self):
        '''
        A helper method to indicate that no such 
        classification is possible. 
        '''
        NO_SUCH_CLASS = False
        return NO_SUCH_CLASS

    def _get_fire_land_water_state(self):
        '''
        Extract the land/water state of FIRE pixels.
        '''
        return _get_bit(self._algorithm_qa, VIIRS.AQA_BIT_FIRE_PIXEL_OVER_WATER)

    def _is_fire_over_land(self):
        '''
        Fire pixels over land.
        '''
        lws = self._get_fire_land_water_state()
        return (lws == VIIRS.AQA_FIRE_PIXEL_OVER_WATER_FALSE)

    def _is_fire_over_water(self):
        '''
        Fire pixels over water.
        '''
        lws = self._get_fire_land_water_state()
        return (lws == VIIRS.AQA_FIRE_PIXEL_OVER_WATER_TRUE)

    def _is_fire_over_coast(self):
        '''
        Coast FIRE pixels.

        VIIRS classification product provides 
        only water|land state, i.e. it is not 
        possible identify coast pixels.  

        This method always returns FALSE.
        '''
        return self._no_such_classification() 

    def _is_fire_residual_bowtie(self):
        '''
        Residual bow-tie FIRE pixels.
        '''
        bit = _get_bit(self._algorithm_qa, VIIRS.AQA_BIT_RESIDUAL_BOWTIE_PIXEL)
        return (bit == VIIRS.AQA_RESIDUAL_BOWTIE_PIXEL_TRUE)

    def _is_fire_not_residual_bowtie(self):
        '''
        Residual bow-tie FIRE pixels.
        '''
        bit = _get_bit(self._algorithm_qa, VIIRS.AQA_BIT_RESIDUAL_BOWTIE_PIXEL)
        return (bit == VIIRS.AQA_RESIDUAL_BOWTIE_PIXEL_FALSE)

    def _fire_place(self, pixel):
        not_residual_bowtie = self._is_fire_not_residual_bowtie()
 
        result = {}
        result['land' ] = pixel & self._is_fire_over_land()  & not_residual_bowtie
        result['coast'] = pixel & self._is_fire_over_coast() & not_residual_bowtie
        result['water'] = pixel & self._is_fire_over_water() & not_residual_bowtie
        result['unknown'] = pixel & self._no_such_classification()
        return result  

    def _place(self, pixel):
        result = {}
        result['land' ] = pixel & self._no_such_classification()
        result['coast'] = pixel & self._no_such_classification()
        result['water'] = pixel & self._no_such_classification()
        result['unknown'] = pixel
        return result  

    def _info(self, class_name, result):
        '''
        Helper method that shows debug info.
        '''
        for surface in ('land', 'coast', 'water', 'unknown'):
            label = f"{class_name:>13}({surface:<7})"
            count = np.sum(result[surface])
            logging.debug(f"{label:>18} : {count = }")

    def get_not_processed(self):
        '''
        Pixels that could not be processed due to missing or 
        poor quality input data.
        '''

        pixel = self._fire_mask == VIIRS.NOT_PROCESSED
        result = self._place(pixel)
        self._info('not processed', result) 
        return result

    def get_unclassified(self):
        '''
        Pixels (land or water) that could not be definitively classified.
        '''
        pixel = (self._fire_mask == VIIRS.UNCLASSIFIED)

        result = self._place(pixel)
        self._info('unclassified', result)
        return result
 
    def get_cloud(self):
        '''
        Cloud pixels. Can occur either over land, coast or water.
        '''
        pixel = (self._fire_mask == VIIRS.CLOUD)

        result = self._place(pixel)
        self._info('cloud', result)
        return result

    def get_clear_sky(self):
        '''
        Non-fire clear sky pixels. Can occur either over land, coast or water.
        '''
        result = {}

        result['land' ] = self._fire_mask == VIIRS.NON_FIRE_LAND
        result['water'] = (self._fire_mask == VIIRS.NON_FIRE_WATER) | \
                          (self._fire_mask == VIIRS.SUN_GLINT)
        result['coast'] = self._fire_mask & self._no_such_classification()
        result['unknown'] = self._fire_mask & self._no_such_classification()

        self._info('clear sky', result)
        return result

    def _get_fire_confidence_low(self):
        '''
        Fire pixels - low confidence.
        Can occur either over land, coast or water.

        Note: all low confidence fire pixels are returned, 
              regardless of the flags indicating Glint condition
              and/or Potential South Atlantic magnetic anomaly.
        '''
        return (self._fire_mask == VIIRS.FIRE_LOW_CONFIDENCE)

    def _get_fire_confidence_nominal(self):
        '''
        Fire pixels - nominal confidence.
        Can occur either over land, coast or water.
        '''
        return (self._fire_mask == VIIRS.FIRE_NOMINAL_CONFIDENCE)

    def _get_fire_confidence_high(self):
        '''
        Fire pixels - high confidence.
        Can occur either over land, coast or water.
        ''' 
        return (self._fire_mask == VIIRS.FIRE_HIGH_CONFIDENCE)

    def _get_fire_all(self):
        '''
        Fire pixels - all fires regardless of the detection 
        confidence.
        Can occur either over land, coast or water.
        ''' 
        return self._get_fire_confidence_low()     | \
               self._get_fire_confidence_nominal() | \
               self._get_fire_confidence_high()

    def get_fire(self, confidence=''):
        '''
        Fire pixels. Use the optional argument to select 
        fires with either low|nominal|high or any confidence.
        Can occur either over land, coast or water.
        '''
        select = {'low'    : self._get_fire_confidence_low,
                  'nominal': self._get_fire_confidence_nominal,
                  'high'   : self._get_fire_confidence_high,
                  ''       : self._get_fire_all,
                  'all'    : self._get_fire_all,
                  'any'    : self._get_fire_all,} 
       
        pixel = select[confidence]()
        logging.debug(f"fires({confidence} confidence) = {np.sum(pixel)}")

        not_residual_bowtie = self._is_fire_not_residual_bowtie()
        pixel = pixel & not_residual_bowtie
        logging.debug(f"fires({confidence} confidence) without residual bowtie = {np.sum(pixel)}")

        result = self._fire_place(pixel)
        self._info('fire', result)
        return result

    def get_area(self):
        '''
        Calculate pixel area.
        '''
        area = np.zeros_like(self._fire_mask)
        n_lines, n_samples = area.shape

        sample = np.arange(n_samples)
        area[:] = viirs_pixel_area(sample)
        return area

    def __visualize(self): 
        import matplotlib.pyplot as plt
        from matplotlib.colors import NoNorm
        data = self._fire_mask == VIIRS.NON_FIRE_WATER
        plt.imsave('water.fire_mask.png', data, cmap='gray')#, norm=NoNorm())
        plt.clf()
        data = self._is_residual_bowtie() 
        data = self._is_over_water()
        data = (self._algorithm_qa >> 10 & 0b1) == 1
        plt.imsave('water.algorithm_qa.bit10.png', data, cmap='gray')#, norm=NoNorm())
     

def create(instrument, satellite):
    '''
    Classifier product factory.
    '''

    if instrument == Instrument.MODIS and \
       satellite in (Satellite.AQUA, Satellite.TERRA):
        engine = DatasetAccessEngine_HDF4()
        return MODIS(engine)

    if instrument == Instrument.VIIRS and \
       satellite in (Satellite.JPSS1, Satellite.NOAA20):
        engine = DatasetAccessEngine_NetCDF4()
        return VIIRS(engine)

    if instrument == Instrument.VIIRS and \
       satellite in (Satellite.NPP, Satellite.SNPP, Satellite.SuomiNPP):
        engine = DatasetAccessEngine_NetCDF4()
        return VIIRS(engine)

    msg = ("Unrecognized satellite observing system platform: "
           "{0:s} on board of {1:s}.".format(instrument, satellite))
    raise ValueError(msg)


