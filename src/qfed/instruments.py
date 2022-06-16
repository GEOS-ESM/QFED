'''
Satellites and instruments
'''

from enum import Enum, unique

@unique
class Instrument(Enum):
    MODIS = 'modis'
    VIIRS = 'viirs'

@unique
class Satellite(Enum):
    TERRA = 'terra'
    AQUA  = 'aqua'

    JPSS1 = 'jpss-1'
    NPP   = 'npp'


