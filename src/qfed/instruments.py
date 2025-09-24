'''
Satellites and instruments
'''

from enum import Enum, unique

import numpy as np

@unique
class Instrument(Enum):
    MODIS = 'modis'
    VIIRS = 'viirs'

@unique
class Satellite(Enum):
    TERRA = 'terra'
    AQUA  = 'aqua'

    JPSS1 = 'jpss-1'
    NOAA20 = 'noaa-20'
    VJ1 = 'vj1'

    JPSS2  = 'jpss-2'
    NOAA21 = 'noaa-21'
    VJ2 = 'vj2'

    NPP = 'npp'
    SNPP = 's-npp'
    SuomiNPP = 'suomi-npp'
    VNP = 'vnp'

# Canonical codes for variable naming
# Map ALL satellite enum members (including aliases) to the desired short code
canonical_instrument = {Instrument.MODIS: "modis", Instrument.VIIRS: "viirs"}
canonical_satellite  = {
    Satellite.TERRA: "terra", Satellite.AQUA: "aqua",
    Satellite.NPP: "npp", Satellite.SNPP: "npp", Satellite.SuomiNPP: "npp",
    Satellite.JPSS1: "vj1", Satellite.NOAA20: "vj1",
    Satellite.JPSS2: "vj2", Satellite.NOAA21: "vj2",
}

def sensor_code(inst: Instrument, sat: Satellite) -> str:
    """Single tag that already implies instrument, e.g., mod14/myd14/vnp/vj1/vj2."""
    if inst is Instrument.MODIS:
        return "mod14" if canonical_satellite.get(sat) == "terra" else "myd14"
    # VIIRS family
    code = canonical_satellite.get(sat, "unknown")
    return "vnp" if code == "npp" else code

# VIIRS SDR User Guide: The bow-tie effect leads to 
# scan-to-scan overlap, which start to show visibly at 
# scan angles greater than approximately 19 degrees. 
# The size of overlap is more than 1 and 2 M-band pixels 
# at scan angle greater than 31.72 and 44.86 degrees, 
# respectively. 
VIIRS_SCAN_OVERLAP_ANGLE = (19.00, 31.72, 44.86)


def modis_pixel_area(sample):
    """
    Compute pixel area given the sample number. 

    Based on 'MODIS Collection 5 Active Fire Product 
    User's Guide Version 2.4', Giglio, L., p. 44, 2010.,
    and C. Ichoku and Y. J. Kaufman, 'A method to derive 
    smoke emission rates from MODIS fire radiative energy 
    measurements', IEEE Transactions on Geoscience and 
    Remote Sensing, vol. 43, no. 11, 2005, 
    doi:10.1109/TGRS.2005.857328.

    Note that sample is zero-based.
    """

    # parameters
    s  = 0.0014184397
    Re = 6378.137                # Earth radius, km
    h  = 705.0                   # satellite altitude, km

    sa = s * (sample - 676.5)    # scan angle, radians

    r  = Re + h
    q  = Re / r

    cos_sa    = np.cos(sa)
    sin_sa    = np.sin(sa)
    sqrt_term = np.sqrt(q*q - sin_sa*sin_sa)
    
    dS = s * Re * (cos_sa / sqrt_term - 1)
    dT = s * r  * (cos_sa - sqrt_term)

    area = dS * dT

    return area


def viirs_pixel_area(sample):
    '''
    A. Darmenov - Segmented polynomial regression model 
    fitted to VIIRS 375m pixel dimensions data. VIIRS 
    data was provided by Wilfrid Schroeder (NOAA; personal 
    communication).
 
    Note that in the tabular data:
        sample = 0     at nadir
        sample = 3199  at the edge (either left or right)
    whereas in the VIIRS granules/images the samples are 
    in the interval [0, 6399].

    The tabular data has two points of discontinuity:
        breakpoint = (1184, 1920)    
    hence there are three intervals:   
        interval = ((0, 1184), (1184, 1920), (1920, 3199))
    
    Pixel area data in each interval was fitted to a polynomial
    of degree 5. The coefficients of the three polynomials, 
    in decreasing powers, are included in the code.
    '''

    # (re)construct VIIRS pixel area with a polynomial of degree 5
    _area = np.zeros(3200)

    interval = ((0, 1184), (1184, 1920), (1920, 3199))

    p_fit = (np.poly1d((  7.50716369,     -2.80021144,   
                          9.60939363e-01,  4.45427672e-01, 
                          4.19643416e-03,  1.38396491e-01)),
             np.poly1d(( 31.83780167,    -63.73430877,
                         53.82525724,    -22.64615702, 
                          4.97063667,     -0.32549506)),
             np.poly1d(( 36.99433091,   -132.42437447,
                        191.84304112,   -139.19738588,
                         50.69193310,     -7.28395961)))

    for ((s_i, s_e), p) in zip(interval, p_fit):
        _area[s_i:s_e] = p(np.arange(s_i, s_e)/3200.0)
 
    # sample the reconstructed data
    area = np.concatenate((_area[::-1], _area[:]))[sample]
    return area



