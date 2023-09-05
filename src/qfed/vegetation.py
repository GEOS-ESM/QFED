"""
Simplified vegetation categories.
"""

import os
from enum import IntEnum, unique

from pyobs import IGBP_


@unique
class VegetationCategory(IntEnum):
    """
    Simplified representation of vegetation types.

    The symbolic names and values need to be
    consistent with simplified_vegetation().
    """

    TROPICAL_FOREST = 1
    EXTRATROPICAL_FOREST = 2
    SAVANNA = 3
    GRASSLAND = 4


def simplified_vegetation(lon, lat, Path, nonVeg=None):
    """
    Helper method - aggregates IGBP vegetation classes
    into a reduced set of vegetation types:

      1 -> Tropical Forest:         IGBP  1, 30S < lat < 30N
      2 -> Extra-tropical Forests:  IGBP  1, 2(lat <=30S or lat >= 30N), 3, 4, 5
      3 -> Cerrado/Woody Savanna:   IGBP  6, 7, 8, 9
      4 -> Grassland/Cropland:      IGBP 10, 11, ..., 17
    """

    nobs = lon.shape[0]
    veg = IGBP_.getsimpleveg(lon, lat, os.path.join(Path, 'IGBP'), nobs)

    # substitute non vegetation (water, snow/ice) data with
    # another type, e.g. GRASSLAND by default
    if nonVeg is not None:
        i = (veg == -15) | (veg == -17)
        veg[i] = nonVeg  # could be one of the veg. categories, i.e., 1, 2, 3 or 4

    return veg


def get_category(lon, lat, Path, nonVeg=None):
    veg = simplified_vegetation(lon, lat, Path, nonVeg)

    category = {}
    for c in VegetationCategory:
        category[c] = veg == c.value

    return category
