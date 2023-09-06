"""
Basic representation of lat/lon and cubed sphere grids.
"""


import re
from enum import Enum, unique
import numpy as np


@unique
class GridType(Enum):
    LATLON_GEOS = 'lat/lon (GEOS)'
    LATLON_3600x1800 = 'lat/lon (3600x1800)'
    CUBEDSPHERE = 'cubed sphere'


_REFINE_FACTOR = {
    'a': 1,
    'b': 2,
    'c': 4,
    'd': 8,
    'e': 16,
    'f': 32,
    '0.1x0.1': None,
    '3600x1800': None,
}


class Grid:
    def __init__(self, alias):
        """
        Creates a grid object.

        Name aliases:
        -- single letter denoting GEOS resolution:
           'a'    produces a      4x5        grid
           'b'    produces a      2x2.50     grid
           'c'    produces a      1x1.25     grid
           'd'    produces a   0.50x0.625    grid
           'e'    produces a   0.25x0.3125   grid
           'f'    produces a  0.125x0.15625  grid
        -- cubed sphere GEOS notation:
           'c48'  produces approx.    2x2    grid
           'c90'  produces approx.    1x1    grid
           'c180' produces approx.  0.5x0.5  grid
           'c360' produces approx. 0.25x0.25 grid
        -- or
           '0.1x0.1'   produces     0.1x0.1  grid
           '3600x1800' produces     0.1x0.1  grid
        """
        self._set_spec(alias)
        self._set_coordinates()

    def _set_spec(self, alias):
        """
        Map alias to either a lat/lon grid spec or
        a cubed-sphere grid spec.
        """
        if isinstance(alias, int):
            # feature: can take refine factor as input
            refine = alias
            is_cubed_sphere = False
        else:
            cubed = re.compile("c[0-9]+")

            if cubed.match(alias):
                refine = None
                is_cubed_sphere = True
            else:
                refine = _REFINE_FACTOR[alias]
                is_cubed_sphere = False

        self._alias = alias
        self._refine = refine
        self._is_cubed_sphere = is_cubed_sphere

    def _set_coordinates(self):
        """
        Sets coordinates as per the grid spec.
        """
        if self._is_cubed_sphere:
            self._set_cubed_sphere()
        else:
            if self._refine is not None:
                self._set_latlon()
            else:
                self._set_PE_DE(3600, 1800)

    def _set_cubed_sphere(self):
        """
        Sets 'dummy' coordinates of a cubed sphere grid.
        """
        self.type = GridType.CUBEDSPHERE

        im = int(self._alias[1:])
        jm = 6 * im
        self._glon = np.arange(im)
        self._glat = np.arange(jm)

    def _set_latlon(self):
        """
        Sets coordinates of a GEOS lon/lat grid.
        """
        self.type = GridType.LATLON_GEOS

        dx = 5.0 / self._refine
        dy = 4.0 / self._refine
        im = int(360.0 / dx)
        jm = int(180.0 / dy + 1)

        self._glon = np.linspace(-180.0, 180.0, im, endpoint=False)
        self._glat = np.linspace(-90.0, 90.0, jm)

    def _set_PE_DE(self, im=3600, jm=1800):
        """
        Sets coordinates of a PE-DE lon/lat grid.
        """
        self.type = GridType.LATLON_3600x1800

        d_x = 360.0 / im
        d_y = 180.0 / jm

        o_x = 0.5 * d_x
        o_y = 0.5 * d_y

        self._glon = np.linspace(-180.0 + o_x, 180.0 - o_x, im)
        self._glat = np.linspace(-90.0 + o_y, 90.0 - o_y, jm)

    def dimensions(self):
        return {'x': len(self._glon), 'y': len(self._glat)}

    def lon(self):
        return self._glon

    def lat(self):
        return self._glat
