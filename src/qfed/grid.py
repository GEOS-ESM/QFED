'''
Basic representation of lat/lon and cubed sphere grids.
'''


import re
from enum import Enum
import numpy as np



class GridType(Enum):
    LATLON = 'lat/lon'
    CUBEDSPHERE = 'cubed sphere'


class Grid:
    def __init__(self, alias):
        '''
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
           '0.1x0.1' produces       0.1x0.1  grid
        '''

        self.__parse(alias)
        self.__set()


    def __parse(self, alias):
        '''
        Map name alias with a refine factor and a CubedSphere flag.
        '''

        if isinstance(alias, int):
            # undocumented feature: can take refine factor as input
            refine = alias
            is_cubed_sphere = False
        else:
           cubed = re.compile('c[0-9]+')

           if cubed.match(alias):
               refine = None
               is_cubed_sphere = True
           else:
               factor = {'a':1, 'b':2, 'c': 4, 'd': 8, 'e':16, 'f':32, '0.1x0.1':None}
               refine = factor[alias]
               is_cubed_sphere = False
        
        self.__alias = alias
        self.__refine = refine
        self.__is_cubed_sphere = is_cubed_sphere


    def __set(self):
        if self.__is_cubed_sphere:
            # Cubed Sphere grid
            self.__set_cubed_sphere()
        else:
            if self.__refine is not None:
                # GEOS lat/lon grid 
                self.__set_latlon()
            else:
                # 0.1x0.1 lat/lon grid
                self.__set_01x01()


    def __set_cubed_sphere(self):
        self.type = GridType.CUBEDSPHERE

        im = int(self.__alias[1:])
        jm = 6*im
        self.__glon = np.arange(im)
        self.__glat = np.arange(jm)


    def __set_latlon(self):
        self.type = GridType.LATLON

        dx = 5.0 / self.__refine
        dy = 4.0 / self.__refine
        im = int(360.0 / dx)
        jm = int(180.0 / dy + 1)

        self.__glon = np.linspace(-180.0, 180.0, im, endpoint=False)
        self.__glat = np.linspace( -90.0,  90.0, jm)


    def __set_01x01(self):
        self.type = GridType.LATLON

        im = 3600
        jm = 1800

        d_lon = 360.0 / im
        d_lat = 180.0 / jm

        self.__glon = np.linspace(-180.0 + d_lon/2, 180.0 - d_lon/2, im)
        self.__glat = np.linspace( -90.0 + d_lat/2,  90.0 - d_lat/2, jm)


    def dimensions(self):
        return {'x': len(self.__glon), 'y': len(self.__glat)}


    def lon(self):
        return self.__glon


    def lat(self):
        return self.__glat



def __test__():
    '''
    Test basic functionality of Grid instances. 
    '''

    for name in ('c', 'e', '0.1x0.1', 'c90', 'c360'):
        grid = Grid(name)
       
        print ('name: {0:s}'.format(name))
        print ('type: {0:s}'.format(grid.type))
        print ('dimensions: x={x:d}, y={y:d}'.format(**grid.dimensions()))
        print ('lon: ', grid.lon())
        print ('lat: ', grid.lat())
        print ('')


if __name__ == '__main__':

    __test__()
