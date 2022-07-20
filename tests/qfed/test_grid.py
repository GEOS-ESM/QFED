import unittest

import os
import sys

import qfed.grid

class test(unittest.TestCase):
    def test_basic_functionality(self):
        '''
        Test basic functionality of Grid instances. 
        '''

        for name in ('c', 'e', '0.1x0.1', 'c90', 'c360'):
            grid = qfed.grid.Grid(name)

            print ('name: {0:s}'.format(name))
            print ('type: {0:s}'.format(grid.type))
            print ('dimensions: x={x:d}, y={y:d}'.format(**grid.dimensions()))
            print ('lon: ', grid.lon())
            print ('lat: ', grid.lat())
            print ('')


if __name__ == '__main__':
    unittest.main()
