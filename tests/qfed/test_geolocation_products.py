import unittest

import os
import sys

from qfed.instruments import Instrument, Satellite
import qfed.geolocation_products


class test(unittest.TestCase):
    def test_basic_functionality(self):
        """
        Test basic functionality of GeolocationProduct instances.
        """

        import os
        import numpy as np

        modis_dir = '/discover/nobackup/dao_ops/intermediate/flk/modis/061'
        viirs_dir = '/css/viirs/data/Level1'

        queue = {
            (Instrument.MODIS, Satellite.TERRA): os.path.join(
                modis_dir, 
                'MOD03',
                '2020',
                '300',
                'MOD03.A2020300.1215.061.NRT.hdf'
            ),
            (Instrument.MODIS, Satellite.AQUA): os.path.join(
                modis_dir,
                'MYD03',
                '2020',
                '300',
                'MYD03.A2020300.1215.061.NRT.hdf'
            ),
            (Instrument.VIIRS, Satellite.NPP): os.path.join(
                viirs_dir,
                'VNP03IMG.trimmed',
                '2020',
                '300',
                'VNP03IMG.A2020300.1218.002.2021126045958.nc',
            ),
            (Instrument.VIIRS, Satellite.SNPP): os.path.join(
                viirs_dir,
                'VNP03IMG.trimmed',
                '2020',
                '300',
                'VNP03IMG.A2020300.1218.002.2021126045958.nc',
            ),
            (Instrument.VIIRS, Satellite.SuomiNPP): os.path.join(
                viirs_dir,
                'VNP03IMG.trimmed',
                '2020',
                '300',
                'VNP03IMG.A2020300.1218.002.2021126045958.nc',
            ),
            (Instrument.VIIRS, Satellite.JPSS1): os.path.join(
                viirs_dir,
                'VJ103IMG.trimmed',
                '2020',
                '300',
                'VJ103IMG.A2020300.1218.002.2020300180500.nc',
            ),
            (Instrument.VIIRS, Satellite.NOAA20): os.path.join(
                viirs_dir,
                'VJ103IMG.trimmed',
                '2020',
                '300',
                'VJ103IMG.A2020300.1218.002.2020300180500.nc',
            ),
        }

        for (instrument, satellite), file in queue.items():
            print(f'{instrument = }')
            print(f'{satellite = }')
            print(f'file = {os.path.basename(file)}')
            print(f'directory = {os.path.dirname(file)}')

            reader = qfed.geolocation_products.create(instrument, satellite)

            lon, lat, valid, lon_range, lat_range = reader.get_coordinates(file)
            print(f'lon:')
            print(f'  min = {np.min(lon)}')
            print(f'  max = {np.max(lon)}')
            print(f'  range = {lon_range}')
            print(f'lat:')
            print(f'  min = {np.min(lat)}')
            print(f'  max = {np.max(lat)}')
            print(f'  range = {lat_range}')
            print('')


if __name__ == '__main__':
    unittest.main()
