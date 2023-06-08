import unittest

import os
import sys

from qfed.instruments import Instrument, Satellite
import qfed.fire_products

class test(unittest.TestCase):
    def test_basic_functionality(self):
        '''
        Test basic functionality of ActiveFireProduct instances.
        '''
    
        modis_dir = '/discover/nobackup/dao_ops/intermediate/flk/modis/006'
        viirs_dir = '/css/viirs/data/Level2'
    
        queue = {
            (Instrument.MODIS, Satellite.TERRA): os.path.join(modis_dir, 'MOD14',    '2020', '300', 'MOD14.A2020300.0850.006.NRT.hdf'),
            (Instrument.MODIS, Satellite.AQUA) : os.path.join(modis_dir, 'MYD14',    '2020', '300', 'MYD14.A2020300.1325.006.NRT.hdf'),
            (Instrument.VIIRS, Satellite.NPP)  : os.path.join(viirs_dir, 'VNP14IMG', '2020', '300', 'VNP14IMG.A2020300.1142.001.2020300194419.nc'),
            (Instrument.VIIRS, Satellite.JPSS1): os.path.join(viirs_dir, 'VJ114IMG', '2020', '300', 'VJ114IMG.A2020300.1054.002.2020300170004.nc')
            }

        print(queue)

        for (instrument, satellite), file in queue.items():
            print(f"{instrument=}")
            print(f"{satellite=}")
            print(f"file: {os.path.basename(file)}")
            print(f"direcory: {os.path.dirname(file)}")

            reader = qfed.fire_products.create(instrument, satellite)

            gp_file = reader.get_geolocation_file(file)
            print(f"geolocation file: {gp_file}")

            n_fires = reader.get_num_fire_pixels(file)
            print(f"number of fire pixels: {n_fires}")

            if n_fires > 0:
                print('fire longitude : ', reader.get_fire_longitude(file))
                print('fire latitude  : ', reader.get_fire_latitude(file))
                print('fire frp       : ', reader.get_fire_frp(file))
                print('fire pixel area: ', reader.get_fire_pixel_area(file))
            print('')


if __name__ == '__main__':
    unittest.main()
