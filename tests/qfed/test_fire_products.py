import unittest

import os
import sys

import qfed.fire_products

class test(unittest.TestCase):
    def test_basic_functionality(self):
        '''
        Test basic functionality of ActiveFireProduct instances.
        '''
    
        modis_dir = '/discover/nobackup/dao_ops/intermediate/flk/modis/006'
        viirs_dir = '/discover/nobackup/projects/eis_fire/data/VIIRS/Level2'
    
        instance = {'modis/terra' : os.path.join(modis_dir, 'MOD14',    '2020', '300', 'MOD14.A2020300.1215.006.NRT.hdf'),
                    'modis/aqua'  : os.path.join(modis_dir, 'MYD14',    '2020', '300', 'MYD14.A2020300.1215.006.NRT.hdf'),
                    'viirs/npp'   : os.path.join(viirs_dir, 'VNP14IMG', '2020', '300', 'VNP14IMG.A2020300.1142.001.2020300194419.nc'),
                    'viirs/jpss-1': os.path.join(viirs_dir, 'VJ114IMG', '2020', '300', 'VJ114IMG.A2020300.1054.002.2020300170004.nc')}
    
        for id, file in instance.items():
            instrument, satellite = id.split('/')
            print('instrument: {0:s}'.format(id))
            print('file: {0:s}'.format(os.path.basename(file)), file)
    
            reader = qfed.fire_products.create(instrument, satellite, verbosity=10)
    
            print('geolocation file: {0:s}'.format(reader.get_geolocation_file(file)))
            print('number of fire pixels: {0:d}'.format(reader.get_num_fire_pixels(file)))
            if reader.get_num_fire_pixels(file) > 0:
                print('fire longitude : ', reader.get_fire_longitude(file))
                print('fire latitude  : ', reader.get_fire_latitude(file))
                print('fire frp       : ', reader.get_fire_frp(file))
                print('fire pixel area: ', reader.get_fire_pixel_area(file))
            print('')


if __name__ == '__main__':
    unittest.main()
