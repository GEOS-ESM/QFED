import unittest

import os
from datetime import  datetime, timedelta

from qfed.inventory import Finder


class test(unittest.TestCase):
    def test_basic_functionality(self):
        '''
        Test basic functionality of PathFinder instances.
        '''

        modis_dir = '/discover/nobackup/dao_ops/intermediate/flk/modis'
        viirs_dir = '/css/viirs/data/'
        igbp_dir  = '/discover/nobackup/projects/gmao/share/gmao_ops/qfed/Emissions/Vegetation/GL_IGBP_INPE/'
    
        date_start = datetime(2020, 10, 26, 0)
        date_end   = datetime(2020, 10, 26, 1)

        # MODIS/Terra
        finder = Finder(
            os.path.join(modis_dir, '061', 'MOD03', '{0:%Y}', '{0:%j}', 
                'MOD03.A{0:%Y%j}.{0:%H%M}.061.*.hdf'),
            os.path.join(modis_dir, '006', 'MOD14', '{0:%Y}', '{0:%j}', 
                'MOD14.A{0:%Y%j}.{0:%H%M}.006.*.hdf'),
            igbp_dir)

        files = finder.find(date_start, date_end)
        print('MODIS/Terra: \n', files, '\n\n')

        # MODIS/Aqua
        finder = Finder(
            os.path.join(modis_dir, '061', 'MYD03', 
                '{0:%Y}', '{0:%j}', 'MYD03.A{0:%Y%j}.{0:%H%M}.061.*.hdf'), 
            os.path.join(modis_dir, '006', 'MYD14', 
                '{0:%Y}', '{0:%j}', 'MYD14.A{0:%Y%j}.{0:%H%M}.006.*.hdf'),
            igbp_dir)

        files = finder.find(date_start, date_end)
        print('MODIS/Aqua: \n', files, '\n\n')

        # VIIRS/NPP
        finder = Finder(
            os.path.join(viirs_dir, 'Level1', 'VNP03IMG.trimmed', '{0:%Y}', '{0:%j}', 
                'VNP03IMG.A{0:%Y%j}.{0:%H%M}.002.*.nc'),
            os.path.join(viirs_dir, 'Level2', 'VNP14IMG', '{0:%Y}', '{0:%j}',
                'VNP14IMG.A{0:%Y%j}.{0:%H%M}.001.*.nc'),
            igbp_dir)

        files = finder.find(date_start, date_end)
        print('VIIRS/NPP: \n', files, '\n\n')

        # VIIRS/JPSS1
        finder = Finder(
            os.path.join(viirs_dir, 'Level1', 'VJ103IMG.trimmed', '{0:%Y}', '{0:%j}',
                'VJ103IMG.A{0:%Y%j}.{0:%H%M}.002.*.nc'),
            os.path.join(viirs_dir, 'Level2', 'VJ114IMG', '{0:%Y}', '{0:%j}',
                'VJ114IMG.A{0:%Y%j}.{0:%H%M}.002.*.nc'),
            igbp_dir)

        files = finder.find(date_start, date_end)
        print('VIIRS/JPSS1: \n', files, '\n\n')



if __name__ == '__main__':
    unittest.main()
