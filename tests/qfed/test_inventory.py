import unittest

import os
from datetime import  datetime, timedelta

from qfed.pathfinder import PathFinder


class test(unittest.TestCase):
    def test_basic_functionality(self):
        '''
        Test basic functionality of PathFinder instances.
        '''

        modis_dir = '/discover/nobackup/dao_ops/intermediate/flk/modis'
        viirs_dir = '/discover/nobackup/projects/eis_fire/data/VIIRS'
    
        date_start = datetime(2020, 10, 26, 0)
        date_end   = datetime(2020, 10, 26, 1)

        # MODIS/Terra
        finder = PathFinder(os.path.join(modis_dir, '061', 'MOD03', '{0:%Y}', '{0:%j}'),
                            os.path.join(modis_dir, '006', 'MOD14', '{0:%Y}', '{0:%j}'),
                            'MOD14.A{0:%Y%j}.{0:%H%M}.006.*.hdf')

        files = finder.find(date_start, date_end)
        print('MODIS/Terra: \n',files, '\n\n')

        # MODIS/Aqua
        finder = PathFinder(os.path.join(modis_dir, '061', 'MYD03', '{0:%Y}', '{0:%j}'),
                            os.path.join(modis_dir, '006', 'MYD14', '{0:%Y}', '{0:%j}'),
                            'MYD14.A{0:%Y%j}.{0:%H%M}.006.*.hdf')

        files = finder.find(date_start, date_end)
        print('MODIS/Aqua: \n', files, '\n\n')

        # VIIRS/NPP
        finder = PathFinder(os.path.join(viirs_dir, 'Level1', 'NPP_IMFTS_L1', '{0:%Y}', '{0:%j}'),
                            os.path.join(viirs_dir, 'Level2', 'VNP14IMG', '{0:%Y}', '{0:%j}'),
                            'VNP14IMG.A{0:%Y%j}.{0:%H%M}.001.*.nc')

        files = finder.find(date_start, date_end)
        print('VIIRS/NPP: \n', files, '\n\n')

        # VIIRS/JPSS1
        finder = PathFinder(os.path.join(viirs_dir, 'Level1', 'VJ103IMG', '{0:%Y}', '{0:%j}'),
                            os.path.join(viirs_dir, 'Level2', 'VJ114IMG', '{0:%Y}', '{0:%j}'),
                            'VJ114IMG.A{0:%Y%j}.{0:%H%M}.002.*.nc')

        files = finder.find(date_start, date_end)
        print('VIIRS/JPSS1: \n', files, '\n\n')



if __name__ == '__main__':
    unittest.main()
