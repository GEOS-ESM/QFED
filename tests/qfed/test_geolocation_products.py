import unittest

import os
import sys

from qfed.instruments import Instrument, Satellite
import qfed.geolocation_products

class test(unittest.TestCase):
    def test_basic_functionality(self):
        '''
        Test basic functionality of GeolocationProduct instances. 
        '''
    
        import os
        import numpy as np
    
        modis_dir = '/discover/nobackup/dao_ops/intermediate/flk/modis/061'
        viirs_dir = '/discover/nobackup/projects/eis_fire/data/VIIRS/Level1'
    
        instance = {(Instrument.MODIS, Satellite.TERRA)   : os.path.join(modis_dir, 'MOD03',        '2020', '300', 'MOD03.A2020300.1215.061.NRT.hdf'),
                    (Instrument.MODIS, Satellite.AQUA)    : os.path.join(modis_dir, 'MYD03',        '2020', '300', 'MYD03.A2020300.1215.061.NRT.hdf'),
                    (Instrument.VIIRS, Satellite.NPP)     : os.path.join(viirs_dir, 'NPP_IMFTS_L1', '2020', '300', 'NPP_IMFTS_L1.A2020300.1218.001.2020300185829.hdf'),
                    (Instrument.VIIRS, Satellite.SNPP)    : os.path.join(viirs_dir, 'NPP_IMFTS_L1', '2020', '300', 'NPP_IMFTS_L1.A2020300.1218.001.2020300185829.hdf'),
                    (Instrument.VIIRS, Satellite.SuomiNPP): os.path.join(viirs_dir, 'NPP_IMFTS_L1', '2020', '300', 'NPP_IMFTS_L1.A2020300.1218.001.2020300185829.hdf'),
                    (Instrument.VIIRS, Satellite.JPSS1)   : os.path.join(viirs_dir, 'VJ103IMG',     '2020', '300', 'VJ103IMG.A2020300.1218.002.2020300180500.nc'),
                    (Instrument.VIIRS, Satellite.NOAA20)  : os.path.join(viirs_dir, 'VJ103IMG',     '2020', '300', 'VJ103IMG.A2020300.1218.002.2020300180500.nc')}
    
        for id, file in instance.items():
            instrument, satellite = id
            print('name: {0:s}/{1:s}'.format(instrument.value, satellite.value))
            print('satellite: {0:s}'.format(satellite))
            print('instrument: {0:s}'.format(instrument))
            print('file: {0:s}'.format(os.path.basename(file)))
    
            reader = qfed.geolocation_products.create(instrument, satellite, verbosity=10)
            lon, lat, valid, lon_range, lat_range = reader.get_coordinates(file)
            print(' lon: min={0:f}, max={1:f}, range={2:s}'.format(np.min(lon), np.max(lon), str(lon_range)))
            print(' lat: min={0:f}, max={1:f}, range={2:s}'.format(np.min(lat), np.max(lat), str(lat_range)))
            print('')


if __name__ == '__main__':
    unittest.main()
