import unittest

import os
import logging

from qfed.instruments import Instrument, Satellite
import qfed.classification_products


class test(unittest.TestCase):
    def test_basic_functionality(self):
        """
        Test basic functionality of ClassificationProduct instances.
        """

        modis_dir = '/discover/nobackup/dao_ops/intermediate/flk/modis/006'
        viirs_dir = '/css/viirs/data/Level2/'

        queue = {
            (Instrument.MODIS, Satellite.TERRA): os.path.join(
                modis_dir,
                'MOD14',
                '2020',
                '300',
                'MOD14.A2020300.0850.006.NRT.hdf'
            ),
            (Instrument.MODIS, Satellite.AQUA): os.path.join(
                modis_dir,
                'MYD14',
                '2020',
                '300',
                'MYD14.A2020300.1325.006.NRT.hdf'
            ),
            (Instrument.VIIRS, Satellite.NPP): os.path.join(
                viirs_dir,
                'VNP14IMG',
                '2020',
                '300',
                'VNP14IMG.A2020300.1142.001.2020300194419.nc',
            ),
            (Instrument.VIIRS, Satellite.JPSS1): os.path.join(
                viirs_dir,
                'VJ114IMG',
                '2020',
                '300',
                'VJ114IMG.A2020300.1054.002.2020300170004.nc',
            ),
        }

        for (instrument, satellite), file in queue.items():
            logging.debug(f'{instrument = }')
            logging.debug(f'{satellite = }')
            logging.debug(f'file = {os.path.basename(file)}')
            logging.debug(f'direcory = {os.path.dirname(file)}')

            reader = qfed.classification_products.create(instrument, satellite)

            reader.read(file)

            reader.get_not_processed()
            reader.get_unclassified()
            reader.get_cloud()
            reader.get_cloud_free()
            reader.get_fire(confidence='low')
            reader.get_fire(confidence='nominal')
            reader.get_fire(confidence='high')
            reader.get_fire(confidence='non-zero')

            logging.debug('\n\n')
            


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        # filename='classification.log',
    )

    unittest.main()
