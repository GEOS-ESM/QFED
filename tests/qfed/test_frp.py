import unittest

import os
import logging
from datetime import datetime, timedelta

from qfed import grid
from qfed import geolocation_products
from qfed import fire_products
from qfed import classification_products
from qfed.inventory import Finder
from qfed.instruments import Instrument, Satellite
from qfed.frp import GriddedFRP


class test(unittest.TestCase):
    def test_gridded_frp(self):
        """
        Test basic functionality of GriddedFRP instances.
        """

        modis_dir = '/discover/nobackup/dao_ops/intermediate/flk/modis'
        viirs_dir = '/css/viirs/data/'
        igbp_dir  = '/discover/nobackup/projects/gmao/share/gmao_ops/qfed/Emissions/Vegetation/GL_IGBP_INPE/'

        time = datetime(2021, 2, 1, 12)
        time_window = timedelta(hours=24)

        time_s = time - 0.5 * time_window
        time_e = time + 0.5 * time_window

        grid_ = grid.Grid('c')

        # MODIS/Terra
        gp_file = os.path.join(
            modis_dir,
            '061',
            'MOD03',
            '{0:%Y}',
            '{0:%j}',
            'MOD03.A{0:%Y%j}.{0:%H%M}.061.NRT.hdf',
        )

        fp_file = os.path.join(
            modis_dir,
            '006',
            'MOD14',
            '{0:%Y}',
            '{0:%j}',
            'MOD14.A{0:%Y%j}.{0:%H%M}.006.*.hdf',
        )

        vg_file = igbp_dir

        finder = Finder(gp_file, fp_file, vg_file, time_interval=60.0)
        fp_reader = fire_products.create(Instrument.MODIS, Satellite.TERRA)
        gp_reader = geolocation_products.create(Instrument.MODIS, Satellite.TERRA)
        cp_reader = classification_products.create(Instrument.MODIS, Satellite.TERRA)

        frp = GriddedFRP(grid_, finder, gp_reader, fp_reader, cp_reader)
        frp.ingest(time_s, time_e)
        frp.save(
            filename='qfed3-foo.frp.modis-terra.nc4',
            timestamp=time,
            bootstrap=True,
            qc=False,
        )

        # MODIS/Aqua
        gp_file = os.path.join(
            modis_dir,
            '061',
            'MYD03',
            '{0:%Y}',
            '{0:%j}',
            'MYD03.A{0:%Y%j}.{0:%H%M}.061.NRT.hdf',
        )

        fp_file = os.path.join(
            modis_dir,
            '006',
            'MYD14',
            '{0:%Y}',
            '{0:%j}',
            'MYD14.A{0:%Y%j}.{0:%H%M}.006.*.hdf',
        )

        vg_file = igbp_dir

        finder = Finder(gp_file, fp_file, vg_file, time_interval=300.0)
        fp_reader = fire_products.create(Instrument.MODIS, Satellite.AQUA)
        gp_reader = geolocation_products.create(Instrument.MODIS, Satellite.AQUA)
        cp_reader = classification_products.create(Instrument.MODIS, Satellite.TERRA)

        frp = GriddedFRP(grid_, finder, gp_reader, fp_reader, cp_reader)
        frp.ingest(time_s, time_e)
        frp.save(
            filename='qfed3-foo.frp.modis-aqua.nc4',
            timestamp=time,
            bootstrap=True,
            qc=False,
        )

        # VIIRS-NPP
        # gp_dir = os.path.join(viirs_dir, 'Level1', 'NPP_IMFTS_L1', '{0:%Y}', '{0:%j}')
        gp_file = os.path.join(
            viirs_dir,
            'Level1',
            'VNP03IMG.trimmed',
            '{0:%Y}',
            '{0:%j}',
            'VNP03IMG.A{0:%Y%j}.{0:%H%M}.002.*.nc',
        )

        fp_file = os.path.join(
            viirs_dir,
            'Level2',
            'VNP14IMG',
            '{0:%Y}',
            '{0:%j}',
            'VNP14IMG.A{0:%Y%j}.{0:%H%M}.001.*.nc',
        )

        vg_file = igbp_dir

        finder = Finder(gp_file, fp_file, vg_file, time_interval=360.0)
        fp_reader = fire_products.create(Instrument.VIIRS, Satellite.NPP)
        gp_reader = geolocation_products.create(Instrument.VIIRS, Satellite.NPP)
        cp_reader = classification_products.create(Instrument.VIIRS, Satellite.NPP)

        frp = GriddedFRP(grid_, finder, gp_reader, fp_reader, cp_reader)
        frp.ingest(time_s, time_e)
        frp.save(
            filename='qfed3-foo.frp.viirs-npp.nc4',
            timestamp=time,
            bootstrap=True,
            qc=False,
        )

        # VIIRS-JPSS1
        gp_file = os.path.join(
            viirs_dir,
            'Level1',
            'VJ103IMG.trimmed',
            '{0:%Y}',
            '{0:%j}',
            'VJ103IMG.A{0:%Y%j}.{0:%H%M}.002.*.nc',
        )

        fp_file = os.path.join(
            viirs_dir,
            'Level2',
            'VJ114IMG',
            '{0:%Y}',
            '{0:%j}',
            'VJ114IMG.A{0:%Y%j}.{0:%H%M}.002.*.nc',
        )

        vg_file = igbp_dir

        finder = Finder(gp_file, fp_file, vg_file, time_interval=360.0)
        fp_reader = fire_products.create(Instrument.VIIRS, Satellite.JPSS1)
        gp_reader = geolocation_products.create(Instrument.VIIRS, Satellite.JPSS1)
        cp_reader = classification_products.create(Instrument.VIIRS, Satellite.JPSS1)

        frp = GriddedFRP(grid_, finder, gp_reader, fp_reader, cp_reader)
        frp.ingest(time_s, time_e)
        frp.save(
            filename='qfed3-foo.frp.viirs-jpss1.nc4',
            timestamp=time,
            bootstrap=True,
            qc=False,
        )


if __name__ == '__main__':

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # filename='frp.log',
    )

    unittest.main()
