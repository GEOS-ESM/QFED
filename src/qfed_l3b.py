#!/usr/bin/env python3

"""
A script that creates QFED Level 3B files.
"""

import os
import sys
import logging
import argparse
import yaml
from datetime import datetime, timedelta
from glob import glob

import numpy as np
import netCDF4 as nc

from qfed import utils
from qfed import grid
from qfed.instruments import Instrument, Satellite
from qfed.emissions import Emissions
from qfed import fire
from qfed import VERSION


def parse_arguments(default, version):
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        prog='qfed_l3b.py',
        description='Create QFED Level 3B files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=f'QFED {version} (%(prog)s)',
    )

    parser.add_argument(
        '-c',
        '--config',
        dest='config',
        default=default['config'],
        help='config file',
    )

    parser.add_argument(
        '-s',
        '--obs',
        nargs='+',
        metavar='platform',
        dest='obs',
        default=default['obs'],
        choices=('modis/terra', 'modis/aqua', 'viirs/npp', 'viirs/jpss-1'),
        help='fire observing system (default: %(default)s)',
    )

    parser.add_argument(
        '--compress',
        dest='compress',
        action='store_true',
        help='compress output files (default: %(default)s)',
    )

    parser.add_argument(
        '-n',
        '--ndays',
        dest='ndays',
        type=int,
        default=default['fill_days'],
        help='Number of days to fill in',
    )

    parser.add_argument(
        '-l',
        '--log',
        dest='log_level',
        default=default['log_level'],
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='logging level',
    )

    parser.add_argument(
        'year',
        type=int,
        help="year specified in 'yyyy' format",
    )

    parser.add_argument(
        'doy', nargs='+', type=int, help='single DOY, or start and end DOYs'
    )

    args = parser.parse_args()

    # modify the Namespace object to set the 'doy' argument value
    if len(args.doy) == 1:
        doy_beg = args.doy[0]
        doy_end = doy_beg
    elif len(args.doy) == 2:
        doy_beg = min(args.doy[0], args.doy[1])
        doy_end = max(args.doy[0], args.doy[1])
    else:
        parser.error("must have one or two DOY arguments: doy | 'start doy' 'end doy'")

    args.doy = [doy_beg, doy_end]

    return args


def search(file_l3a, logging):
    """
    Search for a L3A file in the filesystem.
    """
    match = glob(file_l3a)

    if not match:
        logging.warning(
            f"The QFED L3A file '{os.path.basename(file_l3a)}' "
            f"was not found and cannot be included in the "
            f"QFED L3B processing."
        )
        return ()

    if len(match) > 1:
        logging.warning(
            f"Found multiple files matching "
            f"pattern '{os.path.basename(file_l3a)}' "
            f"in directory '{os.path.dirname(file_l3a)}': "
            f"{match}."
        )
        logging.warning(
            f"Retaining file {match[0]}. The remaining files "
            f"{match[1:]} will not be included in the processing."
        )

    return match[0]


def main():
    """
    Processes QFED L3B files according to command line arguments,
    and a configuration file.
    """
    defaults = dict(
        obs=['modis/aqua', 'modis/terra', 'viirs/npp', 'viirs/jpss-1'],
        fill_days=1,
        config='config.yaml',
        log_level='INFO',
    )

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # filename='qfed_l3a.log',
    )

    args = parse_arguments(defaults, VERSION)
    config = utils.read_config(args.config)

    logging.getLogger().setLevel(args.log_level)
    utils.display_description(VERSION, 'QFED Level 3B - Gridded Emissions')

    resolution = config['qfed']['output']['grid']['resolution']
    if resolution not in grid.CLI_ALIAS_CHOICES:
        logging.critical(
            f"Invalid choice of resolution: '{resolution}' "
            f"(choose from {str(grid.CLI_ALIAS_CHOICES).strip('()')} "
            f"in '{args.config}')."
        )
        return

    output_grid = grid.Grid(resolution)

    doy_s = args.doy[0]
    doy_e = args.doy[1] + 1

    for doy in range(doy_s, doy_e):

        d = datetime(args.year, 1, 1) + timedelta(days=(doy - 1)) + timedelta(hours=12)

        frp = {}
        area = {}
        frp_density = {}

        for component in args.obs:
            instrument, satellite = component.split('/')
            platform = Instrument(instrument), Satellite(satellite)

            search_path = utils.get_path(
                config['qfed']['output']['frp'][component]['file'],
                timestamp=d,
            )

            logging.debug(
                f"Searching for QFED L3A file "
                f"matching pattern '{os.path.basename(search_path)}' "
                f"in directory '{os.path.dirname(search_path)}'."
            )

            l3a_file = search(search_path, logging)
            if not l3a_file:
                continue

            logging.info(f"Reading QFED L3A file '{os.path.basename(l3a_file)}'.")
            f = nc.Dataset(l3a_file, 'r')

            area[platform] = {
                v: np.transpose(f.variables[v][0, :, :])
                for v in ('land', 'water', 'cloud', 'unknown')
            }

            frp[platform] = {
                bb: np.transpose(f.variables[f'frp_{bb.type.value}'][0, :, :])
                for bb in fire.BIOMASS_BURNING
            }

            frp_density[platform] = {
                # TODO: read the predicted FRP
                bb: np.zeros_like(frp[platform][bb])
                for bb in fire.BIOMASS_BURNING
            }

        # TODO: FRP density forecast files
        d_fcst = d + timedelta(days=1)
        l3a_fcst_files = {}
        for component in args.obs:
            instrument, satellite = component.split('/')
            platform = Instrument(instrument), Satellite(satellite)

            search_path = utils.get_path(
                config['qfed']['output']['frp'][component]['file'],
                timestamp=d_fcst,
            )

            match = glob(search_path)
            if match:
                l3a_fcst_files[component] = match[0]
            else:
                l3a_fcst_files[component] = None

        # emissions and output
        output_file = utils.get_path(
            config['qfed']['output']['emissions']['file'],
            timestamp=d,
        )

        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        emission_factors_file = os.path.join(
            os.path.dirname(sys.argv[0]), '..', 'etc', 'emission_factors.yaml'
        )

        emissions = Emissions(d, frp, frp_density, area, emission_factors_file)
        emissions.calculate(('co2', 'oc'))
        emissions.save(
            output_file,
            forecast=l3a_fcst_files,
            ndays=args.ndays,
            compress=args.compress,
        )


if __name__ == '__main__':
    main()
