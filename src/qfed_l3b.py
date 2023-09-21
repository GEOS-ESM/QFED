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
        description='Creates QFED Level 3B files',
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
        '-p',
        '--products',
        dest='products',
        default=default['products'],
        help='list of active fire products',
    )

    parser.add_argument(
        '-u',
        '--uncompressed',
        action='store_true',
        help='do not compress output files (default=False)',
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


def read_config(config):
    """
    Parses the QFED config file into a dictionary.
    """
    with open(config) as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            data = None
            logging.critical(exc)

    return data


def display_banner(version):
    logging.info('')
    logging.info(f'QFED {version}')
    logging.info('')
    logging.info('QFED Level 3B - Gridded Emissions')
    logging.info('')


if __name__ == '__main__':

    defaults = dict(
        products='modis/aqua,modis/terra,viirs/npp,viirs/jpss-1',
        fill_days=1,
        log_level='INFO',
        config='config.yaml',
    )

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # filename='qfed_l3a.log',
    )

    args = parse_arguments(defaults, VERSION)
    config = read_config(args.config)

    logging.getLogger().setLevel(args.log_level)
    display_banner(VERSION)

    products = args.products.replace(' ', '').split(',')

    doy_s = args.doy[0]
    doy_e = args.doy[1] + 1

    for doy in range(doy_s, doy_e):

        d = datetime(args.year, 1, 1) + timedelta(days=(doy - 1)) + timedelta(hours=12)

        frp = {}
        area = {}
        frp_density = {}

        for p in products:
            instrument, satellite = p.split('/')
            platform = Instrument(instrument), Satellite(satellite)

            # TODO
            l3a_dir = './validation/v3.0.0-geos-esm/FRP/'
            l3a_filename = config[p]['frp'].format(d)

            search_path = os.path.join(l3a_dir, l3a_filename)

            logging.debug(
                (
                    f"Searching for QFED L3A file "
                    f"matching pattern '{os.path.basename(search_path)}' "
                    f"in directory '{os.path.dirname(search_path)}'."
                )
            )

            match = glob(search_path)
            if match:
                if len(match) > 1:
                    logging.warning(
                        (
                            f"Found multiple files matching "
                            f"pattern '{os.path.basename(search_path)}' "
                            f"in directory '{os.path.dirname(search_path)}': "
                            f"{match}."
                        )
                    )

                    logging.warning(
                        (
                            f"Retaining file {match[0]}. The remaining files "
                            f"{match[1:]} will not be included in the processing."
                        )
                    )
            else:
                logging.warning(
                    (
                        f"Did not find QFED L3A file '{search_path}'. "
                        f"This file will not be included in the processing."
                    )
                )

                continue

            l3a_file = match[0]

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
                bb: np.transpose(f.variables[f'fb_{bb.type.value}'][0, :, :])
                for bb in fire.BIOMASS_BURNING
            }

        # TODO: FRP density forecast files
        d_fcst = d + timedelta(days=1)
        l3a_fcst_files = {}
        for p in products:
            instrument, satellite = p.split('/')
            platform = Instrument(instrument), Satellite(satellite)

            l3a_fcst_dir = './validation/v3.0.0-geos-esm/FRP/'
            l3a_fcst_filename = config[p]['frp'].format(d_fcst)

            search_path = os.path.join(l3a_fcst_dir, l3a_fcst_filename)
            match = glob(search_path)

            if match:
                l3a_fcst_files[p] = match[0]
            else:
                l3a_fcst_files[p] = None

        # output
        output_dir = config['emissions'][0]
        output_template = config['emissions'][1]
        output_file = os.path.join(output_dir, output_template.format(d))
        os.makedirs(output_dir, exist_ok=True)

        species = ('co2', 'co', 'oc', 'bc', 'so2')
        emission_factors_file = os.path.join(
            os.path.dirname(sys.argv[0]), '..', 'etc', 'emission_factors.yaml'
        )

        emissions = Emissions(d, frp, frp_density, area, emission_factors_file)
        emissions.calculate(species)
        emissions.save(
            filename=output_file,
            dir=output_dir,
            forecast=l3a_fcst_files,
            ndays=args.ndays,
            uncompressed=args.uncompressed,
        )
