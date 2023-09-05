#!/usr/bin/env python3

"""
A script that creates QFED Level 3A files.
"""

import os
import logging
from datetime import datetime, timedelta
import yaml
import argparse

from qfed import grid
from qfed import geolocation_products
from qfed import classification_products
from qfed import fire_products
from qfed.inventory import Finder
from qfed.instruments import Instrument, Satellite
from qfed.frp import GriddedFRP
from qfed import VERSION


def parse_arguments(default, version):
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        prog='qfed_l3a.py',
        description='Creates QFED Level 3A files',
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
        '-o',
        '--output-dir',
        dest='output_dir',
        default=default['output_dir'],
        help='directory for output files',
    )

    parser.add_argument(
        '-p',
        '--products',
        dest='products',
        default=default['products'],
        help='list of active fire products',
    )

    parser.add_argument(
        '-r',
        '--resolution',
        dest='resolution',
        default=default['resolution'],
        help='horizontal resolution',
    )

    parser.add_argument(
        '-l',
        '--log',
        dest='log_level',
        default=default['log_level'],
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='logging level',
    )

    args = parser.parse_args()

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
    logging.info('QFED Level 3A - Gridded FRP')
    logging.info('')


if __name__ == "__main__":

    defaults = dict(
        products='modis/aqua,modis/terra,viirs/npp,viirs/jpss-1',
        resolution='e',
        output_dir='./',
        log_level='INFO',
        config='config.yaml',
    )

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # filename='qfed_l3a.log',
    )

    time = datetime(2021, 2, 1, 12)
    time_window = timedelta(hours=24)

    time_s = time - 0.5 * time_window
    time_e = time + 0.5 * time_window

    args = parse_arguments(defaults, VERSION)
    config = read_config(args.config)

    logging.getLogger().setLevel(args.log_level)

    display_banner(VERSION)

    output_grid = grid.Grid(args.resolution)
    products = args.products.replace(' ', '').split(',')

    for p in products:
        instrument, satellite = p.split('/')
        platform = Instrument(instrument), Satellite(satellite)

        # input files
        gp_dir, gp_template = config[p]['geolocation']
        fp_dir, fp_template = config[p]['fires']
        vg_dir = config['igbp']

        gp_file = os.path.join(gp_dir, '{0:%Y}', '{0:%j}', gp_template)
        fp_file = os.path.join(fp_dir, '{0:%Y}', '{0:%j}', fp_template)

        # output file
        output_template = config[p]['frp']
        output_file = os.path.join(args.output_dir, output_template.format(time))

        # product readers
        finder = Finder(gp_file, fp_file, vg_dir)
        gp_reader = geolocation_products.create(*platform)
        fp_reader = fire_products.create(*platform)
        cp_reader = classification_products.create(*platform)

        # generate gridded FRP and areas
        frp = GriddedFRP(output_grid, finder, gp_reader, fp_reader, cp_reader)
        frp.ingest(time_s, time_e)
        frp.save(
            filename=output_file,
            timestamp=time,
            bootstrap=True,
            qc=False,
            fill_value=1e20,
        )
