#!/usr/bin/env python3

"""
A script that creates QFED Level 3A files.
"""

import os
import logging
from datetime import datetime, timedelta
import yaml
import argparse
import pathlib
import textwrap

import netCDF4 as nc

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
        description='Create QFED Level 3A files',
        epilog=textwrap.dedent(
            '''
            examples:
              process single date of MODIS and VIIRS fire observations
              $ %(prog)s --obs modis/aqua modis/terra viirs/npp viirs/jpss-1 2021-08-21

              process several months of VIIRS/JPSS1 fire observations and compress the output files
              $ %(prog)s --obs viirs/jpss-1 --compress 2020-08-01 2021-04-01
            '''
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help='config file (default: %(default)s)',
    )

    parser.add_argument(
        '-o',
        '--output-dir',
        type=pathlib.Path,
        dest='output_dir',
        default=default['output_dir'],
        help='directory for output files (default: %(default)s)',
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
        '-r',
        '--resolution',
        dest='resolution',
        default=default['resolution'],
        choices=('c', 'd', 'e', 'f', '0.1x0.1'),
        help='horizontal resolution (default: %(default)s)',
    )

    parser.add_argument(
        '-l',
        '--log-level',
        dest='log_level',
        default=default['log_level'],
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='logging level (default: %(default)s)',
    )

    parser.add_argument(
        '--compress',
        dest='compress',
        action='store_true',
        help='compress the output files (default: %(default)s)',
    )

    parser.add_argument(
        'date_start',
        type=datetime.fromisoformat,
        metavar='start',
        help='start date in the format YYYY-MM-DD',
    )

    parser.add_argument(
        'date_end',
        type=datetime.fromisoformat,
        nargs='?',
        metavar='end',
        help='end date in the format YYYY-MM-DD',
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


def display_description(version):
    """
    Displays the QFED version and a brief description
    of this script.
    """
    logging.info('')
    logging.info(f'QFED {version}')
    logging.info('')
    logging.info('QFED Level 3A - Gridded FRP and Areas')
    logging.info('')


def get_auxiliary_watermask(file):
    """
    Reads auxiliary watermask from a file.
    """
    logging.info(f"Reading auxiliary watermask from file '{file}'.")
    f = nc.Dataset(file)
    watermask = f.variables['watermask'][...]
    f.close()
    logging.debug(
        f'The auxiliary watermask uses {1e-6*watermask.nbytes:.1f} MB of RAM.'
    )
    return watermask


def get_entire_time_interval(args):
    """
    Parses args and returns the start and end
    of the entire time interval that needs to be
    processed.
    """
    time_start = args.date_start
    time_end = args.date_end

    if time_end is None:
        time_end = time_start

    return time_start, time_end


def get_timestamped_time_intervals(time_start, time_end, time_window):
    """
    Returns a list of timestamped time intervals.

    Use with caution. This is very basic... sub-intervals may
    end up outside of the complete time interval.
    """
    result = []

    t = time_start
    while t <= time_end:
        t_s = t
        t_e = t + time_window
        t_stamp = t + 0.5 * time_window

        result.append((t, t_e, t_stamp))
        t = t + time_window

    return result


def process(
    t_start,
    t_end,
    timestamp,
    output_grid,
    output_dir,
    obs_system,
    igbp,
    watermask,
    compress,
):
    """
    Processes single timestamped time interval.
    """
    for component in obs_system.keys():
        instrument, satellite = component.split('/')
        platform = Instrument(instrument), Satellite(satellite)

        # input files
        gp_dir, gp_template = obs_system[component]['geolocation']
        fp_dir, fp_template = obs_system[component]['fires']
        vg_dir = igbp

        gp_file = os.path.join(gp_dir, '{0:%Y}', '{0:%j}', gp_template)
        fp_file = os.path.join(fp_dir, '{0:%Y}', '{0:%j}', fp_template)

        # output file
        output_template = obs_system[component]['frp']
        output_file = os.path.join(output_dir, output_template.format(timestamp))

        # product readers
        finder = Finder(gp_file, fp_file, vg_dir)
        gp_reader = geolocation_products.create(*platform)
        fp_reader = fire_products.create(*platform)
        cp_reader = classification_products.create(*platform)

        cp_reader.set_auxiliary(watermask=watermask)

        # generate gridded FRP and areas
        frp = GriddedFRP(output_grid, finder, gp_reader, fp_reader, cp_reader)
        frp.ingest(t_start, t_end)
        frp.save(
            output_file,
            timestamp,
            qc=False,
            compress=compress,
            source=f'{instrument}/{satellite}'.upper(),
            instrument=instrument.upper(),
            satellite=satellite.upper(),
            fill_value=1e20,
        )


def main():
    """
    Processes QFED L3A files according to command line arguments,
    and a configuration file.
    """
    defaults = dict(
        obs=['modis/aqua', 'modis/terra', 'viirs/npp', 'viirs/jpss-1'],
        resolution='e',
        output_dir='./frp/',
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
    display_description(VERSION)

    output_grid = grid.Grid(args.resolution)

    watermask = get_auxiliary_watermask(config['watermask'])

    start, end = get_entire_time_interval(args)
    intervals = get_timestamped_time_intervals(start, end, timedelta(hours=24))

    obs = {platform: config[platform] for platform in args.obs}

    for t_start, t_end, timestamp in intervals:
        process(
            t_start,
            t_end,
            timestamp,
            output_grid,
            args.output_dir,
            obs,
            config['igbp'],
            watermask,
            args.compress,
        )


if __name__ == '__main__':
    main()
