#!/usr/bin/env python3

"""
A script that creates QFED Level 3A files.
"""

import os
import logging
from datetime import datetime, timedelta
import yaml
import argparse
import textwrap

import netCDF4 as nc

from qfed import cli_utils
from qfed import grid
from qfed import geolocation_products
from qfed import classification_products
from qfed import fire_products
from qfed.inventory import Finder
from qfed.instruments import Instrument, Satellite
from qfed.frp import GriddedFRP
from qfed import VERSION
from qfed.vegetation import IGBPNetCDF

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
              $ %(prog)s --obs mod myd vnp vj1 vj2 2021-08-21

              process several months of VIIRS/JPSS1 fire observations and compress the output files
              $ %(prog)s --obs vj1 --compress 2020-08-01 2021-04-01
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
        '-s',
        '--obs',
        nargs='+',
        metavar='platform',
        dest='obs',
        default=default['obs'],
        choices=('mod', 'myd', 'vnp', 'vj1', 'vj2'),
        help=("Fire observing system(s). Accepts short or long names: "
              "mod|modis/terra, myd|modis/aqua, "
              "vnp|viirs/npp or s-npp"
              "vj1|viirs/jpss-1 or noaa-20, "
              "vj2|viirs/jpss-2 or noaa-21"),
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
        help='compress output files (default: %(default)s)',
    )

    parser.add_argument(
        '--dry-run',
        dest='dry_run',
        action='store_true',
        help='perform a trial run without modifying output files (default: %(default)s)',
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


def process(
    t_start,
    t_end,
    timestamp,
    output_grid,
    output,
    obs_system,
    igbp,
    version,
    watermask,
    compress,
    dry_run,
):
    """
    Processes single timestamped time interval.
    """
    for satellite in obs_system.keys():

        platform = Satellite(satellite) #Instrument(instrument)
        
        # input files
        gp_file = cli_utils.get_path(obs_system[satellite]['geolocation']['file'])
        fp_file = cli_utils.get_path(obs_system[satellite]['fires']['file'])
        
#         vg_dir = igbp

        # output file
        output_file = cli_utils.get_path(output[satellite], timestamp=timestamp, 
                                         version=version, sat=satellite)

      
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

        # product readers
#         finder = Finder(gp_file, fp_file, vg_dir)
        finder = Finder(gp_file, fp_file)
        gp_reader = geolocation_products.create(platform)
        fp_reader = fire_products.create(platform)
        cp_reader = classification_products.create(platform)

        cp_reader.set_auxiliary(watermask=watermask)

        # generate gridded FRP and areas
        frp = GriddedFRP(satellite, output_grid, finder, gp_reader, fp_reader, cp_reader, igbp)
        frp.ingest(t_start, t_end)
        frp.save(
            output_file,
            timestamp,
            qc=True,
            compress=compress,
            satellite=satellite,
            fill_value=1e20,
            diskless=dry_run,
        )


def main():
    """
    Processes QFED L3A files according to command line arguments,
    and a configuration file.
    """
    defaults = dict(
        obs=['mod', 'myd', 'vnp', 'vj1', 'vj2'],
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
    config = cli_utils.read_config(args.config)

    logging.getLogger().setLevel(args.log_level)
    cli_utils.display_description(VERSION, 'QFED Level 3A - Gridded FRP and Areas')

    resolution = config['qfed']['output']['grid']['resolution']
    if resolution not in grid.CLI_ALIAS_CHOICES:
        logging.critical(
            f"Invalid choice of resolution: '{resolution}' "
            f"(choose from {str(grid.CLI_ALIAS_CHOICES).strip('()')} "
            f"in '{args.config}')."
        )
        return

    output_grid = grid.Grid(resolution)

    watermask = get_auxiliary_watermask(config['qfed']['with']['watermask'])

	# Option if want to remove the gas flaring and other static sources
#     igbp = IGBPNetCDF( config['qfed']['with']['igbp'], 
#                       static_heat=True,
#                       gasflaring=True,
#                       volcano=True,
#                       drops = [0, 21, 22, 23])

    igbp = IGBPNetCDF(config['qfed']['with']['igbp'],
                      drops = [0])
    

    obs = {platform: config['qfed']['with'][platform] for platform in args.obs}

    output = {
        platform: config['qfed']['output']['frp']['file'] for platform in args.obs
    }
    
    version = f'v{VERSION.replace(".", "_")}'
    
    start, end = cli_utils.get_entire_time_interval(args)
    intervals = cli_utils.get_timestamped_time_intervals(start, end, timedelta(hours=24))

    for t_start, t_end, timestamp in intervals:
        process(
            t_start,
            t_end,
            timestamp,
            output_grid,
            output,
            obs,
            igbp,
            version,
            watermask,
            args.compress,
            args.dry_run,
        )



if __name__ == '__main__':
    main()
