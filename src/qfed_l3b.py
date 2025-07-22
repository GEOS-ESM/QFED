#!/usr/bin/env python3

"""
A script that creates QFED Level 3B files.
"""

import os
import sys
import logging
import argparse
import yaml
import textwrap
from datetime import datetime, timedelta
from glob import glob

import numpy as np
import xarray as xr

from qfed import cli_utils
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
        epilog=textwrap.dedent(
            '''
            examples:
              generate emissions for a single date using MODIS and VIIRS L3A data
              $ %(prog)s --obs modis/aqua modis/terra viirs/npp viirs/jpss-1 2021-08-21

              generate and persist emissions for 7 consecutive days
              $ %(prog)s --obs viirs/jpss-1 --ndays 7 2021-08-21

              generate emissions for several months of VIIRS/JPSS1 L3A data and compress the output files
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
        '-n',
        '--ndays',
        dest='ndays',
        type=int,
        default=default['fill_days'],
        help='number of days to fill in (default: %(default)s)',
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
        'date_start',
        type=datetime.fromisoformat,
        metavar='start',
        help='start date in the format YYYY-MM-DD',
    )

    parser.add_argument(
        '--dry-run',
        dest='dry_run',
        action='store_true',
        help='perform a trial run without modifying output files (default: %(default)s)',
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


def process(
    time,
    output_grid,
    output_file,
    obs_system,
    emission_factors_file,
    species,
    ndays,
    compress,
    dry_run,
    doi,
):
    """
    Processes single time/date.
    """

    frp = {}
    area = {}
    frp_density = {}
    l3a_files = {}

    for component in obs_system:
        instrument, satellite = component.split('/')
        platform = (Instrument(instrument), Satellite(satellite))

        # Load current day's data
        search_path = cli_utils.get_path(
            obs_system[component]['file'],
            timestamp=time,
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
        l3a_files[platform] = l3a_file
        
        current_ds = xr.open_dataset(l3a_file)
        
        area[platform] = {
            v: current_ds[v][0, :, :].values.T
            for v in ('land', 'water', 'cloud', 'unknown')
        }

        frp[platform] = {
            bb: current_ds[f'frp_{bb.type.value}'][0, :, :].values.T
            for bb in fire.BIOMASS_BURNING
        }
        
        # Load previous day's FRP density for Background
        previous_day = time - timedelta(days=1)
        previous_search_path = cli_utils.get_path(
            obs_system[component]['file'],
            timestamp=previous_day,
        )
        
        previous_l3a_file = search(previous_search_path, logging)
        
        if previous_l3a_file:
            logging.info(f"Reading previous day's dampened FRP density from '{os.path.basename(previous_l3a_file)}' for sequential method.")
            previous_ds = xr.open_dataset(previous_l3a_file)
            frp_density[platform] = {
                bb: previous_ds[f'fb_{bb.type.value}'][0, :, :].values.T
                for bb in fire.BIOMASS_BURNING
            }
            
            previous_ds.close()
        else:
            logging.warning(f"Previous day's L3A file or background not found for {component}. Using zeros for background state.")
            frp_density[platform] = {
                bb: np.zeros_like(frp[platform][bb])
                for bb in fire.BIOMASS_BURNING
            }
        
        current_ds.close()

    # emissions and output
    output_file = cli_utils.get_path(
        output_file,
        timestamp=time,
    )

    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    emissions = Emissions(time, frp, frp_density, area, emission_factors_file)
    emissions.calculate(species)
    emissions.save_forecast_density(l3a_files)
    emissions.save(
        output_file,
        doi,
        forecast=None,
        ndays=ndays,
        compress=compress,
        diskless=dry_run,
    )


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
    config = cli_utils.read_config(args.config)

    logging.getLogger().setLevel(args.log_level)
    cli_utils.display_description(VERSION, 'QFED Level 3B - Gridded Emissions')

    resolution = config['qfed']['output']['grid']['resolution']
    if resolution not in grid.CLI_ALIAS_CHOICES:
        logging.critical(
            f"Invalid choice of resolution: '{resolution}' "
            f"(choose from {str(grid.CLI_ALIAS_CHOICES).strip('()')} "
            f"in '{args.config}')."
        )
        return

    output_grid = grid.Grid(resolution)

    obs = {platform: config['qfed']['output']['frp'][platform] for platform in args.obs}
    output_file = config['qfed']['output']['emissions']['file']
    doi = config['qfed']['output']['emissions']['doi']

    emission_factors_file = os.path.join(
        os.path.dirname(sys.argv[0]), 'emission_factors.yaml'
    )

    species = ('co2', 'oc', 'so2', 'nh3', 'bc', 'co','acet','ald2','alk4','c2h6','c3h8','ch2o','mek','no','c3h6','pm25','tpm','ch4')

    start, end = cli_utils.get_entire_time_interval(args)
    intervals = cli_utils.get_timestamped_time_intervals(start, end, timedelta(hours=24))

    for t_start, t_end, timestamp in intervals:
        process(
            timestamp,
            output_grid,
            output_file,
            obs,
            emission_factors_file,
            species,
            args.ndays,
            args.compress,
            args.dry_run,
            doi,
        )


if __name__ == '__main__':
    main()
