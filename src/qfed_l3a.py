#!/usr/bin/env python3

"""
A script that creates QFED Level 3A files.
"""

import os
import logging
from datetime import datetime, timedelta
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
              "vnp|viirs/npp or s-npp, "
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
        '--max-workers',
        dest='max_workers',
        type=int,
        default=None,
        help=(
            'maximum number of parallel worker processes for granule '
            'processing (default: auto-detect based on CPU count)'
        ),
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


def process(
    t_start,
    t_end,
    timestamp,
    output_grid,
    output,
    obs_system,
    igbp_template,
    peat_file,
    version,
    watermask_file,
    compress,
    dry_run,
    FRPcapping=True,
    max_workers=None,
):
    """
    Processes all satellites sequentially for a single timestamped
    time interval.

    Granule-level parallelism is handled inside GriddedFRP.ingest(),
    which spawns one worker process per granule up to max_workers.
    IGBP and watermask data are loaded from file paths inside each
    worker process and cached after first use, so there is no
    pickling of large numpy arrays across process boundaries.

    Parameters
    ----------
    t_start : datetime
        Start of the processing window (inclusive).
    t_end : datetime
        End of the processing window (exclusive).
    timestamp : datetime
        Timestamp used to label the output file.
    output_grid : grid.Grid
        Output grid specification.
    output : dict
        Mapping of satellite name -> output file path template.
    obs_system : dict
        Mapping of satellite name -> geolocation/fire file templates.
    igbp_template : str
        Raw format string for the IGBP file path. Formatted with t_start.
    version : str
        QFED version string used in output file names.
    watermask_file : str
        Path to the auxiliary watermask NetCDF file.
    compress : bool
        If True, compress output NetCDF variables with zlib.
    dry_run : bool
        If True, create diskless (in-memory) output files only.
    max_workers : int or None
        Maximum number of parallel granule worker processes.
        None defers to the GriddedFRP default (cpu_count - 1).
    """
    # Format the IGBP path for the year of t_start.
    # The path string is passed directly to GriddedFRP — workers load
    # and cache their own IGBPNetCDF instances from this path.
    igbp_path = igbp_template.format(t_start)
    logging.info(f"Using IGBP file: {igbp_path}")


    for satellite in obs_system.keys():

        platform = Satellite(satellite)

        # Input file path templates
        gp_file = cli_utils.get_path(obs_system[satellite]['geolocation']['file'])
        fp_file = cli_utils.get_path(obs_system[satellite]['fires']['file'])

        # Output file path
        output_file = cli_utils.get_path(
            output[satellite],
            timestamp=timestamp,
            version=version,
            sat=satellite,
        )

        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        # Product readers
        # These are passed to GriddedFRP for API compatibility but are
        # not used internally by the parallel implementation — each
        # worker process creates its own fresh reader instances.
        finder    = Finder(gp_file, fp_file)
        gp_reader = geolocation_products.create(platform)
        fp_reader = fire_products.create(platform)
        cp_reader = classification_products.create(platform)


        # Generate gridded FRP and areas.
        # Pass igbp_path and watermask_file as strings — workers load
        # and cache their own instances, avoiding large array pickling.
        frp = GriddedFRP(
            satellite,
            output_grid,
            finder,
            gp_reader,
            fp_reader,
            cp_reader,
            igbp_path,
            watermask_file=watermask_file,
            peat_file=peat_file,
            max_workers=max_workers,
        )
        frp.ingest(t_start, t_end)
        frp.save(
            output_file,
            timestamp,
            qc=FRPcapping,
            compress=compress,
            satellite=satellite,
            fill_value=1e20,
            diskless=dry_run,
        )


def main():
    """
    Processes QFED L3A files according to command line arguments
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
    FRPcapping=config['qfed']['with']['FRPcapping']
    # Pass the watermask file path to process() — workers load their
    # own copies from this path, so we do not read it into memory here.
    watermask_file = config['qfed']['with']['watermask']

    # Keep as a raw template string; formatted with t_start inside process()
    igbp_template = config['qfed']['with']['igbp']
    peat_file = config['qfed']['with'].get('peat', None) 

    obs = {platform: config['qfed']['with'][platform] for platform in args.obs}

    output = {
        platform: config['qfed']['output']['frp']['file'] for platform in args.obs
    }

    version = f'v{VERSION.replace(".", "_")}'

    start, end = cli_utils.get_entire_time_interval(args)
    intervals = cli_utils.get_timestamped_time_intervals(
        start, end, timedelta(hours=24)
    )

    logging.info(
        f"Processing {len(intervals)} date(s) "
        f"with {len(args.obs)} satellite(s) each."
    )

    for t_start, t_end, timestamp in intervals:
        logging.info(f"\n{'='*70}")
        logging.info(f"Processing date: {timestamp:%Y-%m-%d}")
        logging.info(f"{'='*70}\n")

        process(
            t_start,
            t_end,
            timestamp,
            output_grid,
            output,
            obs,
            igbp_template,
            peat_file,
            version,
            watermask_file,
            args.compress,
            args.dry_run,
            FRPcapping,
            max_workers=args.max_workers,
        )

    logging.info("\n" + "="*70)
    logging.info("All processing complete.")
    logging.info("="*70)


if __name__ == '__main__':
    main()
