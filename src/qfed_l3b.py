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
import re

import numpy as np
import xarray as xr

from qfed import cli_utils
from qfed import grid
from qfed.instruments import Instrument, Satellite, canonical_instrument, canonical_satellite
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
        choices=('modis/terra', 'modis/aqua', 'viirs/npp', 'viirs/jpss-1', 'viirs/jpss-2'),
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


def search(file_glob: str, logging):
    """
    Search for a L3A file in the filesystem.

    Returns
    -------
    First matching file path as str, or None if not found.
    """
    match = glob(file_glob)

    if not match:
        logging.warning(
            f"The QFED L3A file '{os.path.basename(file_glob)}' "
            f"was not found and cannot be included in the QFED L3B processing."
        )
        return None

    if len(match) > 1:
        logging.warning(
            f"Found multiple files matching pattern '{os.path.basename(file_glob)}' "
            f"in directory '{os.path.dirname(file_glob)}': {match}."
        )
        logging.warning(
            f"Retaining file {match[0]}. The remaining files {match[1:]} "
            f"will not be included in the processing."
        )

    return match[0]

def _combined_var_names(inst: Instrument, sat: Satellite, bb_code: str):

    def _sanitize(s: str) -> str:
        s = re.sub(r'[^0-9a-z_]+', '_', str(s).lower())
        return re.sub(r'_+', '_', s).strip('_')

    """Return likely var names for combined forecast files."""
    inst_code = canonical_instrument.get(inst, _sanitize(inst.value))
    sat_code  = canonical_satellite.get(sat, _sanitize(sat.value))
    bb_code   = _sanitize(bb_code)
    # try both orders
    return (
        f"{sat_code}_{inst_code}_{bb_code}",  # sat_inst_biome (e.g., vj1_viirs_gl)
        f"{inst_code}_{sat_code}_{bb_code}",  # inst_sat_biome (e.g., viirs_vj1_gl)
    )

def load_frp_density(
    *,
    platform: tuple[Instrument, Satellite],
    fcs_this_l3a_file: str | None,
    frp: dict,     # frp[platform][bb] exists; used for shapes
    fire,          # provides BIOMASS_BURNING
    logging,
    use_forecast: bool = True,
    ) -> dict:
    """
    Load background FRP density for one platform from today's combined forecast file.
    If the file or variables are missing, return zeros (per biome).
    """
    inst_enum, sat_enum = platform
    bg = {}

    if not use_forecast:
        logging.info(f"Background set to zeros for {inst_enum.name}/{sat_enum.name} (use_forecast=False).")
        return {bb: np.zeros_like(frp[platform][bb]) for bb in fire.BIOMASS_BURNING}

    if fcs_this_l3a_file and os.path.exists(fcs_this_l3a_file):
        try:
            with xr.open_dataset(fcs_this_l3a_file) as ds:
                for bb in fire.BIOMASS_BURNING:
                    found = False
                    for vn in _combined_var_names(inst_enum, sat_enum, bb.type.value):
                        if vn in ds.variables:
                            bg[bb] = ds[vn][0, :, :].values.T
                            found = True
                            break
                    if not found:
                        logging.warning(
                            f"Background var not found for {inst_enum.name}/{sat_enum.name}/{bb.type.value} "
                            f"in '{os.path.basename(fcs_this_l3a_file)}'; using zeros."
                        )
                        bg[bb] = np.zeros_like(frp[platform][bb])
            logging.info(f"Background loaded from forecast file '{os.path.basename(fcs_this_l3a_file)}'.")
            return bg
        except Exception as e:
            logging.warning(
                f"Failed to read forecast background '{os.path.basename(fcs_this_l3a_file)}' ({e}). Using zeros."
            )

    # No file or failed read -> zeros
    for bb in fire.BIOMASS_BURNING:
        bg[bb] = np.zeros_like(frp[platform][bb])
    return bg



def process(
    time,
    output_grid,
    output_file,
    obs_system,
    fcs_bkg,
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

    # ---- Forecast background paths (today: read; tomorrow: write) ----
    fcs_this_l3a_file = None
    fcs_next_l3a_file = None
    if fcs_bkg is not None:
        # today's forecast background (search because it SHOULD already exist)
        fcs_today_glob = cli_utils.get_path(fcs_bkg['file'], timestamp=time)
        fcs_this_l3a_file = search(fcs_today_glob, logging)  # Optional[str]

        # tomorrow's forecast background output (do NOT search; we'll create it)
        fcs_day = time + timedelta(days=1)
        fcs_next_l3a_file = cli_utils.get_path(fcs_bkg['file'], timestamp=fcs_day)

    # ---- Iterate components, read L3A, fill FRP/area/background ----
    for component in obs_system:
        instrument, satellite = component.split('/')
        platform = (Instrument(instrument), Satellite(satellite))

        # current-day per-platform L3A (observations)
        search_path = cli_utils.get_path(obs_system[component]['file'], timestamp=time)
        logging.debug(
            f"Searching for QFED L3A file matching '{os.path.basename(search_path)}' "
            f"in '{os.path.dirname(search_path)}'."
        )
        l3a_file = search(search_path, logging)
        if not l3a_file:
            continue

        logging.info(f"Reading QFED L3A file '{os.path.basename(l3a_file)}'.")
        l3a_files[platform] = l3a_file

        ds = xr.open_dataset(l3a_file)
        area[platform] = {
            v: ds[v][0, :, :].values.T
            for v in ('land', 'water', 'cloud', 'unknown')
        }
        frp[platform] = {
            bb: ds[f'frp_{bb.type.value}'][0, :, :].values.T
            for bb in fire.BIOMASS_BURNING
        }
        ds.close()

        # Load today's FRP density background (forecast) or zeros
        frp_density[platform] = load_frp_density(
            platform=platform,
            frp=frp,
            fire=fire,
            logging=logging,
            use_forecast=(fcs_bkg is not None),          # toggle read vs zeros
            fcs_this_l3a_file=fcs_this_l3a_file,         # Optional[str]
        )

    # ---- Emissions & outputs ----
    output_file = cli_utils.get_path(output_file, timestamp=time)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    emissions = Emissions(time, frp, frp_density, area, emission_factors_file)
    emissions.calculate(species)

    # Write combined forecast density file for tomorrow, only if configured
    if fcs_next_l3a_file is not None:
        os.makedirs(os.path.dirname(fcs_next_l3a_file), exist_ok=True)
        emissions._save_forecast(fcs_next_l3a_file, compress=compress, diskless=dry_run)

    emissions.save(
        output_file,
        doi,
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
    fcs_bkg = config['qfed']['output']['frp_fcs']
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
            fcs_bkg,
            emission_factors_file,
            species,
            args.ndays,
            args.compress,
            args.dry_run,
            doi,
        )


if __name__ == '__main__':
    main()
