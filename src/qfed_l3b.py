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
from qfed.instruments import Instrument, Satellite
from qfed.emissions import Emissions
from qfed import fire
from qfed import VERSION
from qfed.scaletov2 import apply_regional_scaling


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
              $ %(prog)s --obs mod myd vnp vj1 vj2 2021-08-21

              generate and persist emissions for 7 consecutive days
              $ %(prog)s --obs vj1 --ndays 7 2021-08-21

              generate emissions for several months of VIIRS/JPSS1 L3A data and compress the output files
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


def load_frp_density(
    *,
    platform,                 # (Instrument, Satellite)
    fcs_this_l3a_file: str | None,
    frp: dict,                # shapes via frp[platform][bb]
    fire,                     # BIOMASS_BURNING
    logging,
    use_forecast: bool = True,
) -> dict:
    """Read today's per-sensor FRP-FCST file (vars: fb_{biome}); else zeros."""
    if not use_forecast or not (fcs_this_l3a_file and os.path.exists(fcs_this_l3a_file)):
        return {bb: np.zeros_like(frp[platform][bb]) for bb in fire.BIOMASS_BURNING}

    bg = {}
    try:
        with xr.open_dataset(fcs_this_l3a_file) as ds:
            for bb in fire.BIOMASS_BURNING:
                vn = f"fb_{bb.type.value}"
                if vn in ds.variables:
                    bg[bb] = ds[vn][0, :, :].values.T
                else:
                    logging.warning(f"'{vn}' missing in {os.path.basename(fcs_this_l3a_file)}; using zeros.")
                    bg[bb] = np.zeros_like(frp[platform][bb])
        logging.info(f"Background loaded: {os.path.basename(fcs_this_l3a_file)}")
    except Exception as e:
        logging.warning(f"Failed to read {os.path.basename(fcs_this_l3a_file)} ({e}); using zeros.")
        bg = {bb: np.zeros_like(frp[platform][bb]) for bb in fire.BIOMASS_BURNING}
    return bg


def process(
    time,
    output_grid,
    output_file,
    obs_system,
    fcs_bkg,
    emission_factors_file,
    alpha_factor_file,
    species,
    ndays,
    compress,
    dry_run,
    doi,
    scaling_config,
    dt = 1.0, 
    tau = 3.0
):
    frp = {}
    area = {}
    frp_density = {}
    l3a_files = {}
    l3a_fsc_files = {}
    number_of_l2b_file = {}

    # Iterate platforms
    for satellite in obs_system.keys():
    
        platform = Satellite(satellite)
        # Current-day per-platform L3A (observations)
        search_path = cli_utils.get_path(obs_system[satellite]['file'], 
                                         timestamp=time, 
                                         sat=satellite, 
                                         version = f'v{VERSION.replace(".", "_")}')

        logging.debug(f"Searching for L3A '{os.path.basename(search_path)}' in '{os.path.dirname(search_path)}'")
        l3a_file = search(search_path, logging)
        
        if not l3a_file:
            raise FileNotFoundError(f"Failed to find '{os.path.basename(search_path)}' ")
            
        else:
            logging.info(f"Reading L3A '{os.path.basename(l3a_file)}'")
            l3a_files[platform] = l3a_file

            ds = xr.open_dataset(l3a_file)
            area[platform] = {v: ds[v][0, :, :].values.T for v in ('land', 'water', 'cloud', 'unknown')}
            frp[platform] = {bb: ds[f'frp_{bb.type.value}'][0, :, :].values.T for bb in fire.BIOMASS_BURNING}
            number_of_l2b_file[platform] = ds.number_of_input_files
            ds.close()

        # Today's FRP-FCS (per sensor): format with {sat}
        fcs_this_l3a_file = None
        
        if fcs_bkg is not None:
            today_path = cli_utils.get_path(fcs_bkg["file"], 
                             timestamp=time-timedelta(days=1), 
                             sat=satellite, 
                             version = f'v{VERSION.replace(".", "_")}')
            fcs_this_l3a_file = search(today_path, logging)
            
            # Background (forecast or zeros)
            frp_density[platform] = load_frp_density(
                platform=platform,
                fcs_this_l3a_file=fcs_this_l3a_file,
                frp=frp,
                fire=fire,
                logging=logging,
                use_forecast=(fcs_bkg is not None),
                )            
            l3a_fsc_files[platform] = fcs_this_l3a_file
    
    no_l2b_flag = all(value == 0 for value in number_of_l2b_file.values())
    
    if no_l2b_flag:
        logging.warning(f"No valid l3b grided FRP available, apply Full persistence to FPR.")
        
        im = output_grid.dimensions()['x']
        jm = output_grid.dimensions()['y']
        # overwrite the cloud area to all one
        for satellite in obs_system.keys():
            platform = Satellite(satellite)
            area[platform]['cloud'] = np.ones((im, jm))  
            for biome in frp_density[platform].keys():
                frp_density[platform][biome] *=np.exp(dt/tau)
 
    # Emissions & outputs
    # Build a template with {date} placeholder for emissions.save() to use
    # First, get the path with directory structure
    sample_path = cli_utils.get_path(output_file, timestamp=time)
    os.makedirs(os.path.dirname(sample_path), exist_ok=True)
    
    # Create a template by replacing the date in the filename with {date}
    # Extract just the filename part and replace the date pattern
    date_str = time.strftime('%Y%m%d')
    out_template = sample_path.replace(date_str, '{date}')
    
    logging.debug(f"Output template: {out_template}")
    
    emissions = Emissions(time, frp, frp_density, area, emission_factors_file, alpha_factor_file)
    emissions.calculate(species, dt=dt, tau=tau)

    # Write per-sensor FRP-FCS for tomorrow
    if fcs_bkg is not None:
        fcs_out_map = {}
        for platform in frp.keys():
            tomorrow_path = cli_utils.get_path(fcs_bkg["file"], 
                                timestamp=time, 
                                sat=platform.value, 
                                version = f'v{VERSION.replace(".", "_")}')
            os.makedirs(os.path.dirname(tomorrow_path), exist_ok=True)
            fcs_out_map[platform] = tomorrow_path

        emissions._save_forecast(fcs_out_map, compress=compress, diskless=dry_run)


    emissions.save(
        out_template,
        number_of_l2b_file,
        doi,
        ndays=ndays,
        compress=compress,
        diskless=dry_run,
    )

    # Apply scaling based on precomputed mask if provided in config.yaml
    if scaling_config is None:
        logging.info("No scaling configuration found, skipping regional scaling")
    elif scaling_config.get('file')== '/dev/null':
        logging.info("Regional scaling disabled (mask file set to /dev/null)")
    else:        
        try:
            logging.info(f"Applying scaling using mask: {scaling_config['file']}")
            
            # Get save_original flag from config, default to False
            save_original = scaling_config.get('save_original', False)
            
            apply_regional_scaling(
                emissions_file_template=output_file,
                timestamp=time,
                species_list=scaling_config['species'],
                scaling_mask_file=scaling_config['file'],
                scaled_output_dir=scaling_config.get('output_dir', None),
                ndays=ndays,
                save_original=save_original,
            )
        except Exception as e:
            logging.error(f"Regional scaling failed: {e}")

def main():
    """
    Processes QFED L3B files according to command line arguments,
    and a configuration file.
    """
    defaults = dict(
        obs=['mod', 'myd', 'vnp', 'vj1', 'vj2'],
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

    obs = {platform: config['qfed']['output']['frp'] for platform in args.obs}
    
    fcs_bkg = config['qfed']['output']['frp_fcs']
    
    output_file = config['qfed']['output']['emissions']['file']
    
    doi = config['qfed']['output']['emissions']['doi']

    # Extract scaling configuration using consistent pattern
    scaling_config = config['qfed'].get('scaling', None)

    emission_factors_file = os.path.join(
        os.path.dirname(sys.argv[0]), 'emission_factors.yaml'
    )

    alpha_factor_file = os.path.join(
        os.path.dirname(sys.argv[0]), 'alpha_factor.yaml'
    )

    species = config['qfed']['output']['emissions']['species']    
      
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
            alpha_factor_file,
            species,
            args.ndays,
            args.compress,
            args.dry_run,
            doi,
            scaling_config,
        )


if __name__ == '__main__':
    main()
