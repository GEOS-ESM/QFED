#!/usr/bin/env python3

import os, copy
import numpy as np
import pandas as pd
import importlib
from datetime import datetime
import lib_frp_scaling as lfs 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import yaml
import argparse

plt.switch_backend('agg')


if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run analysis with a YAML config file.")
    parser.add_argument(
        "config",
        type=str,
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Analysis year
    year = cfg["year"]

    # Satellite configuration
    satellites_to_process = cfg["satellites_to_process"]
    baseline_satellite    = cfg["baseline_satellite"]

    # Data loading behavior
    fresh_load    = cfg["fresh_load"]
    plot_analysis = cfg["plot_analysis"]

    # Output & cache directories
    figure_dir = cfg["figure_dir"]
    cache_dir  = cfg["cache_dir"]

    # Parallelism
    n_workers = cfg["n_workers"]

    # QFED FRP input configuration
    lfs.base_path = cfg["base_path"]
    prefix        = cfg["prefix"]

    # Confirmation printout
    print(f"Config loaded from : {args.config}")
    print(f"Year               : {year}")
    print(f"Satellites         : {satellites_to_process}")
    print(f"Baseline satellite : {baseline_satellite}")
    print(f"Fresh load         : {fresh_load}")
    print(f"Plot analysis      : {plot_analysis}")
    print(f"Figure directory   : {figure_dir}")
    print(f"Cache directory    : {cache_dir}")
    print(f"Workers            : {n_workers}")
    print(f"QFED base path     : {lfs.base_path}")
    print(f"Prefix             : {prefix}")


    satellites = satellites_to_process + [baseline_satellite]
    biome_map = lfs.BIOMES
    biomes = lfs.FRP_VARS

    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    date_range = pd.date_range(start_date, end_date, freq='D')

    os.makedirs(cache_dir, exist_ok=True)

    if fresh_load:
        data_dict = {}
        for satellite in satellites:
            biome_dataframes, biome_names = lfs.load_time_series_by_biome_parallel([satellite], start_date, end_date, n_workers)

            df = pd.DataFrame(index=date_range)
            df.index.name = 'date'
            for biome, bdf in biome_dataframes.items():
                # bdf could be: (a) DataFrame with columns per sat, or (b) dict of Series per sat
                if isinstance(bdf, dict):
                    ser = bdf.get(satellite)
                else:
                    ser = bdf[satellite]  # column named by satellite

                if ser is None:
                    # no data for this biome/sat; create empty series
                    ser = pd.Series(dtype=float)

                # ensure it's a Series with a datetime index
                ser = pd.Series(ser)
                if not isinstance(ser.index, pd.DatetimeIndex):
                    ser.index = pd.to_datetime(ser.index)

                # align to our desired date range
                df[biome] = ser.reindex(date_range)  # .fillna(0) if you prefer zeros

            df.to_csv(f'{cache_dir}/{satellite}_{start_date.year}.csv')
            data_dict[satellite] = df
            data_dict[satellite]['date'] = data_dict[satellite].index
    else:
        data_dict = {}
        for satellite in satellites:
            filename = f'{cache_dir}/{satellite}_{year}.csv'
            data_dict[satellite] = pd.read_csv(filename)

    scale_dicts = {}
    print(f' - Process {year}')
    print(f"{'    biome':<{15}}|{'logc':^{15}}|{'c':^{15}}|{'final c':^{15}}|")
    for sat in satellites_to_process:
        
        data = copy.deepcopy(data_dict[sat]) 
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
          
        base_data = copy.deepcopy(data_dict[baseline_satellite])
        base_data['date'] = pd.to_datetime(base_data['date'])
        base_data.set_index('date', inplace=True)
        
        scale_dict = {}
        
        print(f' {"- " * 15} Sensor {sat.upper()}  {"- " * 15} ')
        for biome in biomes:
            ma_biome_data = np.log(data.resample('16D').mean()[biome].values)
            ma_biome_bl_data = np.log(base_data.resample('16D').mean()[biome].values)
            
            valid = np.where((ma_biome_data == ma_biome_data) & (ma_biome_bl_data==ma_biome_bl_data))
            
            logc = np.nanmean(ma_biome_bl_data)-np.nanmean(ma_biome_data)
            
            print(f" - {biome:<{12}} "f"{logc:^{15}.6f} "f"{np.exp(logc):^{15}.6f} ")
            scale_dict[biome] = np.exp(logc)
        scale_dicts[sat] = scale_dict


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    plot_analysis = True
    print(' - Plot scatter plot analysis')
    if plot_analysis:
        os.makedirs(figure_dir, exist_ok=True)
        for sel_sat in satellites_to_process:
            fig, axes = lfs.multiFigure(2, 2, figsize = (10, 10), gap = 0.25)

            xRange = [-10, -5]
            yRange = [-10, -5]

            delta_ticks_X = (xRange[1] - xRange[0]) / 5
            delta_ticks_Y = (yRange[1] - yRange[0]) / 5
            for i, biome in enumerate(biomes):
            
                panel_letter = f'({chr(97 + i)})'
                x = np.log(data_dict[baseline_satellite][biome].values)
                y = np.log(data_dict[sel_sat][biome].values)

                axes[i] = lfs.scatter(axes[i], x, y, stdOn = False, one2one_line = True,  
                                  markersize = 15, eeOn= False, alpha = 1, xRange = xRange, yRange = yRange,
                                  model = 'RMA', regress_line = False, 
                                  delta_ticks_X = delta_ticks_X, delta_ticks_Y = delta_ticks_Y )

                x = np.log(data_dict[baseline_satellite][biome].values)
                y = np.log(data_dict[sel_sat][biome].values * scale_dicts[sel_sat][biome])
                axes[i] = lfs.scatter(axes[i], x, y, stdOn = False, one2one_line = True, 
                                  markersize = 2, eeOn= False, alpha = 1, xRange = xRange, yRange = yRange,
                                  delta_ticks_X = delta_ticks_X, delta_ticks_Y = delta_ticks_Y, 
                                  model = 'RMA', regress_line = False, 
                                  color = 'crimson', label_p = 'lower right', case_str='After Scaling:')

                axes[i].set_xlabel(lfs.SATELLITES[baseline_satellite] + ' ln($\\rho_{FRP}$) (MW$\\cdot km^{-2}$)')
                axes[i].set_ylabel(lfs.SATELLITES[sel_sat] + ' ln($\\rho_{FRP}$) (MW$\\cdot km^{-2}$)')

                axes[i].set_title(f"{panel_letter} {prefix} {biome_map[biome]} ({year})\nScaling: {scale_dicts[sel_sat][biome]:4.4f}")

            plt.savefig(f'{figure_dir}/SCATTER_Scaling_factor_{sel_sat}_{year}')
            plt.close()
            
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    print(' - Plot timeseries analysis')
    if plot_analysis:
        os.makedirs(figure_dir, exist_ok=True)
        for sat in satellites_to_process:

            product = sat.upper()

            fig, axes = lfs.multiFigure(4, 1, figsize = (9, 12), gap = 0.25, left = 0.1)

            for idx, biome in enumerate(biomes):
                x = pd.to_datetime(data_dict[baseline_satellite]['date'])

                y1 = data_dict[baseline_satellite][biome].values * 1e6
                y2 = data_dict[sat][biome].values * 1e6
                y3 = y2 * scale_dicts[sat][biome]

                axes[idx].plot(x, y1, lw = 1.5, color = 'k', label = f'{prefix} {lfs.SATELLITES[baseline_satellite]}')

                axes[idx].plot(x, y2, lw = 1.5, color = 'C0', label = f'{prefix} {lfs.SATELLITES[sat]}')

                axes[idx].plot(x, y3, lw = 1.5, color = 'C1', label = f'{prefix} {lfs.SATELLITES[sat]} scaled')

                panel_letter = f'({chr(97 + idx)})'
                axes[idx].set_title(f"{panel_letter} {prefix} {product} - {biome_map[biome]} {year}")
                axes[idx].set_ylabel(r'$\rho_{FRP}$ (W $\cdot$ km$^{-2}$)')

                # Set major ticks to every month
                axes[idx].xaxis.set_major_locator(mdates.MonthLocator())
                axes[idx].xaxis.set_major_formatter(mdates.DateFormatter('%b') )  # e.g., Jan 2022

                axes[idx].legend(frameon = False, loc = 'upper left', ncol = 2)
                axes[idx].set_xlim(start_date, end_date)

                axes[idx].tick_params(axis='both', which='major')
                axes[idx].tick_params(axis='both', which='minor')

                axes[idx].grid(ls = '-.', color='gray',  alpha = 0.4)

            plt.savefig(f'{figure_dir}/TS_FPR_Density_{product}_{year}.png', dpi = 300)
            plt.close()


