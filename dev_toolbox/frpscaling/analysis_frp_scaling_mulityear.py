import os, copy
import numpy as np
import pandas as pd
import importlib
from datetime import datetime
import lib_frp_scaling as lfs 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
import sys
from io import StringIO

plt.switch_backend('agg')

# Analysis years (calendar years for data/plots)
year_range = (2024, 2025)  # Start and end year (inclusive)
years = list(range(year_range[0], year_range[1] + 1))

# Which satellite streams to process (keys used in your data_dict)
# 'vnp' = SNPP VIIRS, 'vj1' = NOAA-20 VIIRS, 'vj2' = NOAA-21 VIIRS, 'mod' = Terra MODIS
satellites_to_process = ['vnp', 'vj1', 'mod']  # order controls output order

# Reference/baseline satellite against which others are compared
baseline_satellite = 'myd'  # 'myd' = Aqua MODIS

# Data loading behavior
fresh_load = False  # True: force re-read from source; False: use cached/intermediate files if present
plot_analysis = True  # True: generate/show/save plots; False: skip plotting steps

# Output & cache directories
figure_dir = "./v3.2"  # destination for saved figures
cache_dir = "./cache"   # directory for serialized intermediates (e.g., npz/pkl)

# Parallelism for heavy I/O/compute steps
n_workers = 20  # number of worker processes/threads (depends on your implementation)

# Root path to QFED FRP inputs used by the loader functions
lfs.base_path = '/discover/nobackup/projects/gmao/iesa/pub/aerosol/emissions/QFED/.v3.2/0.1/FRP/'

prefix = 'QFED V3.2'

# Optional biome-specific scaling factors (dimensionless) for FRP density
# Keys match columns in data_dict[...] (units of rho_FRP typically W·km^-2 after your 1e6 scaling)
c6scale = {}
c6scale['frp_tf'] = 1  # Tropical Forests
c6scale['frp_xf'] = 1  # Extra-tropical Forests
c6scale['frp_sv'] = 1  # Savanna
c6scale['frp_gl'] = 1  # Grasslands

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
satellites = satellites_to_process + [baseline_satellite]
biome_map = lfs.BIOMES
biomes = lfs.FRP_VARS

os.makedirs(cache_dir, exist_ok=True)
os.makedirs(figure_dir, exist_ok=True)

# Dictionary to store all data by year and satellite
all_data_dict = {}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load data for all years
print("=" * 80)
print("Loading data for all years...")
print("=" * 80)

for year in years:
    all_data_dict[year] = {}
    
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    
    if fresh_load:
        for satellite in satellites:
            print(f"Loading {satellite.upper()} data for {year}...", end=" ", flush=True)
            try:
                # Suppress stderr to hide error messages from parallel processing
                old_stderr = sys.stderr
                sys.stderr = StringIO()
                
                result = lfs.load_time_series_by_biome_parallel([satellite], start_date, end_date, n_workers)
                
                sys.stderr = old_stderr
                
                # Handle case where function returns None
                if result is None:
                    print(f"✗ WARNING: load_time_series_by_biome_parallel returned None")
                    continue
                
                biome_dataframes, biome_names = result
                
                # Handle case where either return value is None
                if biome_dataframes is None or biome_names is None:
                    print(f"✗ WARNING: biome_dataframes or biome_names is None")
                    continue

                # Combine biome dataframes into a single dataframe
                combined_df = pd.DataFrame(index=biome_dataframes[list(biome_dataframes.keys())[0]].index)
                
                for biome, bdf in biome_dataframes.items():
                    if isinstance(bdf, pd.DataFrame):
                        # The DataFrame should have the satellite as a column
                        if satellite in bdf.columns:
                            combined_df[biome] = bdf[satellite].values
                        else:
                            # If satellite not in columns, maybe it's the only column
                            cols = [c for c in bdf.columns if c != 'date']
                            if len(cols) == 1:
                                combined_df[biome] = bdf[cols[0]].values
                            else:
                                print(f"    WARNING: Could not find {satellite} in {biome}. Available: {list(bdf.columns)}")
                                combined_df[biome] = np.nan
                    else:
                        print(f"    WARNING: {biome} value is {type(bdf)}, not a DataFrame")

                # Save to cache
                combined_df.to_csv(f'{cache_dir}/{satellite}_{year}.csv')
                all_data_dict[year][satellite] = combined_df
                print(f"✓ Successfully loaded ({len(combined_df)} days, {len(biomes)} biomes)")
                
            except Exception as e:
                sys.stderr = old_stderr
                print(f"✗ ERROR: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    else:
        for satellite in satellites:
            filename = f'{cache_dir}/{satellite}_{year}.csv'
            print(f"Loading cached {satellite.upper()} data for {year}...", end=" ", flush=True)
            try:
                if os.path.exists(filename):
                    all_data_dict[year][satellite] = pd.read_csv(filename, index_col=0)
                    print(f"✓ Loaded")
                else:
                    print(f"✗ Cache file not found: {filename}")
            except Exception as e:
                print(f"✗ ERROR reading {filename}: {str(e)}")
                continue

print("=" * 80)
print()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Combine all years into single dataframes per satellite
print("Combining all years into single dataset...")
print("=" * 80)

combined_all_data = {}
for satellite in satellites:
    dfs_list = []
    for year in years:
        if year in all_data_dict and satellite in all_data_dict[year]:
            dfs_list.append(all_data_dict[year][satellite])
    
    if dfs_list:
        combined_all_data[satellite] = pd.concat(dfs_list, axis=0)
        print(f"{satellite.upper()}: {combined_all_data[satellite].shape[0]} total days")
    else:
        print(f"{satellite.upper()}: No data found")

print("=" * 80)
print()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Compute single scaling factors across all years
print("Computing scaling factors across all years (2018-2022)...")
print("=" * 80)

all_scale_dicts = {}

# Check if we have baseline data
if baseline_satellite not in combined_all_data:
    print(f"ERROR: Baseline satellite {baseline_satellite.upper()} not found!")
else:
    base_data = copy.deepcopy(combined_all_data[baseline_satellite])
    
    # Ensure index is datetime
    if not isinstance(base_data.index, pd.DatetimeIndex):
        base_data.index = pd.to_datetime(base_data.index)
    
    print(f'\n - Combined period: 2018-2022')
    print(f"{'    biome':<{15}}|{'logc':^{15}}|{'c':^{15}}|{'final c':^{15}}|")
    
    for sat in satellites_to_process:
        
        # Check if satellite data exists
        if sat not in combined_all_data:
            print(f"  Skipping {sat.upper()} - no data found")
            continue
        
        data = copy.deepcopy(combined_all_data[sat])
        
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        scale_dict = {}
        
        print(f' {"- " * 15} Sensor {sat.upper()}  {"- " * 15} ')
        for biome in biomes:
            
            # Check if biome exists in dataframes
            if biome not in data.columns or biome not in base_data.columns:
                print(f" - {biome:<{12}} SKIPPED - column not found")
                scale_dict[biome] = 1.0
                continue
            
            try:
                # Get the data for this biome
                sat_data = data[biome].values
                base_data_vals = base_data[biome].values
                
                # Remove NaN values
                mask = ~(np.isnan(sat_data) | np.isnan(base_data_vals))
                sat_data_clean = sat_data[mask]
                base_data_clean = base_data_vals[mask]
                
                if len(sat_data_clean) == 0:
                    print(f" - {biome:<{12}} NO VALID DATA")
                    scale_dict[biome] = 1.0
                    continue
                
                # Resample to 16-day means
                sat_df = pd.DataFrame({'val': sat_data_clean}, index=data.index[mask])
                base_df = pd.DataFrame({'val': base_data_clean}, index=base_data.index[mask])
                
                sat_16d = np.log(sat_df.resample('16D').mean()['val'].values)
                base_16d = np.log(base_df.resample('16D').mean()['val'].values)
                
                # Remove NaN from resampled data
                mask_16d = ~(np.isnan(sat_16d) | np.isnan(base_16d))
                sat_16d_clean = sat_16d[mask_16d]
                base_16d_clean = base_16d[mask_16d]
                
                if len(sat_16d_clean) == 0:
                    print(f" - {biome:<{12}} NO VALID 16D DATA")
                    scale_dict[biome] = 1.0
                    continue
                
                logc = np.nanmean(base_16d_clean) - np.nanmean(sat_16d_clean)
                scaling_factor = np.exp(logc)
                
                print(f" - {biome:<{12}} "f"{logc:^{15}.6f} "f"{scaling_factor:^{15}.6f} "f"{(scaling_factor*c6scale[biome]):^{15}.6f}")
                scale_dict[biome] = scaling_factor
                
            except Exception as e:
                print(f" - {biome:<{12}} ERROR: {str(e)}")
                scale_dict[biome] = 1.0
                continue
        
        all_scale_dicts[sat] = scale_dict

print("=" * 80)
print()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot scatter plot analysis (combined all years)
print("Creating scatter plot analysis (all years combined)...")
print("=" * 80)

if plot_analysis:
    
    for sel_sat in satellites_to_process:
        
        # Check if satellite data exists
        if sel_sat not in combined_all_data or sel_sat not in all_scale_dicts:
            print(f"  Skipping scatter plot for {sel_sat.upper()}")
            continue
        
        try:
            fig, axes = lfs.multiFigure(2, 2, figsize=(10, 10), gap=0.25)

            xRange = [-10, -5]
            yRange = [-10, -5]

            delta_ticks_X = (xRange[1] - xRange[0]) / 5
            delta_ticks_Y = (yRange[1] - yRange[0]) / 5
            
            for i, biome in enumerate(biomes):
            
                panel_letter = f'({chr(97 + i)})'
                
                # Check if biome data exists
                if biome not in combined_all_data[baseline_satellite].columns or biome not in combined_all_data[sel_sat].columns:
                    axes[i].text(0.5, 0.5, f'Data not available for {biome}', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    continue
                
                # Get raw data
                baseline_vals = combined_all_data[baseline_satellite][biome].values
                sat_vals = combined_all_data[sel_sat][biome].values
                
                # Filter out invalid values (NaN, zero, negative)
                mask = (baseline_vals > 0) & (sat_vals > 0) & ~np.isnan(baseline_vals) & ~np.isnan(sat_vals)
                
                if np.sum(mask) < 10:
                    axes[i].text(0.5, 0.5, f'Insufficient valid data for {biome}\n({np.sum(mask)} points)', 
                               ha='center', va='center', transform=axes[i].transAxes)
                    continue
                
                # Apply mask
                baseline_filtered = baseline_vals[mask]
                sat_filtered = sat_vals[mask]
                
                # Take logarithms
                x = np.log(baseline_filtered)
                y = np.log(sat_filtered)

                axes[i] = lfs.scatter(axes[i], x, y, stdOn=False, one2one_line=True,  
                                  markersize=15, eeOn=False, alpha=1, xRange=xRange, yRange=yRange,
                                  model='RMA', regress_line=False, 
                                  delta_ticks_X=delta_ticks_X, delta_ticks_Y=delta_ticks_Y)

                # Scaled version
                x = np.log(baseline_filtered)
                y = np.log(sat_filtered * all_scale_dicts[sel_sat][biome])
                axes[i] = lfs.scatter(axes[i], x, y, stdOn=False, one2one_line=True, 
                                  markersize=2, eeOn=False, alpha=1, xRange=xRange, yRange=yRange,
                                  delta_ticks_X=delta_ticks_X, delta_ticks_Y=delta_ticks_Y, 
                                  model='RMA', regress_line=False, 
                                  color='crimson', label_p='lower right', case_str='After Scaling:')

                axes[i].set_xlabel(lfs.SATELLITES[baseline_satellite] + ' ln($\\rho_{FRP}$) (MW$\\cdot km^{-2}$)')
                axes[i].set_ylabel(lfs.SATELLITES[sel_sat] + ' ln($\\rho_{FRP}$) (MW$\\cdot km^{-2}$)')

                axes[i].set_title(f"{panel_letter} {prefix} {biome_map[biome]} (2018-2022)\nScaling: {all_scale_dicts[sel_sat][biome]:4.4f}")

            plt.savefig(f'{figure_dir}/SCATTER_Scaling_factor_{sel_sat}_2018-2022.png', dpi=300)
            plt.close()
            print(f"  ✓ Saved scatter plot for {sel_sat.upper()}")
            
        except Exception as e:
            print(f"  ✗ ERROR creating scatter plot for {sel_sat.upper()}: {str(e)}")
            import traceback
            traceback.print_exc()
            plt.close()
            continue

print("=" * 80)
print()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot timeseries analysis (combined all years)
print("Creating time series analysis plots (all years combined)...")
print("=" * 80)

if plot_analysis:
    
    for sat in satellites_to_process:
        
        # Check if satellite data exists
        if sat not in combined_all_data or sat not in all_scale_dicts:
            print(f"  Skipping timeseries plot for {sat.upper()}")
            continue
        
        try:
            product = sat.upper()

            fig, axes = lfs.multiFigure(4, 1, figsize=(9, 12), gap=0.25, left=0.1)

            # Get the date index for plotting
            x = combined_all_data[baseline_satellite].index
            if not isinstance(x, pd.DatetimeIndex):
                x = pd.to_datetime(x)

            for idx, biome in enumerate(biomes):
                
                # Check if biome data exists
                if biome not in combined_all_data[baseline_satellite].columns or biome not in combined_all_data[sat].columns:
                    axes[idx].text(0.5, 0.5, f'Data not available for {biome}', 
                                 ha='center', va='center', transform=axes[idx].transAxes)
                    continue
                
                # Get data with proper filtering
                baseline_vals = combined_all_data[baseline_satellite][biome].values
                sat_vals = combined_all_data[sat][biome].values
                
                # Filter out invalid values for scaling
                mask = (baseline_vals > 0) & (sat_vals > 0) & ~np.isnan(baseline_vals) & ~np.isnan(sat_vals)
                sat_vals_scaled = sat_vals.copy()
                sat_vals_scaled[mask] = sat_vals[mask] * all_scale_dicts[sat][biome]
                
                y1 = baseline_vals * 1e6
                y2 = sat_vals * 1e6
                y3 = sat_vals_scaled * 1e6

                axes[idx].plot(x, y1, lw=1.5, color='k', label=f'{prefix} {lfs.SATELLITES[baseline_satellite]}')
                axes[idx].plot(x, y2, lw=1.5, color='C0', label=f'{prefix} {lfs.SATELLITES[sat]}')
                axes[idx].plot(x, y3, lw=1.5, color='C1', label=f'{prefix} {lfs.SATELLITES[sat]} scaled')

                panel_letter = f'({chr(97 + idx)})'
                axes[idx].set_title(f"{panel_letter} {prefix} {product} - {biome_map[biome]} (2018-2022)")
                axes[idx].set_ylabel(r'$\rho_{FRP}$ (W $\cdot$ km$^{-2}$)')

                # Set major ticks to every 6 months
                axes[idx].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                axes[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

                axes[idx].legend(frameon=False, loc='upper left', ncol=2)
                axes[idx].set_xlim(x.min(), x.max())

                axes[idx].tick_params(axis='both', which='major')
                axes[idx].tick_params(axis='both', which='minor')

                axes[idx].grid(ls='-.', color='gray', alpha=0.4)

            plt.savefig(f'{figure_dir}/TS_FPR_Density_{product}_2018-2022.png', dpi=300)
            plt.close()
            print(f"  ✓ Saved timeseries plot for {sat.upper()}")
            
        except Exception as e:
            print(f"  ✗ ERROR creating timeseries plot for {sat.upper()}: {str(e)}")
            import traceback
            traceback.print_exc()
            plt.close()
            continue

print("=" * 80)
print("\n✓ Analysis complete!")
print("\nGenerated files:")
for sat in satellites_to_process:
    print(f"  - {figure_dir}/SCATTER_Scaling_factor_{sat}_2018-2022.png")
    print(f"  - {figure_dir}/TS_FPR_Density_{sat.upper()}_2018-2022.png")
