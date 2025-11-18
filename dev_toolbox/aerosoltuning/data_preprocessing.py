"""
Data preprocessing module for AOD scaling analysis.
Highly parallelized version to utilize all available processors.
"""

import numpy as np
import xarray as xr
from pathlib import Path
import logging
from datetime import datetime, timedelta
import calendar
import re
from typing import Dict, List, Tuple
import yaml
import concurrent.futures
from functools import partial
import os
import tools.eisf_functions as eisf
from itertools import product
import multiprocessing as mp

class AODDataProcessor:
    """Process AOD data from multiple sources with highly parallel processing."""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.year = config['analysis']['year']
        self.bb_threshold = config['analysis']['biomass_burning_fraction']
        self.model_grid = None
        
    def _initialize_model_grid(self):
        """Initialize the model grid from a sample model file."""
        if self.model_grid is not None:
            return True
            
        # Get the first experiment
        experiments = list(self.config['model']['experiments'].keys())
        if not experiments:
            return False
            
        first_exp = experiments[0]
        
        # Try to get a sample file
        sample_dates = [
            datetime(self.year, 1, 1, 12),
            datetime(self.year, 1, 2, 12),
            datetime(self.year, 1, 3, 12)
        ]
        
        for sample_date in sample_dates:
            file_path = self._get_file_path(first_exp, sample_date)
            
            if Path(file_path).exists():
                try:
                    with xr.open_dataset(file_path) as ds:
                        if 'lat' in ds.coords and 'lon' in ds.coords:
                            self.model_grid = {
                                'lat': ds.lat.values.copy(),
                                'lon': ds.lon.values.copy()
                            }
                        elif 'latitude' in ds.coords and 'longitude' in ds.coords:
                            self.model_grid = {
                                'lat': ds.latitude.values.copy(),
                                'lon': ds.longitude.values.copy()
                            }
                        else:
                            continue
                        return True
                except Exception:
                    continue
        return False
    
    def _regrid_to_model_grid(self, dataset: xr.Dataset) -> xr.Dataset:
        """Regrid dataset to match model grid using xarray interpolation."""
        
        if self.model_grid is None:
            if not self._initialize_model_grid():
                return dataset
        
        # Get original coordinates
        if 'lat' in dataset.coords and 'lon' in dataset.coords:
            orig_lat_name = 'lat'
            orig_lon_name = 'lon'
        elif 'latitude' in dataset.coords and 'longitude' in dataset.coords:
            orig_lat_name = 'latitude'
            orig_lon_name = 'longitude'
        else:
            return dataset
        
        orig_lat = dataset[orig_lat_name].values
        orig_lon = dataset[orig_lon_name].values
        target_lat = self.model_grid['lat']
        target_lon = self.model_grid['lon']
        
        # Check if regridding is needed
        if (len(orig_lat) == len(target_lat) and len(orig_lon) == len(target_lon) and
            np.allclose(orig_lat, target_lat, atol=1e-5) and 
            np.allclose(orig_lon, target_lon, atol=1e-5)):
            if orig_lat_name != 'lat' or orig_lon_name != 'lon':
                dataset = dataset.rename({orig_lat_name: 'lat', orig_lon_name: 'lon'})
            return dataset
        
        # Create regridded dataset
        regridded = xr.Dataset()
        
        # Process each data variable
        for var_name in dataset.data_vars:
            var = dataset[var_name]
            
            if orig_lat_name not in var.dims or orig_lon_name not in var.dims:
                continue
                
            try:
                regridded_var = var.interp(
                    {orig_lat_name: target_lat, orig_lon_name: target_lon},
                    method='linear',
                    kwargs={"fill_value": np.nan}
                )
                
                if orig_lat_name != 'lat':
                    regridded_var = regridded_var.rename({orig_lat_name: 'lat'})
                if orig_lon_name != 'lon':
                    regridded_var = regridded_var.rename({orig_lon_name: 'lon'})
                
                regridded[var_name] = regridded_var
                
            except Exception:
                if len(var.dims) == 2:
                    regridded[var_name] = (['lat', 'lon'], 
                                         np.full((len(target_lat), len(target_lon)), np.nan, dtype=np.float32))
        
        regridded = regridded.assign_coords({
            'lat': ('lat', target_lat),
            'lon': ('lon', target_lon)
        })
        
        regridded.attrs = dataset.attrs.copy()
        return regridded
    
    def _get_file_path(self, experiment: str, date: datetime) -> str:
        """Generate file path for a given experiment and date."""
        if experiment in ['aqua', 'terra']:
            template = self.config['observations'][experiment]['path_template']
        else:
            exp_config = self.config['model']['experiments'][experiment]
            template = exp_config['path_template'].format(
                base_path=self.config['model']['base_path'],
                collection=self.config['model']['collection']
            )
        
        # Replace date placeholders
        path = template.replace('%y4', f"{date.year:04d}")
        path = path.replace('%m2', f"{date.month:02d}")
        path = path.replace('%d2', f"{date.day:02d}")
        path = path.replace('%h2', f"{date.hour:02d}")
        path = path.replace('%h4', f"{date.hour:04d}")
        
        return path

def process_timestep_worker(timestep_info, config):
    """
    Worker function to process a single 3-hourly timestep.
    
    Parameters:
    -----------
    timestep_info : tuple
        (timestep_date, month) tuple
    config : dict
        Configuration dictionary
    """
    
    timestep_date, month = timestep_info
    
    try:
        processor = AODDataProcessor(config)
        
        # Process observations for this timestep
        obs_data = process_observations_timestep(timestep_date, config, processor)
        
        # Process model experiments for this timestep
        model_data = process_model_timestep(timestep_date, config, processor)
        
        if obs_data is None or not model_data:
            return None
            
        # Combine observation and model data
        combined_data = combine_timestep_data(obs_data, model_data)
        
        # Calculate BB fraction
        combined_data = calculate_bb_fraction_3hourly(combined_data)
        
        # Apply observation mask
        combined_data = apply_observation_mask(combined_data)
        
        # Check if timestep meets BB threshold
        if timestep_meets_bb_threshold(combined_data, config['analysis']['biomass_burning_fraction']):
            # Apply final combined mask before returning
            obs_mask = np.isfinite(combined_data['OBS_totexttau'])
            bb_mask = combined_data['f_bb'] >= config['analysis']['biomass_burning_fraction']
            combined_mask = obs_mask & bb_mask
            
            # Apply mask to all variables
            for var_name in combined_data.data_vars:
                combined_data[var_name] = combined_data[var_name].where(combined_mask)
            
            # Add metadata for aggregation
            combined_data.attrs['timestep_date'] = timestep_date.isoformat()
            combined_data.attrs['month'] = month
            
            return combined_data
        else:
            return None
            
    except Exception as e:
        # Log error but don't crash the worker
        print(f"Error processing timestep {timestep_date}: {e}")
        return None

def process_observations_timestep(timestep_date, config, processor):
    """Process observation data for a single timestep."""
    
    obs_sources = list(config['observations'].keys())
    obs_datasets = {}
    
    for obs_source in obs_sources:
        obs_config = config['observations'][obs_source]
        file_path = processor._get_file_path(obs_source, timestep_date)
        
        if Path(file_path).exists():
            try:
                with xr.open_dataset(file_path) as ds:
                    aod_var = obs_config.get('variable_name', 'tau')
                    if aod_var in ds:
                        obs_ds = xr.Dataset()
                        obs_ds[aod_var] = ds[aod_var].squeeze()
                        
                        for coord_name in ['lat', 'lon', 'latitude', 'longitude']:
                            if coord_name in ds.coords:
                                obs_ds[coord_name] = ds[coord_name]
                        
                        regridded_obs = processor._regrid_to_model_grid(obs_ds)
                        
                        if aod_var in regridded_obs:
                            obs_datasets[obs_source] = regridded_obs[aod_var]
                            
            except Exception:
                pass
    
    if obs_datasets:
        # Create dataset with individual sources
        result_ds = xr.Dataset()
        
        # Add coordinates from first dataset
        first_data = list(obs_datasets.values())[0]
        result_ds = result_ds.assign_coords({
            'lat': first_data.lat,
            'lon': first_data.lon
        })
        
        # Add individual source data
        for source, data in obs_datasets.items():
            result_ds[f'{source}_totexttau'] = data
        
        # Combine using average method
        obs_list = list(obs_datasets.values())
        obs_stack = xr.concat(obs_list, dim='obs_source')
        combined = obs_stack.mean(dim='obs_source', skipna=True)
        result_ds['OBS_totexttau'] = combined
        
        return result_ds
    else:
        return None

def process_model_timestep(timestep_date, config, processor):
    """Process model data for a single timestep."""
    
    experiments = list(config['model']['experiments'].keys())
    model_data = {}
    
    for experiment in experiments:
        try:
            hourly_data = []
            
            # Average 3 hours of hourly data
            base_hour = timestep_date.hour
            hours_to_average = [base_hour-1, base_hour, base_hour+1]
            
            for hour_offset in hours_to_average:
                try:
                    if hour_offset < 0:
                        file_date = timestep_date.replace(hour=23) - timedelta(days=1)
                    elif hour_offset >= 24:
                        file_date = timestep_date.replace(hour=hour_offset-24) + timedelta(days=1)
                    else:
                        file_date = timestep_date.replace(hour=hour_offset)
                    
                    file_path = processor._get_file_path(experiment, file_date)
                    
                    if Path(file_path).exists():
                        with xr.open_dataset(file_path) as ds:
                            aod_var = config['model']['variable_name']
                            if aod_var in ds:
                                model_ds = xr.Dataset()
                                model_ds[aod_var] = ds[aod_var].squeeze()
                                
                                for coord_name in ['lat', 'lon', 'latitude', 'longitude']:
                                    if coord_name in ds.coords:
                                        model_ds[coord_name] = ds[coord_name]
                                
                                regridded_model = processor._regrid_to_model_grid(model_ds)
                                
                                if aod_var in regridded_model:
                                    hourly_data.append(regridded_model[aod_var])
                                    
                except Exception:
                    pass
            
            if hourly_data:
                avg_data = xr.concat(hourly_data, dim='time').mean(dim='time')
                model_data[f'{experiment}_totexttau'] = avg_data
                
        except Exception:
            pass
    
    return model_data

def combine_timestep_data(obs_data, model_data):
    """Combine observation and model data for a timestep."""
    combined = obs_data.copy()
    
    for var_name, data_array in model_data.items():
        combined[var_name] = data_array
    
    return combined

def calculate_bb_fraction_3hourly(timestep_data):
    """Calculate biomass burning fraction."""
    
    if 'allviirs_totexttau' in timestep_data and 'noBB_totexttau' in timestep_data:
        allviirs = timestep_data['allviirs_totexttau']
        nobb = timestep_data['noBB_totexttau']
        
        bb_fraction = xr.where(
            allviirs > 0.01,
            (allviirs - nobb) / allviirs,
            0.0
        )
        
        bb_fraction = xr.where(bb_fraction < 0, 0, bb_fraction)
        bb_fraction = xr.where(bb_fraction > 1, 1, bb_fraction)
        
        timestep_data['f_bb'] = bb_fraction
    else:
        if len(timestep_data.data_vars) > 0:
            template = list(timestep_data.data_vars.values())[0]
            timestep_data['f_bb'] = xr.zeros_like(template)
    
    return timestep_data

def apply_observation_mask(timestep_data):
    """Apply observation mask."""
    
    if 'OBS_totexttau' not in timestep_data:
        return timestep_data
    
    obs_mask = np.isfinite(timestep_data['OBS_totexttau'])
    masked_data = timestep_data.copy()
    
    for var_name in timestep_data.data_vars:
        if var_name not in ['aqua_totexttau', 'terra_totexttau'] and 'totexttau' in var_name:
            masked_data[var_name] = timestep_data[var_name].where(obs_mask)
        elif var_name == 'f_bb':
            masked_data[var_name] = timestep_data[var_name].where(obs_mask)
    
    return masked_data

def timestep_meets_bb_threshold(timestep_data, bb_threshold):
    """Check if timestep meets BB threshold."""
    
    if 'f_bb' not in timestep_data or 'OBS_totexttau' not in timestep_data:
        return False
    
    obs_mask = np.isfinite(timestep_data['OBS_totexttau'])
    obs_points = np.sum(obs_mask.values)
    
    if obs_points == 0:
        return False
    
    bb_mask = timestep_data['f_bb'] >= bb_threshold
    combined_mask = obs_mask & bb_mask
    qualifying_points = np.sum(combined_mask.values)
    
    return qualifying_points > 0

def aggregate_timesteps_to_monthly(timestep_results, month):
    """Aggregate timestep results into monthly data."""
    
    # Filter results for this month and remove None values
    month_timesteps = [result for result in timestep_results 
                      if result is not None and result.attrs.get('month') == month]
    
    if not month_timesteps:
        return None
    
    print(f"Aggregating {len(month_timesteps)} timesteps for month {month}")
    
    # Concatenate and average
    combined = xr.concat(month_timesteps, dim='timestep')
    monthly_avg = combined.mean(dim='timestep', skipna=True)
    
    return monthly_avg

def create_grads_compatible_file(dataset, output_file):
    """Create a simple GrADS-compatible NetCDF file."""
    
    # Get coordinates, handling both possible names
    if 'lat' in dataset.coords:
        lats = dataset.lat.values
        lons = dataset.lon.values
    elif 'latitude' in dataset.coords:
        lats = dataset.latitude.values  
        lons = dataset.longitude.values
    else:
        raise ValueError("No latitude coordinate found")
    
    # Ensure proper coordinate order
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        flip_lat = True
    else:
        flip_lat = False
    
    # Ensure longitude is -180 to 180
    if lons.max() > 180:
        lons = ((lons + 180) % 360) - 180
        lon_sort_idx = np.argsort(lons)
        lons = lons[lon_sort_idx]
        sort_lon = True
    else:
        sort_lon = False
        lon_sort_idx = None
    
    # Create time coordinate from month dimension
    if 'month' in dataset.coords:
        months = dataset.month.values
        times = np.arange(len(months), dtype=np.float64)
    else:
        times = np.array([0.0])
    
    # Create new dataset with standard coordinates
    new_ds = xr.Dataset()
    new_ds = new_ds.assign_coords({
        'time': ('time', times),
        'lat': ('lat', lats),
        'lon': ('lon', lons)
    })
    
    # Add coordinate attributes
    new_ds.time.attrs = {
        'long_name': 'time', 
        'units': 'days since 2024-01-01 00:00:00',
        'calendar': 'standard'
    }
    new_ds.lat.attrs = {
        'long_name': 'latitude', 
        'units': 'degrees_north',
        'axis': 'Y'
    }
    new_ds.lon.attrs = {
        'long_name': 'longitude', 
        'units': 'degrees_east',
        'axis': 'X'
    }
    
    # Process all data variables
    for var_name in dataset.data_vars:
        if 'totexttau' in var_name or var_name == 'f_bb':
            
            data = dataset[var_name].values
            original_dims = dataset[var_name].dims
            
            # Handle coordinate transformations
            if flip_lat and 'lat' in original_dims:
                lat_axis = original_dims.index('lat')
                data = np.flip(data, axis=lat_axis)
            
            if sort_lon and 'lon' in original_dims:
                lon_axis = original_dims.index('lon')
                data = np.take(data, lon_sort_idx, axis=lon_axis)
            
            # Reshape to standard (time, lat, lon) order
            if 'month' in original_dims:
                # 3D data - reorder dimensions
                if original_dims == ('month', 'lat', 'lon'):
                    pass  # Already correct
                elif original_dims == ('lat', 'lon', 'month'):
                    data = data.transpose(2, 0, 1)
                elif original_dims == ('lon', 'lat', 'month'):
                    data = data.transpose(2, 1, 0)
                
                target_dims = ('time', 'lat', 'lon')
            else:
                # 2D data - add time dimension
                data = data[np.newaxis, :, :]
                target_dims = ('time', 'lat', 'lon')
            
            # Add to new dataset
            new_ds[var_name] = (target_dims, data.astype(np.float32))
            
            # Add variable attributes
            if 'totexttau' in var_name:
                experiment = var_name.replace('_totexttau', '')
                new_ds[var_name].attrs = {
                    'long_name': f'Aerosol Optical Depth {experiment}',
                    'units': '1'
                }
            elif var_name == 'f_bb':
                new_ds[var_name].attrs = {
                    'long_name': 'Biomass Burning Fraction',
                    'units': '1'
                }
    
    # Add global attributes
    new_ds.attrs = {
        'title': 'AOD Analysis Data - GrADS Compatible',
        'institution': 'NASA GSFC',
        'conventions': 'CF-1.6',
        'history': f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    }
    
    # Save with simple encoding
    encoding = {}
    for var in new_ds.data_vars:
        encoding[var] = {
            'dtype': 'float32', 
            '_FillValue': -999.0,
            'zlib': True,
            'complevel': 1
        }
    
    # Coordinate encoding
    encoding['time'] = {'dtype': 'float64'}
    encoding['lat'] = {'dtype': 'float32'}
    encoding['lon'] = {'dtype': 'float32'}
    
    new_ds.to_netcdf(output_file, format='NETCDF4_CLASSIC', encoding=encoding)
    
    print(f"Created GrADS-compatible file: {output_file}")
    print(f"Time dimension: {len(times)} points")
    print(f"Lat dimension: {len(lats)} points ({lats.min():.3f} to {lats.max():.3f})")
    print(f"Lon dimension: {len(lons)} points ({lons.min():.3f} to {lons.max():.3f})")
    print(f"Variables: {list(new_ds.data_vars.keys())}")
    
    return output_file

def _generate_filename_components(config):
    """Generate filename components from config."""
    
    # Extract observation identifiers
    obs_sources = list(config['observations'].keys())
    obs_ids = []
    for source in obs_sources:
        if 'MYD04' in config['observations'][source]['path_template']:
            obs_ids.append('MYD')
        elif 'MOD04' in config['observations'][source]['path_template']:
            obs_ids.append('MOD')
        else:
            obs_ids.append(source.upper())
    obs_string = '_'.join(sorted(obs_ids))
    
    # Extract experiment basename
    experiment_basename = "c180R_qfed3igbp"  # Updated default
    if config['model']['experiments']:
        # Extract from first experiment path
        first_exp = list(config['model']['experiments'].values())[0]
        path_template = first_exp['path_template']
        # Look for pattern like "c180R_qfed3igbp_"
        match = re.search(r'(c\d+R?[^/]*qfed[^/]*?)_', path_template)
        if match:
            experiment_basename = match.group(1)
    
    return obs_string, experiment_basename

def _generate_output_filename(config, months_to_process, is_monthly=False, month=None):
    """Generate output filename based on config settings."""
    
    year = config['analysis']['year']
    bb_threshold = config['analysis']['biomass_burning_fraction']
    obs_string, experiment_basename = _generate_filename_components(config)
    
    # Check if custom filenames are specified in config
    output_config = config.get('output', {})
    
    if is_monthly and month is not None:
        # Monthly filename
        if 'monthly_filename_template' in output_config:
            template = output_config['monthly_filename_template']
            return template.format(
                experiment_basename=experiment_basename,
                obs_string=obs_string,
                year=year,
                month=month,
                bb_threshold=bb_threshold
            )
    else:
        # Annual filename
        if 'annual_filename' in output_config:
            return output_config['annual_filename']
    
    # Use template from config if available
    if 'filename_template' in output_config:
        template = output_config['filename_template']
        
        # Determine time period string
        if len(months_to_process) == 12 and months_to_process == list(range(1, 13)):
            time_period = "annual"
        else:
            months_str = "_".join([f"{m:02d}" for m in sorted(months_to_process)])
            time_period = f"months{months_str}"
        
        filename = template.format(
            experiment_basename=experiment_basename,
            obs_string=obs_string,
            time_period=time_period,
            bb_threshold=bb_threshold,
            year=year
        )
        
        return filename
    
    # Fallback to original naming scheme
    if len(months_to_process) == 12 and months_to_process == list(range(1, 13)):
        return f"{experiment_basename}_{obs_string}_annual_bb{bb_threshold}_{year}"
    else:
        months_str = "_".join([f"{m:02d}" for m in sorted(months_to_process)])
        return f"{experiment_basename}_{obs_string}_months{months_str}_bb{bb_threshold}_{year}"

def debug_grid_consistency(filename):
    """Debug function to check grid consistency in processed file."""
    
    print(f"Debugging grid consistency for: {filename}")
    
    try:
        with xr.open_dataset(filename) as ds:
            print(f"Dataset dimensions: {dict(ds.sizes)}")
            print(f"Coordinates: {list(ds.coords.keys())}")
            
            # Check each variable's valid data coverage
            for var_name in ds.data_vars:
                if 'totexttau' in var_name or var_name == 'f_bb':
                    var_data = ds[var_name]
                    
                    if 'month' in var_data.dims:
                        # Check coverage for each month
                        for month_idx in range(var_data.sizes['month']):
                            month_data = var_data.isel(month=month_idx)
                            valid_count = np.sum(np.isfinite(month_data.values))
                            total_count = month_data.size
                            month_val = ds.month.values[month_idx]
                            
                            print(f"{var_name} month {month_val}: {valid_count}/{total_count} valid points ({100*valid_count/total_count:.1f}%)")
                    else:
                        # 2D variable
                        valid_count = np.sum(np.isfinite(var_data.values))
                        total_count = var_data.size
                        print(f"{var_name}: {valid_count}/{total_count} valid points ({100*valid_count/total_count:.1f}%)")
                        
    except Exception as e:
        print(f"Error debugging file: {e}")

def process_full_year_parallel(config_path: str = "config.yaml", max_workers: int = None):
    """
    Process data with maximum parallelization at the timestep level.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    max_workers : int, optional
        Maximum number of workers (defaults to CPU count)
    """
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('preprocessing_parallel.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    output_dir = Path(config['output']['base_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    year = config['analysis']['year']
    bb_threshold = config['analysis']['biomass_burning_fraction']
    months_to_process = config.get('analysis', {}).get('months_to_process', list(range(1, 13)))
    
    # Use all available processors if not specified
    if max_workers is None:
        max_workers = min(48, os.cpu_count())  # Cap at 48 as requested
    
    logger.info(f"Starting highly parallel preprocessing for year {year}")
    logger.info(f"BB fraction threshold: {bb_threshold}")
    logger.info(f"Months to process: {sorted(months_to_process)}")
    logger.info(f"Using {max_workers} parallel workers")
    
    # Generate all timestep combinations
    all_timesteps = []
    for month in months_to_process:
        days_in_month = calendar.monthrange(year, month)[1]
        for day in range(1, days_in_month + 1):
            for hour in range(0, 24, 3):  # 3-hourly
                timestep_date = datetime(year, month, day, hour)
                all_timesteps.append((timestep_date, month))
    
    logger.info(f"Processing {len(all_timesteps)} total 3-hourly timesteps")
    
    # Process all timesteps in parallel
    timestep_results = []
    
    print(f"Starting parallel processing of {len(all_timesteps)} timesteps...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all timesteps for processing
        future_to_timestep = {
            executor.submit(process_timestep_worker, timestep_info, config): timestep_info 
            for timestep_info in all_timesteps
        }
        
        # Collect results with progress tracking
        completed = 0
        for future in concurrent.futures.as_completed(future_to_timestep):
            timestep_info = future_to_timestep[future]
            timestep_date, month = timestep_info
            
            try:
                result = future.result()
                timestep_results.append(result)
                
                completed += 1
                if completed % 100 == 0:  # Progress every 100 timesteps
                    logger.info(f"Completed {completed}/{len(all_timesteps)} timesteps ({100*completed/len(all_timesteps):.1f}%)")
                    
            except Exception as e:
                logger.error(f"Exception processing timestep {timestep_date}: {e}")
    
    logger.info(f"Completed processing all timesteps")
    
    # Count successful timesteps by month
    successful_by_month = {}
    for result in timestep_results:
        if result is not None:
            month = result.attrs['month']
            successful_by_month[month] = successful_by_month.get(month, 0) + 1
    
    for month in months_to_process:
        count = successful_by_month.get(month, 0)
        total_for_month = sum(1 for ts, m in all_timesteps if m == month)
        logger.info(f"Month {month}: {count}/{total_for_month} successful timesteps")
    
    # Aggregate timesteps into monthly data
    monthly_datasets = []
    
    for month in months_to_process:
        logger.info(f"Aggregating timesteps for month {month}")
        
        monthly_data = aggregate_timesteps_to_monthly(timestep_results, month)
        
        if monthly_data is not None:
            # Apply biome combinations and quality control
            if 'combine_biomes' in config and config['combine_biomes']:
                monthly_data = eisf.combine_biomes(monthly_data, config['combine_biomes'])
            
            # Apply quality control
            qc_config = config['quality_control']
            for var_name in monthly_data.data_vars:
                if 'totexttau' in var_name:
                    original_data = monthly_data[var_name]
                    filtered_data = xr.where(
                        (original_data >= qc_config['min_aod_threshold']) & 
                        (original_data <= qc_config['max_aod_threshold']),
                        original_data,
                        np.nan
                    )
                    monthly_data[var_name] = filtered_data
            
            # Add month coordinate
            monthly_data = monthly_data.expand_dims('month')
            monthly_data = monthly_data.assign_coords(month=[month])
            monthly_datasets.append(monthly_data)
            
            logger.info(f"Month {month} aggregation complete")
        else:
            logger.warning(f"No data for month {month}")
    
    # Combine all months and save
    if monthly_datasets:
        monthly_datasets.sort(key=lambda ds: int(ds.month.values[0]))
        
        logger.info("Combining all monthly datasets")
        annual_data = xr.concat(monthly_datasets, dim='month')
        
        # Generate output filename
        annual_filename = _generate_output_filename(config, months_to_process)
        
        # Save files
        original_nc_file = output_dir / f"{annual_filename}.nc"
        annual_data.to_netcdf(original_nc_file)
        logger.info(f"Saved original NetCDF format: {original_nc_file}")
        
        # Save GrADS version
        grads_filename = f"{annual_filename}_grads.nc"
        grads_file = output_dir / grads_filename
        
        try:
            create_grads_compatible_file(annual_data, grads_file)
            logger.info(f"Created GrADS-compatible file: {grads_file}")
            
            # Print GrADS usage instructions
            logger.info("\n" + "="*60)
            logger.info("GrADS USAGE INSTRUCTIONS:")
            logger.info("="*60)
            logger.info(f"To open in GrADS:")
            logger.info(f"  ga-> sdfopen {grads_filename}")
            logger.info(f"  ga-> q file")
            logger.info(f"  ga-> q dims")
            logger.info(f"")
            logger.info(f"Available variables:")
            
            # List available variables
            for var_name in annual_data.data_vars:
                if 'totexttau' in var_name or var_name == 'f_bb':
                    logger.info(f"  - {var_name}")
            
            logger.info(f"")
            logger.info(f"Example GrADS commands:")
            if len(months_to_process) > 1:
                logger.info(f"  ga-> set t 1")
                logger.info(f"  ga-> d f_bb")
                logger.info(f"  ga-> set t 1 {len(months_to_process)}")
                logger.info(f"  ga-> d ave(f_bb,t=1,t={len(months_to_process)})")
                logger.info(f"  ga-> d ave(allviirs_totexttau-OBS_totexttau,t=1,t={len(months_to_process)})")
            else:
                logger.info(f"  ga-> d f_bb")
                logger.info(f"  ga-> d allviirs_totexttau-OBS_totexttau")
            
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Failed to create GrADS-compatible file: {e}")
            import traceback
            traceback.print_exc()
            grads_file = None
        
        # Print summary statistics
        logger.info("Processing complete! Summary statistics:")
        for var_name in annual_data.data_vars:
            if 'totexttau' in var_name or var_name == 'f_bb':
                data_array = annual_data[var_name]
                valid_data = data_array.values[np.isfinite(data_array.values)]
                if len(valid_data) > 0:
                    logger.info(f"{var_name}: mean={np.mean(valid_data):.4f}, "
                              f"std={np.std(valid_data):.4f}, "
                              f"valid_points={len(valid_data)}")
        
        # Additional diagnostics for grid consistency
        logger.info("\n" + "="*60)
        logger.info("GRID DIAGNOSTICS:")
        logger.info("="*60)
        logger.info(f"Final grid dimensions: {dict(annual_data.sizes)}")
        if 'lat' in annual_data.coords and 'lon' in annual_data.coords:
            logger.info(f"Latitude range: {annual_data.lat.values.min():.6f} to {annual_data.lat.values.max():.6f}")
            logger.info(f"Longitude range: {annual_data.lon.values.min():.6f} to {annual_data.lon.values.max():.6f}")
            
            # Check for uniform grid spacing
            lat_diff = np.diff(annual_data.lat.values)
            lon_diff = np.diff(annual_data.lon.values)
            logger.info(f"Latitude spacing: {lat_diff.min():.6f} to {lat_diff.max():.6f} (uniform: {np.allclose(lat_diff, lat_diff[0], rtol=1e-6)})")
            logger.info(f"Longitude spacing: {lon_diff.min():.6f} to {lon_diff.max():.6f} (uniform: {np.allclose(lon_diff, lon_diff[0], rtol=1e-6)})")
        logger.info("="*60)
        
        return original_nc_file
    else:
        logger.error("No monthly data was successfully processed!")
        return None

if __name__ == "__main__":
    # Run the highly parallel version
    output_file = process_full_year_parallel(max_workers=48)
    
    if output_file:
        debug_grid_consistency(output_file)
