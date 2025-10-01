import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import hashlib

warnings.filterwarnings('ignore')

@dataclass
class ScalingResult:
    """Data class to store scaling results"""
    scaling_factor: float
    log_scaling_factor: float  # Added log scaling factor
    r_squared: float
    correlation: float
    rmse: float
    n_points: int
    scaled_data: pd.Series
    original_data: pd.Series
    baseline_data: pd.Series

class FRPScaler:
    """Fire Radiative Power density scaling between satellites"""
    
    # Class constants
    SATELLITES = {
        'mod': 'Terra MODIS',
        'myd': 'Aqua MODIS', 
        'vnp': 'SNPP VIIRS',
        'vj1': 'NOAA-20 VIIRS',
        'vj2': 'NOAA-21 VIIRS'
    }
    
    BIOMES = {
        'frp_tf': 'Tropical Forests',
        'frp_xf': 'Extra-tropical Forests', 
        'frp_sv': 'Savanna',
        'frp_gl': 'Grasslands'
    }
    
    FRP_VARS = list(BIOMES.keys())
    EARTH_RADIUS_KM = 6371.0
    MW_TO_W_CONVERSION = 1_000_000

    def __init__(self, base_path: str = "/discover/nobackup/acollow/geos_aerosols/acollow/QFED/FRP"):
        """Initialize FRP Scaler"""
        self.base_path = base_path
        self.area_weights = None

    def _validate_satellites(self, satellites: List[str]) -> Tuple[List[str], List[str]]:
        """Validate satellite codes"""
        valid = [sat for sat in satellites if sat in self.SATELLITES]
        invalid = [sat for sat in satellites if sat not in self.SATELLITES]
        return valid, invalid

    def _calculate_area_weights(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Calculate area weights for each grid cell accounting for latitude"""
        if self.area_weights is not None:
            return self.area_weights
            
        dlat = abs(lat[1] - lat[0])
        dlon = abs(lon[1] - lon[0])
        
        lat_rad = np.radians(lat)
        dlat_rad = np.radians(dlat)
        dlon_rad = np.radians(dlon)
        
        lat_upper = lat_rad + dlat_rad/2
        lat_lower = lat_rad - dlat_rad/2
        area_per_lat = self.EARTH_RADIUS_KM**2 * dlon_rad * (np.sin(lat_upper) - np.sin(lat_lower))
        
        self.area_weights = np.tile(area_per_lat[:, np.newaxis], (1, len(lon)))
        return self.area_weights

    def _get_file_path(self, satellite: str, date: datetime) -> str:
        """Generate file path for given satellite and date"""
        year, month, day = date.strftime('%Y'), date.strftime('%m'), date.strftime('%d')
        
        # Handle special naming for MODIS satellites
        filename_codes = {'myd': 'MYD14', 'mod': 'MOD14'}
        filename_code = filename_codes.get(satellite, satellite)
        filename = f"qfed3_2.frp.{filename_code}.{year}{month}{day}.nc4"
        
        return os.path.join(self.base_path, f"Y{year}", f"M{month}", filename)

    def _load_daily_data(self, satellite: str, date: datetime) -> Optional[Dict]:
        """Optimized daily data loading"""
        filepath = self._get_file_path(satellite, date)
        
        if not os.path.exists(filepath):
            return None
            
        try:
            # Use chunks and only load what we need
            with xr.open_dataset(filepath, chunks={'time': 1}) as ds:
                # Only load required variables
                required_vars = ['lat', 'lon', 'land'] + self.FRP_VARS
                available_vars = [var for var in required_vars if var in ds.variables]
                
                ds_subset = ds[available_vars]
                
                lat, lon = ds_subset['lat'].values, ds_subset['lon'].values
                
                if self.area_weights is None:
                    self._calculate_area_weights(lat, lon)
                
                frp_density = {'lat': lat, 'lon': lon}
                
                if 'land' in ds_subset.variables:
                    land = ds_subset['land'].values[0]
                    
                    for frp_var in self.FRP_VARS:
                        if frp_var in ds_subset.variables:
                            frp = ds_subset[frp_var].values[0]
                            
                            valid_mask = (frp < 1e19) & (land < 1e19) & (land > 0)
                            density = np.full_like(frp, np.nan)
                            
                            if np.any(valid_mask):
                                density[valid_mask] = frp[valid_mask] / land[valid_mask]
                            
                            frp_density[frp_var] = density
                
                return frp_density
                
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None

    def _calculate_biome_global_mean(self, frp_density_dict: Optional[Dict], biome: str) -> float:
        """Calculate area-weighted global mean FRP density for a specific biome"""
        if not frp_density_dict or biome not in frp_density_dict:
            return np.nan
        
        density = frp_density_dict[biome]
        if density is None:
            return np.nan
        
        valid_mask = ~np.isnan(density)
        if not np.any(valid_mask):
            return np.nan
        
        weighted_sum = np.sum(density[valid_mask] * self.area_weights[valid_mask])
        total_area = np.sum(self.area_weights[valid_mask])
        
        # Convert MW/km² to W/km²
        return (weighted_sum / total_area) * self.MW_TO_W_CONVERSION

    def _load_single_day(self, args):
        """Helper function for parallel loading"""
        date, satellites = args
        daily_results = {}
        
        for sat in satellites:
            frp_density = self._load_daily_data(sat, date)
            daily_results[sat] = {}
            
            for biome in self.FRP_VARS:
                global_mean = self._calculate_biome_global_mean(frp_density, biome)
                daily_results[sat][biome] = global_mean
        
        return date, daily_results

    def _get_cache_key(self, satellites, start_date, end_date):
        """Generate cache key for data loading"""
        key_str = f"{satellites}_{start_date}_{end_date}_{self.base_path}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def load_time_series_by_biome_parallel(self, satellites: List[str], start_date: datetime, 
                                         end_date: datetime, n_workers: int = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """Load time series data using parallel processing"""
        valid_satellites, invalid_satellites = self._validate_satellites(satellites)
        
        if invalid_satellites:
            print(f"Warning: Invalid satellite codes: {invalid_satellites}")
        
        if not valid_satellites:
            print("Error: No valid satellites specified!")
            return {}, {}
        
        print(f"Processing satellites: {[f'{sat} ({self.SATELLITES[sat]})' for sat in valid_satellites]}")
        
        date_range = pd.date_range(start_date, end_date, freq='D')
        
        # Use all available cores minus 1 by default
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)
        
        print(f"Using {n_workers} parallel workers to process {len(date_range)} days...")
        
        # Prepare arguments for parallel processing
        args_list = [(date, valid_satellites) for date in date_range]
        
        # Initialize data structure
        biome_data = {biome: {sat: [np.nan] * len(date_range) for sat in valid_satellites} for biome in self.FRP_VARS}
        
        # Process in parallel
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit all jobs
            future_to_idx = {executor.submit(self._load_single_day, args): idx for idx, args in enumerate(args_list)}
            
            # Collect results
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    date, daily_results = future.result()
                    
                    # Store results
                    for sat in valid_satellites:
                        for biome in self.FRP_VARS:
                            if sat in daily_results:
                                biome_data[biome][sat][idx] = daily_results[sat].get(biome, np.nan)
                    
                    completed += 1
                    if completed % 50 == 0 or completed == len(date_range):
                        print(f"  Completed {completed}/{len(date_range)} days...")
                        
                except Exception as e:
                    print(f"Error processing day {idx}: {e}")
        
        # Convert to DataFrames
        biome_dataframes = {}
        for biome in self.FRP_VARS:
            df = pd.DataFrame(biome_data[biome], index=date_range)
            biome_dataframes[biome] = df
            
            print(f"\n{self.BIOMES[biome]} data loading summary:")
            print(f"  DataFrame shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
        
        return biome_dataframes, self.BIOMES

    def load_time_series_by_biome_cached(self, satellites: List[str], start_date: datetime, 
                                       end_date: datetime, cache_dir: str = "./cache", 
                                       n_workers: int = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """Load with caching support"""
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        cache_key = self._get_cache_key(satellites, start_date, end_date)
        cache_file = os.path.join(cache_dir, f"frp_data_{cache_key}.pkl")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            print("Loading data from cache...")
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                print("Successfully loaded from cache!")
                return cached_data['biome_dataframes'], cached_data['biome_names']
            except Exception as e:
                print(f"Cache loading failed: {e}, loading fresh data...")
        
        # Load fresh data (use parallel version)
        biome_dataframes, biome_names = self.load_time_series_by_biome_parallel(satellites, start_date, end_date, n_workers)
        
        # Save to cache
        try:
            cache_data = {
                'biome_dataframes': biome_dataframes,
                'biome_names': biome_names,
                'satellites': satellites,
                'start_date': start_date,
                'end_date': end_date
            }
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Data saved to cache: {cache_file}")
        except Exception as e:
            print(f"Cache saving failed: {e}")
        
        return biome_dataframes, biome_names

    def load_time_series_by_biome(self, satellites: List[str], start_date: datetime, 
                                  end_date: datetime) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """Original sequential loading method (kept for compatibility)"""
        return self.load_time_series_by_biome_cached(satellites, start_date, end_date)

    def perform_geometric_scaling(self, df: pd.DataFrame, baseline_satellite: str, 
                                 target_satellites: Optional[List[str]] = None) -> Dict[str, ScalingResult]:
        """Perform geometric mean scaling between satellites
        
        This method scales target satellite data to match the baseline satellite using
        log-space calculations: log(c) = mean(log(baseline)) - mean(log(target))
        
        Parameters:
        df: DataFrame with satellite data
        baseline_satellite: str, baseline satellite designation
        target_satellites: list, satellites to scale (if None, uses all except baseline)
        
        Returns:
        Dictionary of scaling results keyed by satellite code
        """
        if baseline_satellite not in df.columns:
            print(f"Error: Baseline satellite '{baseline_satellite}' not found in data.")
            print(f"Available satellites: {list(df.columns)}")
            return {}
        
        if target_satellites is None:
            target_satellites = [sat for sat in df.columns if sat != baseline_satellite]
        
        valid_targets = [sat for sat in target_satellites 
                        if sat in df.columns and sat != baseline_satellite]
        
        if not valid_targets:
            print("Error: No valid target satellites found!")
            return {}
        
        print(f"Baseline: {baseline_satellite} ({self.SATELLITES[baseline_satellite]})")
        print(f"Targets: {[f'{sat} ({self.SATELLITES[sat]})' for sat in valid_targets]}")
        print("Using log-space geometric mean scaling: log(c) = mean(log(baseline)) - mean(log(target))")
        
        results = {}
        
        for target_sat in valid_targets:
            try:
                # Get data and handle missing values
                baseline_data = df[baseline_satellite]
                target_data = df[target_sat]
                
                # Filter to positive values for log calculations
                pos_mask = (baseline_data > 0) & (target_data > 0)
                baseline_pos = baseline_data[pos_mask]
                target_pos = target_data[pos_mask]
                
                if len(baseline_pos) < 10:
                    print(f"Warning: Insufficient positive data for {baseline_satellite} vs {target_sat}")
                    continue
                
                # Calculate means in log space
                log_mean_baseline = np.mean(np.log(baseline_pos))
                log_mean_target = np.mean(np.log(target_pos))
                
                # Calculate log scaling factor: log(c) = mean(log(baseline)) - mean(log(target))
                log_scaling_factor = log_mean_baseline - log_mean_target
                
                # Convert to linear scaling factor: c = exp(log(c))
                scaling_factor = np.exp(log_scaling_factor)
                
                # Verification (these should be equivalent to the above):
                geo_mean_baseline = np.exp(log_mean_baseline)
                geo_mean_target = np.exp(log_mean_target)
                scaling_factor_verification = geo_mean_baseline / geo_mean_target
                
                # Apply scaling to all target data
                scaled_data = target_data * scaling_factor
                
                # Calculate statistics
                common_mask = np.isfinite(baseline_data) & np.isfinite(scaled_data)
                
                if np.sum(common_mask) < 10:
                    print(f"    Warning: Few valid points for {target_sat} statistics")
                    correlation = np.nan
                    rmse = np.nan
                    r_squared = np.nan
                else:
                    baseline_vals = baseline_data[common_mask].values
                    scaled_vals = scaled_data[common_mask].values
                    
                    # Remove any remaining NaN/Inf
                    final_mask = np.isfinite(baseline_vals) & np.isfinite(scaled_vals)
                    baseline_vals = baseline_vals[final_mask]
                    scaled_vals = scaled_vals[final_mask]
                    
                    print(f"    After cleaning: {len(baseline_vals)} points")
                    print(f"    Baseline mean: {np.mean(baseline_vals):.6f}")
                    print(f"    Scaled mean: {np.mean(scaled_vals):.6f}")
                    
                    if len(baseline_vals) >= 10:
                        # Calculate correlation
                        try:
                            correlation = np.corrcoef(baseline_vals, scaled_vals)[0, 1]
                        except:
                            correlation = np.nan
                        
                        # Calculate RMSE
                        residuals = scaled_vals - baseline_vals
                        rmse = np.sqrt(np.mean(residuals ** 2))
                        
                        # Calculate R² with better error handling
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((baseline_vals - np.mean(baseline_vals)) ** 2)
                        
                        print(f"    SS_res: {ss_res:.6f}")
                        print(f"    SS_tot: {ss_tot:.6f}")
                        
                        if ss_tot > 1e-10:  # Avoid division by very small numbers
                            r_squared = max(0.0, 1 - (ss_res / ss_tot))
                        else:
                            r_squared = 1.0 if ss_res < 1e-10 else 0.0
                            print(f"    WARNING: No variance in baseline data")
                        
                        print(f"    FINAL - R²: {r_squared:.6f}, Corr: {correlation:.6f}, RMSE: {rmse:.6f}")
                    else:
                        correlation = np.nan
                        rmse = np.nan
                        r_squared = np.nan
                
                # Log-space correlation
                log_common_mask = common_mask & (baseline_data > 0) & (scaled_data > 0)
                if np.sum(log_common_mask) >= 10:
                    log_baseline = np.log(baseline_data[log_common_mask])
                    log_scaled = np.log(scaled_data[log_common_mask])
                    log_correlation = stats.pearsonr(log_baseline, log_scaled)[0]
                    log_rmse = np.sqrt(np.mean((log_scaled - log_baseline) ** 2))
                else:
                    log_correlation = np.nan
                    log_rmse = np.nan
                
                # Print results with explicit log-space calculations
                print(f"  {target_sat}:")
                print(f"    Log mean baseline: {log_mean_baseline:.6f}")
                print(f"    Log mean target: {log_mean_target:.6f}")
                print(f"    Log scaling factor: {log_scaling_factor:.6f}")
                print(f"    Linear scaling factor: {scaling_factor:.6f}")
                print(f"    Verification (should match): {scaling_factor_verification:.6f}")
                print(f"    Geometric mean baseline: {geo_mean_baseline:.4f}")
                print(f"    Geometric mean target: {geo_mean_target:.4f}")
                print(f"    Linear correlation: {correlation:.4f}")
                print(f"    Log-space correlation: {log_correlation:.4f}")
                print(f"    R²: {r_squared:.4f}")
                print(f"    RMSE: {rmse:.4f}")
                print(f"    Log-space RMSE: {log_rmse:.4f}")
                print(f"    Valid data points: {np.sum(common_mask)}")
                
                # Store the results
                results[target_sat] = ScalingResult(
                    scaling_factor=scaling_factor,
                    log_scaling_factor=log_scaling_factor,  # Store log scaling factor
                    r_squared=r_squared,
                    correlation=correlation,
                    rmse=rmse,
                    n_points=np.sum(common_mask),
                    scaled_data=scaled_data,
                    original_data=df[target_sat],
                    baseline_data=df[baseline_satellite]
                )
                
            except Exception as e:
                print(f"Error processing {target_sat}: {str(e)}")
                continue
        
        return results

    def perform_geometric_scaling_by_biome(self, biome_dataframes: Dict[str, pd.DataFrame], 
                                          baseline_satellite: str, 
                                          target_satellites: Optional[List[str]] = None) -> Dict[str, Dict[str, ScalingResult]]:
        """Perform geometric scaling for each biome"""
        all_results = {}
        
        for biome, df in biome_dataframes.items():
            print(f"\nProcessing biome: {biome}")
            results = self.perform_geometric_scaling(df, baseline_satellite, target_satellites)
            all_results[biome] = results
        
        return all_results

    def diagnose_scaling(self, df: pd.DataFrame, baseline_sat: str, target_sat: str):
        """Enhanced diagnostic of scaling relationship with log-space calculations"""
        baseline_data = df[baseline_sat].dropna()
        target_data = df[target_sat].dropna()
        
        # Find common valid indices  
        common_idx = baseline_data.index.intersection(target_data.index)
        baseline_common = baseline_data[common_idx]
        target_common = target_data[common_idx]
        
        print(f"\n=== DIAGNOSTIC: {target_sat} vs {baseline_sat} ===")
        print(f"Common data points: {len(common_idx)}")
        print(f"{baseline_sat} range: {baseline_common.min():.4f} to {baseline_common.max():.4f}")
        print(f"{target_sat} range: {target_common.min():.4f} to {target_common.max():.4f}")
        print(f"{baseline_sat} mean: {baseline_common.mean():.4f}")
        print(f"{target_sat} mean: {target_common.mean():.4f}")
        
        # Simple ratio
        mean_ratio = target_common.mean() / baseline_common.mean()
        print(f"Mean ratio (target/baseline): {mean_ratio:.4f}")
        
        # Correlation
        corr = target_common.corr(baseline_common)
        print(f"Correlation: {corr:.4f}")
        
        # Positive data only
        pos_baseline = baseline_common[baseline_common > 0]
        pos_target = target_common[target_common > 0]
        common_pos_idx = pos_baseline.index.intersection(pos_target.index)
        
        if len(common_pos_idx) > 10:
            print("\nPositive values only:")
            pos_baseline = pos_baseline[common_pos_idx]
            pos_target = pos_target[common_pos_idx]
            
            print(f"Common positive data points: {len(common_pos_idx)}")
            print(f"Positive {baseline_sat} mean: {pos_baseline.mean():.4f}")
            print(f"Positive {target_sat} mean: {pos_target.mean():.4f}")
            
            # Log-space calculations
            log_mean_baseline = np.mean(np.log(pos_baseline))
            log_mean_target = np.mean(np.log(pos_target))
            log_scaling_factor = log_mean_baseline - log_mean_target
            scaling_factor = np.exp(log_scaling_factor)
            
            print(f"\nLog-space calculations:")
            print(f"Log mean {baseline_sat}: {log_mean_baseline:.6f}")
            print(f"Log mean {target_sat}: {log_mean_target:.6f}")
            print(f"Log scaling factor (log(c)): {log_scaling_factor:.6f}")
            print(f"Linear scaling factor (c): {scaling_factor:.6f}")
            
            # Geometric means (should match exp of log means)
            geo_mean_baseline = np.exp(log_mean_baseline)
            geo_mean_target = np.exp(log_mean_target)
            
            print(f"\nGeometric means (verification):")
            print(f"Geometric mean {baseline_sat}: {geo_mean_baseline:.4f}")
            print(f"Geometric mean {target_sat}: {geo_mean_target:.4f}")
            print(f"Geometric scaling factor (verification): {geo_mean_baseline/geo_mean_target:.6f}")
        
        return scaling_factor if 'scaling_factor' in locals() else mean_ratio

    def _create_stats_text(self, baseline_data: pd.Series, comparison_data: pd.Series, 
                          result: Optional[ScalingResult] = None, log_space: bool = False) -> str:
        """Create statistics text for plots"""
        # Make sure we handle NaN and zero values properly
        valid_mask = np.isfinite(baseline_data) & np.isfinite(comparison_data)
        
        if log_space:
            # For log space, also exclude zeros and negative values
            valid_mask = valid_mask & (baseline_data > 0) & (comparison_data > 0)
            
            if np.sum(valid_mask) == 0:
                return "No valid data"
            
            baseline_vals = np.log(baseline_data[valid_mask])
            comparison_vals = np.log(comparison_data[valid_mask])
            
            # Additional check for log results
            final_mask = np.isfinite(baseline_vals) & np.isfinite(comparison_vals)
            if np.sum(final_mask) == 0:
                return "No valid log data"
                
            baseline_vals = baseline_vals[final_mask]
            comparison_vals = comparison_vals[final_mask]
        else:
            if np.sum(valid_mask) == 0:
                return "No valid data"
            baseline_vals = baseline_data[valid_mask]
            comparison_vals = comparison_data[valid_mask]
        
        corr = stats.pearsonr(baseline_vals, comparison_vals)[0]
        bias = np.mean(comparison_vals - baseline_vals)
        rmse = np.sqrt(np.mean((comparison_vals - baseline_vals) ** 2))
        
        stats_text = f'r = {corr:.3f}\nBias = {bias:.4f}\nRMSE = {rmse:.4f}\nn = {len(baseline_vals)}'
        
        if result and not log_space:
            stats_text += f'\nScaling = {result.scaling_factor:.3f}'
            stats_text += f'\nLog(c) = {result.log_scaling_factor:.4f}'
        
        return stats_text

    def plot_biome_analysis(self, biome_dataframes: Dict[str, pd.DataFrame], 
                           biome_results: Dict[str, Dict[str, ScalingResult]], 
                           baseline_satellite: str, plot_type: str = 'timeseries', 
                           save_path: Optional[str] = None):
        """Create plots for biome analysis"""
        
        # Extract date range from the first available dataframe
        date_range_str = ""
        if biome_dataframes:
            first_df = next(iter(biome_dataframes.values()))
            start_date = first_df.index.min().strftime('%Y%m%d')
            end_date = first_df.index.max().strftime('%Y%m%d')
            date_range_str = f"{start_date}_{end_date}"
        
        for biome, df in biome_dataframes.items():
            if biome not in biome_results or not biome_results[biome]:
                continue
            
            results = biome_results[biome]
            biome_name = self.BIOMES[biome]
            
            for sat, result in results.items():
                # Increased figure size and adjusted subplot spacing
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.subplots_adjust(hspace=0.35, wspace=0.25)
                
                baseline_data = df[baseline_satellite]
                original_data = result.original_data
                scaled_data = result.scaled_data
                
                if plot_type == 'timeseries':
                    self._plot_timeseries_2x2(axes, baseline_data, original_data, scaled_data, 
                                            biome_name, sat, baseline_satellite, result)
                else:  # scatter
                    self._plot_scatter_2x2(axes, baseline_data, original_data, scaled_data, 
                                         result, biome_name, sat, baseline_satellite)
                
                # Improve x-axis label formatting for all subplots
                for ax in axes.flat:
                    if plot_type == 'timeseries':
                        # Reduce number of x-axis ticks and rotate labels
                        ax.tick_params(axis='x', rotation=45, labelsize=9)
                        ax.locator_params(axis='x', nbins=6)
                    else:
                        ax.tick_params(axis='x', labelsize=10)
                        ax.tick_params(axis='y', labelsize=10)
                
                # Adjust overall title
                fig.suptitle(f'{biome_name}: {self.SATELLITES[sat]} vs {self.SATELLITES[baseline_satellite]}', 
                           fontsize=14, y=0.98)
                
                # Use tight_layout with extra padding
                plt.tight_layout(pad=2.0, rect=[0, 0.02, 1, 0.96])
                
                if save_path:
                    # Extract biome code from the full biome key (e.g., 'frp_tf' -> 'tf')
                    biome_code = biome.replace('frp_', '')
                    
                    # Create new filename format: plottype_biomeCode_satellite_vs_baseline_dateRange.png
                    new_filename = f"frp_{plot_type}_{biome_code}_{sat}_vs_{baseline_satellite}_{date_range_str}.png"
                    
                    # Replace the original save_path with the new filename
                    if '/' in save_path:
                        directory = os.path.dirname(save_path)
                        sat_save_path = os.path.join(directory, new_filename)
                    else:
                        sat_save_path = new_filename
                    
                    plt.savefig(sat_save_path, dpi=300, bbox_inches='tight', facecolor='white')
                    print(f"Plot saved to: {sat_save_path}")
                
                plt.close()

    def _plot_timeseries_2x2(self, axes, baseline_data, original_data, scaled_data, 
                           biome_name, sat, baseline_satellite, result):
        """Create 2x2 timeseries plots with better formatting"""
        # Linear before (top left)
        axes[0, 0].plot(baseline_data.index, baseline_data.values, 'b-', 
                       label=f'{self.SATELLITES[baseline_satellite]} (Baseline)', alpha=0.8, linewidth=1.5)
        axes[0, 0].plot(original_data.index, original_data.values, 'r-', 
                       label=f'{self.SATELLITES[sat]} (Original)', alpha=0.8, linewidth=1.5)
        axes[0, 0].set_ylabel('FRP Density (W/km²)', fontsize=10)
        axes[0, 0].set_title('Before Scaling (Linear)', fontsize=11, pad=10)
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)
        
        stats_text = self._create_stats_text(baseline_data, original_data)
        axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes, 
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Linear after (top right)
        axes[0, 1].plot(baseline_data.index, baseline_data.values, 'b-', 
                       label=f'{self.SATELLITES[baseline_satellite]} (Baseline)', alpha=0.8, linewidth=1.5)
        axes[0, 1].plot(scaled_data.index, scaled_data.values, 'g-', 
                       label=f'{self.SATELLITES[sat]} (Scaled)', alpha=0.8, linewidth=1.5)
        axes[0, 1].set_ylabel('FRP Density (W/km²)', fontsize=10)
        axes[0, 1].set_title('After Scaling (Linear)', fontsize=11, pad=10)
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        
        stats_text = self._create_stats_text(baseline_data, scaled_data, result)
        axes[0, 1].text(0.02, 0.98, stats_text, transform=axes[0, 1].transAxes, 
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Log before (bottom left)
        pos_mask_baseline = baseline_data > 0
        pos_mask_original = original_data > 0
        
        axes[1, 0].plot(baseline_data.index[pos_mask_baseline], np.log(baseline_data[pos_mask_baseline]), 'b-', 
                       label=f'{self.SATELLITES[baseline_satellite]} (Baseline)', alpha=0.8, linewidth=1.5)
        axes[1, 0].plot(original_data.index[pos_mask_original], np.log(original_data[pos_mask_original]), 'r-', 
                       label=f'{self.SATELLITES[sat]} (Original)', alpha=0.8, linewidth=1.5)
        axes[1, 0].set_ylabel('ln(FRP Density (W/km²))', fontsize=10)
        axes[1, 0].set_xlabel('Date', fontsize=10)
        axes[1, 0].set_title('Before Scaling (ln)', fontsize=11, pad=10)
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        
        stats_text = self._create_stats_text(baseline_data, original_data, log_space=True)
        axes[1, 0].text(0.02, 0.98, stats_text, transform=axes[1, 0].transAxes, 
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Log after (bottom right)
        pos_mask_scaled = scaled_data > 0
        
        axes[1, 1].plot(baseline_data.index[pos_mask_baseline], np.log(baseline_data[pos_mask_baseline]), 'b-', 
                       label=f'{self.SATELLITES[baseline_satellite]} (Baseline)', alpha=0.8, linewidth=1.5)
        axes[1, 1].plot(scaled_data.index[pos_mask_scaled], np.log(scaled_data[pos_mask_scaled]), 'g-', 
                       label=f'{self.SATELLITES[sat]} (Scaled)', alpha=0.8, linewidth=1.5)
        axes[1, 1].set_ylabel('ln(FRP Density (W/km²))', fontsize=10)
        axes[1, 1].set_xlabel('Date', fontsize=10)
        axes[1, 1].set_title('After Scaling (ln)', fontsize=11, pad=10)
        axes[1, 1].legend(fontsize=9)
        axes[1, 1].grid(True, alpha=0.3)
        
        stats_text = self._create_stats_text(baseline_data, scaled_data, result, log_space=True)
        axes[1, 1].text(0.02, 0.98, stats_text, transform=axes[1, 1].transAxes, 
                       verticalalignment='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    def _plot_scatter_2x2(self, axes, baseline_data, original_data, scaled_data, 
                        result, biome_name, sat, baseline_satellite):
        """Create 2x2 scatter plots with better formatting"""
        # Linear before (top left)
        valid_mask = np.isfinite(baseline_data) & np.isfinite(original_data)
        if np.sum(valid_mask) > 0:
            axes[0, 0].scatter(baseline_data[valid_mask], original_data[valid_mask], alpha=0.6, s=15)
            min_val = min(baseline_data[valid_mask].min(), original_data[valid_mask].min())
            max_val = max(baseline_data[valid_mask].max(), original_data[valid_mask].max())
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1)
        
        axes[0, 0].set_xlabel(f'{self.SATELLITES[baseline_satellite]} FRP Density (W/km²)', fontsize=9)
        axes[0, 0].set_ylabel(f'{self.SATELLITES[sat]} FRP Density (W/km²)', fontsize=9)
        axes[0, 0].set_title('Before Scaling (Linear)', fontsize=11, pad=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Calculate statistics for original data
        if np.sum(valid_mask) > 0:
            orig_corr = stats.pearsonr(baseline_data[valid_mask], original_data[valid_mask])[0]
            orig_rmse = np.sqrt(np.mean((original_data[valid_mask] - baseline_data[valid_mask]) ** 2))
            
            stats_text = f'r = {orig_corr:.3f}\nRMSE = {orig_rmse:.3f}\nn = {np.sum(valid_mask)}'
            axes[0, 0].text(0.05, 0.95, stats_text, transform=axes[0, 0].transAxes, 
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Linear after (top right)
        valid_mask = np.isfinite(baseline_data) & np.isfinite(scaled_data)
        if np.sum(valid_mask) > 0:
            axes[0, 1].scatter(baseline_data[valid_mask], scaled_data[valid_mask], alpha=0.6, s=15, color='green')
            min_val = min(baseline_data[valid_mask].min(), scaled_data[valid_mask].min())
            max_val = max(baseline_data[valid_mask].max(), scaled_data[valid_mask].max())
            axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1)
        
        axes[0, 1].set_xlabel(f'{self.SATELLITES[baseline_satellite]} FRP Density (W/km²)', fontsize=9)
        axes[0, 1].set_ylabel(f'{self.SATELLITES[sat]} FRP Density (W/km²) (Scaled)', fontsize=9)
        axes[0, 1].set_title('After Scaling (Linear)', fontsize=11, pad=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        if np.sum(valid_mask) > 0:
            stats_text = f'r = {result.correlation:.3f}\nR² = {result.r_squared:.3f}\n'
            stats_text += f'RMSE = {result.rmse:.3f}\nScaling = {result.scaling_factor:.3f}\nlog(c) = {result.log_scaling_factor:.4f}'
            axes[0, 1].text(0.05, 0.95, stats_text, transform=axes[0, 1].transAxes, 
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Log before (bottom left)
        log_valid_mask = np.isfinite(baseline_data) & np.isfinite(original_data) & (baseline_data > 0) & (original_data > 0)
        if np.sum(log_valid_mask) > 0:
            ln_baseline = np.log(baseline_data[log_valid_mask])
            ln_original = np.log(original_data[log_valid_mask])
            axes[1, 0].scatter(ln_baseline, ln_original, alpha=0.6, s=15)
            min_val = min(ln_baseline.min(), ln_original.min())
            max_val = max(ln_baseline.max(), ln_original.max())
            axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1)
        
        axes[1, 0].set_xlabel(f'ln({self.SATELLITES[baseline_satellite]} FRP Density (W/km²))', fontsize=9)
        axes[1, 0].set_ylabel(f'ln({self.SATELLITES[sat]} FRP Density (W/km²))', fontsize=9)
        axes[1, 0].set_title('Before Scaling (ln)', fontsize=11, pad=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        if np.sum(log_valid_mask) > 0:
            log_corr = stats.pearsonr(ln_baseline, ln_original)[0]
            log_rmse = np.sqrt(np.mean((ln_original - ln_baseline) ** 2))
            
            stats_text = f'r = {log_corr:.3f}\nRMSE = {log_rmse:.3f}\nn = {np.sum(log_valid_mask)}'
            axes[1, 0].text(0.05, 0.95, stats_text, transform=axes[1, 0].transAxes, 
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Log after (bottom right)
        log_valid_mask_scaled = np.isfinite(baseline_data) & np.isfinite(scaled_data) & (baseline_data > 0) & (scaled_data > 0)
        if np.sum(log_valid_mask_scaled) > 0:
            ln_baseline = np.log(baseline_data[log_valid_mask_scaled])
            ln_scaled = np.log(scaled_data[log_valid_mask_scaled])
            axes[1, 1].scatter(ln_baseline, ln_scaled, alpha=0.6, s=15, color='green')
            min_val = min(ln_baseline.min(), ln_scaled.min())
            max_val = max(ln_baseline.max(), ln_scaled.max())
            axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=1)
        
        axes[1, 1].set_xlabel(f'ln({self.SATELLITES[baseline_satellite]} FRP Density (W/km²))', fontsize=9)
        axes[1, 1].set_ylabel(f'ln({self.SATELLITES[sat]} FRP Density (W/km²)) (Scaled)', fontsize=9)
        axes[1, 1].set_title('After Scaling (ln)', fontsize=11, pad=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        if np.sum(log_valid_mask_scaled) > 0:
            log_corr = stats.pearsonr(ln_baseline, ln_scaled)[0]
            log_rmse = np.sqrt(np.mean((ln_scaled - ln_baseline) ** 2))
            
            stats_text = f'r = {log_corr:.3f}\nRMSE = {log_rmse:.3f}\nlog(c) = {result.log_scaling_factor:.4f}'
            axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def print_summary(self, biome_dataframes: Dict[str, pd.DataFrame], 
                     biome_results: Dict[str, Dict[str, ScalingResult]]):
        """Print comprehensive analysis summary with log scaling factors"""
        print("\n" + "="*60)
        print("FRP LOG-SPACE GEOMETRIC SCALING ANALYSIS SUMMARY")
        print("="*60)
        
        # Data availability
        print("\nDATA AVAILABILITY BY BIOME:")
        for biome, df in biome_dataframes.items():
            print(f"\n{self.BIOMES[biome]}:")
            for sat in df.columns:
                valid_days = df[sat].notna().sum()
                total_days = len(df)
                print(f"  {self.SATELLITES[sat]}: {valid_days}/{total_days} days ({100*valid_days/total_days:.1f}%)")
        
        # Scaling results
        print("\nLOG-SPACE GEOMETRIC SCALING RESULTS BY BIOME:")
        print("Formula: log(c) = mean(log(baseline)) - mean(log(target))")
        for biome, results in biome_results.items():
            if results:
                print(f"\n{self.BIOMES[biome]} ({biome}):")
                print("-" * 60)
                for sat, result in results.items():
                    print(f"  {self.SATELLITES[sat]}:")
                    print(f"    Log Scaling Factor (log(c)): {result.log_scaling_factor:.6f}")
                    print(f"    Linear Scaling Factor (c):   {result.scaling_factor:.6f}")
                    print(f"    R²: {result.r_squared:.4f}")
                    print(f"    Correlation: {result.correlation:.4f}")
                    print(f"    RMSE: {result.rmse:.4f}")
                    print(f"    Data points: {result.n_points}")
                    print(f"    Scaling Equation: scaled_{sat} = exp({result.log_scaling_factor:.6f}) × original_{sat}")
                    print(f"                     scaled_{sat} = {result.scaling_factor:.6f} × original_{sat}")


def main():
    """Main analysis function with explicit log-space geometric scaling"""
    # Configuration
    scaler = FRPScaler()
    satellites_to_process = ['vj1', 'vj2', 'vnp', 'myd', 'mod']
    baseline_satellite = 'myd'
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    print("FRP SCALING ANALYSIS BY BIOME (Log-Space Geometric Mean Scaling)")
    print("Formula: log(c) = mean(log(FRP_baseline)) - mean(log(FRP_target))")
    print(f"Satellites: {satellites_to_process}")
    print(f"Baseline: {baseline_satellite} ({scaler.SATELLITES[baseline_satellite]})")
    print(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Load data with caching and parallel processing
    print("\nLoading time series data by biome (optimized with caching)...")
    biome_dataframes, biome_names = scaler.load_time_series_by_biome_cached(
        satellites_to_process, start_date, end_date
    )
    
    if not biome_dataframes:
        print("No data loaded. Please check file paths and data availability.")
        return
    
    # Optional: Run diagnostics first (uncomment to use)
    # print("\nRUNNING DIAGNOSTICS...")
    # for biome, df in biome_dataframes.items():
    #     print(f"\nDiagnostics for {biome} ({scaler.BIOMES[biome]}):")
    #     for target in [sat for sat in satellites_to_process if sat != baseline_satellite]:
    #         scaler.diagnose_scaling(df, baseline_satellite, target)
    
    # Perform geometric scaling
    print("\nPerforming log-space geometric scaling by biome...")
    biome_results = scaler.perform_geometric_scaling_by_biome(
        biome_dataframes, baseline_satellite
    )
    
    # Check if we got any results
    print(f"\nLog-space geometric scaling completed. Results for {len(biome_results)} biomes.")
    for biome, results in biome_results.items():
        print(f"  {biome}: {len(results)} satellite pairs processed")
    
    if not any(biome_results.values()):
        print("No scaling results obtained. Check data availability and parameters.")
        return
    
    # Print summary
    print("\n" + "="*60)
    print("PRINTING SUMMARY...")
    scaler.print_summary(biome_dataframes, biome_results)
    
    # Create plots
    print("\n" + "="*60)
    print("CREATING PLOTS...")
    print("="*60)
    
    print("\nCreating time series plots by biome...")
    try:
        scaler.plot_biome_analysis(
            biome_dataframes, biome_results, baseline_satellite, 
            plot_type='timeseries', save_path='frp_logspace_scaling.png'
        )
        print("Time series plots completed successfully.")
    except Exception as e:
        print(f"Error creating time series plots: {e}")
    
    print("\nCreating scatter plots by biome...")
    try:
        scaler.plot_biome_analysis(
            biome_dataframes, biome_results, baseline_satellite, 
            plot_type='scatter', save_path='frp_logspace_scaling.png'
        )
        print("Scatter plots completed successfully.")
    except Exception as e:
        print(f"Error creating scatter plots: {e}")
    
    print("\nAnalysis complete!")


def quick_analysis(satellites: List[str], baseline: str, start_date: datetime, 
                  end_date: datetime, biomes: Optional[List[str]] = None):
    """Quick analysis function for specific parameters"""
    scaler = FRPScaler()
    
    biome_dataframes, _ = scaler.load_time_series_by_biome_cached(satellites, start_date, end_date)
    
    if biomes:
        # Filter to specific biomes
        biome_dataframes = {k: v for k, v in biome_dataframes.items() if k in biomes}
    
    biome_results = scaler.perform_geometric_scaling_by_biome(biome_dataframes, baseline)
    scaler.print_summary(biome_dataframes, biome_results)
    
    return biome_dataframes, biome_results


if __name__ == "__main__":
    main()
