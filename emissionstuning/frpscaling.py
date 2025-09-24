import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import os
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import hashlib

warnings.filterwarnings('ignore')

@dataclass
class RegressionResult:
    """Data class to store regression results"""
    slope: float
    intercept: float
    r_squared: float
    correlation: float
    p_value: float
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

    def __init__(self, base_path: str = "path_to_l3a_FRP"):
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
        
        # Handle special naming for Aqua MODIS
        satellite = 'MYD14' if satellite == 'myd' else satellite
        satellite = 'MOD14' if satellite == 'mod' else satellite
        filename = f"qfed3_2.frp.{satellite}.{year}{month}{day}.nc4"
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

    # Keep the original method for compatibility
    def load_time_series_by_biome(self, satellites: List[str], start_date: datetime, 
                                  end_date: datetime) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
        """Original sequential loading method (kept for compatibility)"""
        return self.load_time_series_by_biome_cached(satellites, start_date, end_date)

    def perform_regression(self, df: pd.DataFrame, baseline_satellite: str, 
                          target_satellites: Optional[List[str]] = None, 
                          force_zero_intercept: bool = True,
                          linear_scaling: bool = True) -> Dict[str, RegressionResult]:
        """Perform linear regression to scale satellites to baseline
        
        Parameters:
        df: DataFrame with satellite data
        baseline_satellite: str, baseline satellite designation
        target_satellites: list, satellites to scale (if None, uses all except baseline)
        force_zero_intercept: bool, whether to force intercept=0
        linear_scaling: bool, use linear scaling instead of power-law
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
        print(f"Force zero intercept: {force_zero_intercept}")
        print(f"Linear scaling: {linear_scaling}")
        
        results = {}
        
        for target_sat in valid_targets:
            if linear_scaling:
                # Direct linear regression between original values (no log transform)
                valid_mask = np.isfinite(df[baseline_satellite]) & np.isfinite(df[target_sat])
                
                if np.sum(valid_mask) < 10:
                    print(f"Warning: Insufficient data for {baseline_satellite} vs {target_sat}")
                    continue
                
                X = df[baseline_satellite].values[valid_mask].reshape(-1, 1)
                y = df[target_sat].values[valid_mask]
                
                if force_zero_intercept:
                    # Force regression through origin (intercept = 0)
                    reg = LinearRegression(fit_intercept=False).fit(X, y)
                    intercept = 0.0
                else:
                    # Standard regression with intercept
                    reg = LinearRegression().fit(X, y)
                    intercept = reg.intercept_
                
                y_pred = reg.predict(X)
                r_squared = reg.score(X, y)
                correlation, p_value = stats.pearsonr(X.flatten(), y)
                rmse = np.sqrt(np.mean((y - y_pred) ** 2))
                
                # Apply linear scaling
                if force_zero_intercept:
                    scaled_data = reg.coef_[0] * df[baseline_satellite]
                else:
                    scaled_data = intercept + reg.coef_[0] * df[baseline_satellite]
                
            else:
                # Log-space regression (power-law scaling)
                log_baseline = np.log(df[baseline_satellite])
                log_target = np.log(df[target_sat])
                valid_mask = np.isfinite(log_baseline) & np.isfinite(log_target)
                
                if np.sum(valid_mask) < 10:
                    print(f"Warning: Insufficient data for {baseline_satellite} vs {target_sat}")
                    continue
                
                X = log_baseline[valid_mask].values.reshape(-1, 1)
                y = log_target[valid_mask].values
                
                if force_zero_intercept:
                    # Force regression through origin (intercept = 0)
                    reg = LinearRegression(fit_intercept=False).fit(X, y)
                    intercept = 0.0
                else:
                    # Standard regression with intercept
                    reg = LinearRegression().fit(X, y)
                    intercept = reg.intercept_
                
                y_pred = reg.predict(X)
                
                # Calculate R² manually for zero-intercept case
                if force_zero_intercept:
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum(y ** 2)  # Different for zero-intercept
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                else:
                    r_squared = reg.score(X, y)
                
                correlation, p_value = stats.pearsonr(X.flatten(), y)
                rmse = np.sqrt(np.mean((y - y_pred) ** 2))
                
                # Apply scaling
                if force_zero_intercept:
                    log_scaled = reg.coef_[0] * log_baseline
                else:
                    log_scaled = intercept + reg.coef_[0] * log_baseline
                
                scaled_data = np.exp(log_scaled)
            
            results[target_sat] = RegressionResult(
                slope=reg.coef_[0],
                intercept=intercept,
                r_squared=r_squared,
                correlation=correlation,
                p_value=p_value,
                rmse=rmse,
                n_points=np.sum(valid_mask),
                scaled_data=scaled_data,
                original_data=df[target_sat],
                baseline_data=df[baseline_satellite]
            )
        
        return results

    def perform_regression_by_biome(self, biome_dataframes: Dict[str, pd.DataFrame], 
                                   baseline_satellite: str, 
                                   target_satellites: Optional[List[str]] = None,
                                   force_zero_intercept: bool = True,
                                   linear_scaling: bool = True) -> Dict[str, Dict[str, RegressionResult]]:
        """Perform regression for each biome"""
        all_results = {}
        
        for biome, df in biome_dataframes.items():
            print(f"\nProcessing biome: {biome}")
            results = self.perform_regression(df, baseline_satellite, target_satellites, 
                                            force_zero_intercept, linear_scaling)
            all_results[biome] = results
        
        return all_results

    def _create_stats_text(self, baseline_data: pd.Series, comparison_data: pd.Series, 
                          result: Optional[RegressionResult] = None, log_space: bool = False) -> str:
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
            stats_text += f'\nSlope = {result.slope:.3f}\nIntercept = {result.intercept:.3f}'
        
        return stats_text

    def plot_biome_analysis(self, biome_dataframes: Dict[str, pd.DataFrame], 
                           biome_results: Dict[str, Dict[str, RegressionResult]], 
                           baseline_satellite: str, plot_type: str = 'timeseries', 
                           save_path: Optional[str] = None):
        """Create plots for biome analysis"""
        for biome, df in biome_dataframes.items():
            if biome not in biome_results or not biome_results[biome]:
                continue
            
            results = biome_results[biome]
            biome_name = self.BIOMES[biome]
            
            for sat, result in results.items():
                # Increased figure size and adjusted subplot spacing
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                fig.subplots_adjust(hspace=0.35, wspace=0.25)  # Add more space between subplots
                
                baseline_data = df[baseline_satellite]
                original_data = result.original_data
                scaled_data = result.scaled_data
                
                if plot_type == 'timeseries':
                    self._plot_timeseries_2x2(axes, baseline_data, original_data, scaled_data, 
                                            biome_name, sat, baseline_satellite)
                else:  # scatter
                    self._plot_scatter_2x2(axes, baseline_data, original_data, scaled_data, 
                                         result, biome_name, sat, baseline_satellite)
                
                # Improve x-axis label formatting for all subplots
                for ax in axes.flat:
                    if plot_type == 'timeseries':
                        # Reduce number of x-axis ticks and rotate labels
                        ax.tick_params(axis='x', rotation=45, labelsize=9)
                        # Set fewer ticks to reduce overlap
                        ax.locator_params(axis='x', nbins=6)
                    else:
                        # For scatter plots, normal tick formatting
                        ax.tick_params(axis='x', labelsize=10)
                        ax.tick_params(axis='y', labelsize=10)
                
                # Adjust overall title
                fig.suptitle(f'{biome_name}: {self.SATELLITES[sat]} vs {self.SATELLITES[baseline_satellite]}', 
                           fontsize=14, y=0.98)
                
                # Use tight_layout with extra padding
                plt.tight_layout(pad=2.0, rect=[0, 0.02, 1, 0.96])
                
                if save_path:
                    plot_suffix = 'timeseries' if plot_type == 'timeseries' else 'scatter'
                    sat_save_path = save_path.replace('.png', f'_{plot_suffix}_{biome}_{sat}.png')
                    plt.savefig(sat_save_path, dpi=300, bbox_inches='tight', facecolor='white')
                    print(f"Plot saved to: {sat_save_path}")
                
                plt.close()

    def _plot_timeseries_2x2(self, axes, baseline_data, original_data, scaled_data, 
                           biome_name, sat, baseline_satellite):
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
        
        stats_text = self._create_stats_text(baseline_data, scaled_data)
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
        
        stats_text = self._create_stats_text(baseline_data, scaled_data, log_space=True)
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
        
        stats_text = f'R² = {result.r_squared:.3f}\nr = {result.correlation:.3f}\n'
        stats_text += f'RMSE = {result.rmse:.3f}\nn = {result.n_points}'
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
            post_corr = stats.pearsonr(baseline_data[valid_mask], scaled_data[valid_mask])[0]
            post_rmse = np.sqrt(np.mean((baseline_data[valid_mask] - scaled_data[valid_mask]) ** 2))
            
            stats_text = f'R² = {post_corr**2:.3f}\nr = {post_corr:.3f}\n'
            stats_text += f'RMSE = {post_rmse:.3f}\nSlope = {result.slope:.3f}\nIntercept = {result.intercept:.3f}'
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
            
            stats_text = f'R² = {log_corr**2:.3f}\nr = {log_corr:.3f}\n'
            stats_text += f'RMSE = {log_rmse:.3f}\nn = {np.sum(log_valid_mask)}'
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
            
            stats_text = f'R² = {log_corr**2:.3f}\nr = {log_corr:.3f}\n'
            stats_text += f'RMSE = {log_rmse:.3f}\nSlope = {result.slope:.3f}\nIntercept = {result.intercept:.3f}'
            axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes, 
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def print_summary(self, biome_dataframes: Dict[str, pd.DataFrame], 
                     biome_results: Dict[str, Dict[str, RegressionResult]]):
        """Print comprehensive analysis summary"""
        print("\n" + "="*60)
        print("FRP SCALING ANALYSIS SUMMARY")
        print("="*60)
        
        # Data availability
        print("\nDATA AVAILABILITY BY BIOME:")
        for biome, df in biome_dataframes.items():
            print(f"\n{self.BIOMES[biome]}:")
            for sat in df.columns:
                valid_days = df[sat].notna().sum()
                total_days = len(df)
                print(f"  {self.SATELLITES[sat]}: {valid_days}/{total_days} days ({100*valid_days/total_days:.1f}%)")
        
        # Regression results
        print("\nREGRESSION RESULTS BY BIOME:")
        for biome, results in biome_results.items():
            if results:
                print(f"\n{self.BIOMES[biome]} ({biome}):")
                print("-" * 40)
                for sat, result in results.items():
                    # Calculate the actual scaling factor (what to multiply target by)
                    if result.slope != 0:
                        scaling_factor = 1.0 / result.slope
                    else:
                        scaling_factor = np.nan
                    
                    print(f"  {self.SATELLITES[sat]}:")
                    print(f"    Scaling Factor: {scaling_factor:.4f}  (multiply {sat} by this to match baseline)")
                    print(f"    Regression Slope: {result.slope:.4f}  (baseline = slope × {sat})")
                    print(f"    Intercept: {result.intercept:.4f}")
                    print(f"    R²: {result.r_squared:.4f}")
                    print(f"    Correlation: {result.correlation:.4f}")
                    print(f"    RMSE: {result.rmse:.4f}")
                    print(f"    Data points: {result.n_points}")
                    
                    # Show the scaling equation
                    print(f"    Scaling Equation: scaled_{sat} = {scaling_factor:.4f} × original_{sat}")


def main():
    """Main analysis function with linear scaling"""
    # Configuration
    scaler = FRPScaler()
    satellites_to_process = ['vnp', 'myd'] # 'mod', 'vj1', 'vnp',
    baseline_satellite = 'myd'
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2019, 12, 31)
    
    print("FRP SCALING ANALYSIS BY BIOME (Linear Scaling)")
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
    
    # Perform linear regression (not log space) with zero intercept
    print("\nPerforming linear regression by biome (linear scaling, zero intercept)...")
    biome_results = scaler.perform_regression_by_biome(
        biome_dataframes, baseline_satellite, 
        force_zero_intercept=True, 
        linear_scaling=True  # Use linear scaling
    )
    
    # Check if we got any results
    print(f"\nRegression completed. Results for {len(biome_results)} biomes.")
    for biome, results in biome_results.items():
        print(f"  {biome}: {len(results)} satellite pairs processed")
    
    if not any(biome_results.values()):
        print("No regression results obtained. Check data availability and parameters.")
        return
    
    # Print summary
    print("\n" + "="*60)
    print("PRINTING SUMMARY...")
    scaler.print_summary(biome_dataframes, biome_results)
    
    # Debug section
    print("\n" + "="*60)
    print("DEBUG: CHECKING DATA QUALITY AND CORRELATIONS")
    print("="*60)
    
    for biome, results in biome_results.items():
        for sat, result in results.items():
            print(f"\n=== DEBUGGING {scaler.BIOMES[biome]} - {scaler.SATELLITES[sat]} ===")
            
            baseline_data = result.baseline_data
            original_data = result.original_data
            scaled_data = result.scaled_data
            
            # Check correlation after linear scaling
            valid_mask = np.isfinite(baseline_data) & np.isfinite(scaled_data)
            
            if np.sum(valid_mask) > 10:
                corr_after = stats.pearsonr(baseline_data[valid_mask], scaled_data[valid_mask])[0]
                print(f"Post-scaling correlation (linear): {corr_after:.6f}")
                print(f"Scaling factor: {result.slope:.6f}")
                print(f"Mean baseline: {np.nanmean(baseline_data):.2f}")
                print(f"Mean original: {np.nanmean(original_data):.2f}")
                print(f"Mean scaled: {np.nanmean(scaled_data):.2f}")
                print(f"R²: {result.r_squared:.4f}")
                print(f"RMSE: {result.rmse:.4f}")
    
    # Create plots
    print("\n" + "="*60)
    print("CREATING PLOTS...")
    print("="*60)
    
    print("\nCreating time series plots by biome...")
    try:
        scaler.plot_biome_analysis(
            biome_dataframes, biome_results, baseline_satellite, 
            plot_type='timeseries', save_path='frp_timeseries_biome_linear_scaling.png'
        )
        print("Time series plots completed successfully.")
    except Exception as e:
        print(f"Error creating time series plots: {e}")
    
    print("\nCreating scatter plots by biome...")
    try:
        scaler.plot_biome_analysis(
            biome_dataframes, biome_results, baseline_satellite, 
            plot_type='scatter', save_path='frp_scatter_biome_linear_scaling.png'
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
    
    biome_results = scaler.perform_regression_by_biome(biome_dataframes, baseline, force_zero_intercept=True)
    scaler.print_summary(biome_dataframes, biome_results)
    
    return biome_dataframes, biome_results


if __name__ == "__main__":
    main()