"""
Main analysis script for AOD-based emission scaling factor calculation.
Takes preprocessed data (linear AOD) and calculates optimal scaling factors using log-transformed AOD.
"""

import numpy as np
import xarray as xr
import yaml
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import logging
import re
from datetime import datetime
import pandas as pd
import cartopy.crs as ccrs
from typing import Dict, List, Tuple, Optional

from config_loader import ConfigLoader
import tools.eisf_functions as eisf
from tools.biome_tuning import scale_biomes

class AODScalingAnalysis:
    """
    Analyze preprocessed AOD data to derive optimal scaling factors using log-transformed AOD.
    """
    
    def __init__(self, config: ConfigLoader):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data = None
        self.scaling_factors = None
        self.year = config.config['analysis']['year']
    
    def load_data(self) -> bool:
        """Load preprocessed data for analysis."""
        # Determine the filename based on config
        output_dir = Path(self.config.config['output']['base_directory'])
        
        # Extract observation identifiers
        obs_sources = self.config.get_observation_sources()
        obs_ids = []
        for source in obs_sources:
            if 'MYD04' in self.config.config['observations'][source]['path_template']:
                obs_ids.append('MYD')
            elif 'MOD04' in self.config.config['observations'][source]['path_template']:
                obs_ids.append('MOD')
            else:
                obs_ids.append(source.upper())
        obs_string = '_'.join(sorted(obs_ids))  # e.g., "MOD_MYD"
        
        # Extract experiment basename
        experiment_basename = "c180R_qfed3-2"  # Default
        if self.config.config['model']['experiments']:
            # Extract from first experiment path
            first_exp = list(self.config.config['model']['experiments'].values())[0]
            path_template = first_exp['path_template']
            # Look for pattern like "c180R_qfed3-2_"
            match = re.search(r'(c\d+R?[^/]*qfed[^/]*?)_', path_template)
            if match:
                experiment_basename = match.group(1)
        
        # Get BB threshold from config to match preprocessing filename
        bb_threshold = self.config.config['analysis']['biomass_burning_fraction']
        
        # Check what months were processed to determine correct filename
        months_to_process = self.config.config['analysis'].get('months_to_process')
        if months_to_process is None:
            months_to_process = list(range(1, 13))
        
        # Try to use custom filename from config first
        output_config = self.config.config.get('output', {})
        if 'annual_filename' in output_config:
            annual_file = output_dir / output_config['annual_filename']
        elif 'filename_template' in output_config:
            # Use template
            template = output_config['filename_template']
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
                year=self.year
            ) + ".nc"
            annual_file = output_dir / filename
        else:
            # Fallback to default naming
            if len(months_to_process) == 12 and months_to_process == list(range(1, 13)):
                annual_file = output_dir / f"{experiment_basename}_{obs_string}_annual_bb{bb_threshold}_{self.year}.nc"
            else:
                months_str = "_".join([f"{m:02d}" for m in sorted(months_to_process)])
                annual_file = output_dir / f"{experiment_basename}_{obs_string}_months{months_str}_bb{bb_threshold}_{self.year}.nc"
        
        if not annual_file.exists():
            self.logger.error(f"Preprocessed annual data not found: {annual_file}")
            self.logger.info("Please run data preprocessing first")
            return False
        
        self.logger.info(f"Loading data from {annual_file}")
        self.data = xr.open_dataset(annual_file)
        
        # Log available variables
        self.logger.info(f"Available variables: {list(self.data.data_vars)}")
        self.logger.info(f"Data dimensions: {self.data.dims}")
        
        # Check required variables
        required_vars = ['OBS_totexttau', 'noBB_totexttau', 'f_bb']
        biome_vars = [v for v in self.data.data_vars if v.endswith('_totexttau') and v not in ['OBS_totexttau', 'noBB_totexttau', 'allviirs_totexttau', 'aqua_totexttau', 'terra_totexttau']]
        
        missing_vars = [v for v in required_vars if v not in self.data]
        if missing_vars:
            self.logger.error(f"Missing required variables: {missing_vars}")
            return False
        
        if not biome_vars:
            self.logger.error("No biome-specific AOD variables found")
            return False
        
        self.logger.info(f"Found biome variables: {biome_vars}")
        return True
    
    def run_analysis(self, month: Optional[int] = None) -> Dict:
        """Run the scaling factor analysis - pass linear AOD to optimization."""
        if self.data is None:
            self.logger.error("No data loaded. Call load_data() first.")
            return {}
        
        # Get surface area weighting using original data shape (before any averaging)
        sa_file = self.config.config.get('surface_area_file')
        if sa_file and Path(sa_file).exists():
            self.logger.info(f"Using surface area weighting from: {sa_file}")
            wsa = eisf.get_sa_weight(sa_file, self.data['OBS_totexttau'].shape)
        else:
            self.logger.info("No surface area weighting file specified, using uniform weights")
            wsa = np.ones_like(self.data['OBS_totexttau'].values)
        
        # Filter for specific month or take annual average using log-space methods
        if month is not None:
            if month < 1 or month > 12:
                self.logger.error(f"Invalid month: {month}")
                return {}
                
            self.logger.info(f"Analyzing month {month}")
            # For single month, use the linear data directly (no log averaging needed)
            analysis_data = self.data.sel(month=month)
            if wsa.ndim == 3:
                wsa = wsa[month-1]
        else:
            self.logger.info("Analyzing annual average using log-space averaging")
            # Use log-space averaging for annual mean, but convert back to linear for optimization
            analysis_data = self._compute_annual_average_log_space_return_linear()
            if wsa.ndim == 3:
                wsa = wsa.mean(axis=0)
        
        # Get biomass burning fraction threshold
        bb_fraction_threshold = self.config.config['analysis']['biomass_burning_fraction']
        self.logger.info(f"Using biomass burning fraction threshold: {bb_fraction_threshold}")
        
        # Apply biomass burning fraction weighting
        wbb = eisf.get_fbb_weight(analysis_data.f_bb.values, bb_fraction_threshold)
        
        # Create combined weights (could include wbb if desired)
        wobs = wsa  # * wbb (optional: include biomass burning weighting)
        
        # Get biome variables (after any combinations have been applied during preprocessing)
        biome_vars = []
        for v in analysis_data.data_vars:
            if (v.endswith('_totexttau') and 
                v not in ['OBS_totexttau', 'noBB_totexttau', 'allviirs_totexttau', 
                         'aqua_totexttau', 'terra_totexttau']):
                biome_name = v.replace('_totexttau', '')
                biome_vars.append(biome_name)
        
        self.logger.info(f"Identified biome variables: {biome_vars}")
        
        # Check data availability and create masks
        obs_mask = np.isfinite(analysis_data.OBS_totexttau.values)
        nobb_mask = np.isfinite(analysis_data.noBB_totexttau.values)
        bb_mask = analysis_data.f_bb.values >= bb_fraction_threshold
        
        # Create combined mask for all requirements
        combined_mask = obs_mask & nobb_mask & bb_mask
        
        # Check biome data availability
        for biome in biome_vars:
            var_name = f'{biome}_totexttau'
            if var_name in analysis_data:
                biome_mask = np.isfinite(analysis_data[var_name].values)
                combined_mask &= biome_mask
        
        n_valid = np.sum(combined_mask)
        total_points = combined_mask.size
        self.logger.info(f"Valid data points meeting all criteria: {n_valid}/{total_points}")
        
        min_valid_points = self.config.config['quality_control']['min_valid_points']
        if n_valid < min_valid_points:
            self.logger.error(f"Insufficient valid data points: {n_valid} < {min_valid_points}")
            return {}
        
        # Extract data where mask is True - KEEP IN LINEAR SPACE for optimization
        self.logger.info("Preparing linear AOD data for optimization (log transformation handled internally)")
        
        # Extract flattened arrays where mask is True (LINEAR AOD)
        obs_linear = analysis_data.OBS_totexttau.values[combined_mask]
        nobb_linear = analysis_data.noBB_totexttau.values[combined_mask]
        weights = wobs.flatten()[combined_mask.flatten()] if wobs is not None else None
        
        # Prepare biome data dictionary (LINEAR AOD)
        tau_bio = {}
        for biome in biome_vars:
            var_name = f'{biome}_totexttau'
            if var_name in analysis_data:
                biome_linear = analysis_data[var_name].values[combined_mask]
                tau_bio[biome] = biome_linear
                self.logger.info(f"Loaded {biome} data (linear): {len(biome_linear)} valid points")
        
        if not tau_bio:
            self.logger.error("No biome data found for optimization")
            return {}
        
        # Debug: Check data types before optimization
        self.logger.info("=== DEBUGGING DATA TYPES ===")
        self.logger.info(f"obs_linear type: {type(obs_linear)}, dtype: {obs_linear.dtype if hasattr(obs_linear, 'dtype') else 'N/A'}")
        self.logger.info(f"nobb_linear type: {type(nobb_linear)}, dtype: {nobb_linear.dtype if hasattr(nobb_linear, 'dtype') else 'N/A'}")
        self.logger.info(f"weights type: {type(weights)}, dtype: {weights.dtype if hasattr(weights, 'dtype') else 'N/A'}")

        self.logger.info(f"tau_bio type: {type(tau_bio)}")
        for biome, data in tau_bio.items():
            self.logger.info(f"  {biome}: type={type(data)}, dtype={data.dtype if hasattr(data, 'dtype') else 'N/A'}")
            self.logger.info(f"    shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
            self.logger.info(f"    sample values: {data[:5] if hasattr(data, '__getitem__') else 'N/A'}")

        # Check for any string contamination
        self.logger.info("=== CHECKING FOR STRING CONTAMINATION ===")
        try:
            obs_finite_check = np.isfinite(obs_linear)
            self.logger.info(f"obs_linear finite check passed: {np.sum(obs_finite_check)} finite values")
        except Exception as e:
            self.logger.error(f"Error checking obs_linear: {e}")

        try:
            nobb_finite_check = np.isfinite(nobb_linear)
            self.logger.info(f"nobb_linear finite check passed: {np.sum(nobb_finite_check)} finite values")
        except Exception as e:
            self.logger.error(f"Error checking nobb_linear: {e}")

        for biome, data in tau_bio.items():
            try:
                finite_check = np.isfinite(data)
                self.logger.info(f"{biome} finite check passed: {np.sum(finite_check)} finite values")
            except Exception as e:
                self.logger.error(f"Error checking {biome}: {e}")

        self.logger.info("=== END DEBUGGING ===")
        # Diagnostic: Check raw AOD relationships
        self.logger.info("=== RAW AOD DIAGNOSTICS ===")
        self.logger.info(f"Observations - mean: {np.mean(obs_linear):.4f}, std: {np.std(obs_linear):.4f}")
        self.logger.info(f"No-BB model - mean: {np.mean(nobb_linear):.4f}, std: {np.std(nobb_linear):.4f}")

        for biome, data in tau_bio.items():
            self.logger.info(f"{biome} - mean: {np.mean(data):.4f}, std: {np.std(data):.4f}")

        # Check correlations
        from scipy.stats import pearsonr
        for biome, data in tau_bio.items():
            corr, _ = pearsonr(obs_linear, data)
            self.logger.info(f"Correlation obs vs {biome}: {corr:.4f}")

        # Check what happens with original scaling (all 1.0)
        original_total = nobb_linear + sum(tau_bio.values())
        orig_corr, _ = pearsonr(obs_linear, original_total)
        self.logger.info(f"Original model correlation: {orig_corr:.4f}")
        self.logger.info(f"Original model - mean: {np.mean(original_total):.4f}, obs mean: {np.mean(obs_linear):.4f}")
        self.logger.info("=== END DIAGNOSTICS ===")

        # Run the optimization with LINEAR AOD (optimization function handles log transformation)
        self.logger.info("Running optimization with linear AOD (log transformation handled by optimization function)...")
        min_alpha = self.config.config['optimization']['min_alpha_regularization']
        max_iter = self.config.config['optimization'].get('max_iterations', 2000)
        tolerance = self.config.config['optimization'].get('tolerance', 1e-8)
        
        try:
            alphas = scale_biomes(
                obs_linear,      # Pass linear AOD
                nobb_linear,     # Pass linear AOD
                tau_bio,         # Pass linear AOD
                wobs=weights,
                min_alpha=min_alpha,
                max_iter=max_iter,
                ftol=tolerance,
                xtol=tolerance
            )
            
            self.scaling_factors = alphas
            
            self.logger.info("Optimization complete!")
            for biome, alpha in alphas.items():
                self.logger.info(f"{biome}: {alpha:.4f}")
                
            return alphas
            
        except Exception as e:
            self.logger.error(f"Error in optimization: {str(e)}")
            return {}
    
    def plot_biome_scatter_analysis(self, output_dir: Optional[Path] = None) -> None:
        """Create comprehensive scatter plots for all biomes analyzed."""
        if not self.scaling_factors or self.data is None:
            self.logger.warning("Missing data or scaling factors for biome scatter plots")
            return
        
        if output_dir is None:
            output_dir = Path(self.config.config['output']['base_directory'])
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract identifiers for filenames
        obs_sources = self.config.get_observation_sources()
        obs_ids = []
        for source in obs_sources:
            if 'MYD04' in self.config.config['observations'][source]['path_template']:
                obs_ids.append('MYD')
            elif 'MOD04' in self.config.config['observations'][source]['path_template']:
                obs_ids.append('MOD')
            else:
                obs_ids.append(source.upper())
        obs_string = '_'.join(sorted(obs_ids))
        
        # Extract experiment basename
        experiment_basename = "c180R_qfed3-2"  # Default
        if self.config.config['model']['experiments']:
            first_exp = list(self.config.config['model']['experiments'].values())[0]
            path_template = first_exp['path_template']
            match = re.search(r'(c\d+R?[^/]*qfed[^/]*?)_', path_template)
            if match:
                experiment_basename = match.group(1)
        
        # Get BB threshold for filename
        bb_threshold = self.config.config['analysis']['biomass_burning_fraction']
        
        try:
            # Use annual means computed with log-space averaging
            annual_data = self._compute_annual_average_log_space()
            
            # Apply BB fraction mask
            bb_mask = annual_data.f_bb.values >= bb_threshold
            
            # Get observations
            obs_mask = np.isfinite(annual_data.OBS_totexttau.values)
            
            # Combine masks
            base_mask = obs_mask & bb_mask
            
            if np.sum(base_mask) < 100:
                self.logger.warning("Insufficient data for biome scatter analysis")
                return
            
            # Get list of biomes (excluding individual biomes that were combined)
            combine_biomes_config = self.config.config.get('combine_biomes', {})
            excluded_biomes = set()
            for combined_name, biome_list in combine_biomes_config.items():
                excluded_biomes.update(biome_list)
            
            # Filter biomes for plotting
            biomes_to_plot = []
            for biome in self.scaling_factors.keys():
                if biome not in excluded_biomes:
                    biome_var = f"{biome}_totexttau"
                    if biome_var in annual_data:
                        biomes_to_plot.append(biome)
            
            self.logger.info(f"Creating scatter plots for biomes: {biomes_to_plot}")
            
            if not biomes_to_plot:
                self.logger.warning("No biomes available for scatter plotting")
                return
            
            # Determine subplot layout
            n_biomes = len(biomes_to_plot)
            if n_biomes <= 2:
                ncols = n_biomes
                nrows = 1
                figsize = (8 * ncols, 6)
            elif n_biomes <= 4:
                ncols = 2
                nrows = 2
                figsize = (16, 12)
            elif n_biomes <= 6:
                ncols = 3
                nrows = 2
                figsize = (24, 12)
            else:
                ncols = 3
                nrows = (n_biomes + 2) // 3  # Ceiling division
                figsize = (24, 6 * nrows)
            
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
            if n_biomes == 1:
                axes = [axes]
            elif nrows == 1:
                axes = axes if n_biomes > 1 else [axes]
            else:
                axes = axes.flatten()
            
            # Get observations for all plots
            obs_flat = annual_data.OBS_totexttau.values[base_mask]
            
            # Colors for different biomes
            biome_colors = plt.cm.tab10(np.linspace(0, 1, n_biomes))
            
            for i, biome in enumerate(biomes_to_plot):
                ax = axes[i]
                biome_var = f"{biome}_totexttau"
                scaling_factor = self.scaling_factors[biome]
                
                # Get biome data
                biome_mask = np.isfinite(annual_data[biome_var].values)
                combined_mask = base_mask & biome_mask
                
                if np.sum(combined_mask) < 50:
                    ax.text(0.5, 0.5, f'Insufficient Data\n({np.sum(combined_mask)} points)', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{biome} (α={scaling_factor:.3f})')
                    continue
                
                # Get data for this biome
                obs_biome = annual_data.OBS_totexttau.values[combined_mask]
                biome_original = annual_data[biome_var].values[combined_mask]
                biome_scaled = biome_original * scaling_factor
                
                # Create scaled total AOD for this biome's contribution
                total_scaled = annual_data.noBB_totexttau.values[combined_mask].copy()
                
                # Add all biome contributions with their scaling factors
                for other_biome, other_factor in self.scaling_factors.items():
                    other_var = f"{other_biome}_totexttau"
                    if other_var in annual_data:
                        other_data = annual_data[other_var].values[combined_mask]
                        total_scaled += other_data * other_factor
                
                # Plotting limits
                max_obs = np.percentile(obs_biome, 99)
                max_biome_orig = np.percentile(biome_original, 99) 
                max_biome_scaled = np.percentile(biome_scaled, 99)
                max_total = np.percentile(total_scaled, 99)
                
                max_plot = min(2.0, max(max_obs, max_biome_orig, max_biome_scaled, max_total))
                
                # Plot 1: Individual biome contribution vs observations
                ax.scatter(obs_biome, biome_scaled, alpha=0.4, s=8, c=biome_colors[i], 
                          label=f'Scaled {biome}', edgecolors='none')
                
                # Plot 2: Total scaled AOD vs observations (fainter)
                ax.scatter(obs_biome, total_scaled, alpha=0.2, s=4, c='gray', 
                          label='Total Scaled', edgecolors='none')
                
                # Reference lines
                ax.plot([0, max_plot], [0, max_plot], 'k--', alpha=0.5, linewidth=1)
                
                # Set limits and labels
                ax.set_xlim(0, max_plot)
                ax.set_ylim(0, max_plot)
                ax.grid(True, alpha=0.3)
                ax.set_xlabel('Observed AOD')
                ax.set_ylabel('Model AOD')
                ax.set_title(f'{biome} (α={scaling_factor:.3f})')
                
                # Calculate statistics
                try:
                    # Statistics for individual biome contribution
                    biome_corr = np.corrcoef(obs_biome, biome_scaled)[0, 1] if len(obs_biome) > 1 else 0
                    biome_rmse = np.sqrt(np.mean((obs_biome - biome_scaled)**2))
                    
                    # Statistics for total scaled AOD
                    total_corr = np.corrcoef(obs_biome, total_scaled)[0, 1] if len(obs_biome) > 1 else 0
                    total_rmse = np.sqrt(np.mean((obs_biome - total_scaled)**2))
                    
                    # Statistics using eisf functions (log-space)
                    try:
                        biome_ratio, _, biome_r = eisf.log_aod_stats(obs_biome, biome_scaled)
                        total_ratio, _, total_r = eisf.log_aod_stats(obs_biome, total_scaled)
                        
                        stats_text = (f'Biome: r²={biome_r[0]**2:.2f}, ratio={biome_ratio:.2f}\n'
                                     f'Total: r²={total_r[0]**2:.2f}, ratio={total_ratio:.2f}\n'
                                     f'Points: {len(obs_biome)}')
                    except:
                        stats_text = (f'Biome: r={biome_corr:.2f}, RMSE={biome_rmse:.3f}\n'
                                     f'Total: r={total_corr:.2f}, RMSE={total_rmse:.3f}\n'
                                     f'Points: {len(obs_biome)}')
                    
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                    
                except Exception as e:
                    self.logger.debug(f"Could not calculate statistics for {biome}: {e}")
                
                # Add legend for first plot
                if i == 0:
                    ax.legend(fontsize=8, loc='lower right')
            
            # Hide unused subplots
            for i in range(n_biomes, len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'Biome-Specific AOD Analysis - BB Regions (f_bb ≥ {bb_threshold}) - {self.year}', 
                        fontsize=16)
            plt.tight_layout()
            
            # Include BB threshold in filename
            biome_scatter_file = output_dir / f"{experiment_basename}_{obs_string}_biome_scatter_bb{bb_threshold}_{self.year}.png"
            plt.savefig(biome_scatter_file, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Saved biome scatter analysis to: {biome_scatter_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating biome scatter plots: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _compute_annual_average_log_space_return_linear(self) -> xr.Dataset:
        """Compute annual average using log-space averaging for AOD variables, but return in linear space."""
        
        self.logger.info("Computing annual average using log-space averaging for AOD variables (returning linear)")
        
        # Initialize result dataset
        annual_avg = xr.Dataset()
        
        # Process each variable
        for var_name in self.data.data_vars:
            if 'totexttau' in var_name:
                # Use eisf function for AOD variables (handles log transformation internally)
                self.logger.debug(f"Computing log-space annual average for {var_name}")
                
                # Get the data for this variable across all months
                var_data = self.data[var_name]  # Shape: [month, lat, lon]
                
                # Create mask for invalid data (NaN values)
                mask_bad = np.isnan(var_data.values)
                
                # Compute mean using eisf function (averages over month dimension)
                # This function returns LINEAR AOD after doing log-space averaging
                mean_aod, std_aod = eisf.aod_mean_std(var_data.values, mask_bad, iax=0)
                
                # Create new DataArray with same coordinates as original (minus month)
                coords_dict = {k: v for k, v in var_data.coords.items() if k != 'month'}
                annual_avg[var_name] = xr.DataArray(
                    mean_aod,  # This is LINEAR AOD (eisf.aod_mean_std returns linear)
                    coords=coords_dict,
                    dims=[d for d in var_data.dims if d != 'month']
                )
                
            elif var_name == 'f_bb':
                # Regular averaging for biomass burning fraction
                annual_avg[var_name] = self.data[var_name].mean(dim='month', skipna=True)
            else:
                # Regular averaging for other variables
                annual_avg[var_name] = self.data[var_name].mean(dim='month', skipna=True)
        
        return annual_avg
    
    def _compute_annual_average_log_space(self) -> xr.Dataset:
        """Compute annual average using log-space averaging for AOD variables (for plotting)."""
        
        self.logger.info("Computing annual average using log-space averaging for AOD variables")
        
        # Initialize result dataset
        annual_avg = xr.Dataset()
        
        # Process each variable
        for var_name in self.data.data_vars:
            if 'totexttau' in var_name:
                # Use eisf function for AOD variables (handles log transformation internally)
                self.logger.debug(f"Computing log-space annual average for {var_name}")
                
                # Get the data for this variable across all months
                var_data = self.data[var_name]  # Shape: [month, lat, lon]
                
                # Create mask for invalid data (NaN values)
                mask_bad = np.isnan(var_data.values)
                
                # Compute mean using eisf function (averages over month dimension)
                mean_aod, std_aod = eisf.aod_mean_std(var_data.values, mask_bad, iax=0)
                
                # Create new DataArray with same coordinates as original (minus month)
                coords_dict = {k: v for k, v in var_data.coords.items() if k != 'month'}
                annual_avg[var_name] = xr.DataArray(
                    mean_aod,
                    coords=coords_dict,
                    dims=[d for d in var_data.dims if d != 'month']
                )
                
            elif var_name == 'f_bb':
                # Regular averaging for biomass burning fraction
                annual_avg[var_name] = self.data[var_name].mean(dim='month', skipna=True)
            else:
                # Regular averaging for other variables
                annual_avg[var_name] = self.data[var_name].mean(dim='month', skipna=True)
        
        return annual_avg
    
    def save_results(self, output_dir: Optional[Path] = None) -> None:
        """Save analysis results to files."""
        if not self.scaling_factors:
            self.logger.warning("No scaling factors to save")
            return
        
        if output_dir is None:
            output_dir = Path(self.config.config['output']['base_directory'])
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract same obs and experiment identifiers
        obs_sources = self.config.get_observation_sources()
        obs_ids = []
        for source in obs_sources:
            if 'MYD04' in self.config.config['observations'][source]['path_template']:
                obs_ids.append('MYD')
            elif 'MOD04' in self.config.config['observations'][source]['path_template']:
                obs_ids.append('MOD')
            else:
                obs_ids.append(source.upper())
        obs_string = '_'.join(sorted(obs_ids))
        
        # Extract experiment basename
        experiment_basename = "c180R_qfed3-2"  # Default
        if self.config.config['model']['experiments']:
            first_exp = list(self.config.config['model']['experiments'].values())[0]
            path_template = first_exp['path_template']
            match = re.search(r'(c\d+R?[^/]*qfed[^/]*?)_', path_template)
            if match:
                experiment_basename = match.group(1)
        
        # Get BB threshold for filename
        bb_threshold = self.config.config['analysis']['biomass_burning_fraction']
        
        # Save as YAML
        results = {
            'scaling_factors': {k: float(v) for k, v in self.scaling_factors.items()},
            'analysis_parameters': {
                'year': self.year,
                'biomass_burning_fraction': bb_threshold,
                'min_alpha_regularization': self.config.config['optimization']['min_alpha_regularization'],
                'log_transformed': True
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Include BB threshold in filename
        yaml_file = output_dir / f"{experiment_basename}_{obs_string}_scaling_factors_bb{bb_threshold}_{self.year}.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        self.logger.info(f"Saved scaling factors to: {yaml_file}")
        
        # Create a simple CSV also
        df = pd.DataFrame({
            'biome': list(self.scaling_factors.keys()),
            'scaling_factor': list(self.scaling_factors.values())
        })
        
        # Include BB threshold in CSV filename
        csv_file = output_dir / f"{experiment_basename}_{obs_string}_scaling_factors_bb{bb_threshold}_{self.year}.csv"
        df.to_csv(csv_file, index=False)
        self.logger.info(f"Saved scaling factors CSV to: {csv_file}")
    
    def plot_results(self, output_dir: Optional[Path] = None) -> None:
        """Create visualizations of results using eisf functions for proper log-space statistics."""
        if not self.scaling_factors or self.data is None:
            self.logger.warning("Missing data or scaling factors for plotting")
            return
        
        if output_dir is None:
            output_dir = Path(self.config.config['output']['base_directory'])
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract identifiers for filenames
        obs_sources = self.config.get_observation_sources()
        obs_ids = []
        for source in obs_sources:
            if 'MYD04' in self.config.config['observations'][source]['path_template']:
                obs_ids.append('MYD')
            elif 'MOD04' in self.config.config['observations'][source]['path_template']:
                obs_ids.append('MOD')
            else:
                obs_ids.append(source.upper())
        obs_string = '_'.join(sorted(obs_ids))
        
        # Extract experiment basename
        experiment_basename = "c180R_qfed3-2"  # Default
        if self.config.config['model']['experiments']:
            first_exp = list(self.config.config['model']['experiments'].values())[0]
            path_template = first_exp['path_template']
            match = re.search(r'(c\d+R?[^/]*qfed[^/]*?)_', path_template)
            if match:
                experiment_basename = match.group(1)
        
        # Get BB threshold for all filenames
        bb_threshold = self.config.config['analysis']['biomass_burning_fraction']
        
        # Plot 1: Bar chart of scaling factors
        plt.figure(figsize=(10, 6))
        biomes = list(self.scaling_factors.keys())
        factors = list(self.scaling_factors.values())
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(biomes)))
        
        bars = plt.bar(biomes, factors, color=colors)
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7)
        
        plt.title(f'Biomass Burning Scaling Factors (Log-Space Analysis) - BB≥{bb_threshold} - {self.year}')
        plt.ylabel('Scaling Factor')
        plt.grid(axis='y', alpha=0.3)
        
        # Add values on top of bars
        for bar, factor in zip(bars, factors):
            plt.text(bar.get_x() + bar.get_width()/2, 
                     bar.get_height() + 0.05, 
                     f'{factor:.2f}', 
                     ha='center')
        
        plt.tight_layout()
        # Include BB threshold in filename
        bar_plot_file = output_dir / f"{experiment_basename}_{obs_string}_scaling_factors_bb{bb_threshold}_{self.year}_bar.png"
        plt.savefig(bar_plot_file, dpi=300)
        plt.close()
        self.logger.info(f"Saved scaling factors bar chart to: {bar_plot_file}")
        
        # Plot 2: Time series plots with debugging
        try:
            # Calculate monthly time series using eisf functions
            months = range(1, 13)
            obs_timeseries = []
            original_timeseries = []
            scaled_timeseries = []
            
            # Get surface area weights
            sa_file = self.config.config.get('surface_area_file')
            if sa_file and Path(sa_file).exists():
                wsa = eisf.get_sa_weight(sa_file, self.data['OBS_totexttau'].shape)
                if wsa.ndim == 3:
                    wsa_2d = wsa.mean(axis=0)
                else:
                    wsa_2d = wsa
            else:
                wsa_2d = np.ones((self.data.sizes['lat'], self.data.sizes['lon']))
            
            self.logger.info(f"Time series calculation - BB threshold: {bb_threshold}")
            
            # Get biome combinations from config to filter out individual biomes
            combine_biomes_config = self.config.config.get('combine_biomes', {})
            
            # Create set of individual biomes that were combined (to exclude from plotting)
            excluded_biomes = set()
            for combined_name, biome_list in combine_biomes_config.items():
                excluded_biomes.update(biome_list)
            
            self.logger.info(f"Excluded individual biomes (part of combinations): {excluded_biomes}")
            
            for month in months:
                month_data = self.data.sel(month=month)
                
                # Debug: Check data availability step by step
                obs_finite = np.isfinite(month_data.OBS_totexttau.values)
                obs_count = np.sum(obs_finite)
                
                if 'allviirs_totexttau' in month_data:
                    model_finite = np.isfinite(month_data.allviirs_totexttau.values)
                    model_count = np.sum(model_finite)
                else:
                    model_finite = obs_finite
                    model_count = obs_count
                
                bb_valid = month_data.f_bb.values >= bb_threshold
                bb_count = np.sum(bb_valid)
                
                combined_mask = obs_finite & model_finite & bb_valid
                combined_count = np.sum(combined_mask)
                
                self.logger.info(f"Month {month}: obs_finite={obs_count}, model_finite={model_count}, bb_valid={bb_count}, combined={combined_count}")
                
                # Try with relaxed criteria
                min_points = 50  # Reduce from 100
                
                if combined_count >= min_points:
                    self.logger.info(f"Month {month}: Using strict criteria ({combined_count} points)")
                    valid_mask = combined_mask
                elif obs_count >= min_points and model_count >= min_points:
                    self.logger.info(f"Month {month}: Using relaxed criteria (no BB filter, {obs_count} points)")
                    valid_mask = obs_finite & model_finite
                elif obs_count >= min_points:
                    self.logger.info(f"Month {month}: Using minimal criteria (obs only, {obs_count} points)")
                    valid_mask = obs_finite
                else:
                    self.logger.warning(f"Month {month}: Insufficient data ({obs_count} obs points)")
                    obs_timeseries.append(np.nan)
                    original_timeseries.append(np.nan)
                    scaled_timeseries.append(np.nan)
                    continue
                
                # Calculate simple averages first (fallback if eisf fails)
                try:
                    # Simple weighted average as fallback
                    weights = wsa_2d[valid_mask]
                    total_weight = np.sum(weights)
                    
                    if total_weight > 0:
                        # Normalize weights
                        weights_norm = weights / total_weight
                        
                        # Observations
                        obs_data = month_data.OBS_totexttau.values[valid_mask]
                        obs_mean = np.sum(obs_data * weights_norm)
                        obs_timeseries.append(float(obs_mean))
                        
                        # Original model (if available)
                        if 'allviirs_totexttau' in month_data:
                            orig_data = month_data.allviirs_totexttau.values[valid_mask]
                            orig_mean = np.sum(orig_data * weights_norm)
                            original_timeseries.append(float(orig_mean))
                        else:
                            original_timeseries.append(np.nan)
                        
                        # Scaled model
                        scaled_data = month_data.noBB_totexttau.values[valid_mask].copy()
                        for biome, factor in self.scaling_factors.items():
                            biome_var = f"{biome}_totexttau"
                            if biome_var in month_data:
                                biome_contribution = month_data[biome_var].values[valid_mask] * factor
                                scaled_data += biome_contribution
                        
                        scaled_mean = np.sum(scaled_data * weights_norm)
                        scaled_timeseries.append(float(scaled_mean))
                        
                        self.logger.info(f"Month {month}: obs={obs_mean:.4f}, orig={orig_mean if 'allviirs_totexttau' in month_data else 'N/A'}, scaled={scaled_mean:.4f}")
                    
                    else:
                        self.logger.warning(f"Month {month}: Zero total weight")
                        obs_timeseries.append(np.nan)
                        original_timeseries.append(np.nan)
                        scaled_timeseries.append(np.nan)
                
                except Exception as e:
                    self.logger.error(f"Month {month}: Error in calculation - {e}")
                    obs_timeseries.append(np.nan)
                    original_timeseries.append(np.nan)
                    scaled_timeseries.append(np.nan)
            
            # Check what we got
            valid_obs = [x for x in obs_timeseries if not np.isnan(x)]
            valid_scaled = [x for x in scaled_timeseries if not np.isnan(x)]
            
            self.logger.info(f"Time series summary: {len(valid_obs)} valid obs points, {len(valid_scaled)} valid scaled points")
            
            if len(valid_obs) == 0:
                self.logger.error("No valid time series data - creating placeholder plot")
                plt.figure(figsize=(14, 8))
                plt.text(0.5, 0.5, 'No Valid Time Series Data\nCheck BB fraction threshold and data availability', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
                plt.title(f'Monthly AOD Time Series - No Data - BB≥{bb_threshold} - {self.year}')
                
            else:
                # Create time series plot
                plt.figure(figsize=(14, 8))
                
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                # Convert to numpy arrays 
                obs_array = np.array(obs_timeseries)
                original_array = np.array(original_timeseries)
                scaled_array = np.array(scaled_timeseries)
                
                plt.plot(months, obs_array, 'ko-', label='Observations', linewidth=3, markersize=8)
                if not np.all(np.isnan(original_array)):
                    plt.plot(months, original_array, 'r^-', label='Original Model', linewidth=2, markersize=7)
                plt.plot(months, scaled_array, 'bs-', label='Scaled Model', linewidth=2, markersize=7)
                
                # Calculate and plot individual biome contributions (scaled) - FIXED
                biome_colors = plt.cm.tab10(np.linspace(0, 1, len(self.scaling_factors)))
                
                for i, (biome, factor) in enumerate(self.scaling_factors.items()):
                    # Skip individual biomes that are part of combinations
                    if biome in excluded_biomes:
                        self.logger.info(f"Skipping individual biome {biome} (part of combination)")
                        continue
                    
                    biome_monthly_contributions = []
                    
                    for month in months:
                        month_data = self.data.sel(month=month)
                        biome_var = f"{biome}_totexttau"
                        
                        if biome_var in month_data:
                            # Use same mask logic as main calculation
                            obs_finite = np.isfinite(month_data.OBS_totexttau.values)
                            obs_count = np.sum(obs_finite)
                            
                            if 'allviirs_totexttau' in month_data:
                                model_finite = np.isfinite(month_data.allviirs_totexttau.values)
                                model_count = np.sum(model_finite)
                            else:
                                model_finite = obs_finite
                                model_count = obs_count
                            
                            bb_valid = month_data.f_bb.values >= bb_threshold
                            combined_mask = obs_finite & model_finite & bb_valid
                            combined_count = np.sum(combined_mask)
                            
                            # Apply same filtering logic
                            if combined_count >= 50:
                                valid_mask = combined_mask
                            elif obs_count >= 50 and model_count >= 50:
                                valid_mask = obs_finite & model_finite
                            elif obs_count >= 50:
                                valid_mask = obs_finite
                            else:
                                biome_monthly_contributions.append(np.nan)
                                continue
                            
                            if np.sum(valid_mask) > 0:
                                weights = wsa_2d[valid_mask]
                                total_weight = np.sum(weights)
                                
                                if total_weight > 0:
                                    weights_norm = weights / total_weight
                                    # Scale the biome contribution by its factor
                                    scaled_biome_data = month_data[biome_var].values[valid_mask] * factor
                                    biome_mean = np.sum(scaled_biome_data * weights_norm)
                                    biome_monthly_contributions.append(float(biome_mean))
                                else:
                                    biome_monthly_contributions.append(np.nan)
                            else:
                                biome_monthly_contributions.append(np.nan)
                        else:
                            biome_monthly_contributions.append(np.nan)
                    
                    # Convert to numpy array and plot
                    biome_array = np.array(biome_monthly_contributions)
                    plt.plot(months, biome_array, '--', color=biome_colors[i], 
                            label=f'{biome} (α={factor:.2f})',
                            linewidth=1.5, markersize=5, alpha=0.7)
                
                # Add basic statistics
                if len(valid_obs) >= 3 and len(valid_scaled) >= 3:
                    valid_indices = ~np.isnan(obs_array) & ~np.isnan(scaled_array)
                    if np.sum(valid_indices) >= 3:
                        obs_subset = obs_array[valid_indices]
                        scaled_subset = scaled_array[valid_indices]
                        
                        # Simple correlation
                        correlation = np.corrcoef(obs_subset, scaled_subset)[0, 1]
                        rmse = np.sqrt(np.mean((obs_subset - scaled_subset)**2))
                        bias = np.mean(scaled_subset - obs_subset)
                        
                        stats_text = f'r = {correlation:.3f}\nRMSE = {rmse:.3f}\nBias = {bias:.3f}'
                        
                        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                plt.xlabel('Month')
                plt.ylabel('AOD')
                plt.title(f'Monthly AOD Time Series (Simple Weighted Averages) - BB≥{bb_threshold} - {self.year}')
                plt.grid(True, alpha=0.3)
                plt.xticks(months, month_names)
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
            
            plt.tight_layout()
            # Include BB threshold in filename
            timeseries_plot_file = output_dir / f"{experiment_basename}_{obs_string}_timeseries_bb{bb_threshold}_{self.year}.png"
            plt.savefig(timeseries_plot_file, dpi=300)
            plt.close()
            self.logger.info(f"Saved time series plot to: {timeseries_plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating time series plot: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

        # Plot 3: Four-panel scatter plot - Linear/Log space, Before/After scaling
        try:
            # Use annual means computed with log-space averaging
            annual_data = self._compute_annual_average_log_space()
            
            # Create scaled model AOD
            annual_scaled = annual_data.noBB_totexttau.copy()
            for biome, factor in self.scaling_factors.items():
                biome_aod_var = f"{biome}_totexttau"
                if biome_aod_var in annual_data:
                    annual_scaled += annual_data[biome_aod_var] * factor
            
            # Apply BB fraction mask
            bb_mask = annual_data.f_bb.values >= bb_threshold
            
            # Get original model data if available
            has_original = 'allviirs_totexttau' in annual_data
            
            if has_original:
                valid_mask = (np.isfinite(annual_data.OBS_totexttau.values) & 
                             np.isfinite(annual_data.allviirs_totexttau.values) &
                             np.isfinite(annual_scaled.values) & bb_mask)
            else:
                valid_mask = (np.isfinite(annual_data.OBS_totexttau.values) & 
                             np.isfinite(annual_scaled.values) & bb_mask)
            
            if np.sum(valid_mask) > 100:
                obs_flat = annual_data.OBS_totexttau.values[valid_mask]
                scaled_flat = annual_scaled.values[valid_mask]
                
                if has_original:
                    orig_flat = annual_data.allviirs_totexttau.values[valid_mask]
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                    panels = ['Linear Before', 'Linear After', 'Log Before', 'Log After']
                else:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    panels = ['Linear After', 'Log After']
                
                # Limit to reasonable AOD range for plotting
                max_aod_linear = min(3.0, max(np.percentile(obs_flat, 99), 
                                             np.percentile(scaled_flat, 99)))
                if has_original:
                    max_aod_linear = min(max_aod_linear, np.percentile(orig_flat, 99))
                
                # Convert to log space for log plots
                obs_log = eisf.lin2log_aod(obs_flat)
                scaled_log = eisf.lin2log_aod(scaled_flat)
                if has_original:
                    orig_log = eisf.lin2log_aod(orig_flat)
                
                max_aod_log = max(np.percentile(obs_log, 99), np.percentile(scaled_log, 99))
                if has_original:
                    max_aod_log = max(max_aod_log, np.percentile(orig_log, 99))
                min_aod_log = min(np.percentile(obs_log, 1), np.percentile(scaled_log, 1))
                if has_original:
                    min_aod_log = min(min_aod_log, np.percentile(orig_log, 1))
                
                if has_original:
                    # Panel 1: Linear space - Original vs Observations
                    ax1.scatter(obs_flat, orig_flat, alpha=0.3, s=1, c='red')
                    ax1.plot([0, max_aod_linear], [0, max_aod_linear], 'k--', alpha=0.7)
                    ax1.set_xlabel('Observed AOD (Linear)')
                    ax1.set_ylabel('Original Model AOD (Linear)')
                    ax1.set_title('Linear Space - Before Scaling')
                    ax1.set_xlim(0, max_aod_linear)
                    ax1.set_ylim(0, max_aod_linear)
                    ax1.grid(True, alpha=0.3)
                    
                    # Add statistics for original (linear space)
                    try:
                        orig_corr = np.corrcoef(obs_flat, orig_flat)[0, 1]
                        orig_rmse = np.sqrt(np.mean((obs_flat - orig_flat)**2))
                        orig_bias = np.mean(orig_flat - obs_flat)
                        ax1.text(0.05, 0.95, f'r = {orig_corr:.3f}\nRMSE = {orig_rmse:.3f}\nBias = {orig_bias:.3f}',
                                transform=ax1.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except:
                        pass
                    
                    # Panel 2: Linear space - Scaled vs Observations
                    ax2.scatter(obs_flat, scaled_flat, alpha=0.3, s=1, c='blue')
                    ax2.plot([0, max_aod_linear], [0, max_aod_linear], 'k--', alpha=0.7)
                    ax2.set_xlabel('Observed AOD (Linear)')
                    ax2.set_ylabel('Scaled Model AOD (Linear)')
                    ax2.set_title('Linear Space - After Scaling')
                    ax2.set_xlim(0, max_aod_linear)
                    ax2.set_ylim(0, max_aod_linear)
                    ax2.grid(True, alpha=0.3)
                    
                    # Add statistics for scaled (linear space)
                    try:
                        scaled_corr = np.corrcoef(obs_flat, scaled_flat)[0, 1]
                        scaled_rmse = np.sqrt(np.mean((obs_flat - scaled_flat)**2))
                        scaled_bias = np.mean(scaled_flat - obs_flat)
                        ax2.text(0.05, 0.95, f'r = {scaled_corr:.3f}\nRMSE = {scaled_rmse:.3f}\nBias = {scaled_bias:.3f}',
                                transform=ax2.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except:
                        pass
                    
                    # Panel 3: Log space - Original vs Observations
                    ax3.scatter(obs_log, orig_log, alpha=0.3, s=1, c='red')
                    ax3.plot([min_aod_log, max_aod_log], [min_aod_log, max_aod_log], 'k--', alpha=0.7)
                    ax3.set_xlabel('Observed AOD (Log Space)')
                    ax3.set_ylabel('Original Model AOD (Log Space)')
                    ax3.set_title('Log Space - Before Scaling')
                    ax3.set_xlim(min_aod_log, max_aod_log)
                    ax3.set_ylim(min_aod_log, max_aod_log)
                    ax3.grid(True, alpha=0.3)
                    
                    # Add log-space statistics for original using eisf function
                    try:
                        orig_ratio, orig_log_diff, orig_r = eisf.log_aod_stats(obs_flat, orig_flat)
                        ax3.text(0.05, 0.95, f'r² = {orig_r[0]**2:.3f}\nratio = {orig_ratio:.3f}',
                                transform=ax3.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except:
                        pass
                    
                    # Panel 4: Log space - Scaled vs Observations
                    ax4.scatter(obs_log, scaled_log, alpha=0.3, s=1, c='blue')
                    ax4.plot([min_aod_log, max_aod_log], [min_aod_log, max_aod_log], 'k--', alpha=0.7)
                    ax4.set_xlabel('Observed AOD (Log Space)')
                    ax4.set_ylabel('Scaled Model AOD (Log Space)')
                    ax4.set_title('Log Space - After Scaling')
                    ax4.set_xlim(min_aod_log, max_aod_log)
                    ax4.set_ylim(min_aod_log, max_aod_log)
                    ax4.grid(True, alpha=0.3)
                    
                    # Add log-space statistics for scaled using eisf function
                    try:
                        scaled_ratio, scaled_log_diff, scaled_r = eisf.log_aod_stats(obs_flat, scaled_flat)
                        ax4.text(0.05, 0.95, f'r² = {scaled_r[0]**2:.3f}\nratio = {scaled_ratio:.3f}',
                                transform=ax4.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except:
                        pass
                        
                    plt.suptitle(f'Model vs Observations Comparison - BB Regions (f_bb ≥ {bb_threshold}) - {self.year}', 
                                fontsize=16)
                
                else:
                    # Only 2 panels if no original model data
                    # Panel 1: Linear space - Scaled vs Observations
                    ax1.scatter(obs_flat, scaled_flat, alpha=0.3, s=1, c='blue')
                    ax1.plot([0, max_aod_linear], [0, max_aod_linear], 'k--', alpha=0.7)
                    ax1.set_xlabel('Observed AOD (Linear)')
                    ax1.set_ylabel('Scaled Model AOD (Linear)')
                    ax1.set_title('Linear Space - After Scaling')
                    ax1.set_xlim(0, max_aod_linear)
                    ax1.set_ylim(0, max_aod_linear)
                    ax1.grid(True, alpha=0.3)
                    
                    # Add linear statistics
                    try:
                        scaled_corr = np.corrcoef(obs_flat, scaled_flat)[0, 1]
                        scaled_rmse = np.sqrt(np.mean((obs_flat - scaled_flat)**2))
                        scaled_bias = np.mean(scaled_flat - obs_flat)
                        ax1.text(0.05, 0.95, f'r = {scaled_corr:.3f}\nRMSE = {scaled_rmse:.3f}\nBias = {scaled_bias:.3f}',
                                transform=ax1.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except:
                        pass
                    
                    # Panel 2: Log space - Scaled vs Observations
                    ax2.scatter(obs_log, scaled_log, alpha=0.3, s=1, c='blue')
                    ax2.plot([min_aod_log, max_aod_log], [min_aod_log, max_aod_log], 'k--', alpha=0.7)
                    ax2.set_xlabel('Observed AOD (Log Space)')
                    ax2.set_ylabel('Scaled Model AOD (Log Space)')
                    ax2.set_title('Log Space - After Scaling')
                    ax2.set_xlim(min_aod_log, max_aod_log)
                    ax2.set_ylim(min_aod_log, max_aod_log)
                    ax2.grid(True, alpha=0.3)
                    
                    # Add log-space statistics using eisf function
                    try:
                        scaled_ratio, scaled_log_diff, scaled_r = eisf.log_aod_stats(obs_flat, scaled_flat)
                        ax2.text(0.05, 0.95, f'r² = {scaled_r[0]**2:.3f}\nratio = {scaled_ratio:.3f}',
                                transform=ax2.transAxes, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except:
                        pass
                        
                    plt.suptitle(f'Scaled Model vs Observations - BB Regions (f_bb ≥ {bb_threshold}) - {self.year}', 
                                fontsize=16)
                
                plt.tight_layout()
                # Include BB threshold in filename
                scatter_plot_file = output_dir / f"{experiment_basename}_{obs_string}_scatter_4panel_bb{bb_threshold}_{self.year}.png"
                plt.savefig(scatter_plot_file, dpi=300)
                plt.close()
                self.logger.info(f"Saved 4-panel scatter plots to: {scatter_plot_file}")
            
            else:
                self.logger.warning("Insufficient valid data for scatter plots")
                
        except Exception as e:
            self.logger.error(f"Error creating 4-panel scatter plots: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

        # Plot 4: Simple BB fraction map
        try:
            annual_data = self._compute_annual_average_log_space()
            
            plt.figure(figsize=(12, 8))
            ax = plt.axes(projection=ccrs.Robinson())
            ax.coastlines()
            ax.gridlines(linestyle='--', alpha=0.5)
            
            # Get coordinates
            if 'lat' in annual_data.coords and 'lon' in annual_data.coords:
                lats = annual_data.lat.values
                lons = annual_data.lon.values
                lon2d, lat2d = np.meshgrid(lons, lats)
                
                bb_plot = ax.pcolormesh(lon2d, lat2d, annual_data.f_bb, 
                                       transform=ccrs.PlateCarree(),
                                       cmap='YlOrRd', vmin=0, vmax=1)
                
                plt.colorbar(bb_plot, ax=ax, orientation='horizontal', 
                            label='Biomass Burning Fraction', pad=0.05)
                
                plt.title(f'Biomass Burning Fraction - Threshold: {bb_threshold} ({self.year})')
                plt.tight_layout()
                
                # Include BB threshold in filename
                bb_plot_file = output_dir / f"{experiment_basename}_{obs_string}_bb_fraction_bb{bb_threshold}_{self.year}_map.png"
                plt.savefig(bb_plot_file, dpi=300)
                plt.close()
                self.logger.info(f"Saved BB fraction map to: {bb_plot_file}")
            
        except Exception as e:
            self.logger.error(f"Error creating BB fraction map: {e}")
        
        # Plot 5: AOD difference map (if original model available)
        try:
            annual_data = self._compute_annual_average_log_space()
            
            # Create scaled model AOD
            scaled_aod = annual_data.noBB_totexttau.copy()
            for biome, factor in self.scaling_factors.items():
                biome_aod_var = f"{biome}_totexttau"
                if biome_aod_var in annual_data:
                    scaled_aod += annual_data[biome_aod_var] * factor
                    
            # Original control AOD (if available)
            if 'allviirs_totexttau' in annual_data:
                original_aod = annual_data.allviirs_totexttau
                
                # Calculate difference
                aod_diff = scaled_aod - original_aod
                
                # Plot difference
                plt.figure(figsize=(12, 8))
                ax = plt.axes(projection=ccrs.Robinson())
                ax.coastlines()
                ax.gridlines(linestyle='--', alpha=0.5)
                
                # Get coordinates
                if 'lat' in annual_data.coords and 'lon' in annual_data.coords:
                    lats = annual_data.lat.values
                    lons = annual_data.lon.values
                    lon2d, lat2d = np.meshgrid(lons, lats)
                    
                    # Create a diverging colormap centered at 0
                    max_abs = max(abs(np.nanmin(aod_diff)), abs(np.nanmax(aod_diff)))
                    norm = mpl.colors.Normalize(vmin=-max_abs, vmax=max_abs)
                    
                    diff_plot = ax.pcolormesh(lon2d, lat2d, aod_diff, 
                                             transform=ccrs.PlateCarree(),
                                             cmap='RdBu_r', norm=norm)
                    
                    plt.colorbar(diff_plot, ax=ax, orientation='horizontal', 
                                label='Scaled AOD - Original AOD', pad=0.05)
                    
                    plt.title(f'AOD Change with Scaling Factors - BB≥{bb_threshold} ({self.year})')
                    plt.tight_layout()
                    
                    # Include BB threshold in filename
                    diff_plot_file = output_dir / f"{experiment_basename}_{obs_string}_aod_diff_bb{bb_threshold}_{self.year}_map.png"
                    plt.savefig(diff_plot_file, dpi=300)
                    plt.close()
                    self.logger.info(f"Saved AOD difference map to: {diff_plot_file}")
            else:
                self.logger.info("No original model data available for difference map")
                
        except Exception as e:
            self.logger.error(f"Error creating AOD difference map: {e}")

        # Plot 6: Biome-specific scatter analysis
        self.plot_biome_scatter_analysis(output_dir)            

def main():
    """Run the AOD scaling analysis."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('scaling_analysis.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting AOD scaling factor analysis with log-transformed AOD")
    
    # Load configuration
    config = ConfigLoader()
    
    # Create analysis object
    analyzer = AODScalingAnalysis(config)
    
    # Load data
    if not analyzer.load_data():
        logger.error("Failed to load data, exiting")
        return
    
    # Run analysis
    alphas = analyzer.run_analysis()
    
    if not alphas:
        logger.error("Analysis failed to produce scaling factors")
        return
    
    # Save results
    analyzer.save_results()
    
    # Create plots
    analyzer.plot_results()
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
