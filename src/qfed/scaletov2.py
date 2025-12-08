#!/usr/bin/env python3
"""
QFED Regional Scaling Module

Applies regional scaling factors to QFED biomass emissions for specific species.
"""

import os
import logging
import netCDF4 as nc
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime

# Species that should have scaling applied
SCALABLE_SPECIES = ['oc', 'bc', 'so2', 'nh3']

# Biome variables in QFED files
BIOME_VARS = ['biomass_tf', 'biomass_xf', 'biomass_sv', 'biomass_gl']

def load_scaling_masks(scaling_mask_file):
    """
    Load regional scaling masks from NetCDF file.
    
    Parameters:
    -----------
    scaling_mask_file : str
        Path to the scaling mask NetCDF file
        
    Returns:
    --------
    dict : Dictionary mapping biome names to scaling arrays
    """
    scaling_maps = {}
    
    if not os.path.exists(scaling_mask_file):
        raise FileNotFoundError(f"Scaling mask file not found: {scaling_mask_file}")
    
    logging.info(f"Loading scaling masks from: {scaling_mask_file}")
    
    with nc.Dataset(scaling_mask_file, 'r') as ncfile:
        for biome in BIOME_VARS:
            var_name = f"scaling_{biome}"
            if var_name in ncfile.variables:
                # Remove time dimension if present
                data = ncfile.variables[var_name][:]
                if data.ndim == 3 and data.shape[0] == 1:
                    data = data[0, :, :]
                scaling_maps[biome] = data
                logging.debug(f"Loaded scaling mask for {biome}: range [{np.min(data):.3f}, {np.max(data):.3f}]")
            else:
                logging.warning(f"Scaling mask for {biome} not found in {scaling_mask_file}")
                # Use no scaling (factor of 1.0) if mask not available
                scaling_maps[biome] = None
    
    return scaling_maps

def apply_scaling_to_file(input_file, output_file, scaling_maps, species_name):
    """
    Apply regional scaling to a single QFED file.
    
    Parameters:
    -----------
    input_file : str
        Path to input QFED file
    output_file : str
        Path to output scaled QFED file
    scaling_maps : dict
        Dictionary mapping biome names to scaling arrays
    species_name : str
        Species name for logging
    """
    
    # Create output directory if it doesn't exist
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy the original file to the output location
    shutil.copy2(input_file, output_file)
    logging.debug(f"Copied {input_file} to {output_file}")
    
    # Open the copied file and apply scaling
    with nc.Dataset(output_file, 'r+') as ncfile:
        # Check if this file has biomass variables
        has_biomass_vars = any(var in ncfile.variables for var in BIOME_VARS + ['biomass'])
        
        if not has_biomass_vars:
            logging.debug(f"No biomass variables found in {species_name} file, skipping scaling")
            return
        
        # Apply scaling to each biome
        scaled_total = None
        scaling_applied = False
        
        for biome in BIOME_VARS:
            if biome in ncfile.variables and biome in scaling_maps and scaling_maps[biome] is not None:
                original_data = ncfile.variables[biome][:]
                scaling_map = scaling_maps[biome]
                
                # Apply scaling
                if original_data.ndim == 3:  # time, lat, lon
                    for t in range(original_data.shape[0]):
                        scaled_data = original_data[t] * scaling_map
                        ncfile.variables[biome][t] = scaled_data
                        
                        # Accumulate for total
                        if scaled_total is None:
                            scaled_total = np.zeros_like(original_data)
                        scaled_total[t] += scaled_data
                        
                elif original_data.ndim == 2:  # lat, lon
                    scaled_data = original_data * scaling_map
                    ncfile.variables[biome][:] = scaled_data
                    
                    # Accumulate for total
                    if scaled_total is None:
                        scaled_total = np.zeros_like(original_data)
                    scaled_total += scaled_data
                
                scaling_applied = True
                logging.debug(f"Applied scaling to {biome} in {species_name}")
            
            elif biome in ncfile.variables and scaling_maps[biome] is None:
                # No scaling mask available, add original data to total
                original_data = ncfile.variables[biome][:]
                if scaled_total is None:
                    scaled_total = np.zeros_like(original_data)
                scaled_total += original_data
                logging.debug(f"No scaling mask for {biome}, using original data")
        
        # Update the total biomass variable if scaling was applied
        if scaling_applied and scaled_total is not None and 'biomass' in ncfile.variables:
            ncfile.variables['biomass'][:] = scaled_total
            logging.debug(f"Updated total biomass for {species_name}")
        
        # Update global attributes to indicate scaling
        if scaling_applied:
            ncfile.setncattr('regional_scaling_applied', 'True')
            ncfile.setncattr('scaling_application_date', datetime.now().isoformat())
            ncfile.setncattr('scaling_method', 'Regional geometric mean scaling factors')
            
        logging.info(f"Scaling {'applied' if scaling_applied else 'skipped'} for {species_name}: {os.path.basename(output_file)}")

def apply_regional_scaling(emissions_file_template, timestamp, species_list, scaling_mask_file, 
                         scaled_output_dir=None, version='v3_2_0'):
    """
    Apply regional scaling to QFED emission files for specified species.
    
    Parameters:
    -----------
    emissions_file_template : str
        Template for emissions file path (from config)
    timestamp : datetime
        Date for processing
    species_list : list
        List of all species
    scaling_mask_file : str
        Path to scaling mask NetCDF file
    scaled_output_dir : str, optional
        Output directory for scaled files (if None, replaces original files)
    version : str
        QFED version string
    """
    
    logging.info(f"Applying regional scaling for {timestamp.strftime('%Y-%m-%d')}")
    
    # Load scaling masks
    try:
        scaling_maps = load_scaling_masks(scaling_mask_file)
    except Exception as e:
        logging.error(f"Failed to load scaling masks: {e}")
        return
    
    # Process each species
    for species in species_list:
        if species not in SCALABLE_SPECIES:
            logging.debug(f"Skipping scaling for {species} (not in scalable species list)")
            continue
        
        # Construct input file path
        input_file = cli_utils.get_path(
            emissions_file_template,
            timestamp=timestamp,
            species=species,
            version=version
        )
        
        if not os.path.exists(input_file):
            logging.warning(f"Input file not found: {input_file}")
            continue
        
        # Determine output file path
        if scaled_output_dir:
            # Create scaled version in separate directory
            output_file = input_file.replace(
                os.path.dirname(input_file), 
                os.path.join(scaled_output_dir, os.path.relpath(os.path.dirname(input_file)))
            )
        else:
            # Replace original file (create backup first)
            backup_file = input_file + '.backup'
            if not os.path.exists(backup_file):
                shutil.copy2(input_file, backup_file)
                logging.debug(f"Created backup: {backup_file}")
            output_file = input_file
        
        # Apply scaling
        try:
            apply_scaling_to_file(input_file, output_file, scaling_maps, species)
        except Exception as e:
            logging.error(f"Failed to apply scaling to {species}: {e}")
            continue
    
    logging.info(f"Regional scaling completed for {timestamp.strftime('%Y-%m-%d')}")
