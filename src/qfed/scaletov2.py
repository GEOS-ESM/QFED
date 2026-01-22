#!/usr/bin/env python3
"""
QFED Scaling Module

Applies scaling factors to QFED aerosol emissions for each biome using a precomputed mask.
The mask must match the resolution of the emissions.
"""

import os
import logging
import netCDF4 as nc
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime
from qfed import cli_utils


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
        Path to output scaled QFED file (same as input_file for in-place modification)
    scaling_maps : dict
        Dictionary mapping biome names to scaling arrays
    species_name : str
        Species name for logging
    """
    #Check if scaling has already been applied
    # Open the file first to check if scaling has already been applied
    with nc.Dataset(input_file, 'r') as check_file:
        # Safety check: verify scaling hasn't already been applied
        if hasattr(check_file, 'regional_scaling_applied'):
            scaling_status = getattr(check_file, 'regional_scaling_applied')
            if scaling_status == 'True' or scaling_status is True:
                logging.warning(f"Regional scaling already applied to {species_name} file: {os.path.basename(input_file)}")
                logging.info(f"Scaling application date: {getattr(check_file, 'scaling_application_date', 'Unknown')}")
                return    
    # Save original data before applying the scaling
    originaldata_file = input_file + '.original'
    if not os.path.exists(originaldata_file):
        shutil.copy2(input_file, originaldata_file)
        logging.debug(f"Created copy of original data: {originaldata_file}")
    
    # Open the file and apply scaling directly
    with nc.Dataset(input_file, 'r+') as ncfile:
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
            
        logging.info(f"Scaling {'applied' if scaling_applied else 'skipped'} for {species_name}: {os.path.basename(input_file)}")

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
        Output directory for scaled files (currently unused - scaling is done in-place)
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
        
        # Apply scaling in-place
        try:
            apply_scaling_to_file(input_file, input_file, scaling_maps, species)
        except Exception as e:
            logging.error(f"Failed to apply scaling to {species}: {e}")
            continue
    
    logging.info(f"Regional scaling completed for {timestamp.strftime('%Y-%m-%d')}")
