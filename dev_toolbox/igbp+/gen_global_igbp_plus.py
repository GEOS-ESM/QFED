'''
To run this code, you need 

1. numpy
2. scipy
3. cartopy
4. netCDF4
5. pandas
'''

import argparse
import os, sys, glob
import pandas as pd
from scipy import stats
import numpy as np
from netCDF4 import Dataset
from lib_IGBP_plus import *
import time, copy

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


in_modis_dir = './GL_IGBP_MODIS/'
in_plus_dir  = './GL_STATIC/'
out_dir = './IGBP+/'
os.makedirs(out_dir, exist_ok=True)

num_cells = 2400
num_cells_plus = 480

year = '2024'
IGBP_FILE = f'{in_modis_dir}GL_IGBP_MODIS.{year}.nc'

ncid = Dataset(IGBP_FILE)

ncid.set_auto_mask(False)
igbp_surface_type = ncid['surface_type'][:]
northing_full = ncid['northing'][:]
easting_full = ncid['easting'][:]

x_min = np.min(ncid['easting'][:])
dx = abs(np.mean(np.diff(ncid['easting'][:])))

y_max  = np.max(ncid['northing'][:])
dy = abs(np.mean(np.diff(ncid['northing'][:])))

ncid.close()


# - - - - - - - - - - - - - - - - - - - - - - - - - - -  
VGF_FILE = f'{in_plus_dir}GL_VIIRS_GASFLARING.{year}.nc'
ncid = Dataset(VGF_FILE, 'r')
ncid.set_auto_mask(False)
gasflaring_source = ncid['gasflaring_source_mask'][:]
ncid.close()

# - - - - - - - - - - - - - - - - - - - - - - - - - - -  
VCN_FILE = f'{in_plus_dir}GL_GVP_VOLCANO.nc'
ncid = Dataset(VCN_FILE, 'r')
ncid.set_auto_mask(False)
volcano_source = ncid['volcano_mask'][:]
northing = ncid['northing'][:]
easting = ncid['easting'][:]
ncid.close()

# - - - - - - - - - - - - - - - - - - - - - - - - - - -  
VHS_FILE = f'{in_plus_dir}GL_VIIRS_HEAT_SOURCE.VJ1.{year}.nc'
ncid = Dataset(VHS_FILE, 'r')
ncid.set_auto_mask(False)
VJ1_VHS_source = ncid['heat_source_mask_per_revisiting_cycle'][:].astype(float)
ncid.close()
# - - - - - - - - - - - - 
VHS_FILE = f'{in_plus_dir}GL_VIIRS_HEAT_SOURCE.VNP.{year}.nc'
ncid = Dataset(VHS_FILE, 'r')
ncid.set_auto_mask(False)
VNP_VHS_source = ncid['heat_source_mask_per_revisiting_cycle'][:].astype(float)
ncid.close()

# - - - - - - - - - - - - 
invalid = (VJ1_VHS_source>=255)
VJ1_VHS_source[invalid] = np.nan

# - - - - - - - - - - - - 
invalid = (VNP_VHS_source>=255)
VNP_VHS_source[invalid] = np.nan

# - - - - - - - - - - - - 
viirs_heat_source=np.nanmean([VNP_VHS_source, VJ1_VHS_source], axis = 0)

del VNP_VHS_source
del VJ1_VHS_source

# # - - - - - - - - - - - - 
# viirs_heat_source_extned = np.kron(viirs_heat_source, np.ones((3, 3), dtype=np.uint8))
# gasflaring_source_extned = np.kron(gasflaring_source, np.ones((3, 3), dtype=np.uint8))
# volcano_source_extned = np.kron(volcano_source, np.ones((3, 3), dtype=np.uint8))


# - - - - - - - - - - - - - - - - - - - - - - - - - -
# set up grids

grid_sinu = SinusoidalGrid(num_cells=num_cells)

# set up plus grids
grid_sinu_plus = SinusoidalGrid(num_cells=num_cells_plus)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
savename = f"{out_dir}GL_IGBP_PLUS.MODIS.{year}.nc"
ncid = Dataset(savename, 'w', format='NETCDF4' )
ncid.createDimension('easting', grid_sinu.n_zonal)
ncid.createDimension('northing', grid_sinu.n_meridional)


ncid.createDimension('easting_plus', grid_sinu_plus.n_zonal)
ncid.createDimension('northing_plus', grid_sinu_plus.n_meridional)

crs_var = ncid.createVariable('crs', 'i4')
crs_var.grid_mapping_name = f"MODIS/VIIRS Sinusoidal {grid_sinu.resol_h:6.2f}x{grid_sinu.resol_v:6.2f} m"
crs_var.long_name = "CRS definition"
crs_var.epsg_code = "EPSG:4326"  # WGS84 standard
crs_var.false_easting = "0.0";
crs_var.false_northing = "0.0";
crs_var.GeoTransform = f"{-grid_sinu.halfHoriLength} {grid_sinu.resol_h} -0 {grid_sinu.halfVertLength} -0 -{grid_sinu.resol_v} ";
crs_var.pixel_coordinate_location = "pixel_upper_left_corner";
crs_var.spatial_ref = ( "{PROJCS[\"Sinusoidal\",GEOGCS[\"GCS_ELLIPSE_BASED_1\","
                        "DATUM[\"D_ELLIPSE_BASED_1\",SPHEROID[\"S_ELLIPSE_BASED_1\","
                        "6371007.2,0.0]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\","
                        "0.0174532925199433]],PROJECTION[\"Sinusoidal\"],"
                        "PARAMETER[\"False_Easting\",0.0],"
                        "PARAMETER[\"False_Northing\",0.0],"
                        "PARAMETER[\"Central_Meridian\",0.0],UNIT[\"Meter\",1.0]]}"
                       )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
crs_plus_var = ncid.createVariable('crs_plus', 'i4')
crs_plus_var.grid_mapping_name = f"MODIS/VIIRS Sinusoidal {grid_sinu_plus.resol_h:6.2f}x{grid_sinu_plus.resol_v:6.2f} m"
crs_plus_var.long_name = "CRS definition for IGBP Plus dataset"
crs_plus_var.epsg_code = "EPSG:4326"  # WGS84 standard
crs_plus_var.false_easting = "0.0";
crs_plus_var.false_northing = "0.0";
crs_plus_var.GeoTransform = f"{-grid_sinu_plus.halfHoriLength} {grid_sinu_plus.resol_h} -0 {grid_sinu_plus.halfVertLength} -0 -{grid_sinu_plus.resol_v} ";
crs_plus_var.pixel_coordinate_location = "pixel_upper_left_corner";
crs_plus_var.spatial_ref = ( "{PROJCS[\"Sinusoidal\",GEOGCS[\"GCS_ELLIPSE_BASED_1\","
                             "DATUM[\"D_ELLIPSE_BASED_1\",SPHEROID[\"S_ELLIPSE_BASED_1\","
                             "6371007.2,0.0]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\","
                             "0.0174532925199433]],PROJECTION[\"Sinusoidal\"],"
                             "PARAMETER[\"False_Easting\",0.0],"
                             "PARAMETER[\"False_Northing\",0.0],"
                             "PARAMETER[\"Central_Meridian\",0.0],UNIT[\"Meter\",1.0]]}"
                             )

# - - - - - - - - - - - -
tempInstance = ncid.createVariable('easting', 'f8', ('easting'), zlib=True, complevel = 4 , chunksizes = (grid_sinu.n_zonal,))
tempInstance[:] = grid_sinu.easting[:-1]
tempInstance.standard_name = "easting"
tempInstance.long_name = "easting for IGBP land cover type"
tempInstance.units = "meters"

tempInstance = ncid.createVariable('northing', 'f8', ('northing'), zlib=True, complevel = 4 , chunksizes = (grid_sinu.n_meridional,))
tempInstance[:] = grid_sinu.northing[:-1]
tempInstance.standard_name = "northing"
tempInstance.long_name = "northing for IGBP land cover type"
tempInstance.units = "meters"

# - - - - - - - - - - - -
tempInstance = ncid.createVariable('easting_plus', 'f8', ('easting_plus'), zlib=True, complevel = 4 , chunksizes = (grid_sinu_plus.n_zonal,))
tempInstance[:] = grid_sinu_plus.easting[:-1]
tempInstance.standard_name = "easting plus"
tempInstance.long_name = "easting for IGBP plus dataset"
tempInstance.units = "meters"

tempInstance = ncid.createVariable('northing_plus', 'f8', ('northing_plus'), zlib=True, complevel = 4 , chunksizes = (grid_sinu_plus.n_meridional,))
tempInstance[:] = grid_sinu_plus.northing[:-1]
tempInstance.standard_name = "northing plus"
tempInstance.long_name = "northing for IGBP plus dataset"
tempInstance.units = "meters"


chunksizes = (2400, 4800)
complevel  = 4
shuffle    = True

# - - - - - - - - - - - -
# surface_type
var = ncid.createVariable('surface_type', 'u1', ('northing', 'easting'), 
                          zlib=True, complevel = complevel, 
                          chunksizes = chunksizes, shuffle = shuffle, fill_value = 31)
var[:, :] = igbp_surface_type
for att_name, att_val in legend_dict['LC_Type1'].items():
    setattr(var, att_name, att_val)

# - - - - - - - - - - - -
# static_heat_mask
var = ncid.createVariable('static_heat_mask', 'u1', ('northing_plus', 'easting_plus'), 
                          zlib=True, complevel = complevel, 
                          chunksizes = chunksizes, shuffle = shuffle, fill_value = 255)
var[:, :] = viirs_heat_source
var.long_name = f"VIIRS Global Grided Static Heat Source Occurrence per Revisiting Cycle {year}"
var.valid_range = [0, 255] 
var.grid_mapping = 'crs_plus'
var.comment = 'Suggested threshold: >= 16'

# - - - - - - - - - - - -
# gasflaring_mask
var = ncid.createVariable('gasflaring_mask', 'u1', ('northing_plus', 'easting_plus'), 
                          zlib=True, complevel = complevel, 
                          chunksizes = chunksizes, shuffle = shuffle, fill_value = 255)
var[:, :] = gasflaring_source
var.long_name = f"VIIRS Global Nighttime Gasflaring Classification {year}"
var.legend = "1: detect"
var.valid_range = [0, 255] 
var.grid_mapping = 'crs_plus'

# - - - - - - - - - - - -
# volcano_mask
var = ncid.createVariable('volcano_mask', 'u1', ('northing_plus', 'easting_plus'), 
                          zlib=True, complevel = complevel, 
                          chunksizes = chunksizes, shuffle = shuffle, fill_value = 255)
var[:, :] = volcano_source
var.long_name = f"Global Volcanism Program Global Volcanol List - Holocene period"
var.legend = "1: detect"
var.valid_range = [0, 255] 
var.grid_mapping = 'crs_plus'


# - - - - - - - - - - - -
ncid.description = ( f"VIIRS Global Gridded Annual Surface Type (IGBP), {year},  "
                     "Global Sinusoidal Projection")
ncid.Conventions = 'CF', 
ncid.institution = 'Global Modeling and Assimilation Office, NASA/GSFC'

ncid.igbp_data_source ='MCD12Q1 Version 6.1 (Aqua/Terra MODIS)'
ncid.igbp_primary_documentation = "https://doi.org/10.5067/MODIS/MCD12Q1.061"

ncid.static_heat_source_data_source = f'VIIRS Active Fire Detection Collection 2'
ncid.SHH_primary_documentation = f"HTTPS://DOI.ORG/10.5067/VIIRS/VIIRS14IMG.002"

ncid.volcano_data_source = f'The Global Volcanism Program'
ncid.GVP_primary_documentation = f"https://volcano.si.edu/volcanolist_holocene.cfm"

ncid.viirs_gasflaring_data_source = f'VIIRS Annual Gas Flared Volume'
ncid.VGF_primary_documentation = f"https://eogdata.mines.edu/products/vnf/global_gas_flare.html"


ncid.history = 'M. Zhou created this CF compliant global file'
ncid.contact = 'mzhou16@umbc.edu',
ncid.close()
print(f' - Wrote {savename}\n')