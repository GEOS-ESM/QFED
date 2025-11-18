'''
To run this code, you need 

1. numpy
2. scipy
3. cartopy
4. netCDF4
5. pandas
'''

import argparse
import os, sys, glob, copy
import pandas as pd
from scipy import stats
import numpy as np
from netCDF4 import Dataset
from lib_IGBP_plues import *

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


parser = argparse.ArgumentParser(description="Concatenate MODIS IGBP MCD12Q1")
parser.add_argument("year", help="Product year")
args = parser.parse_args()

year = args.year

FILL_VALUES = 31
num_cells = 2400


SURF_DIR = f'./MCD12Q1/{year}/001/'
out_dir = './GL_IGBP_MODIS/'
os.makedirs(out_dir, exist_ok=True)


hidMax = 35
hidMin = 0
vidMax = 17
vidMin = 0

# update the hid and vid
hids = np.arange(hidMin, hidMax + 1, 1)
vids = np.arange(vidMin, vidMax + 1, 1)

tiles = []
for hid in hids:
    for vid in vids:
        strhid = str(int(hid))
        while len(strhid) < 2:
            strhid = '0' + strhid
        strvid = str(int(vid))
        while len(strvid) < 2:
            strvid = '0' + strvid
        tile = 'h' + strhid + 'v' + strvid
        tiles.append(tile)


PARAMS = ['LC_Type1', 'LW'] #'LC_Type2', 'LC_Type3', 'LC_Type4', 'LC_Type5'
GridDim = ((len(vids)) * num_cells, (len(hids)) * num_cells)

landtype_dict = {}
for param in PARAMS:
    landtype_dict[param] = FILL_VALUES * np.ones(GridDim, np.uint8)


for tile in tiles:
	hid = int(float(tile[1:3]))
	vid = int(float(tile[4:]))
	
	# calculate the coordinates under sinusodal coordinates..
	# they are linear space grids
	hIdx = hid - hidMin 
	vIdx = vid - vidMin
	
	# Then try to find the land surface data...
	# Improvement needs to be make here for after MODIS era...
	filename = glob.glob(SURF_DIR + '*' + tile + '*.hdf')    
	if len(filename) > 0:
		
		ncid = Dataset(filename[0],'r')
		ncid.set_auto_mask(False)
	
		for param in PARAMS:
			landtype_dict[param][vIdx * num_cells : (vIdx + 1) * num_cells, \
								 hIdx * num_cells : (hIdx + 1) * num_cells, ] = ncid[param][:]
		ncid.close()
	else:
		# just for better illustration, fill those coordinates valid but non-existing tile to water
		print(f' - Fill valid coordinates in tile {tile} to water')
		xv, yv  = cal_sinu_xy(tile, num_cells)
		lat, lon = sinu_to_geog(xv, yv)
		valid = (lat<=90) & (lat>=-90) & (lon>=-180) & (lon<=180)
			
		for param in PARAMS:
			
			landtype = water_mapping[param]*np.ones((num_cells, num_cells))
			landtype[~valid] = FILL_VALUES
	
			landtype_dict[param][vIdx * num_cells : (vIdx + 1) * num_cells, \
								 hIdx * num_cells : (hIdx + 1) * num_cells, ] = landtype

print(f' - Done Reading MCD12Q1.')

# set up grids
grid_sinu = SinusoidalGrid(num_cells=num_cells)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
savename = f"{out_dir}GL_IGBP_MODIS.{year}.nc"
ncid = Dataset(savename, 'w', format='NETCDF4' )
ncid.createDimension('easting', grid_sinu.n_zonal)
ncid.createDimension('northing', grid_sinu.n_meridional)

crs_var = ncid.createVariable('crs', 'i4')
crs_var[:] = 0
crs_var.grid_mapping_name = "MODIS/VIIRS Sinusoidal 500 m"
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

tempInstance = ncid.createVariable('easting', 'f8', ('easting'), 
                                   zlib=True, complevel = 4 , 
                                   chunksizes = (grid_sinu.n_zonal,))
tempInstance[:] = grid_sinu.easting[:-1]
tempInstance.standard_name = "easting"
tempInstance.long_name = "easting"
tempInstance.units = "meters"

tempInstance = ncid.createVariable('northing', 'f8', ('northing'), zlib=True, complevel = 4 , chunksizes = (grid_sinu.n_meridional,))
tempInstance[:] = grid_sinu.northing[:-1]
tempInstance.standard_name = "northing"
tempInstance.long_name = "northing"
tempInstance.units = "meters"

chunksizes = (2400, 4800)
complevel  = 8
shuffle    = True

save_params = ['LC_Type1']
for params in save_params:
    
    LT = copy.deepcopy(landtype_dict[params][::,::])
    LW = landtype_dict['LW'][::,::]
    LT[LW==1] = 17
    LT[LT==255] = 31
    
    var = ncid.createVariable(param_mapping[params], 'u1', ('northing', 'easting'), 
                              zlib=True, complevel = complevel, 
                              chunksizes = chunksizes, shuffle = shuffle, fill_value = FILL_VALUES)
    var[:, :] = LT
    
    for att_name, att_val in legend_dict[params].items():
        setattr(var, att_name, att_val)

ncid.description = ( f"{year} MODIS Global Gridded Annual Surface Type (IGBP) from MCD12Q1.061, "
                     "Global Sinusoidal Projection, 500 m, 17 IGBP types")
ncid.Conventions = 'CF', 
ncid.institution = 'Global Modeling and Assimilation Office, NASA/GSFC'
ncid.data_source ='MCD12Q1 Version 6.1 (Aqua/Terra MODIS)'
ncid.primary_documentation = "https://doi.org/10.5067/MODIS/MCD12Q1.061"
ncid.history = 'M. Zhou created this CF compliant global file'
ncid.contact = 'mzhou16@umbc.edu',
ncid.close()
print(f' - Wrote {savename}')




