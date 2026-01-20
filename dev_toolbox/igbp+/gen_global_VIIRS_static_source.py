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

parser = argparse.ArgumentParser(description="Generate daily grided fire data based on l2 viirs activate fire detection V**14IMG")
parser.add_argument("sat", help="Sensor short name (e.g., VNP)")
parser.add_argument("--year", required=True, help="Year of analysis YYYY (e.g., 2024)")
args = parser.parse_args()

sat = args.sat
year = args.year

flag_verify = True

in_dir  = f'./Daily_NC/{sat}{year}/'
out_dir = f'./GL_STATIC/'
os.makedirs(out_dir, exist_ok=True)



num_cells = 480
FILL_VALUES = 255
revisit_cycle = 16

# set up grids
grid_sinu = SinusoidalGrid(num_cells=num_cells)

# - - - - - - - - - - - - - - - - - - - - - - - - - 
doys = np.arange(1,367, revisit_cycle)
start_time = time.time()
accumulator = np.zeros((grid_sinu.n_meridional, grid_sinu.n_zonal), dtype=np.int16)
sub_accumulator = np.zeros((grid_sinu.n_meridional, grid_sinu.n_zonal), dtype=np.int16) 

for ini_doy in doys:
    
    sub_accumulator.fill(0)  # reset in-place
    
    for doy in range(ini_doy, ini_doy+revisit_cycle):
        filenames = glob.glob( f'{in_dir}*{year}*{str(doy).zfill(3)}.nc')
        if len(filenames) > 0 :
            
            ncid = Dataset(filenames[0], 'r')
            ncid.set_auto_mask(False)
            feild = ncid['static_heat_source_mask'][:]

            # turn the sub_accumulator to 1 if in revisit_cycle there is one detection found
            valid = (feild>0) & (feild!=255) 

            sub_accumulator[valid] = 1

            ncid.close()
          
    accumulator += sub_accumulator
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f" - {ini_doy} Elapsed time: {elapsed_time:.2f} seconds")

accumulator_save = copy.deepcopy(accumulator)
accumulator_save[np.where(accumulator_save<=0)] = FILL_VALUES


savename = f"{out_dir}GL_VIIRS_HEAT_SOURCE.{sat}.{year}.nc"
ncid = Dataset(savename, 'w', format='NETCDF4' )
ncid.createDimension('easting', grid_sinu.n_zonal)
ncid.createDimension('northing', grid_sinu.n_meridional)

crs_var = ncid.createVariable('crs', 'i4')
crs_var[:] = 0
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


tempInstance = ncid.createVariable('easting', 'f4', ('easting'), zlib=True, complevel = 4 , chunksizes = (grid_sinu.n_zonal,))
tempInstance[:] = grid_sinu.easting[:-1]
tempInstance.standard_name = "easting"
tempInstance.long_name = "easting"
tempInstance.units = "meters"

tempInstance = ncid.createVariable('northing', 'f4', ('northing'), zlib=True, complevel = 4 , chunksizes = (grid_sinu.n_meridional,))
tempInstance[:] = grid_sinu.northing[:-1]
tempInstance.standard_name = "northing"
tempInstance.long_name = "northing"
tempInstance.units = "meters"

chunksizes = (2400, 4800)
complevel  = 8
shuffle    = True

var = ncid.createVariable('heat_source_mask_per_revisiting_cycle', 'u1', ('northing', 'easting'), 
                          zlib=True, complevel = complevel, 
                          chunksizes = chunksizes, shuffle = shuffle, 
                          fill_value = FILL_VALUES)
var[:, :] = accumulator_save.astype(int)
var.long_name = f"Yearly grided heat source per revisiting cycle - {sat}14IMG"
var.valid_range = [0, 255] 
var.grid_mapping = 'crs'

ncid.description = ( f"VIIRS Global Binary Frided Fire Occurrence from {sat}14IMG, "
                     f"Global Sinusoidal Projection, {grid_sinu.resol_h:6.2f}x{grid_sinu.resol_v:6.2f} m")
ncid.Conventions = 'CF', 
ncid.institution = 'Global Modeling and Assimilation Office, NASA/GSFC'
ncid.data_source = f'VIIRS {sat}14IMG Collection 2'
ncid.primary_documentation = f"HTTPS://DOI.ORG/10.5067/VIIRS/{sat}14IMG.002"
ncid.history = 'M. Zhou created this CF compliant global file'
ncid.contact = 'mzhou16@umbc.edu',
ncid.close()
print(f' - Wrote {savename}\n')


if flag_verify:
	savename = f"{out_dir}GL_VIIRS_HEAT_SOURCE.{sat}.{year}.nc"
	# - - - - - - - - - - - - - - - - - - - - - 
	ncid = Dataset(savename, 'r')	
	ncid.set_auto_mask(False)
	gasflaring_source = ncid['heat_source_mask_per_revisiting_cycle'][:]
	northing = ncid['northing'][:]
	easting = ncid['easting'][:]
	ncid.close()

	idx = np.where((gasflaring_source>=16) & (gasflaring_source<255))
	lat, lon = get_coordinates(northing, easting, idx)

	# - - - - - - - - - - - - - - - - - - - - - 
	fig_dir = './FIG/'
	os.makedirs(fig_dir, exist_ok=True)
	
	map_projection = ccrs.Robinson()
	data_projection = ccrs.PlateCarree()
	
	fontsize = 12
	linewidth = 1	
		
	fig = plt.figure(figsize=(10, 5))
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
	
	# make the map global rather than have it zoom in to
	# the extents of any plotted data
	ax.set_global()
	ax.scatter(lon, lat, s = 3, 
			   color = 'purple', edgecolors='white', linewidths=0.1, 
			   zorder = 599, transform=ccrs.PlateCarree())
			   
	ax.set_title(f'QFED 3.2 Static Heat Source Map ({sat}) - Verification')
	
	VOL  = Line2D([0], [0], label='Static Heat Source', 
				  lw = 1, ls='', marker = 'o', 
				  markersize = 8, color=f"purple")
	handles = [VOL]
	plt.legend(handles=handles, frameon = False, 
			   ncol = 3, fontsize = fontsize,
			   loc = 'lower center', bbox_to_anchor=(0.5, -0.1))
			   
	# - - - - - - - - - - - - - - - - - - - - - 	   	
	# the rests are map decoration...
	cord = [90, -90, -180, 180] # N, S, W, E
	numTicks_Y = 6
	numTicks_X = 6
	labelLat = np.linspace(cord[0], cord[1], numTicks_Y)
	labelLon = np.linspace(cord[2], cord[3], numTicks_X)

	waterClr     = (203/255., 236/255., 254/255.)
	waterEdgeClr = (57/255., 193/255., 242/255.)
	landClr      = (212/255., 212/255., 212/255.)

	lineColor = 'dimgray'
	ax.tick_params( labelsize = fontsize )
	ax.add_feature(cfeature.OCEAN, zorder=0, linewidth=linewidth*0.5,
				   edgecolor = waterEdgeClr, color = waterClr, alpha=1)
	
	ax.add_feature(cfeature.LAKES.with_scale('50m'), linewidth=linewidth,
				   edgecolor = waterEdgeClr, 
				   color = waterClr, 
				   alpha=1)
	
	ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=linewidth*.5, \
				   edgecolor = waterEdgeClr)
	
	ax.add_feature(cfeature.BORDERS.with_scale('50m'), linewidth=linewidth*.5, \
				   edgecolor = lineColor, linestyle=':')
	
	ax.gridlines(linewidth=linewidth*0.75, crs=data_projection, xlocs=labelLon, ylocs=labelLat, 
				 edgecolor = lineColor,linestyle=':', zorder=200)
	
	
	ax.add_feature(cfeature.LAND.with_scale('10m'), linewidth=linewidth, 
				   edgecolor = lineColor,color = landClr)


	plt.savefig(f'{fig_dir}MAP.QFED_Static_heat_source.{sat}.{year}.png', dpi = 300)


