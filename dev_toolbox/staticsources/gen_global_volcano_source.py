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
from lib_IGBP_plues import *



data_path = './GVP_Volcano_List/'
out_dir = './GL_STATIC/'
os.makedirs(out_dir, exist_ok=True)

FILL_VALUES = 255
flag_verify = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Read the global volcano program volcano list
# https://volcano.si.edu/volcanolist_holocene.cfm
df = pd.read_excel(data_path + 'GVP_Volcano_List_Holocene.xlsx', sheet_name='Sheet1')

# set up grids
grid_sinu = SinusoidalGrid(num_cells=480)

# processing...bin data into grids
xs, ys = geog_to_sinu(df['Latitude'].values, df['Longitude'].values)
ones= np.ones_like(xs)
one_sum, xedges, yedges, binnumber = stats.binned_statistic_2d(xs, ys, 
                                                               values=ones,
                                                               statistic='sum', 
                                                               bins=[grid_sinu.easting, grid_sinu.northing[::-1]])   
one_sum = one_sum[:, ::-1]
one_sum[one_sum >0]  = 1
one_sum[one_sum <=0] = FILL_VALUES

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# write the data into a temporary *.nc file...
savename = f"{out_dir}GL_GVP_VOLCANO.nc"
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

tempInstance = ncid.createVariable('easting', 'f8', ('easting'), 
                                   zlib=True, complevel = 4, 
                                   chunksizes = (grid_sinu.n_zonal,))
tempInstance[:] = grid_sinu.easting[:-1]
tempInstance.standard_name = "easting"
tempInstance.long_name = "easting"
tempInstance.units = "meters"

tempInstance = ncid.createVariable('northing', 'f8', ('northing'), 
                                    zlib=True, complevel = 4 , 
                                    chunksizes = (grid_sinu.n_meridional,))
tempInstance[:] = grid_sinu.northing[:-1]
tempInstance.standard_name = "northing"
tempInstance.long_name = "northing"
tempInstance.units = "meters"

chunksizes = (2400, 4800)
complevel  = 8
shuffle    = True

var = ncid.createVariable('volcano_mask', 'u1', ('northing', 'easting'), 
                          zlib=True, complevel = complevel, 
                          chunksizes = chunksizes, shuffle = shuffle, 
                          fill_value = FILL_VALUES)
var[:, :] = one_sum.T
var.long_name = f"Global Volcanol List - Holocene period"
var.legend = "1: detect"
var.valid_range = [0, 255] 
var.grid_mapping = 'crs'

ncid.description = ( f"The Global Volcanism Program database currently contains 1,230 volcanoes with eruptions during the Holocene period, "
                     f"Global Sinusoidal Projection, {grid_sinu.resol_h:6.2f}x{grid_sinu.resol_v:6.2f} m")
ncid.Conventions = 'CF', 
ncid.institution = 'Global Modeling and Assimilation Office, NASA/GSFC'
ncid.data_source = f'Annual Gas Flared Volume'
ncid.primary_documentation = f"https://volcano.si.edu/volcanolist_holocene.cfm"
ncid.history = 'M. Zhou created this CF compliant global file'
ncid.contact = 'mzhou16@umbc.edu',
ncid.close()
print(f' - Wrote {savename}\n')


if flag_verify:
	import cartopy.crs as ccrs
	import cartopy.feature as cfeature
	import matplotlib.pyplot as plt
	from matplotlib.lines import Line2D
	
	# - - - - - - - - - - - - - - - - - - - - - 
	ncid = Dataset(savename, 'r')	
	ncid.set_auto_mask(False)
	volcano_source = ncid['volcano_mask'][:]
	northing = ncid['northing'][:]
	easting = ncid['easting'][:]
	ncid.close()

	idx = np.where(volcano_source==1)
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
			   color = 'crimson', edgecolors='white', linewidths=0.1, 
			   zorder = 599, transform=ccrs.PlateCarree())
			   
	ax.set_title(f'QFED 3.2 Volcano Source Map (Static) - Verification')
	
	VOL  = Line2D([0], [0], label='Volcano', 
				  lw = 1, ls='', marker = 'o', 
				  markersize = 8, color=f"crimson")
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


	plt.savefig(f'{fig_dir}MAP.QFED_Volcano.png', dpi = 300)







