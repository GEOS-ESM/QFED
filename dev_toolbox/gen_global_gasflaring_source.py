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

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
		

in_dir = './GAS_FLARING_SOURCE_DATA/'
out_dir = './GL_STATIC/'
os.makedirs(out_dir, exist_ok=True)

FILL_VALUES = 255
flag_verify = True


filenames = {}
filenames['2012']='VIIRS_Global_flaring_d.7_slope_0.0298_2012-2016_web.xlsx'
filenames['2013']='VIIRS_Global_flaring_d.7_slope_0.0298_2012-2016_web.xlsx'
filenames['2014']='VIIRS_Global_flaring_d.7_slope_0.0298_2012-2016_web.xlsx'
filenames['2015']='VIIRS_Global_flaring_d.7_slope_0.0298_2012-2016_web.xlsx'
filenames['2016']='VIIRS_Global_flaring_d.7_slope_0.0298_2012-2016_web.xlsx'
filenames['2017']='VIIRS_Global_flaring_d.7_slope_0.029353_2017_web_v1.xlsx'
filenames['2018']='VIIRS_Global_flaring_d.7_slope_0.029353_2018_web.xlsx'
filenames['2019']='VIIRS_Global_flaring_d.7_slope_0.029353_2019_web_v20201114.xlsx'
filenames['2020']='VIIRS_Global_flaring_d.7_slope_0.029353_2020_web_v1.xlsx'
filenames['2021']='VIIRS_Global_flaring_d.7_slope_0.029353_2021_web.xlsx'
filenames['2022']='VIIRS_Global_flaring_d.7_slope_0.029353_2022_v20230526_web.xlsx'
filenames['2023']='VIIRS_Global_flaring_d.7_slope_0.029353_2023_v20230614_web_IDmatch.xlsx'
filenames['2024']='VIIRS_Global_flaring_d.7_slope_0.029353_2024_v20240730_web_IDmatch.xlsx'


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Read the global volcano program volcano list
# https://volcano.si.edu/volcanolist_holocene.cfm
dfs = {}
for year in filenames.keys():
    print(f' - Processing {year}')
    data_dict = {}
    data = pd.ExcelFile(in_dir + filenames[year])
    for sheet_name in data.sheet_names:
        temp_data = data.parse(sheet_name)

        if  ('Latitude' in temp_data.keys()) & ('Longitude' in temp_data.keys()):
            print( ' - Appeding dataset', sheet_name)
            data_dict[sheet_name] = temp_data
            
    lats = []
    lons = []
    for key in data_dict.keys():
        lats = lats + data_dict[key]['Latitude'].values.tolist()
        lons = lons + data_dict[key]['Longitude'].values.tolist()
        
    save_dict = {}
    save_dict['latitude'] = lats
    save_dict['longitude'] = lons
    
    dfs[year] = save_dict

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# processing

# set up grids
grid_sinu = SinusoidalGrid(num_cells=480)

# processing...bin data into grids
for year in dfs.keys():
	
	
	xs, ys = geog_to_sinu(np.array(dfs[year]['latitude']),
						  np.array(dfs[year]['longitude']))
	print(f' - Processing {year} - {len(xs)}') 
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
	savename = f"{out_dir}GL_VIIRS_GASFLARING.{year}.nc"
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
	
	var = ncid.createVariable('gasflaring_source_mask', 'u1', ('northing', 'easting'), 
							  zlib=True, complevel = complevel, 
							  chunksizes = chunksizes, shuffle = shuffle, fill_value = FILL_VALUES)
	var[:, :] = one_sum.T
	var.long_name = f"Global gasflaring flag"
	var.legend = "1: detect"
	var.valid_range = [0, 255] 
	var.grid_mapping = 'crs'
	
	ncid.description = ( f"VIIRS Global Nighttime Gasflaring Classification, "
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
		# - - - - - - - - - - - - - - - - - - - - - 
		ncid = Dataset(savename, 'r')	
		ncid.set_auto_mask(False)
		gasflaring_source = ncid['gasflaring_source_mask'][:]
		northing = ncid['northing'][:]
		easting = ncid['easting'][:]
		ncid.close()
	
		idx = np.where(gasflaring_source==1)
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
				   color = 'orange', edgecolors='white', linewidths=0.1, 
				   zorder = 599, transform=ccrs.PlateCarree())
				   
		ax.set_title(f'QFED 3.2 Volcano Source Map (Static) - Verification')
		
		VOL  = Line2D([0], [0], label='Volcano', 
					  lw = 1, ls='', marker = 'o', 
					  markersize = 8, color=f"orange")
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
	
	
		plt.savefig(f'{fig_dir}MAP.QFED_Gasflaring.{year}.png', dpi = 300)







