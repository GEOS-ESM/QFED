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

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import datetime

parser = argparse.ArgumentParser(description="Generate daily grided fire data based on l2 viirs activate fire detection V**14IMG")
parser.add_argument("sat", help="Sensor short name (e.g., VNP)")
parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
parser.add_argument("--fresh_csv", required=True, help="True or False, Create the new daily *.csv file")
args = parser.parse_args()

sat = args.sat
date_start = args.start
date_end = args.end
fresh_csv = args.fresh_csv

year = date_start[0:4]

FILL_VALUES = 255
num_cells = 480
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
l2_dir   = '/Dedicated/jwang-data2/shared_satData/OPNL_FILDA/DATA/LEV1B/' + sat + '14IMG/'
csv_out_dir = f'./Daily_CSV/{sat}{year}/'
nc_out_dir  = f'./Daily_NC/{sat}{year}/'

os.makedirs(csv_out_dir, exist_ok=True)
os.makedirs(nc_out_dir, exist_ok=True)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
date_start = datetime.datetime.strptime(date_start, "%Y-%m-%d").timetuple().tm_yday
date_end = datetime.datetime.strptime(date_end, "%Y-%m-%d").timetuple().tm_yday
jdns = ['A'+year+str(doy).zfill(3) for doy in np.arange(date_start, date_end+1)]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# set up grids
grid_sinu = SinusoidalGrid(num_cells=num_cells)


# read l2 data and save the data into a daily *.csv file
params = ['FP_latitude', 'FP_longitude', 'FP_line', 'FP_sample']	
	
for jdn in jdns:
	
	if fresh_csv == 'True':
		year = jdn[1:5]
		doy  = jdn[5:]
		filenames = glob.glob(l2_dir + year + '/' + doy + '/' + sat + '*' + jdn + '*.nc')

		if len(filenames) <= 0:
			print(f' - Could not find L2 files for {jdn}')
			continue
		
		print( f' - Processing {jdn}, number of L2 files: {len(filenames)} ...')
		dfs = []
		for i, filename in enumerate(filenames):
	
			save_dict = {}
			for param in params:
				save_dict[param] = []
			
			ncid = Dataset(filename, 'r')
			if ncid.FirePix <= 0:
				ncid.close()
				continue

			for param in params:
				save_dict[param] = save_dict[param] + ncid[param][:].tolist()
			algorithm_QA = ncid['algorithm QA'][:]
			ncid.close()
	
			df = pd.DataFrame.from_dict(save_dict, 'columns')
			df['FP_Bowtie'] = is_fire_residual_bowtie(algorithm_QA[ df['FP_line'].astype(int), df['FP_sample'].astype(int)]).astype(int)

			dfs.append(df)
					
		dfs = pd.concat(dfs, ignore_index=True)
		savename = sat + '14IMG.' + jdn + '.csv'
		print(f' - Saving {savename}...')
		dfs.to_csv(csv_out_dir + savename, index = False, na_rep = 'N/A')
	else:
		filenames = sorted(glob.glob(f"{csv_out_dir}*{sat}*{jdn}*.csv"))
		if len(filenames) == 0:
			print(f' - Could not find daily CSV files for {jdn}')
			continue
		else:
			print(f' - Reading {filenames[0]}')
			dfs = pd.read_csv(filenames[0])
		
	# - - - - -
	# bin the data into the given sinu grid
	valid = dfs['FP_Bowtie'] != 1
	
	dfs['ones'] = np.ones_like(dfs['FP_Bowtie']) 
	
	xs, ys = geog_to_sinu(dfs['FP_latitude'][valid].values, 
						  dfs['FP_longitude'][valid].values,)

	one_sum, xedges, yedges, binnumber = stats.binned_statistic_2d(xs, ys,
																   values=dfs['ones'][valid].values,
																   statistic='sum', 
																   bins=[grid_sinu.easting, grid_sinu.northing[::-1]])
																   
	one_sum = one_sum[:, ::-1]
	one_sum[one_sum >0]  = 1
	one_sum[one_sum <=0] = FILL_VALUES

	savename = f"{nc_out_dir}GL_VIIRS_BINARY_OCCURRENCE.{sat}.{jdn}.nc"
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

	var = ncid.createVariable('static_heat_source_mask', 'u1', ('northing', 'easting'), 
							  zlib=True, complevel = complevel, 
							  chunksizes = chunksizes, shuffle = shuffle, fill_value = FILL_VALUES)
	var[:, :] = one_sum.T
	var.long_name = f"Daily binary grided fire occurrence - {sat}14IMG"
	var.legend = "1: detect"
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









