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
import matplotlib.dates as mdates
from datetime import datetime, timedelta

base_path='./'

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

def validate_satellites(satellites: List[str]) -> Tuple[List[str], List[str]]:
	"""Validate satellite codes"""
	valid = [sat for sat in satellites if sat in SATELLITES]
	invalid = [sat for sat in satellites if sat not in SATELLITES]
	return valid, invalid


def _load_single_day(args):
	"""Helper function for parallel loading"""
	date, satellites = args
	daily_results = {}
	
	for sat in satellites:
		frp_density = _load_daily_data(sat, date)
		daily_results[sat] = {}
		
		for biome in FRP_VARS:
			daily_results[sat][biome] = frp_density[biome] # global_mean
	
	return date, daily_results

def _get_file_path(satellite: str, date: datetime) -> str:
	"""Generate file path for given satellite and date"""
	year, month, day = date.strftime('%Y'), date.strftime('%m'), date.strftime('%d')
	
	# Handle special naming for Aqua MODIS
	satellite = 'MYD14' if satellite == 'myd' else satellite
	satellite = 'MOD14' if satellite == 'mod' else satellite
	filename = f"qfed3_2.frp.{satellite}.{year}{month}{day}.nc4"
	return os.path.join(base_path, f"Y{year}", f"M{month}", filename)


def _load_daily_data(satellite: str, date: datetime) -> Optional[Dict]:
	"""Optimized daily data loading"""
	filepath = _get_file_path(satellite, date)
	
	if not os.path.exists(filepath):
		return None
	try:
		# Use chunks and only load what we need
		with xr.open_dataset(filepath, chunks={'time': 1}) as ds:
			# Only load required variables
			required_vars = ['lat', 'lon', 'land'] + FRP_VARS
			available_vars = [var for var in required_vars if var in ds.variables]
			
			ds_subset = ds[available_vars]
			
			lat, lon = ds_subset['lat'].values, ds_subset['lon'].values
			
			frp_density = {}

			if 'land' in ds_subset.variables:
				land = ds_subset['land'].values[0]
				
				for frp_var in FRP_VARS:
					if frp_var in ds_subset.variables:
						frp = ds_subset[frp_var].values[0]
						
						valid_mask = (frp < 1e19) & (land < 1e19) & (land > 0)
						density = np.full_like(frp, np.nan)
						
						if np.any(valid_mask):
							density = np.nansum(frp[valid_mask]) / np.nansum(land[valid_mask])
							frp_density[frp_var] = density
						else:
							frp_density[frp_var] = np.nan
			
			return frp_density
			
	except Exception as e:
		print(f"Error loading {filepath}: {e}")
		return None

def load_time_series_by_biome_parallel(satellites: List[str], start_date: datetime, 
									   end_date: datetime, n_workers: int = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
	"""Load time series data using parallel processing"""
	
	valid_satellites, invalid_satellites = validate_satellites(satellites)
	
	if invalid_satellites:
		print(f"Warning: Invalid satellite codes: {invalid_satellites}")
	
	if not valid_satellites:
		print("Error: No valid satellites specified!")
		return {}, {}
	
	print(f"Processing satellites: {[f'{sat} ({SATELLITES[sat]})' for sat in valid_satellites]}")
	
	date_range = pd.date_range(start_date, end_date, freq='D')
	
	# Use all available cores minus 1 by default
	if n_workers is None:
		n_workers = max(1, mp.cpu_count() - 1)
	n_workers = 10
	print(f"Using {n_workers} parallel workers to process {len(date_range)} days...")
	
	# Prepare arguments for parallel processing
	args_list = [(date, valid_satellites) for date in date_range]
	
	# Initialize data structure
	biome_data = {biome: {sat: [np.nan] * len(date_range) for sat in valid_satellites} for biome in FRP_VARS}
	
	# Process in parallel
	with ProcessPoolExecutor(max_workers=n_workers) as executor:
		# Submit all jobs
		future_to_idx = {executor.submit(_load_single_day, args): idx for idx, args in enumerate(args_list)}
		
		# Collect results
		completed = 0
		for future in as_completed(future_to_idx):
			idx = future_to_idx[future]
			try:
				date, daily_results = future.result()
				
				# Store results
				for sat in valid_satellites:
					for biome in FRP_VARS:
						if sat in daily_results:
							biome_data[biome][sat][idx] = daily_results[sat].get(biome, np.nan)
				
				completed += 1
				if completed % 50 == 0 or completed == len(date_range):
					print(f"  Completed {completed}/{len(date_range)} days...")
					
			except Exception as e:
				print(f"Error processing day {idx}: {e}")
	
	# Convert to DataFrames
	biome_dataframes = {}
	for biome in FRP_VARS:
		df = pd.DataFrame(biome_data[biome], index=date_range)
		biome_dataframes[biome] = df
		
		print(f"\n{BIOMES[biome]} data loading summary:")
		print(f"  DataFrame shape: {df.shape}")
		print(f"  Columns: {list(df.columns)}")
	
	return biome_dataframes, BIOMES
	

	
def multiFigure(nRow, nCol, **kwargs):
    """
    Simple multi-panel figure helper (no projections, no annotations).

    Parameters
    ----------
    nRow, nCol : int
        Grid layout (rows, cols).
    figsize : tuple, optional
        Figure size in inches, default (9, 9).
    gap : float, optional
        Spacing between subplots (passed to wspace/hspace), default 0.08.
    sharex, sharey : bool, optional
        Share axes among subplots, default False.
    nPlot : int or None, optional
        If given, only the first nPlot axes are shown; the rest are hidden.
        Panels are filled row-major.
    as_array : bool, optional
        If True, return a 2D numpy array of axes shaped (nRow, nCol);
        otherwise return a flat list (default False).
    left, right, top, bottom : float, optional
        Figure margins for subplots_adjust. Defaults: 0.08, 0.98, 0.95, 0.08.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list[Axes] or ndarray of Axes
        Visible axes in row-major order (list), or a full 2D array if as_array=True.
        Hidden axes (when nPlot < nRow*nCol) remain in the figure but are set invisible.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    figsize  = kwargs.get('figsize', (9, 9))
    gap      = kwargs.get('gap', 0.08)
    sharex   = kwargs.get('sharex', False)
    sharey   = kwargs.get('sharey', False)
    nPlot    = kwargs.get('nPlot', None)
    as_array = kwargs.get('as_array', False)

    left   = kwargs.get('left', 0.08)
    right  = kwargs.get('right', 0.92)
    top    = kwargs.get('top', 0.92)
    bottom = kwargs.get('bottom', 0.08)

    fig, axs = plt.subplots(nRow, nCol, figsize=figsize, sharex=sharex, sharey=sharey)

    # Normalize axs into a 2D ndarray for consistent handling
    if nRow * nCol == 1:
        axs2d = np.array([[axs]])
    elif isinstance(axs, np.ndarray) and axs.ndim == 2:
        axs2d = axs
    else:
        # 1D array (when one dimension is 1) -> reshape
        axs2d = np.array(axs).reshape(nRow, nCol)

    # Spacing and margins
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom,
                        wspace=gap, hspace=gap)

    total = nRow * nCol
    if nPlot is not None:
        nPlot = max(0, min(int(nPlot), total))
        flat = axs2d.ravel()
        # Hide any extra panels beyond nPlot
        for k in range(nPlot, total):
            flat[k].set_visible(False)
        visible = [flat[k] for k in range(nPlot)]
    else:
        visible = list(axs2d.ravel())

    return (fig, axs2d if as_array else visible)
    
    
def scatter(ax, in_x_data, in_y_data, **kwargs):
    ''' 
    Function to create scatter plots
    '''
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    import scipy
    from scipy import stats
    from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                                   AutoMinorLocator)
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    stdOn	     = kwargs.get('stdOn', False)
    regress_line = kwargs.get('regress_line', True)
    one2one_line  = kwargs.get('one2one_line', True)
    ee_line  = kwargs.get('ee_line', True)

    label_p		= kwargs.get('label_p', None)	
    yerr		= kwargs.get('yerr', 0)
    xerr		= kwargs.get('xerr', 0)
    markersize	= kwargs.get('markersize', 4)
    elinewidth	= kwargs.get('elinewidth', 1)
    capsize		= kwargs.get('capsize', 2)
    color		= kwargs.get('color', 'black')
    statOn		= kwargs.get('statOn', True)
    fmt			= kwargs.get('fmt', 'o')
    fontsize    = kwargs.get('fontsize', 12)
    ylabel	    = kwargs.get('ylabel', '')
    xlabel	    = kwargs.get('xlabel', '')
    rmse_coef_flag	 = kwargs.get('rmse_coef_flag', True)
    mae_coef_flag	 = kwargs.get('mae_coef_flag', True)


    reg_color	 = kwargs.get('reg_color', 'k')
    eeOn  	    = kwargs.get('eeOn', False)
    eeBias 	    = kwargs.get('eeBias', 0.1)
    eeFrac 	    = kwargs.get('eeFrac', 0.15)
    alpha 	    = kwargs.get('alpha', 1)
    case_str 	= kwargs.get('case_str', None)

    dp_x        = kwargs.get('dp_x', 'x')
    dp_y        = kwargs.get('dp_x', 'y')


    ee_line_color = kwargs.get('ee_line_color', 'C1')

    xRange	= kwargs.get('xRange', None)
    yRange	= kwargs.get('yRange', None)


    delta_ticks_X = kwargs.get('delta_ticks_X', None)
    delta_ticks_Y = kwargs.get('delta_ticks_Y', None)

    model = kwargs.get('model', 'OLS')


    ano_box = kwargs.get('ano_box', None)

    zorder = kwargs.get('zorder', 100)


    if ano_box is not None:
        ano_box=dict(boxstyle="square", fc="1", alpha = 0.5, edgecolor= 'None')

    # copy and flatten data
    x_data = in_x_data.flatten()
    y_data = in_y_data.flatten()

    #-- get a mask with those elements posible to compare (non-nan values)
    mask = np.logical_and(np.logical_not(np.isnan(x_data)), np.logical_not(np.isnan(y_data)))
    n_colocations = len(mask[mask==True])
    x_data = x_data[mask]
    y_data = y_data[mask]

    #-- liner regression
    if model == 'OLS':
#         print( ' - Using OLS model...')
        slope, intercept, correlation, p_value_slope, std_error = stats.linregress(x_data, y_data)
    if model == 'RMA':
#         print( ' - Using RMA model...')
        results = regress2(x_data, y_data, _need_intercept=True)
        slope    		= results['slope']
        intercept		= results['intercept']
        correlation		= results['r']
        p_value_slope	= results['pvalue'] 
        std_error		= ''
        
    #-- Calculates a Pearson correlation coefficient and the p-value for testing non-correlation
    r, p_value = stats.pearsonr(x_data, y_data)

    rmse   = mean_squared_error(y_data, x_data)**0.5

    mae = mean_absolute_error(y_data, x_data)

    mean_x = np.mean(x_data)
    mean_y = np.mean(y_data)
    std_x  = np.std(x_data, ddof=1)
    std_y  = np.std(y_data, ddof=1)


    #-- create scatter plot
    if stdOn == False:
        paths = ax.scatter(x_data, y_data, s = markersize, c = color, alpha = alpha, zorder = zorder)

    if stdOn == True:
        paths = ax.errorbar(x_data, y_data, yerr = yerr,  xerr = xerr, c = color, \
                            ecolor = color, fmt=fmt, \
                            markersize = markersize, elinewidth=elinewidth, capsize = capsize, alpha = alpha, zorder = zorder)

    if case_str is not None:
        case_str = case_str + '\n'
    else:
        case_str = ''

    min_x = np.nanmin(x_data)
    max_x = np.nanmax(x_data)
    min_y = np.nanmin(y_data)
    max_y = np.nanmax(y_data)

    if xRange is not None:
        min_x = xRange[0]
        max_x = xRange[1]

    # make the statistics...
    if statOn == True:
        #-- add slope line

        x = np.array((min_x- 2 *np.absolute(min_x),max_x+ 2 * np.absolute(max_x)))
        y = (slope * x) + intercept
        if regress_line:
            ax.plot(x, y, '-', color=reg_color, linewidth=1.2)
        if one2one_line:
            ax.plot(x, x, '--', color='r', linewidth=1)

        ee_string = ''
        if eeOn == True:
            err_frac = eeFrac
            bias = eeBias
            y1 = x - x * err_frac - bias
            y2 = x + x * err_frac + bias
            if ee_line == True:
                ax.plot(x, y1, '--', color = ee_line_color, linewidth=0.8)
                ax.plot(x, y2, '--', color = ee_line_color, linewidth=0.8)
            num = 0
            for i in range(len(x_data)):
                upp = x_data[i] * (1 + err_frac) + bias
                low = x_data[i] * (1 - err_frac) - bias
                if (y_data[i] > low) & (y_data[i] < upp):	
                    num = num + 1
            ee_static = np.round( num / len(x_data) * 100, 3)

            ee_string = '\nEE%: ' + str(ee_static) + '%'

        #-- create strings for equations in the plot
        correlation_string = "R = {:.2f}".format(r)

        sign = " + "
        if intercept < 0:
            sign = " - "

        lineal_eq = "y = " + str(round(slope, 3)) + dp_x + sign + str(round(abs(intercept), 3))+ '\n'
        rmse_coef = "RMSE = {:.5f}".format(rmse) + '\n'
        mae_coef = "MAE = {:.5f}".format(mae,7)  + '\n'

        if rmse_coef_flag == False:
            rmse_coef = ''

        if mae_coef_flag == False:
            mae_coef = ''

        if p_value >= 0.05:
            p_value_s = "(p > 0.05)"
        else:
            if p_value < 0.01:
                p_value_s = "(p < 0.01)"
            else:
                p_value_s = "(p < 0.05)"

        n_collocations = "N = " + str(n_colocations) + '\n' 
        x_mean_std = "x: " + str(round(mean_x, 3)) + " $\pm$ " + str(round(abs(std_x), 3)) + '\n'
        y_mean_std = "y: " + str(round(mean_y, 3)) + " $\pm$ " + str(round(abs(std_y), 3)) + '\n'



        equations0 = case_str + \
                     n_collocations + \
                     x_mean_std + \
                     y_mean_std + \
                     rmse_coef  + \
                     mae_coef + \
                     lineal_eq  + \
                     correlation_string + ' ' + p_value_s + \
                     ee_string

        if label_p == None:
            if r>0:
                label_p = 'upper left'
            else:
                label_p = 'lower left'

        if (label_p == 'upper left'):
            # upper left
            posXY0      = (0, 1)
            posXY_text0 = (5, -5)
            ax.annotate(equations0, xy=posXY0, xytext=posXY_text0, va='top', bbox=ano_box, \
                    xycoords='axes fraction', textcoords='offset points', color=color, fontsize=fontsize)

        elif (label_p == 'lower right'):
            # lower right
            posXY0      = (1, 0)
            posXY_text0 = (-5, 5)
            ax.annotate(equations0, xy=posXY0, xytext=posXY_text0, va='bottom', ha='right', bbox=ano_box, \
                    xycoords='axes fraction', textcoords='offset points', color=color, fontsize=fontsize)


        elif (label_p == 'lower left'):
            # lower right
            posXY0      = (0, 0)
            posXY_text0 = (5, 5)
            ax.annotate(equations0, xy=posXY0, xytext=posXY_text0, va='bottom', ha='left', bbox=ano_box, \
                    xycoords='axes fraction', textcoords='offset points', color=color, fontsize=fontsize)

        elif (label_p == 'upper right'):
            # upper right
            posXY0      = (1, 1)
            posXY_text0 = (-5,-5)
            ax.annotate(equations0, xy=posXY0, xytext=posXY_text0, va='top', ha='right', bbox=ano_box, \
                    xycoords='axes fraction', textcoords='offset points', color=color, fontsize=fontsize)

        else:
            '!!! scatter: label_p error !!!'
            exit()

    if yRange == None:
        yRange 		= []
        yRange.append(min_y)
        yRange.append(max_y)

    if xRange == None:
        xRange 		= []
        xRange.append(min_x)
        xRange.append(max_x)

    if delta_ticks_X == None:
        delta_ticks_X = (xRange[1] - xRange[0])/5
    if delta_ticks_Y == None:
        delta_ticks_Y = (yRange[1] - yRange[0])/5


    ax.set_ylim(yRange[0], yRange[1])
    ax.set_xlim(xRange[0], xRange[1])


    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)	
    ax.tick_params(labelsize=fontsize)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_major_locator(MultipleLocator(delta_ticks_X))
    ax.yaxis.set_major_locator(MultipleLocator(delta_ticks_Y))	
    return ax
	
def regress2(_x, _y, _method_type_1 = "ordinary least square",
			 _method_type_2 = "reduced major axis",
			 _weight_x = [], _weight_y = [], _need_intercept = True):
	# Regression Type II based on statsmodels
	# Type II regressions are recommended if there is variability on both x and y
	# It's computing the linear regression type I for (x,y) and (y,x)
	# and then average relationship with one of the type II methods
	#
	# INPUT:
	#   _x <np.array>
	#   _y <np.array>
	#   _method_type_1 <str> method to use for regression type I:
	#     ordinary least square or OLS <default>
	#     weighted least square or WLS
	#     robust linear model or RLM
	#   _method_type_2 <str> method to use for regression type II:
	#     major axis
	#     reduced major axis <default> (also known as geometric mean)
	#     arithmetic mean
	#   _need_intercept <bool>
	#     True <default> add a constant to relation (y = a x + b)
	#     False force relation by 0 (y = a x)
	#   _weight_x <np.array> containing the weigth of x
	#   _weigth_y <np.array> containing the weigth of y
	#
	# OUTPUT:
	#   slope
	#   intercept
	#   r
	#   std_slope
	#   std_intercept
	#   predict
	#
	# REQUIRE:
	#   numpy
	#   statsmodels
	#
	# The code is based on the matlab function of MBARI.
	# AUTHOR: Nils Haentjens
	# REFERENCE: https://www.mbari.org/products/research-software/matlab-scripts-linear-regressions/

	import statsmodels.api as sm

	# Check input
	if _method_type_2 != "reduced major axis" and _method_type_1 != "ordinary least square":
		raise ValueError("'" + _method_type_2 + "' only supports '" + _method_type_1 + "' method as type 1.")

	# Set x, y depending on intercept requirement
	if _need_intercept:
		x_intercept = sm.add_constant(_x)
		y_intercept = sm.add_constant(_y)

	# Compute Regression Type I (if type II requires it)
	if (_method_type_2 == "reduced major axis" or
		_method_type_2 == "geometric mean"):
		if _method_type_1 == "OLS" or _method_type_1 == "ordinary least square":
			if _need_intercept:
				[intercept_a, slope_a] = sm.OLS(_y, x_intercept).fit().params
				[intercept_b, slope_b] = sm.OLS(_x, y_intercept).fit().params
				pvalue = sm.OLS(_x, y_intercept).fit().pvalues
			
			else:
				slope_a = sm.OLS(_y, _x).fit().params
				slope_b = sm.OLS(_x, _y).fit().params
				pvalue = sm.OLS(_y, _x).fit().pvalues
			
		elif _method_type_1 == "WLS" or _method_type_1 == "weighted least square":
			if _need_intercept:
				[intercept_a, slope_a] = sm.WLS(
					_y, x_intercept, weights=1. / _weight_y).fit().params
				[intercept_b, slope_b] = sm.WLS(
					_x, y_intercept, weights=1. / _weight_x).fit().params
				pvalue = sm.WLS(_x, y_intercept, weights=1. / _weight_x).fit().pvalues
				
			else:
				slope_a = sm.WLS(_y, _x, weights=1. / _weight_y).fit().params
				slope_b = sm.WLS(_x, _y, weights=1. / _weight_x).fit().params
				pvalue  = sm.WLS(_x, _y, weights=1. / _weight_x).fit().pvalues
			
		elif _method_type_1 == "RLM" or _method_type_1 == "robust linear model":
			if _need_intercept:
				[intercept_a, slope_a] = sm.RLM(_y, x_intercept).fit().params
				[intercept_b, slope_b] = sm.RLM(_x, y_intercept).fit().params
				[intercept_b, slope_b] = sm.RLM(_x, y_intercept).fit().params
				pvalue                 = sm.RLM(_x, y_intercept).fit().pvalues
			else:
				slope_a = sm.RLM(_y, _x).fit().params
				slope_b = sm.RLM(_x, _y).fit().params
				pvalue = sm.RLM(_y, _x).fit().pvalues
		else:
			raise ValueError("Invalid literal for _method_type_1: " + _method_type_1)

	# Compute Regression Type II
	if (_method_type_2 == "reduced major axis" or
		_method_type_2 == "geometric mean"):
		# Transpose coefficients
		if _need_intercept:
			intercept_b = -intercept_b / slope_b
		slope_b = 1 / slope_b
		# Check if correlated in same direction
		if np.sign(slope_a) != np.sign(slope_b):
			raise RuntimeError('Type I regressions of opposite sign.')
		# Compute Reduced Major Axis Slope
		slope = np.sign(slope_a) * np.sqrt(slope_a * slope_b)
		if _need_intercept:
			# Compute Intercept (use mean for least square)
			if _method_type_1 == "OLS" or _method_type_1 == "ordinary least square":
				intercept = np.mean(_y) - slope * np.mean(_x)
			else:
				intercept = np.median(_y) - slope * np.median(_x)
		else:
			intercept = 0
		# Compute r
		r = np.sign(slope_a) * np.sqrt(slope_a / slope_b)
		# Compute predicted values
		predict = slope * _x + intercept
		# Compute standard deviation of the slope and the intercept
		n = len(_x)
		diff = _y - predict
		Sx2 = np.sum(np.multiply(_x, _x))
		den = n * Sx2 - np.sum(_x) ** 2
		s2 = np.sum(np.multiply(diff, diff)) / (n - 2)
		std_slope = np.sqrt(n * s2 / den)
		if _need_intercept:
			std_intercept = np.sqrt(Sx2 * s2 / den)
		else:
			std_intercept = 0
	elif (_method_type_2 == "Pearson's major axis" or
		  _method_type_2 == "major axis"):
		if not _need_intercept:
			raise ValueError("Invalid value for _need_intercept: " + str(_need_intercept))
		xm = np.mean(_x)
		ym = np.mean(_y)
		xp = _x - xm
		yp = _y - ym
		sumx2 = np.sum(np.multiply(xp, xp))
		sumy2 = np.sum(np.multiply(yp, yp))
		sumxy = np.sum(np.multiply(xp, yp))
		slope = ((sumy2 - sumx2 + np.sqrt((sumy2 - sumx2)**2 + 4 * sumxy**2)) /
				 (2 * sumxy))
		intercept = ym - slope * xm
		# Compute r
		r = sumxy / np.sqrt(sumx2 * sumy2)
		# Compute standard deviation of the slope and the intercept
		n = len(_x)
		std_slope = (slope / r) * np.sqrt((1 - r ** 2) / n)
		sigx = np.sqrt(sumx2 / (n - 1))
		sigy = np.sqrt(sumy2 / (n - 1))
		std_i1 = (sigy - sigx * slope) ** 2
		std_i2 = (2 * sigx * sigy) + ((xm ** 2 * slope * (1 + r)) / r ** 2)
		std_intercept = np.sqrt((std_i1 + ((1 - r) * slope * std_i2)) / n)
		# Compute predicted values
		predict = slope * _x + intercept
	elif _method_type_2 == "arithmetic mean":
		if not _need_intercept:
			raise ValueError("Invalid value for _need_intercept: " + str(_need_intercept))
		n = len(_x)
		sg = np.floor(n / 2)
		# Sort x and y in order of x
		sorted_index = sorted(range(len(_x)), key=lambda i: _x[i])
		x_w = np.array([_x[i] for i in sorted_index])
		y_w = np.array([_y[i] for i in sorted_index])
		x1 = x_w[1:sg + 1]
		x2 = x_w[sg:n]
		y1 = y_w[1:sg + 1]
		y2 = y_w[sg:n]
		x1m = np.mean(x1)
		x2m = np.mean(x2)
		y1m = np.mean(y1)
		y2m = np.mean(y2)
		xm = (x1m + x2m) / 2
		ym = (y1m + y2m) / 2
		slope = (x2m - x1m) / (y2m - y1m)
		intercept = ym - xm * slope
		# r (to verify)
		r = []
		# Compute predicted values
		predict = slope * _x + intercept
		# Compute standard deviation of the slope and the intercept
		std_slope = []
		std_intercept = []

	# Return all that
	return {"slope": float(slope), "intercept": intercept, "r": r, "std_slope": std_slope, 
			"std_intercept": std_intercept, "predict": predict, "pvalue": pvalue}