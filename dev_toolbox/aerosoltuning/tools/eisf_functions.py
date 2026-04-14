'''  Functions for QFED biomass burning analysis during the EIS-Fire program:
	lin2log_aod: Convert linear AOD to log-transformed AOD
	log2lin_aod: Convert log-transformed AOD to linear AOD
	aod_mean_std: Calculate mean and standard devaition of log-transformed AOD
	aod_wgt_mean_std: Calculate weighted mean and stddv of log-transformed AOD
	combine_biomes: Combine AOD from biomes that have been found to be correlated
	get_fbb_weight: Return a weighting factor based on the fraction of AOD from
		biomass burning.
	get_sa_weight: Return a surface area weighting factor
	log_aod_stats: calculate stats from the log-transformed AOD
	reorder_lon180: reorder longitudes that 0 to 360 degrees to -180 to 180 degrees
'''

import numpy as np
from sys import exit

def lin2log_aod(tau_in):
  ''' Convert linear scale AOD to natural log-transformed AOD
  '''

  tau_in[tau_in <= -0.01] = -0.009
  return np.log(tau_in + 0.01)

def log2lin_aod(eta_in):
  ''' Convert natural log-transformed AOD to linear scale AOD
  '''

  return np.exp(eta_in) - 0.01
	########### end log conversions ###########

def aod_mean_std(xaod, msk_arr, iax = 0):
  ''' Calculates a mean and standard deviation in log space for an AOD array.
      Input: aod array, mask array, (axis to average along)
      Output: mean, log-scale standard deviation
  '''

  xaod_ma = lin2log_aod( np.ma.masked_array(xaod, mask = msk_arr) )
  xmean = np.ma.mean(xaod_ma, axis = iax)

  return log2lin_aod(xmean), np.ma.std(xaod_ma, axis = iax)
### end weighted mean/variance calculation

def aod_wgt_mean_std(xaod, msk_arr, wgt_arr, iax = 0):
  ''' Calculates a weighted mean and standard deviation in log space for an AOD array.
      Input: aod array, mask array, weighting array, (axis to average along)
      Output: mean, log-scale standard deviation
  '''

  wgt_ma = np.ma.masked_array(wgt_arr, mask = msk_arr)
  xaod_ma = np.ma.masked_array(xaod, mask = msk_arr)

  xmean = np.average( lin2log_aod(xaod_ma), axis = iax, weights = wgt_ma)

  diff = (lin2log_aod(xaod_ma) - xmean)**2
  xstd = np.sqrt(np.average(diff, axis = iax, weights = wgt_ma))

  return log2lin_aod(xmean), xstd
### end weighted mean/variance calculation

def combine_biomes(data, biomes_new):
  ''' Combine AOD from biomes that we want to scale together
      Input: data array containing biome specific AODs and dictionary containing 
	the new biome name(s) and array of the original biomes to combine
      Output: data array with new biome variable
  '''
  import xarray as xr
  
  for biome in biomes_new:
    xa_example = data[biomes_new[biome][0] + '_totexttau']

    data[biome + '_totexttau'] = xr.DataArray( np.zeros(xa_example.shape), \
	coords = xa_example.coords, dims = xa_example.dims )

    for ib in biomes_new[biome]:
      data[biome + '_totexttau'].values = data[biome + '_totexttau'].values + \
	data[ib + '_totexttau'].values

  return data

def get_fbb_weight(fbb_arr, fbb_thres):
  ''' Return weighting factors based on the fraction of AOD from biomass burning
  '''

  if fbb_thres > 1:
    xfbb = fbb_thres/100.
  else: xfbb = fbb_thres

  wbb = np.cos(np.pi/2. * (1 - fbb_arr) / (1 - xfbb))
  wbb[fbb_arr == 0] = 0.

  return wbb

def get_sa_weight(file_sa, arr_shape = None):
  ''' Read in surface area, return surface area weighting factors
      Input: file to read in and optional array shape to return the weights in
  '''
  import pandas as pd

  sa_pd = pd.read_csv(file_sa, header = 1)

  sa_wgt = sa_pd.surface_area.to_numpy() / sa_pd.surface_area.max()

  if arr_shape is None:
    return sa_wgt
  else:
    ### assuming the array dimensions are [time, lat, lon]
    sa_arr = np.ones(arr_shape)
    for il in range(0, len(sa_pd.lat)):
      sa_arr[:,il,:] = sa_wgt[il]

    return sa_arr

def log_aod_stats(tau_obs, tau_exp, lat = None):
  ''' Calculate statistics between observed and "experimental" AOD in the 
	log-transformed space. Input and output AOD are in linear scale.
    Input: Observed AOD, Experimental AOD (in linear scale, no missing values)
	Optional: if latitudes are provided, calculate a cos(lat) - weighted mean
    Output: linear mean ratio, logarithmic mean difference, pearson r 
	correlation coefficient
  '''

  from scipy.stats import pearsonr
  if(len(tau_obs) != len(tau_exp)): print('Warning: array lengths are not the same')

  lg_obs = np.log(tau_obs + 0.01)
  lg_exp = np.log(tau_exp + 0.01)
  lg_rat = np.log(tau_exp/tau_obs + 0.01)

  if(lat is None):
    mean_rat = np.exp(np.mean(lg_rat)) - 0.01
    lg_diff = np.mean(lg_exp - lg_obs)
    print('From log-transformed AOD: ')
  else:
    print('From log-transformed, latitude-weighted AOD: ')
    clat = np.cos(lat*np.pi/180.)
    mean_rat = np.exp( np.sum(clat * lg_rat) / np.sum(clat)) - 0.01
    lg_diff = np.sum(clat * (lg_exp - lg_obs))/np.sum(clat)

  r = pearsonr(lg_obs, lg_exp)

  print('  Mean linear AOD ratio = {:5.2f}'.format(mean_rat))
  print('  exp(Mean) log-transformed AOD difference = {:5.2f}'.format(np.exp(lg_diff)))
  print('  R^2 = {:5.2f}'.format(r[0]**2))

  return mean_rat, lg_diff, r

############################# end log_aod_stats ########################### 


def reorder_lon180(lon_in, arr_in):
  '''  Reorder a data array so that the longitude dimension is organized from -180
	to 180 degrees. Assuming a 3 or 4 dimensional (GEOS) array.
    Input: Longitude, Data array
    Output: Reordered data array  [time (and/or) lev, lat, lon]
  '''

  arr_out = np.copy(arr_in)
  ndim = len(np.shape(arr_out))
  ptwest = np.where(lon_in >= 180.)
  if(np.size(ptwest) != 0.):
    pteast = np.where(lon_in < 180.)
    if(ndim == 3):
      arr_out[:,:,pteast[0]] = arr_in[:,:,ptwest[0]]
      arr_out[:,:,ptwest[0]] = arr_in[:,:,pteast[0]]
    elif(ndim == 4):
      arr_out[:,:,:,pteast[0]] = arr_in[:,:,:,ptwest[0]]
      arr_out[:,:,:,ptwest[0]] = arr_in[:,:,:,pteast[0]]
    else:
      print('Unexpected number of dimensions')
      exit()
  else: 
    print('Longitude max is 180, will not reorder')

  return arr_out

####################### END reorder_longitude  ###############




