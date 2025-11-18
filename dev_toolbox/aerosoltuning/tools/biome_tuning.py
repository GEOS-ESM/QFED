'''  Functions for QFED biomass burning analysis during the EIS-Fire program:
        cost_obs_alpha: cost function for minimizing the biome scaling factors
        cost_obs: cost function only includes the J_o (observed) term
	scale_biomes: 
'''
import numpy as np

def cost_obs_alpha(alpha, tau_obs, tau_an, tau_bio, w_o=None, w_b=None):
  '''  Calculate the cost function to minimize the difference between observed
        and experimental AOD (j_o) and the change in alpha scaling factors (j_b). 
	Experimental AOD = AOD anthropogenc + sum(AOD_biomes)
    Input: alpha [m,], tau_obs [n,], tau_an [n,], tau_bio [n, m],
        weight_obs [n,], and weight_biome [n,] or [n, m] (??)
        where n = number of grid points, m = number of biomes
    Output: sum of the residuals, j_o + j_b
  '''

  if(len(tau_obs) < len(alpha)): print('Too few data points in obs!')

  tau_exp = tau_an + np.dot(tau_bio, alpha)
  #print(alpha)

  if(w_o is None):
    w_o = np.ones(np.shape(tau_obs), dtype=np.float)
  ### note: w_b is currently not being used. This was a place holder
  if(w_b is None):
    w_b = np.ones(np.shape(tau_bio), dtype=np.float)

  ### sum over n
  ### Minimize the difference between experimental and observed AOD
  j_o = np.sum(w_o * (np.log(tau_obs+0.01) - np.log(tau_exp+0.01))**2)
  #print('J_o: {:f}'.format(j_o))

  ### not weighted, taking sum over m
  ### Minimize the difference between optimized and current QFED factors
  j_b = np.sum((alpha - 1.)**2)
  #print('J_b: {:f}'.format(j_b))

  return j_o + j_b      # dominated by j_o

def cost_jo(alpha, tau_obs, tau_an, tau_bio, w_o=None):
  '''  Calculate the cost function to minimize the difference between observed
        and experimental AOD (j_o). Experimental AOD = AOD anthropogenc +
        sum(AOD_biomes)
    Input: alpha [m,], tau_obs [n,], tau_an [n,], tau_bio [n, m], and
        weight_obs [n,], where n = number of grid points, m = number of biomes
    Output: the residual, j_o
  '''

  if(len(tau_obs) < len(alpha)): print('Too few data points in obs!')

  tau_exp = tau_an + np.dot(tau_bio, alpha)
  tau_exp[tau_exp <= -0.01] = -0.009

  if(w_o is None):
    w_o = np.ones(np.shape(tau_obs), dtype=np.float)

  ### sum over n
  j_o = np.sum(w_o * (np.log(tau_obs + 0.01) - np.log(tau_exp + 0.01))**2)

  return j_o

def scale_biomes(tau_obs, tau_nobb, tau_bio, wobs=None, min_alpha=False, max_iter=2000, ftol=1e-8, xtol=1e-8):
    ''' Perform a least squares minimization using observed, anthropogenic, and biomass
        burning AOD
        Calls: either cost_jo_alpha or cost_jo functions
        Input: 1D numpy arrays of AODs from observations (GEOS-FP) and 
        anthropogenic emissions (no biomass burning simulation), and a dictionary 
        of biomass burning AOD for each biome.
        Optional input: weighting factors for the observed cost function, do you
        want to include a term that minimizes the change in the scaling factors,
        maximum iterations, function tolerance, parameter tolerance
        Return: dictionary of scaling factors for emissions from each biome
    '''
    from scipy.optimize import least_squares

    ### start the scaling factors at 1 (no change to existing QFED scaling)
    biome_names = list(tau_bio.keys())  # Get the biome names as a list
    nbio = len(biome_names)
    alpha_1s = np.ones((nbio))

    ptext = np.isfinite(tau_obs)
    nobs = len(tau_obs[ptext])

    if nobs <= 1:
        print('Warning, no observations available')
    
    if wobs is None: 
        wobs_arr = np.ones(nobs)
    else: 
        wobs_arr = wobs[ptext]

    obs_arr = tau_obs[ptext]
    nobb_arr = tau_nobb[ptext]

    bio_arr = np.zeros((nobs, nbio))
    for ib, biome in enumerate(biome_names):  # Use enumerate to get integer indices
        bio_arr[:,ib] = tau_bio[biome][ptext] 

    if min_alpha:
        lsq_res = least_squares(cost_obs_alpha, alpha_1s, args=(obs_arr, nobb_arr, 
                                bio_arr, wobs_arr), verbose=1, bounds=[0.01, np.inf],
                                max_nfev=max_iter, ftol=ftol, xtol=xtol)
    else:
        lsq_res = least_squares(cost_jo, alpha_1s, args=(obs_arr, nobb_arr, bio_arr, 
                                wobs_arr), verbose=1, bounds=[0.01, np.inf],
                                max_nfev=max_iter, ftol=ftol, xtol=xtol)

    print(lsq_res)
    alpha = {}
    for ib, biome in enumerate(biome_names):  # Use enumerate here too
        alpha[biome] = lsq_res.x[ib]

    return alpha
