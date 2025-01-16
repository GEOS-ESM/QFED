"""
Planck function and its inverse, derived from J. Joiner IDL code.
Uses GLA TOVS consistent constants.

This software is hereby placed in the public domain.
Arlindo.daSilva@nasa.gov
"""

import numpy as np

# Constant consistent with GLATOVS
C = (1.19104e-5, 1.43833)

def planck(k,T):
    """
    Planck function.
    Returns radiances in [mw/(sr m^2 cm^-1)] given
       k  wave number [cm-1]
       T  temperature [K]
    Shortcut: 
       If k<0, then abs(k) is assumed to be wave length in microns.
    """
    if k<0: 
        k = - 1.0e4 / k
    if not np.isscalar(T):
        k = float(k) * np.ones(T.shape)
    return C[0]*np.power(k,3.0)/(np.exp(C[1]*k/T)-1.0)

def iplanck(k,I):
    """
    Inverse of the Planck function.
    Returns Brightness Temperature [K] given 
       k  wave number [cm-1]
       I  radiance [mw/(sr m^2 cm^-1])
    Shortcut: 
       If k<0, then abs(k) is assumed to be wave length in microns.
    """
    if k<0:
        k = - 1.0e4 / k
    k = float(k) * np.ones(I.shape)
    return C[1]*k/np.log(1.0+C[0]*np.power(k,3.0)/I)

#............................................................

def B21(T):
    return planck(-3.959,T) # MODIS Channel 21

def B31(T):
    return planck(-11.03,T) # MODIS Channel 31

def B32(T):
    return planck(-12.02,T) # MODIS Channel 32

def iB21(I):
    return iplanck(-3.959,I) # MODIS Channel 21

def iB31(I):
    return iplanck(-11.03,I) # MODIS Channel 31

def iB32(I):
    return iplanck(-12.02,I) # MODIS Channel 32

#............................................................


        

