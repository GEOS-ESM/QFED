"""
A Python interface to the Freitas' Plume Rise Model.

"""

import numpy  as np
import xarray as xr

from pyobs.constants import *

from . import eta 
from . import FreitasPlume_ as fp

__VERSION__ = 3.0
__AMISS__   = 1.E+20

#..............................................................................

#
#                                    Static Methods
#                                    --------------
#

def omp_set_num_threads(num_threads):
    """
    Set the number of thereads for OMP.
    """
    fp.setnumthreads(num_threads)

def getPlumesVMD(heatFlux_F, area_F, met, rad2conv=5, ktop=30, Verb=False):
    
    """
    Runs the Plume Rise extension to compute the paramerers of the 
    vertical mass distribution:
    
          z_i, z_d, z_a, z_f  = getPlumesVMD(heatFlux, area, met, ktop=30, Verb=False)

    where 
    
    heatFlux_F --- radiative (flaming) heat flux (kW/m2)
    area_F     --- (flaming) fire area, units of m2
    met        --- meteorological environment (xarray dataset)
    ktop       --- top vertical index of met fields; indices above these 
                   will be trimmed off of met fields.
    rad2conv   --- factor to convert radiative het flux to convective head flux.
    
    On otput, z_i, z_d, z_a, z_f are nd-arrays of shape(N),   
    
    z_i  ---  height of maximum W (bottom of plume)
    z_d  ---  height of maximum detrainment
    z_a  ---  average height in (z_i,z_f), weighted by -dw/dz
    z_f  ---  height where w<1 (top of plume)
    
    N being the number of fires. See getVMD for computing the vertical mass distribution.

    
    """

#   Top pressure
#   ------------        
    N, km = met.U.shape
    pe = eta.getPe(km)     # make sure eta supports this number of levels
    ptop = pe[ktop]
    
#   Trim top of met fields, as fortran arrays to avoid f2py copy-in
#   ---------------------------------------------------------------
    u = np.asfortranarray(met.U[:,ktop:].values.T)
    v = np.asfortranarray(met.V[:,ktop:].values.T)
    T = np.asfortranarray(met.T[:,ktop:].values.T)
    q = np.asfortranarray(met.QV[:,ktop:].values.T)
    delp = np.asfortranarray(met.DELP[:,ktop:].values.T)

#   Units:
#       heatFlux_F - kw/m2 as required by plume rise model
#       area - m2 as required by plume rise model
#   ------------------------------------------------------  
    mode = 1 # index of  mode stat
    hflux_kW = rad2conv * heatFlux_F[:,mode] * 1000  # Convective HF, MW/m2 to kW/m2
    area_m2 = area_F[:,mode] # in m2
    

#   Run plume rise model basing heat flux
#   -------------------------------------
    z_i,z_d,z_a,z_f,rc = fp.plumesvmd(u,v,T,q,delp,ptop,hflux_kW,area_m2)

    if Verb:
        if N>1000:
            Np = list(range(0,N,N//100))
        elif N>100:
            Np = list(range(0,N,N//10))
        else:
            Np = list(range(N))
            
        print("")
        print("                   Plume Rise Estimation")
        print("                   ----------------------")
        print("")
        print("          |    Lon    Lat  |    z_i      z_f   |   z_a      z_d    |  rc")     
        print("  index   |    deg    deg  |     km       km   |    km       km    | ")
        print("--------- |  ------ ------ | -------- -------- | -------- -------- | ---")

        for i in Np:
            print("%8d  | %7.2f %6.2f | %8.2f %8.2f | %8.2f %8.2f | %3d"%\
                  (i,met.lon[i],met.lat[i], \
                   z_i[i]/1000,z_f[i]/1000, \
                   z_a[i]/1000,z_d[i]/1000, \
                   rc[i]))

    return (z_i,z_d,z_a,z_f,rc)

#---
def getVMD ( z_i, z_d, z_a, z_f, met, ktop=30, option='centered' ):
    """
    Compute Vertical Mass Distribution (VMD) for each fire, 
    
    ds = getVMD ( z_i, z_d, z_a, z_f, met, ktop=30, option='centered' )
    
    On input,
    
    z_i  ---  height of maximum W (bottom of plume)
    z_d  ---  height of maximum detrainment
    z_a  ---  average height in (z_i,z_f), weighted by -dw/dz
    z_f  ---  height where w<1 (top of plume)
    
    ktop --- top vertical index of met fields; indices above these 
             will be trimmed off of met fields.
    
    This function assumes a parabolic VMD with parameters (z_c,delta)
    specified by *option*:

    option = 'centered':
   
                     z_c   = (z_f+z_i)/2
                    delta = (z_f-z_i)/2

    option = 'bottom', preserve bottom half:
                     z_c   = z_d
                     delta = z_d - z_i

    option = 'upper', preserve upper half:
                     z_c   = z_d
                     delta = z_f - z_d
                     
    Output is an xarray dataset with the VMD and input parameters.

    """

    # height above surface
    # --------------------
    h = met.H[:,ktop:].values 
    hs = met.PHIS.values / MAPL_GRAV  
    z = np.asfortranarray((h - hs.reshape(hs.size,1)).T) # to avoid f2py copy-in 
    
    if option == 'centered':
        z_c   = (z_f+z_i)/2
        delta = (z_f-z_i)/2
    elif option == 'bottom':
        z_c   = z_d
        delta = z_d - z_i
    elif option == 'upper':
        z_c   = z_d
        delta = z_f - z_d
    else:
        raise ValueError("Invalid option for getVMD: "+option)

    vmd = fp.getvmd(z,z_c,delta)
    
    #print('vmd = ', vmd.shape, vmd.min(), vmd.max())
    
    # Construct xarray Dataset
    # ------------------------
    coords = met.coords.copy()
    coords['lev']  = met.coords['lev'][ktop:]
    time = coords['time']
    lev = coords['lev']
    DA = dict(
          z_i = xr.DataArray(z_i,coords={'time':time},
                       attrs={'units':'m','description':'height of maximum W (bottom of plume)'}),
          z_d = xr.DataArray(z_d,coords={'time':time},
                       attrs={'units':'m','description':'height of maximum detrainment'}),
          z_a = xr.DataArray(z_a,coords={'time':time},
                       attrs={'units':'m','description':'average height in (z_i,z_f), weighted by -dw/dz'}),
          z_f = xr.DataArray(z_f,coords={'time':time},
                       attrs={'units':'m','description':'height where w<1 (top of plume)'}),
          z_c = xr.DataArray(z_c,coords={'time':time},
                       attrs={'units':'m','description':'center of VMD'}),
          delta = xr.DataArray(delta,coords={'time':time},
                       attrs={'units':'m','description':'width of VMD'}),
          vmd = xr.DataArray(vmd.T,coords={'time':time,'lev':lev},
                       attrs={'units':'1','description':'Vertical Mass Distribution'}),
          z = xr.DataArray(z.T,coords={'time':time,'lev':lev},
                       attrs={'units':'m','description':'Height Above Surface'}),
          zs = xr.DataArray(hs,coords={'time':time},
                       attrs={'units':'m','description':'Surface Height'}),
    )
      
    ds = xr.Dataset(DA,coords = coords, attrs={'option':option}) 
    
    return ds

#..............................................................................

if __name__ == "__main__":

    pass

