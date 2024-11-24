"""
A Python interface to the Freitas' Plume Rise Model.

"""

import numpy as np
                       
import FreitasPlume_ as fp

from pyobs.constants import *
from . import eta 

__VERSION__ = 3.0
__AMISS__   = 1.E+20

#..............................................................................

#
#                                    Static Methods
#                                    --------------
#

def getPlumeBiome(farea,biome,met,ktop=30,Verb=False):
    
    """
    Runs the Plume Rise extension to compute the extent of the plume.

          p, z, k = getPlume(farea,biome,met,t,ntd)

    where p, z and k are nd-arrays of shape(N,2), N being the
    number of fires. On input,

    farea --- (flaming) fire area, units of km2
    biome --- biome type index
    met   --- meteorological environment (xarray dataset)
    ktop  --- top vertical index of met fields; indices above these 
              will be trimmed off of met fields.
    
    """

    one_km2 = 1.e6 # 1 km^2: farea is in units of km2

    
#   Top pressure
#   ------------        
    N, km = met.U.shape
    pe = eta.getPe(km)     # make sure eta supports this number of levels
    ptop = pe(ktop)
    
#   Trim top of met fields
#   ----------------------
    u = met.U[:,ktop:].values
    v = met.V[:,ktop:].values
    T = met.T[:,ktop:].values
    q = met.q[:,ktop:].values
    delp = met.delp[:,ktop:].values

#   Units:
#       farea - km2 (must multiply by nominal area for m2)
#       area - m2 as required by plume rise model
#   ------------------------------------------------------   
    area = farea[i] * one_km2

#   Run plume rise model basing heat flux on biome type
#   ---------------------------------------------------
    p1, p2, z1, z2, k1, k2, rc = \
            plumesBiome(u, v, T, q, delp, ptop, area, biome)

    k1, k2 = (k1+ktop-1, k2+ktop-1)

    if Verb:
        if N>100:
            Np = list(range(0,N,N/10))
        elif N>10:
            Np = list(range(0,N,N/10))
        else:
            Np = list(range(N))
            
        print("")
        print("                   Plume Rise Estimation for t=%d"%t)
        print("                   ------------------------------")
        print("")
        print("  %  |    Lon    Lat  b |   p_bot    p_top  |  z_bot z_top  |  k   k")     
        print("     |    deg    deg    |    mb       mb    |   km     km   | bot top")
        print("---- |  ------ ------ - | -------- -------- | ------ ------ | --- ---")


        if i in Np:
            ip = int(0.5+100.*i/N)
            print("%3d%% | %7.2f %6.2f %d | %8.2f %8.2f | %6.2f %6.2f | %3d %3d "%\
                  (ip,met.lon[i],met.lat[i],veg[i], \
                   p2/100,p1/100,z2/1000,z1/1000,k2,k1))

    return (p1, p2, z1, z2, k1, k2, rc)

#..............................................................................

if __name__ == "__main__":

    pass

