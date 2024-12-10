"""
A Python interface to the Freitas' Plume Rise Model.

"""

import numpy  as np
import xarray as xr
import yaml

from pyobs.constants import *
from pyobs           import mcbef as mb

from . import bioma   as bm
from . import eta 
from . import FreitasPlume_ as fp

__VERSION__ = 3.0
__AMISS__   = 1.E+20

mode = mb.STATS['mode']

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

def getPlumesVMD(heatFlux_F, area_F, met, getW=False, rad2conv=5, ktop=30, Verb=False):
    
    """
    Runs the Plume Rise extension to compute the paramerers of the 
    vertical mass distribution:
    
          z_i, z_d, z_a, z_f,
            z_plume, w_plume = getPlumesVMD(heatFlux, area, met, ktop=30, Verb=False)

    where 
    
    heatFlux_F --- radiative (flaming) heat flux (kW/m2)
    area_F     --- (flaming) fire area, units of m2
    met        --- meteorological environment (xarray dataset)
    I          --- fire indices to work on; if None, all fires
    getW       --- Whether to return (z_plume, w_plume)
    ktop       --- top vertical index of met fields; indices above these 
                   will be trimmed off of met fields.
    rad2conv   --- factor to convert radiative het flux to convective head flux.
    
    On output, z_i, z_d, z_a, z_f are nd-arrays of shape(N), N being the number of fires:
    
    z_i  ---  height of maximum W (bottom of plume)
    z_d  ---  height of maximum detrainment
    z_a  ---  average height in (z_i,z_f), weighted by -dw/dz
    z_f  ---  height where w<1 (top of plume)

    See getVMD for computing the vertical mass distribution from (z_i,z_d,z)a,z_f).

    The plume output are the vertical velocity and coordinates in native PR
    model verical coordinates:
    
    z_plume --- (nkp)   native vertical levels (same for every fire)
    w_plume --- (N,nkp) native vertial velocity
    
    
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

#   Actual problem size after subsetting
#   ------------------------------------
    km, N = u.shape
    if Verb:
        print('getPlumesVMD: problem size is N, km:', N, km)
    
#   Units:
#       heatFlux_F - kw/m2 as required by plume rise model
#       area - m2 as required by plume rise model
#   ------------------------------------------------------  
    hflux_kW = rad2conv * heatFlux_F[:] * 1000  # Convective HF, MW/m2 to kW/m2
    area_m2 = area_F[:] # in m2
    

#   Run plume rise model basing heat flux
#   -------------------------------------
    #print("before fp.plumesvmd",N,km,hflux_kW.min(),hflux_kW.max())
    #return np.arange(7)

    (z_i, z_d, z_a, z_f,
         z_plume, w_plume, rc) = fp.plumesvmd(u,v,T,q,delp,ptop,hflux_kW,area_m2)

    ### print('*** plume shapes',z_plume.shape, w_plume.shape)
    
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
        print("          |  hFlux  |    Lon    Lat  |    z_i      z_f   |   z_a      z_d    |  rc")     
        print("  index   |   kW    |    deg    deg  |     km       km   |    km       km    | ")
        print("--------- | ------- |  ------ ------ | -------- -------- | -------- -------- | ---")

        for i in Np:
            print("%8d  | %7.1f | %7.2f %6.2f | %8.2f %8.2f | %8.2f %8.2f | %3d"%\
                  (i,hflux_kW[i], met.lon[i],met.lat[i], \
                   z_i[i]/1000,z_f[i]/1000, \
                   z_a[i]/1000,z_d[i]/1000, \
                   rc[i]))

    if getW:
        return (z_i,z_d,z_a,z_f,z_plume,w_plume.T,rc)
    else:
        return (z_i,z_d,z_a,z_f,rc)
    
#---
def getVMD ( z_i, z_d, z_a, z_f, met, z_plume=None, w_plume=None, ktop=30, option='centered' ):
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
    if w_plume is not None:
        coords['z_plume'] = xr.DataArray(z_plume, dims='z_plume',
                                         attrs={'units':'m','description':'Native Height of PR model'})
    
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

    if w_plume is not None:
        DA['w_plume'] = xr.DataArray(w_plume,coords = {'time':time, 'z_plume':z_plume},
                               attrs={'units':'m','description':'Plume Vertical Velocity'})
      
    ds = xr.Dataset(DA,coords = coords, attrs={'option':option}) 
    
    return ds

def getVMD2 ( z_i, z_d, z_a, z_f, z_plume, option='centered' ):
    """
    A variant of getVMD() whee the calculation is done in native coordinates.
    
    vmd = getVMD2 ( z_i, z_d, z_a, z_f, z_plume )
    """

    # Extend z_plume to be 2D
    # -----------------------
    L = z_plume.shape[0]
    N = z_i.shape[0]
    z = z_plume.reshape((1,L)) + np.zeros((N,L))
    
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
 
    vmd = fp.getvmd(z.T,z_c,delta).T
    
    return vmd

def pblVMD ( z, pblh ):
    """
    Return a Vertical Mass Distribution Function.
    On input,
    
    pblh.dims = [ 'fire']
    z.dims = [ 'fire', 'lev' ] or z.dims['lev'] 
    
    """
    
    # Make sure both z and pblh are 2D
    # Note: can be optimized if need be
    # ----------------------------------
    N = pblh.shape[0]       # number of fires
    if len(z.shape) == 1:   
        L = z.shape[0]      # number of native layers
        z_ = z.reshape((1,L)) + np.zeros((N,L))
    else:
        L = z.shape[1]      # number of GEOS layers
        z_ = z              # assume shape is (N,L)
    
    pblh_ = pblh.reshape((N,1)) + np.zeros((N,L))
 
    # no emission above PBL
    # ---------------------
    vmd = np.ones(z_.shape)
    vmd[z_>pblh_] = 0.0        
    
    # Normalize so that vertical sum is always 1, for each fire
    # ---------------------------------------------------------
    column = vmd.sum(axis=1).reshape(N,1)
    vmd = vmd / column 
    
    return vmd

def decode_EF(table,species):
    """
    Decode emission factors, returning lower bound, mode and upper bound
    """
    
    EF = np.zeros((4,3)) # (biome,stat)
    
    for b in range(4):
    
        biome = bm.NAMES[b]
        mean = table['emission_factors']['species'][species][bm.NAMES[b]][0]
        stdv = table['emission_factors']['species'][species][bm.NAMES[b]][1]

        EF[b][0] = mean - stdv
        EF[b][1] = mean
        EF[b][2] = mean + stdv 
        
    return EF

def Emission_Rate_Profiles ( f, m, p, bf ):
    
    """
    Compute vertically distributed emission rates.
    """
    
    # Kaiser (2009) Emission Coefficient
    # ----------------------------------
    alpha_SI = 1.37 / 10**6   # kg / J = kg / ( W s ) = 10**3 g / ( s MW/10**6 ) = 10+9 g / (MW s)
    alpha    = 1.37 * 10**3   # g / (MW s)

    # Emission Factors from Andrea (2019)
    # -----------------------------------
    EF = yaml.safe_load(open('emission_factors.yaml','r'))
    EF_co  = decode_EF(EF, 'co')    # g(CO) / kg(dry mater)
    EF_co2 = decode_EF(EF, 'co2')   # g(CO2) / kg(dry mater)
    
    # Total Column Emission Rates (biome dependent)
    # ---------------------------------------------
    A = f.FP_Area[:].values # pixel area
    F_s, F_f = f.Power_s[:,mode].values/A, f.Power_f[:,mode].values/A      # FRP density, MW/m2
    mce = f.FP_MCE[:].values
    
    # vertical mass distrbution on native vertical coordinates
    # --------------------------------------------------------
    vmd_f = getVMD2(p.z_i, p.z_d, p.z_a, p.z_f, p.z_plume.values)   # from Freitas PR
    vmd_s = pblVMD(p.z_plume.values, m.PBLH.values)                        # uniform in   PBL
    
    N = f.dims['fire']
    L = p.z_plume.shape[0]
    E_co, E_co2   = np.zeros((N,L)), np.zeros((N,L))
    qE_co, qE_co2 = np.zeros((N,L)), np.zeros((N,L))
    
    for b in range(4):
        
        I = (bf==b+1)  # for this biome index, 1-4
        
        # QFED bulk emission coefficient, no mce modulation
        # -------------------------------------------------
        q_co  = alpha * EF_co[b][mode]  # no MCE modulation
        q_co2 = alpha * EF_co2[b][mode] # no MCE modulation
        
        # McBEF bulk emission coefficient ( alpha * EF )
        # ----------------------------------------------
        a_co  = q_co  * (1-mce[I])  # MCE modulation
        a_co2 = q_co2 *    mce[I]   # MCE modulation  
        
        N = mce[I].shape[0]  # number of fires in this biome
          
        # Vertical profile with MCE modulation
        # ------------------------------------
        zF = F_s[I].reshape(N,1) * vmd_s[I] + F_f[I].reshape(N,1) * vmd_f[I] # vertically distributed FRP density
        E_co[I]  = a_co.reshape(N,1)  * zF
        E_co2[I] = a_co2.reshape(N,1) * zF 
        
        # QFED: no mce modulation, all in pbl
        # ------------------------------------
        qF = (F_s[I] + F_f[I]).reshape(N,1) * vmd_s[I] # all in PBL
        qE_co[I] = q_co  * qF
        qE_co[I] = q_co2 * qF 
        
    # All done
    # --------
    return (E_co, E_co2, qE_co, qE_co2) # g / ( s m2 )
    
    
#..............................................................................

if __name__ == "__main__":

    pass

