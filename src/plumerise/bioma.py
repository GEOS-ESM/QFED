"""

Implement biome-prescribed fire properties such as heat fluxe, fire areas.

"""

import numpy as np

# Aggregated Vegetation types
# ---------------------------
TROPICAL       = 1
EXTRA_TROPICAL = 2
SAVANNA        = 3
GRASSLAND      = 4 

# Fire Propertie per aggregated biome type
#                      tf    xf   sv    gl
#                    -----------------------
FLAMING_FRACTION = [ 0.45, 0.45, 0.75, 0.97 ]
HEAT_FLUX_MIN  =   [  30,    80,   4.,  3.  ]  # kW/m2
HEAT_FLUX_MAX  =   [  80,    80,  23.,  3.  ]  # kW/m2 


# INPE's Aggregation Table (from VegType_Mod)
# Tropical forest: 2 & 4
# Extra tropical forest: 1, 3, 5
# Cerrado/woody savanna: 6 to 9
# ------------------------------------------
AGGREGATED_BIOME = [ 2, 1, 2, 2,                  # floresta tropical 2 and 4 / extra trop fores 1,3,5
                     2, 3, 3, 3, 3,               # cerrado/woody savanna :6 a 9
                     4, 4, 4, 4, 4, -15, 4, -17 ]

def getAggregateBiome ( detailedBiome, lat ):
    """
    Uses the classic biome aggregation criterion as implemented in QFED 3.1 and earlier
    to aggregate the IGBP 17 types:
    
         biome = aggregateBiome(detailedBiome, lat)
         
    where aggregateBiome takes the values defined in table AGGREEGATED_BIOME.
    
     
   IGBP Land Cover Legend:

    Value     Description
    -----     -----------
      1       Evergreen Needleleaf Forest
      2       Evergreen Broadleaf Forest
      3       Deciduous Needleleaf Forest
      4       Deciduous Broadleaf Forest
      5       Mixed Forest
      6       Closed Shrublands
      7       Open Shrublands
      8       Woody Savannas
      9       Savannas
     10       Grasslands
     11       Permanent Wetlands
     12       Croplands
     13       Urban and Built-Up
     14       Cropland/Natural Vegetation Mosaic
     15       Snow and Ice
     16       Barren or Sparsely Vegetated
     17       Water Bodies
     99       Interrupted Areas (Goodes Homolosine Projection)
     100      Missing Data
    
         
    """
    
    # Aggregated biome
    # ----------------
    biome = np.zeros(detailedBiome.shape, dtype=int)
    
    for i in range(len(AGGREGATED_BIOME)):
        b = AGGREGATED_BIOME[i]
        d = i+1
        biome[detailedBiome==d] = b
    
    # Latitude correction: forests poleward of 30 are EXTRA_TRPOPICAL...
    # ------------------------------------------------------------------
    I = (np.abs(lat)>30.) & (biome==TROPICAL)
    biome[I] = EXTRA_TROPICAL
    
    return biome

def _prop(b,PROP):
    return np.array([ PROP[i] for i in b ] )
    
def getHeatFlux ( aggregateBiome, Power ):
    """
    Given the IGBP land cover type, fire radiative Power in MW and pixel area in m2,
    it computes
    
    Power_F, Power_S, HeatFlux_F, area_F = getHeatFlux(detailedBiome, Power_MW, area_m2)
    
    where,
    
    Power_f(:)      --- Flaming portion of the FRP
    Power_s(:)      --- Smoldering/residual portion of the FRP
    HeatFlux_f(:,3) --- Flaming heat flux MW/m2, uniform distribution
                        F0:lower bound, 1:mode/mean, 2:upper bound
    Area_f(:,3)     --- Area of flaming portion of fire
    
    The estimates are based on the assumption that flaming fraction and 
    heat fluxes are deternined by the aggregate biome type, the classic assumption
    in the Freitas Plume Rise model.
    
    """
    
    b = aggregateBiome - 1 # short-hand, 0-based for indexing
    
    # Initialize output to nan
    # ------------------------
    Power_f = np.zeros(b.shape)         + np.nan
    Power_s = np.zeros(b.shape)         + np.nan
    HeatFlux_f = np.zeros(b.shape+(3,)) + np.nan
    Area_f = np.zeros(b.shape+(3,))     + np.nan
   
    # Valid biome indices
    # -------------------
    I = (b>=0)&(b<=3) 
    
    fraction = _prop(b[I],FLAMING_FRACTION)
    hf_min   = _prop(b[I],HEAT_FLUX_MIN)/1000. # MW/m2
    hf_max   = _prop(b[I],HEAT_FLUX_MAX)/1000. # MW/m2      
        
    Power_f[I] =    fraction  * Power[I]
    Power_s[I] = (1-fraction) * Power[I]
        
    HeatFlux_f[I,0] = hf_min             # MW/m2, lower bound
    HeatFlux_f[I,1] = (hf_min+hf_max)/2  # MW/m2, mean
    HeatFlux_f[I,2] = hf_max             # MW/m2, upper bound
    
    for i in range(3):
        Area_f[I,i] = Power_f[I] / HeatFlux_f[I,i] # m2
    
    return (Power_f, Power_s, HeatFlux_f, Area_f)
    
    