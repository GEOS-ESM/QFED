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

def getAggregateBiome ( detailedBiome, lat ):
    """
    Uses the classic biome aggregation criterion as implemented in QFED 3.1 and earlier
    to aggregate the IGBP 17 types:
    
         biome = aggregateBiome(detailedBiome, lat)
         
    ehre aggregateBiome takes the values:
    
         1  Tropical Forests        IGBP  2, 30S < lat < 30N
         2  Extra-tropical Forests  IGBP  1, 2(lat <=30S or lat >= 30N), 
                                          3, 4, 5
         3  cerrado/woody savanna   IGBP  6 thru  9
         4  Grassland/cropland      IGBP 10 thru 17
         
     On input we have the detailed IGBP land types:
     
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
    biome = np.zeros(detailedBiome.shape, astype='int')
    
    # Tropical Forests
    # ----------------
    I = (detailedBiome==2)&(np.abs(lat)<30.)
    biome[I] = TROPICAL
    
    # Extra-tropical forests
    # ----------------------
    I = (detailedBiome==2)&(np.abs(lat)>=30.)&(detailedBiome>2)&(detailedBiome<6)
    biome[I] = EXTRA_TROPICAL
    
    # Savanna
    # -------
    I = (detailedBiome>5)&(detailedBiome<10)
    biome[I] = SAVANNA 
    
    # Grassland
    # ---------
    I = (detailedBiome==0)&(detailedBiome>9)&(detailedBiome<17)
    biome[I] = GRASSLAND  
    
    return biome
    
def getHeatFlux ( aggregateBiome, Power, area ):
    """
    Given the IGBP land cover type, fire radiative Power in MW and pixel area in m2,
    it computes
    
    Power_F, Power_S, HeatFlux_F, area_F = getHeatFlux(detailedBiome, Power_MW, area_m2)
    
    where,
    
    Power_F(:)      --- Flaming portion of the FRP
    Power_S(:)      --- Smoldering/residual portion of the FRP
    HeatFlux_F(:,3) --- Flaming heat flux kW/m2, uniform distribution
                        F0:lower bound, 1:mode/mean, 2:upper bound
    area_F(:,3)     --- Area of flaming portion of fire
    
    The estimates are based on the assumption that flaming fraction and 
    heat fluxes are deternined by the aggregate biome type, the classic assumption
    in the Freitas Plume Rise model.
    
    """
    
    b = aggregateBiome - 1 # short-hand, 0-based for indexing
    
    # Initialize output to nan
    # ------------------------
    Power_F = np.zeros(b.shape)         + np.nan
    Power_S = np.zeros(b.shape)         + np.nan
    HeatFlux_F = np.zeros(b.shape+(3,)) + np.nan
    area_F = np.zeros(b.shape+(3,))     + np.nan
    
    # Valid biome indices
    # -------------------
    I = (b>=0)&(b<=3) 
    
    Power_F[I] =    FLAMING_FRACTION[b[I]]  * Power[I]
    Power_S[I] = (1-FLAMING_FRACTION[b[I]]) * Power[I]
        
    HeatFlux_F[I,0] = HEAT_FLUX_MIN[b[I]] # kW/m2, lower bound
    HeatFlux_F[I,1] = (HEAT_FLUX_MAX[b[I]] + HEAT_FLUX_MIN[b[I]]) / 2. # kW/m2, mean
    HeatFlux_F[I,2] = HEAT_FLUX_MAX[b[I]] # kW/m2, upper bound
    
    area_F = 1000. * Power_F / HeatFlux_F # m2
    
    return (Power_F, Power_S, HeatFlux_F, area_F)
    
    