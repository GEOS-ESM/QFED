The codes in this directory (dev_toolbox/frpscaling) are used to anchor the biome level FRP to a single satellite. 
analysis_frp_scaling.py was originally written for use with QFED v3.2 to anchor all VIIRS satellites to AQUA MODIS 
for the single year of 2024 (source code was found in the directory emissionstuning). 
The v3.2 version accounts for additional scaling that was needed to account for changes between 
MODIS collection 5 and MODIS collection 6.1. This was encoded in a varible called c6scale
For future versions, analysis_frp_scaling_multiyear.py was added so that data from multiple years could be considered for the anchoring.
c6scale was removed as there is no longer a need for this factor.
