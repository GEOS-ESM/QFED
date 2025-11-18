# qfed_biome_emission_tuning
Scripts for tuning the QFED biomass burning biome emissions, prepared during the EIS-Fire 2021 pilot program. During the pilot program, preliminary scripts were prepared to test scaling the QFED biome-dependent emission coefficients from equation (24) in Darmenov and da Silva (2015) using the then current MODIS-based QFED emissions. 

The original tuning experiments were located in /discover/nobackup/projects/eis_fire/experiments/ and included a control run with all biomass burning emissions turned on, a run with biomass burning emissions turned off (nobb), and four experiments with emissions turned on for each of the individual biomes (grassland, savannah, extratropical forests, and tropical forests). These experiments have since been removed from the shared project space, but monthly mean output is available in /discover/nobackup/projects/eis_fire/sandbox/pawales/eis_pilot/c180R_J10p17p1dev_aura/.

For the preliminary tuning, emission coefficients are determined by minimizing the difference between experimental and GEOS-FP AOD where the GEOS-FP assimilation averaging kernel > 0.85 (i.e, more sensitive to observations) and the fraction of the AOD from biomass burning is greater than a specified threshold. Please note that the resulting scaling factors (alpha) are applied to those already in QFED such that alpha = 1 is no change to the existing biome scaling factor.

## In tools/biome_tuning.py:
- cost_jo_alpha: Calculate the cost function as the sum of the difference between observed and experimental AOD (J_o) and the difference between alpha scaling factors and 1 (J_b, i.e., minimizing the difference between optimized and current QFED factors). Option to apply weighting factors to the J_o calculation.

- cost_jo: Calculate the cost function as the sum of the difference between observed and experiemental AOD (J_o) only.

- scale_biomes: Performs a least squares minimization that calls either of the above cost functions. Returns a dictionary of alpha scaling factors for each specified biome.

## Sample scripts:
- example_scaling.wsa_weighted.py: A short example for calculating biome scaling factors. It calls the scale_biomes function and supplies a weighting factor that is based on surface area. In the supplied yaml file, I'm using a monthing mean file that was filtered for where the fraction of AOD from biomass burning (f_bb) > 80%, but we tested other options, which are available in the sandbox directory.

## Scripts not in repo:
I haven't cleaned up any of our plotting scripts yet. There are scripts for checking the correlation of AOD from different biomes to determine which biomes need to be combined (GL, SV, and TF biomes were all found to have r > 0.6 for the 2020 season). 

We also had done a lot of testing on using different weighting factors for the J_o term in the cost function, including weighting based on the fraction of f_bb instead of having a strict f_bb threshold cut off. We looked into a biome-balanced selection of observations (leave some obs for testing) and iterative scaling of the biome factors.
