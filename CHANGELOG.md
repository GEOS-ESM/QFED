# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-09-23

### Fixed 

- This fix addresses the minor problem raised by PR review [2025-09-09]
- This fix fixes issue of accidental exclusion of wetlands fires. [2025-09-03]
- This fix ensures the syntax compatibility across different python version. [2025-09-03]
- This fix ensures that fire detections across the anti-meridian are retained correctly without accidental exclusion. [2025-08-28]
- The standard name for time in the emissions output files

### Added 
- Added handling of no L2 fire detections in `inventory.py`, `qfed_l3a.py`, and `qfed_l3b.py`, write placeholder L3A and let L3B proceed [25-10-14]
- Added the `analysis_frp_scaling.py` and `lib_frp_scaling.py` to for log-log FRP density regression tuning [25-10-07]
- Added the `alpha_factor.yaml` to save all the emission coefficient related parameter [25-10-02]
- Added the `config_NRT.yaml` adapted for the the l3b [2025-09-23]
- Added the bug fixed `frpscaling.py` [2025-09-23]
- Added the background e-folding time into the global attribute [2025-09-23]
- Added NOAA-21/VJ2 option in `classification_products.py`, `fire_products.py`, `geolocation_products.py`, `config.yaml`, and `qcscalingfactors.yaml`
- Added the configuration to provide the path and the naming convention for the forecast FRP density in config.yaml
- Added several functions in emissions.py [2025-09-18]
  - Added `Emissions._save_forecast(l3a_density_file, compress=False, fill_value=1e15, diskless=False)` in emissions.py to save the FRP density
  - Added helpers function `_make_var_name(inst, sat, biome_code)` to build safe, canonical var names and `_platform_label(...)` to produce per-variable provenance strings like `viirs/vj1`
- Added several maps `canonical_instrument` and `canonical_satellite` in instruments.py
- Added Emissions._save_forecast(l3a_density_file, compress=False, fill_value=1e15, diskless=False) in emissions.py to save the FRP density [2025-09-18]
- Added metadata of **number_of_input_files** in the l3a gridded FRP output
- Code to qfed_l3a.py to generate output directories for FRP if they do not already exist
- Trace gases needed for GEOS CF to the list of emissions in qfed_l3b.py
- A second emission factors yaml file with values from Andreae 2019
- time_increment was added as an attribute to the emissions output files as it is needed to compute monthly means
- QC for erroneously large values of FRP in the input files (max value selected is 24000 MW)
- Capability to include dampened FRP density in tomorrow's emissions calculation in obscured pixels
	
### Changed 

- Modified `fire_products.py`, `cli_utils.py` to fix unclosed NetCDF file handles and improve I/O lifecycle management [25-10-14]
- Modified `config_NRT.yaml`, `classification_products.py`, `geolocation_products.py`, `instruments.py`, `utils.py`, `frp.py`, and `emissions.py` to Unify variable and file naming across QFED v3 (align with Code 619 conventions) [25-10-14]
- Updated values in `alpha_factor.yaml` based new log-log regression analysis. [25-10-06]
- Modified the `set_parameters` in `emissions.py` [25-10-02]
- Modified the `qfed_l3b.py`, `emission.py`, `instruments.py`, and `cli_utils.py` to save the FRP density forecast per sensor [2025-09-23]
- Streamlined process() preamble for forecast backgrounds [2025-09-18]
  - Background FRP density loading replaced with a small helper `load_frp_density` that prefers today’s forecast combined file, else zeros
  - Writing combined forecast into a **qfedvxx.frp_fcs.YYYYMMDD.nc4**
- modified **get_category** in vegetation.py to allow the list of biome type for all the fire detection [2025-09-03]
- modified helper function **_process_fire**, **_process_fire_water**, **_process_fire_coast**, and **_process_fire_land** in frp.py to use open water information in IGBP for land pixel restoring [2025-09-03]
- modified helper function **_fire_place** in classification_products.py to support VIIRS processing [2025-09-03]
- modified helper function **_place** in classification_products.py to support MODIS processing [2025-09-03]
- Contact information in netcdf output files from Anton to the GMAO website
- Changed version in version.py to 3.1.1 in response to changes to the output files and additional QC

### Removed 

- Removed the emission coefficient related parameters in `qcscalingfactor.yaml` [25-10-02]
- removed the commented line made during the developments [2025-09-04]

### Deprecated 



## [3.1.0] - 2025-01-16 

### Added 
- Restored and updated plumerise capability to work with McBEF. 
- Notebooks directory to collect usage examples, etc. 
- Safegueard against negative FRP
- An exception for corrupted input files
	
### Changed
- Updated pyobs to v1.2.0

### Removed 

### Deprecated 


## [3.0.0] - 2022-06-16

### Added
- Complete refactoring since QFED 2.x on CVS
- VIIRS

 
