# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-02-20

### Fixed 
- The standard name for time in the emissions output files

### Added 
- Code to qfed_l3a.py to generate output directories for FRP if they do not already exist
- Trace gases needed for GEOS CF to the list of emissions in qfed_l3b.py
- A second emission factors yaml file with values from Andreae 2019
- time_increment was added as an attribute to the emissions output files as it is needed to compute monthly means
- QC for erroneously large values of FRP in the input files (max value selected is 24000 MW)
	
### Changed 
- Contact information in netcdf output files from Anton to the GMAO website
- Changed version in version.py to 3.1.1 in response to changes to the output files and additional QC

### Removed 

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

 
