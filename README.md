# QFED

The Quick Fire Emissions Dataset (QFED) uses satellite-retrieved fire radiative power (FRP) to estimate fire emissions globally. The principal QFED product (Level 3A) contains gridded FRP stratified by biome. The Level 3B product provides gridded biomass burning emissions for a number of pyrogenic species. 

Two QFED product systems are maintained by the NASA Global Modeling and Assimilation Office (GMAO):
- Near real-time daily emissions used operationally in the GEOS Forward Processing (FP) system and in the GEOS Composition Forecasting (CF) system.
- An extended historical dataset of daily emissions from March 2000 to present.

  
## Steps to Build QFED
### Load Modules
#### NCCS
```
module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15
module load GEOSenv
```
#### GMAO Desktops
```
module use -a /ford1/share/gmao_SIteam/modulefiles
module load GEOSenv
```
### Clone the Repository
```
mepo clone git@github.com:GEOS-ESM/QFED.git
```
This will clone the main branch. Use -b to clone a specific branch.

### Build and Install
```
cd QFED
source @env/g5_modules.sh
./cmake_it
cmake --build build --target install -j 6
```





