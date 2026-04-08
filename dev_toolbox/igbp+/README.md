# QFED IGBP+ tool box

Utilities to generate the IGBP, static heat source, gas flaring, and volcano classification for QFED fire pixil screening
If running on NCCS, as of 6 Apr 2026, there is python dependency error with loading excel files. As a temporary fix, ml use -a /home/mathomp4/modulefiles-SLES15
  ml python/MINIpyD/3.14
however, this is a mini environment that can change at any time.

---

## Repository Structure

```
├── README.md
├── config.yaml                                # User config (raw_data_root, igbp_output_root, figure_root)
├── gen_global_MODIS_IGBP                      # Generates GL_IGBP_MODIS.YYYY.nc
├── gen_global_volcano_source.py               # Generates GL_GVP_VOLCANO.nc
├── gen_global_gasflaring_source.py            # Generates GL_VIIRS_GASFLARING.YYYY.nc
├── gen_global_daily_VIIRS_static_source.py    # Generates daily CSV with binary status of fire detection
└── gen_global_VIIRS_static_source.py          # Generates counts of annual "fires" considered static sources
└── gen_global_igbp_plus.py                    # Concatenates IGBP, volcanoes, gas flaring, and static sources to a single file to be used as input to QFED

```

---

## Configuration

### `config.yaml`

Defines global defaults so command lines stay simple:

```yaml
raw_data_root: "./IGBP_SUPPORT_FILES"
igbp_input: "/Dedicated/jwang-data/shared_satData/OPNL_FILDA/DATA/LAND_COVER_SINO"
igbp_output_root: "./IGBP+"
vnp14img_root: "/Dedicated/jwang-data2/shared_satData/OPNL_FILDA/DATA/LEV1B"
figure_root: "./FIG"
IGBP_resolution: 2400
PLUS_resolution: 480
```
- `raw_data_root`: where the intermedia files are stored
- `igbp_input`: where the MODIS IGBP files are stored
- `igbp_output_root`: where the IGBP+ file will be output
- `vnp14img_root`:  where the VIIRS I-band active fire files are stored
- `figure_root`: where the verification files are stored
- `IGBP_resolution`: resolution (per 10 degree tile) of the MODIS IGBP data
- `PLUS_resolution`: resolution (per 10 degree tile) of the volcano, gasflaring, and static heat source data

---
## Usage

### 1. MODIS IGBP data

Execute the ``gen_global_MODIS_IGBP.py`` to concatenate [MODIS IGBP file](https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/61/MCD12Q1/) from discreet the sinusodial tiles to a global file 

```
python gen_global_MODIS_IGBP.py --year 2019
```

### 2. Global Volcanism data

Execute the ``gen_global_volcano_source.py`` to processthe [Global Volcanism Program database](https://volcano.si.edu/volcanolist_holocene.cfm) into global grided data.

```
python gen_global_volcano_source.py
```

### 3. Global Gasflaring data

Execute the ``gen_global_gasflaring_source.py`` to convert the [Global Gas Flaring Dataset](https://eogdata.mines.edu/products/vnf/global_gas_flare.html) into global grided data.
```
python gen_global_gasflaring_source.py
```


### 4. Global Static heat source data
This needs to be run separately for each satellite you want to find static sources for. Examples for $sat are VNP, VJ1, and VJ2. The input files are from HTTPS://DOI.ORG/10.5067/VIIRS/{sat}14IMG.002.
- Step one: generate the daily static heat source based on daily VIIRS I-band fire detection
```
python gen_global_daily_VIIRS_static_source.py $sat --start 2019-01-01 --end 2019-12-31 --fresh_csv True
```
- Step two: conduct revisit cycle analysis on the daily VIIRS static source 
```
python gen_global_VIIRS_static_source.py --sat $sat --year 2019
```

### 5. Global IGBP+ file
With data from step 1 ~ step 4 ready, excute the ``gen_global_igbp_plus.py`` to concatenate all dataset into one

```
python gen_global_igbp_plus.py --year 2019
```
- --year: the year of the IGBP+ file to be generated









