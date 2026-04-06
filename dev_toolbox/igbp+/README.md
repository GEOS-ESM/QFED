# QFED IGBP+ tool box

Utilities to generate the IGBP, static heat source, gas flaring, and volcano classification for QFED fire pixil screening

---

## Repository Structure

```
├── README.md
├── config.yaml                                # User config (raw_data_root, igbp_output_root, figure_root)
├── gen_global_MODIS_IGBP                      # Generate INFOR/<product>/<YYYY>/<YYYY-MM-DD>.csv
├── gen_global_volcano_source.py               # Download missing/corrupted files (wget + token; supports --output_dir)
├── gen_global_gasflaring_source.py            # Compare INFOR vs local files; list missing (with downloadsLink)
├── gen_global_daily_VIIRS_static_source.py    # Check missing HHMM granules per day (VIIRS 6-min / MODIS 5-min)
└── gen_global_VIIRS_static_source.py          # Daily bars arranged by month; highlight missing days
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
- Step one: generate the daily static heat source based on daily VIIRS I-band fire detection
```
python gen_global_daily_VIIRS_static_source.py $sat --start 2019-01-01 --end 2019-12-31 --fresh_csv True
```
- Step two: conduct revisit cycle analysis on the daily VIIRS static source 
```
python gen_global_VIIRS_static_source.py --sat VNP --year 2019
```














