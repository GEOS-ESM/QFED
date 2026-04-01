import os
import numpy as np
import rasterio
import rasterio.windows
from scipy import stats
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
import xml.etree.ElementTree as ET
from lib_IGBP_plus import *

# ================================================================
# CONFIGURATION
# ================================================================
data_dir   = '/discover/nobackup/acollow/GPM/'
tif_path   = os.path.join(data_dir, 'peatGPA22WGS_2cl.tif')
aux_xml    = os.path.join(data_dir, 'peatGPA22WGS_2cl.tif.aux.xml')
out_dir    = './GL_STATIC/'
fig_dir    = './FIG/'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

FILL_VALUE  = 255
NUM_CELLS   = 480       # 480 cells/tile = ~500 m QFED sinusoidal grid
CHUNK_ROWS  = 500       # rows to process at a time (increase if RAM allows)
flag_verify = True

# Class definitions — swap labels if .aux.xml confirms otherwise
CLASS_CONFIG = {
    1: {
        'name'        : 'peat_dominated',
        'description' : 'Peat-dominated (continuous peatland, >50% peat)',
        'color'       : 'saddlebrown',
    },
    2: {
        'name'        : 'peat_mosaic',
        'description' : 'Peat in soil mosaic (<50% peat)',
        'color'       : 'peru',
    },
}
PEAT_CLASSES = [1, 2]

# ================================================================
# STEP 0 — Print .aux.xml class info if available
# ================================================================
print("=" * 60)
print("STEP 0 — Reading auxiliary metadata")
print("=" * 60)
if os.path.exists(aux_xml):
    try:
        tree = ET.parse(aux_xml)
        root = tree.getroot()

        def print_xml(element, indent=0):
            text = element.text.strip() if element.text and element.text.strip() else ''
            print(' ' * indent + f"<{element.tag}>" +
                  (f" {text}" if text else ''))
            for child in element:
                print_xml(child, indent + 2)

        print_xml(root)
    except Exception as e:
        print(f"  Could not parse .aux.xml: {e}")
        print("  Reading as raw text instead:")
        with open(aux_xml, 'r') as f:
            print(f.read())
else:
    print(f"  {aux_xml} not found — skipping metadata check.")

# ================================================================
# STEP 1 — Inspect the GeoTIFF
# ================================================================
print("\n" + "=" * 60)
print("STEP 1 — GeoTIFF inspection")
print("=" * 60)
with rasterio.open(tif_path) as src:
    nrows, ncols = src.shape
    transform    = src.transform
    nodata       = src.nodata
    crs          = src.crs
    bounds       = src.bounds
    dtype        = src.dtypes[0]
    d            = src.read(1)

print(f"  File      : {tif_path}")
print(f"  CRS       : {crs}")
print(f"  Bounds    : {bounds}")
print(f"  Shape     : {nrows} rows x {ncols} cols")
print(f"  Data type : {dtype}")
print(f"  NoData    : {nodata}")
print(f"  Pixel size: {transform.a:.6f} deg (lon) x "
      f"{abs(transform.e):.6f} deg (lat)")

vals, counts = np.unique(d, return_counts=True)
print(f"\n  Value distribution:")
for v, c in zip(vals, counts):
    pct   = 100.0 * c / d.size
    label = CLASS_CONFIG.get(int(v), {}).get('description', 'NoData/Unknown')
    print(f"    Value {int(v):>5} : {c:>14,} pixels  "
          f"({pct:6.2f}%)  — {label}")
del d   # free memory

# ================================================================
# STEP 2 — Initialise sinusoidal grid and accumulators
# ================================================================
print("\n" + "=" * 60)
print("STEP 2 — Sinusoidal grid setup")
print("=" * 60)
grid_sinu = SinusoidalGrid(num_cells=NUM_CELLS)
print(f"  Grid size  : {grid_sinu.n_zonal} (E) x "
      f"{grid_sinu.n_meridional} (N) cells")
print(f"  Resolution : {grid_sinu.resol_h:.2f} m (H) x "
      f"{grid_sinu.resol_v:.2f} m (V)")
print(f"  Extent H   : {-grid_sinu.halfHoriLength:.0f} to "
      f"{grid_sinu.halfHoriLength:.0f} m")
print(f"  Extent V   : {-grid_sinu.halfVertLength:.0f} to "
      f"{grid_sinu.halfVertLength:.0f} m")

# Accumulators — one per peat class + total valid pixels
acc = {
    cls: np.zeros((grid_sinu.n_zonal, grid_sinu.n_meridional),
                  dtype=np.int32)
    for cls in PEAT_CLASSES
}
total_acc = np.zeros((grid_sinu.n_zonal, grid_sinu.n_meridional),
                     dtype=np.int32)

# Bin edges for scipy binned_statistic_2d
e_bins = grid_sinu.easting           # (n_zonal+1,)    ascending
n_bins = grid_sinu.northing[::-1]    # (n_meridional+1,) ascending

# ================================================================
# STEP 3 — Chunked read, project, and bin
# ================================================================
print("\n" + "=" * 60)
print(f"STEP 3 — Chunked processing ({CHUNK_ROWS} rows/chunk)")
print("=" * 60)

with rasterio.open(tif_path) as src:
    col_idx  = np.arange(ncols)
    # Pixel-centre longitudes — constant for every row
    lons_row = transform.c + (col_idx + 0.5) * transform.a   # (ncols,)

    n_chunks           = int(np.ceil(nrows / CHUNK_ROWS))
    total_valid_pixels = 0

    for chunk_i in range(n_chunks):
        row_start = chunk_i * CHUNK_ROWS
        row_end   = min(row_start + CHUNK_ROWS, nrows)
        n_this    = row_end - row_start

        if chunk_i % 20 == 0 or chunk_i == n_chunks - 1:
            pct_done = 100.0 * chunk_i / n_chunks
            print(f"  Chunk {chunk_i+1:5d}/{n_chunks} "
                  f"| rows {row_start:6d}–{row_end:6d} "
                  f"| {pct_done:5.1f}% done")

        # --- Read chunk ---
        window = rasterio.windows.Window(
            col_off=0, row_off=row_start,
            width=ncols, height=n_this
        )
        chunk = src.read(1, window=window)   # (n_this, ncols)

        # --- Build pixel-centre latitudes for this chunk ---
        row_idx  = np.arange(row_start, row_end)
        lats_col = transform.f + (row_idx + 0.5) * transform.e  # (n_this,)

        # Broadcast lat/lon to full chunk shape
        lats_2d = np.repeat(lats_col[:, np.newaxis], ncols, axis=1)
        lons_2d = np.tile(lons_row[np.newaxis, :],   (n_this,  1))

        # Flatten
        vals_flat = chunk.ravel()
        lats_flat = lats_2d.ravel()
        lons_flat = lons_2d.ravel()

        # --- Mask nodata (255) ---
        valid = (vals_flat != FILL_VALUE)
        if not np.any(valid):
            continue

        lats_v = lats_flat[valid]
        lons_v = lons_flat[valid]
        vals_v = vals_flat[valid]
        total_valid_pixels += valid.sum()

        # --- Project to sinusoidal ---
        xs_sinu, ys_sinu = geog_to_sinu(lats_v, lons_v)

        # --- Bin each peat class ---
        for cls in PEAT_CLASSES:
            cls_mask = (vals_v == cls)
            if not np.any(cls_mask):
                continue
            c_count, _, _, _ = stats.binned_statistic_2d(
                xs_sinu, ys_sinu,
                values=cls_mask.astype(np.int32),
                statistic='sum',
                bins=[e_bins, n_bins]
            )
            acc[cls] += c_count.astype(np.int32)

        # --- Bin total valid pixels ---
        t_count, _, _, _ = stats.binned_statistic_2d(
            xs_sinu, ys_sinu,
            values=np.ones(len(xs_sinu), dtype=np.int32),
            statistic='sum',
            bins=[e_bins, n_bins]
        )
        total_acc += t_count.astype(np.int32)

print(f"\n  Total valid pixels processed : {total_valid_pixels:,}")
print(f"  Total peat-dominated pixels  : {acc[1].sum():,}")
print(f"  Total peat-mosaic pixels     : {acc[2].sum():,}")

# ================================================================
# STEP 4 — Derive output fields
# ================================================================
print("\n" + "=" * 60)
print("STEP 4 — Computing output fields")
print("=" * 60)

has_data     = (total_acc > 0)
any_peat_acc = sum(acc[cls] for cls in PEAT_CLASSES)

# --- Per-class fractional cover [0, 1] ---
frac = {}
for cls in PEAT_CLASSES:
    f = np.full(acc[cls].shape, -9999.0, dtype=np.float32)
    f[has_data] = (acc[cls][has_data].astype(np.float32) /
                   total_acc[has_data].astype(np.float32))
    frac[cls] = f[:, ::-1].T     # → (northing, easting)

# --- Combined any-peat fraction ---
frac_any = np.full(any_peat_acc.shape, -9999.0, dtype=np.float32)
frac_any[has_data] = (any_peat_acc[has_data].astype(np.float32) /
                       total_acc[has_data].astype(np.float32))
frac_any = frac_any[:, ::-1].T

# --- Dominant-class mask ---
#   0   = non-peat land
#   1   = peat-dominated
#   2   = peat-in-soil-mosaic
#   255 = no data
peat_mask = np.full(any_peat_acc.shape, FILL_VALUE, dtype=np.uint8)
peat_mask[has_data & (any_peat_acc == 0)] = 0       # non-peat land
peat_mask[has_data & (acc[2] > 0)]        = 2       # mosaic
peat_mask[has_data &                                # dominated wins
          (acc[1] > 0) &
          (acc[1] >= acc[2])]             = 1
peat_mask = peat_mask[:, ::-1].T

print(f"  Peat-dominated cells  (1) : {(peat_mask == 1).sum():,}")
print(f"  Peat-mosaic cells     (2) : {(peat_mask == 2).sum():,}")
print(f"  Non-peat land cells   (0) : {(peat_mask == 0).sum():,}")
print(f"  No-data cells       (255) : {(peat_mask == 255).sum():,}")
print(f"  Max frac (dominated)      : "
      f"{frac[1][frac[1] > -9999].max():.4f}")
print(f"  Max frac (mosaic)         : "
      f"{frac[2][frac[2] > -9999].max():.4f}")
print(f"  Max frac (any peat)       : "
      f"{frac_any[frac_any > -9999].max():.4f}")

# ================================================================
# STEP 5 — Write CF-compliant NetCDF
# ================================================================
print("\n" + "=" * 60)
print("STEP 5 — Writing NetCDF")
print("=" * 60)

savename = f"{out_dir}GL_PEAT_GPA22.nc"
ncid     = Dataset(savename, 'w', format='NETCDF4')

ncid.createDimension('easting',  grid_sinu.n_zonal)
ncid.createDimension('northing', grid_sinu.n_meridional)

# --- CRS variable ---
crs_var                           = ncid.createVariable('crs', 'i4')
crs_var[:]                        = 0
crs_var.grid_mapping_name         = (f"MODIS/VIIRS Sinusoidal "
                                      f"{grid_sinu.resol_h:6.2f}x"
                                      f"{grid_sinu.resol_v:6.2f} m")
crs_var.long_name                 = "CRS definition"
crs_var.epsg_code                 = "EPSG:4326"
crs_var.false_easting             = "0.0"
crs_var.false_northing            = "0.0"
crs_var.GeoTransform              = (f"{-grid_sinu.halfHoriLength} "
                                      f"{grid_sinu.resol_h} -0 "
                                      f"{grid_sinu.halfVertLength} -0 "
                                      f"-{grid_sinu.resol_v} ")
crs_var.pixel_coordinate_location = "pixel_upper_left_corner"
crs_var.spatial_ref               = (
    "{PROJCS[\"Sinusoidal\",GEOGCS[\"GCS_ELLIPSE_BASED_1\","
    "DATUM[\"D_ELLIPSE_BASED_1\",SPHEROID[\"S_ELLIPSE_BASED_1\","
    "6371007.2,0.0]],PRIMEM[\"Greenwich\",0.0],UNIT[\"Degree\","
    "0.0174532925199433]],PROJECTION[\"Sinusoidal\"],"
    "PARAMETER[\"False_Easting\",0.0],"
    "PARAMETER[\"False_Northing\",0.0],"
    "PARAMETER[\"Central_Meridian\",0.0],UNIT[\"Meter\",1.0]]}"
)

# --- Coordinate variables ---
e_var               = ncid.createVariable(
    'easting', 'f8', ('easting',),
    zlib=True, complevel=4, chunksizes=(grid_sinu.n_zonal,)
)
e_var[:]            = grid_sinu.easting[:-1]
e_var.standard_name = "easting"
e_var.long_name     = "easting"
e_var.units         = "meters"

n_var               = ncid.createVariable(
    'northing', 'f8', ('northing',),
    zlib=True, complevel=4, chunksizes=(grid_sinu.n_meridional,)
)
n_var[:]            = grid_sinu.northing[:-1]
n_var.standard_name = "northing"
n_var.long_name     = "northing"
n_var.units         = "meters"

# --- 2D data variables ---
chunksizes_2d = (grid_sinu.n_meridional, grid_sinu.n_zonal)

# Per-class fractions
for cls in PEAT_CLASSES:
    vname           = f"peat_fraction_class{cls}"
    v               = ncid.createVariable(
        vname, 'f4', ('northing', 'easting'),
        zlib=True, complevel=8, shuffle=True,
        chunksizes=chunksizes_2d, fill_value=-9999.0
    )
    v[:, :]         = frac[cls]
    v.long_name     = (f"Sub-grid fractional cover — "
                       f"{CLASS_CONFIG[cls]['description']}")
    v.units         = "1"
    v.valid_range   = np.array([0.0, 1.0], dtype='f4')
    v.grid_mapping  = 'crs'
    v.comment       = (f"Fraction of sinusoidal grid cell pixels "
                       f"classified as class {cls} in source GeoTIFF")

# Combined any-peat fraction
v               = ncid.createVariable(
    'peat_fraction_any', 'f4', ('northing', 'easting'),
    zlib=True, complevel=8, shuffle=True,
    chunksizes=chunksizes_2d, fill_value=-9999.0
)
v[:, :]         = frac_any
v.long_name     = "Total peat fractional cover (class 1 + class 2)"
v.units         = "1"
v.valid_range   = np.array([0.0, 1.0], dtype='f4')
v.grid_mapping  = 'crs'
v.comment       = "Sum of peat_fraction_class1 and peat_fraction_class2"

# Dominant-class peat mask
v               = ncid.createVariable(
    'peat_mask', 'u1', ('northing', 'easting'),
    zlib=True, complevel=8, shuffle=True,
    chunksizes=chunksizes_2d, fill_value=FILL_VALUE
)
v[:, :]         = peat_mask
v.long_name     = "Dominant peat class per sinusoidal grid cell (GPA22)"
v.legend        = ("0: non-peat land, "
                   "1: peat-dominated (continuous), "
                   "2: peat-in-soil-mosaic, "
                   "255: no data")
v.valid_range   = np.array([0, 255], dtype='u1')
v.grid_mapping  = 'crs'

# --- Global attributes ---
ncid.description    = (
    "Global Peatland Area 2022 (GPA22) remapped to QFED MODIS/VIIRS "
    f"sinusoidal grid ({grid_sinu.resol_h:.1f} x "
    f"{grid_sinu.resol_v:.1f} m). "
    "Class 1 = peat-dominated; Class 2 = peat-in-soil-mosaic; "
    "255 = no data."
)
ncid.Conventions    = 'CF-1.8'
ncid.institution    = ('Global Modeling and Assimilation Office, '
                        'NASA/GSFC')
ncid.data_source    = ('peatGPA22WGS_2cl.tif '
                        '(WGS84 2-class peatland, GPA22)')
ncid.source_classes = ('1: peat-dominated; '
                        '2: peat-in-soil-mosaic; '
                        '255: nodata')
ncid.history        = ('Reprojected from WGS84 GeoTIFF to '
                        'QFED MODIS/VIIRS sinusoidal grid using '
                        'scipy.stats.binned_statistic_2d')
ncid.close()
print(f"  Saved → {savename}")

# ================================================================
# STEP 6 — Verification plots
# ================================================================
if flag_verify:
    print("\n" + "=" * 60)
    print("STEP 6 — Verification plots")
    print("=" * 60)

    ncid       = Dataset(savename, 'r')
    ncid.set_auto_mask(False)
    peat_src   = ncid['peat_mask'][:]
    frac_any_v = ncid['peat_fraction_any'][:]
    northing_v = ncid['northing'][:]
    easting_v  = ncid['easting'][:]
    ncid.close()

    # Shared map decoration function
    waterClr     = (203/255., 236/255., 254/255.)
    waterEdgeClr = (57/255.,  193/255., 242/255.)
    landClr      = (212/255., 212/255., 212/255.)
    lineColor    = 'dimgray'
    linewidth    = 1

    def add_map_features(ax):
        ax.set_global()
        ax.add_feature(cfeature.OCEAN,
                       zorder=0, edgecolor=waterEdgeClr, color=waterClr)
        ax.add_feature(cfeature.LAND.with_scale('10m'),
                       edgecolor=lineColor, color=landClr, zorder=1)
        ax.add_feature(cfeature.LAKES.with_scale('50m'),
                       edgecolor=waterEdgeClr, color=waterClr, zorder=2)
        ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                       linewidth=linewidth * 0.5,
                       edgecolor=waterEdgeClr, zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'),
                       linewidth=linewidth * 0.5,
                       edgecolor=lineColor, linestyle=':', zorder=4)

    # ------------------------------------------------------------------
    # PLOT 1 — Binary class map (class 1 vs class 2)
    # ------------------------------------------------------------------
    print("  Generating Plot 1: class map ...")
    fig, ax = plt.subplots(
        1, 1, figsize=(14, 7),
        subplot_kw={'projection': ccrs.Robinson()}
    )
    add_map_features(ax)

    for cls in PEAT_CLASSES:
        idx = np.where(peat_src == cls)
        if len(idx[0]) == 0:
            continue
        lat_p, lon_p = get_coordinates(northing_v, easting_v, idx)
        ax.scatter(
            lon_p, lat_p,
            s=0.3,
            color=CLASS_CONFIG[cls]['color'],
            transform=ccrs.PlateCarree(),
            zorder=599, rasterized=True,
            label=CLASS_CONFIG[cls]['description']
        )

    ax.set_title('QFED Peatland Mask (GPA22, ~500 m Sinusoidal)',
                 fontsize=13)
    handles = [
        Line2D([0], [0],
               label=CLASS_CONFIG[cls]['description'],
               lw=1, ls='', marker='o', markersize=8,
               color=CLASS_CONFIG[cls]['color'])
        for cls in PEAT_CLASSES
    ]
    ax.legend(handles=handles, frameon=False, ncol=2,
              fontsize=10, loc='lower center',
              bbox_to_anchor=(0.5, -0.07))

    plt.tight_layout()
    out1 = f'{fig_dir}MAP.QFED_Peatland_GPA22_classes.png'
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved → {out1}")

    # ------------------------------------------------------------------
    # PLOT 2 — Fractional peat cover heatmap
    # ------------------------------------------------------------------
    print("  Generating Plot 2: fractional cover map ...")

    STRIDE     = 4
    frac_disp  = frac_any_v[::STRIDE, ::STRIDE].copy()
    frac_disp[frac_disp < 0] = np.nan

    northing_sub     = northing_v[::STRIDE]
    easting_sub      = easting_v[::STRIDE]
    nn, ee           = np.meshgrid(northing_sub, easting_sub, indexing='ij')
    lat_img, lon_img = sinu_to_geog(ee, nn)

    fig, ax = plt.subplots(
        1, 1, figsize=(14, 7),
        subplot_kw={'projection': ccrs.Robinson()}
    )
    add_map_features(ax)

    sc = ax.scatter(
        lon_img.ravel(), lat_img.ravel(),
        c=frac_disp.ravel(),
        s=0.05, cmap='YlOrBr',
        vmin=0, vmax=1,
        transform=ccrs.PlateCarree(),
        zorder=599, rasterized=True
    )
    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal',
                        pad=0.03, shrink=0.6, aspect=40)
    cbar.set_label('Peat fractional cover (any class)', fontsize=10)
    ax.set_title(
        'QFED Peatland Fractional Cover (GPA22, ~500 m Sinusoidal)',
        fontsize=13
    )

    plt.tight_layout()
    out2 = f'{fig_dir}MAP.QFED_Peatland_GPA22_fraction.png'
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved → {out2}")

    print("\n" + "=" * 60)
    print("All done!")
    print(f"  NetCDF : {savename}")
    print(f"  Plot 1 : {out1}")
    print(f"  Plot 2 : {out2}")
    print("=" * 60)
