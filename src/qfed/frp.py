"""
Gridded FRP products.
"""

import os
import logging
from datetime import datetime
from glob import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import netCDF4 as nc
import yaml
from binObs_ import binareas, binareasnr

from qfed import grid
from qfed import instruments
from qfed import fire
from qfed import VERSION


# Global variable to hold shared data
_SHARED_IGBP = None
_SHARED_WATERMASK = None

# ---------------------------------------------------------------------------
# Module-level QC scaling cache — read once per process
# ---------------------------------------------------------------------------
_QC_SCALING_CACHE: dict | None = None

def _get_qc_scaling(path: str = 'qcscalingfactors.yaml') -> dict:
    """
    Cached read of QC scaling factors.
    Safe in ProcessPoolExecutor because each process has its own globals.
    """
    global _QC_SCALING_CACHE
    if _QC_SCALING_CACHE is None:
        with open(path) as fh:
            _QC_SCALING_CACHE = yaml.safe_load(fh)
        logging.debug("Loaded QC scaling factors from disk (cached).")
    return _QC_SCALING_CACHE


# ---------------------------------------------------------------------------
# Binning helper
# ---------------------------------------------------------------------------
def _binareas(
    lon: np.ndarray,
    lat: np.ndarray,
    data: np.ndarray,
    im: int,
    jm: int,
    grid_type,
) -> np.ndarray:
    """Bin pixel data (area or FRP) onto the output grid."""
    if len(data) == 0:
        return np.zeros((im, jm))

    if grid_type == grid.GridType.LATLON_GEOS:
        return binareas(lon, lat, data, im, jm)
    elif grid_type == grid.GridType.LATLON_3600x1800:
        return binareasnr(lon, lat, data, im, jm)
    else:
        raise NotImplementedError(
            f"Data binning does not support grid type '{grid_type}'."
        )


# ---------------------------------------------------------------------------
# Helper: build the FRP string-keyed dict
# ---------------------------------------------------------------------------
def _make_frp_dict(im: int, jm: int) -> dict:
    """
    Return a dict mapping biome string key -> zero array.

    Using plain strings as keys (e.g. 'tf', 'xf', 'sv', 'gl', 'pt')
    guarantees no object-identity ambiguity when dicts are pickled
    across process boundaries with ProcessPoolExecutor.

    The dict is built from fire.BIOMASS_BURNING so it automatically
    reflects any biomes added there (including the new peat biome).
    """
    return {bb.type.value: np.zeros((im, jm)) for bb in fire.BIOMASS_BURNING}


# ---------------------------------------------------------------------------
# Per-granule worker — runs in a separate process
# ---------------------------------------------------------------------------
def _process_granule(
    satellite: str,
    gp_file_pattern: str,
    fp_file: str,
    im: int,
    jm: int,
    grid_type,
) -> dict | None:
    """
    Process one (geolocation, fire-product) granule pair.

    Runs in a worker process. All HDF5/NetCDF4 I/O is isolated to the
    worker process, so there is no shared HDF5 library state. Reader
    instances are created fresh per granule — they are stateless with
    respect to files (no persistent handles) so construction is cheap.

    IGBP and Watermask are loaded from global shared variables 

    FRP is keyed by plain strings (bb.type.value, e.g. 'tf', 'xf',
    'sv', 'gl', 'pt') to avoid object-identity issues when the result dict
    is unpickled in the main process.

    Returns:
        dict  — partial accumulator arrays on success
        None  — granule legitimately skipped (no geo file or no fires)

    Raises:
        Exception — on any processing error, allowing the main process
                    to log the full traceback via future.result()
    """
    import qfed.geolocation_products as geolocation_products
    import qfed.fire_products as fire_products
    import qfed.classification_products as classification_products
    from qfed.instruments import Satellite

    fp_filename = os.path.basename(fp_file)

    # ---- locate geolocation file ----------------------------------------
    match = glob(gp_file_pattern)
    if not match:
        logging.warning(
            f"Skipping '{fp_filename}': geolocation file "
            f"'{gp_file_pattern}' not found."
        )
        return None

    gp_file = match[0]

    # ---- create reader instances ----------------------------------------
    # Cheap: no file handles are stored. Each process gets its own
    # instances, so there is no shared mutable state between workers.
    platform  = Satellite(satellite)
    gp_reader = geolocation_products.create(platform)
    fp_reader = fire_products.create(platform)
    cp_reader = classification_products.create(platform)

    logging.info(f"Processing '{fp_filename}'.")

    # No try/except here — exceptions propagate to the main process where
    # they are caught by future.result() and logged with a full traceback.

    # ---- partial result buffers -----------------------------------------
    # FRP uses plain string keys to avoid object-identity issues when
    # this dict is pickled back to the main process.
    result = {
        'area_land':    np.zeros((im, jm)),
        'area_water':   np.zeros((im, jm)),
        'area_cloud':   np.zeros((im, jm)),
        'area_unknown': np.zeros((im, jm)),
        'frp':          _make_frp_dict(im, jm),
    }

    # ---- geolocation ----------------------------------------------------
    try:
        lon, lat, valid_coords, _, _ = gp_reader.get_coordinates(gp_file)
    except OSError as e:
        logging.error(f"Skipping '{fp_filename}': {e}")
        return None

    # ---- watermask (shared copy) ----------------------------------------
    global _SHARED_WATERMASK
    watermask = _SHARED_WATERMASK

    # ---- classification -------------------------------------------------
    # set_auxiliary and read both mutate cp_reader, but each process
    # owns its own instance so there is no cross-worker interference.
    cp_reader.set_auxiliary(lon=lon, lat=lat, watermask=watermask)
    cp_reader.read(fp_file)

    is_cloud      = cp_reader.get_cloud()
    is_cloud_free = cp_reader.get_cloud_free()
    area_px       = cp_reader.get_area()

    # ---- helper: apply mask + valid_coords, then flatten ----------------
    _false = np.zeros_like(valid_coords)

    def _select_area(mask):
        idx = mask & valid_coords
        return lon[idx].ravel(), lat[idx].ravel(), area_px[idx].ravel()

    # ==================================================================
    # NON-FIRE area accumulation
    # ==================================================================

    # cloud-free land
    result['area_land']    += _binareas(
        *_select_area(is_cloud_free['land']), im, jm, grid_type
    )
    # cloud-free coast → water bucket
    result['area_water']   += _binareas(
        *_select_area(is_cloud_free['coast']), im, jm, grid_type
    )
    # cloud-free water
    result['area_water']   += _binareas(
        *_select_area(is_cloud_free['water']), im, jm, grid_type
    )
    # cloud land
    result['area_cloud']   += _binareas(
        *_select_area(is_cloud['land']), im, jm, grid_type
    )
    # cloud coast → water bucket
    result['area_water']   += _binareas(
        *_select_area(is_cloud['coast']), im, jm, grid_type
    )
    # cloud water → water bucket
    result['area_water']   += _binareas(
        *_select_area(is_cloud['water']), im, jm, grid_type
    )
    # cloud unknown
    result['area_unknown'] += _binareas(
        *_select_area(is_cloud.get('unknown', _false)), im, jm, grid_type
    )

    # sanity check: cloud-free unknown should always be empty
    _, _, unk_area = _select_area(is_cloud_free.get('unknown', _false))
    if len(unk_area) > 0:
        logging.critical(
            f"Found {len(unk_area)} cloud-free unknown pixels in "
            f"'{fp_filename}'! Excluding them."
        )

    # ---- early exit if granule has no fires (after area accumulation) ---
    n_fires = fp_reader.get_num_fire_pixels(fp_file)
    if n_fires == 0:
        logging.info(f"Successfully processed '{fp_filename}' (No fires, accumulated clear/cloud area).")
        return result

    # ==================================================================
    # FIRE pixel accumulation
    # ==================================================================

    # ---- build combined fire masks once, reused 3x below ----------------
    fire_low  = cp_reader.get_fire(confidence='low')
    fire_nom  = cp_reader.get_fire(confidence='nominal')
    fire_high = cp_reader.get_fire(confidence='high')
    fire_any  = cp_reader.get_fire(confidence='non-zero')

    combined_fire = {
        surf: (fire_low[surf] | fire_nom[surf] | fire_high[surf])
        for surf in ('land', 'water', 'coast')
    }

    f_lon    = fp_reader.get_fire_longitude(fp_file)
    f_lat    = fp_reader.get_fire_latitude(fp_file)
    f_frp    = fp_reader.get_fire_frp(fp_file).copy()
    f_line   = fp_reader.get_fire_line(fp_file)
    f_sample = fp_reader.get_fire_sample(fp_file)
    f_area   = fp_reader.get_fire_pixel_area(fp_file)

    # single vectorised pass to clip FRP outliers
    np.clip(f_frp, 0, 40_000, out=f_frp)

    # ---- IGBP vegetation category (shared copy) ------------------
    global _SHARED_IGBP
    veg_masks, veg_codes = _SHARED_IGBP.get_category(f_lat, f_lon, return_codes=True)

    # promote IGBP-land pixels that QA misclassified as water/coast
    overwrite_to_land = (
        (veg_codes != 0)
        & fire_any['valid'][f_line, f_sample]   # exclude residual bowtie
    )

    i_valid = valid_coords[f_line, f_sample]

    # --- water fire pixels -----------------------------------------------
    i = (
        combined_fire['water'][f_line, f_sample]
        & ~overwrite_to_land
        & i_valid
    )
    result['area_water'] += _binareas(
        f_lon[i], f_lat[i], f_area[i], im, jm, grid_type
    )
    logging.info(
        f"Found {int(i.sum())} water fire pixels in '{fp_filename}'."
    )

    # --- coast fire pixels -----------------------------------------------
    i = (
        combined_fire['coast'][f_line, f_sample]
        & ~overwrite_to_land
        & i_valid
    )
    result['area_water'] += _binareas(
        f_lon[i], f_lat[i], f_area[i], im, jm, grid_type
    )
    logging.info(
        f"Found {int(i.sum())} coast fire pixels in '{fp_filename}'."
    )

    # --- land fire pixels ------------------------------------------------
    i_land = (
        combined_fire['land'][f_line, f_sample] | overwrite_to_land
    ) & i_valid

    result['area_land'] += _binareas(
        f_lon[i_land], f_lat[i_land], f_area[i_land], im, jm, grid_type
    )
    n_land = int(i_land.sum())
    logging.info(
        f"Found {n_land} land fire pixels in '{fp_filename}'."
    )

    # --- per-biome FRP ---------------------------------------------------
    # Use bb.type.value (plain string) as key — matches _make_frp_dict()
    if n_land > 0:
        for bb in fire.BIOMASS_BURNING:
            j = veg_masks[bb.vegetation] & i_land
            if np.any(j):
                result['frp'][bb.type.value] += _binareas(
                    f_lon[j], f_lat[j], f_frp[j], im, jm, grid_type
                )

    logging.info(f"Successfully processed '{fp_filename}'.")
    return result


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class GriddedFRP:
    """
    Grids FRP, areas of fire pixels and areas of non-fire pixels
    by aggregating data from multiple granules.
    """

    def __init__(
        self,
        sat: str,
        grid,
        finder,
        gp_reader_factory,      # accepted for API compatibility, not used internally
        fp_reader_factory,      # accepted for API compatibility, not used internally
        cp_reader_factory,      # accepted for API compatibility, not used internally
        igbp,                   # IGBPNetCDF instance or path string
        watermask_file: str = '',
        max_workers: int = 4,
        peat_file: str | None = None,
    ):
        """
        Parameters
        ----------
        peat_file : str or None, optional
            Path to the GL_PEAT_GPA22 NetCDF file.  When None (default),
            peat reclassification is disabled.
        """
        self._grid           = grid
        self._finder         = finder
        self.sat             = sat
        self._watermask_file = watermask_file
        self._max_workers    = max_workers
        self._peat_file      = peat_file

        # Defensive check in case 'None' is passed as a string from YAML config
        if self._peat_file == "None":
            self._peat_file = None

        # Accept either an IGBPNetCDF instance or a raw path string.
        # Workers always receive the path so they can build their own
        # instance without pickling large numpy arrays.
        if isinstance(igbp, str):
            self._igbp_file = igbp
        else:
            # Extract the file path from an already-instantiated object.
            self._igbp_file = igbp.file

    # ------------------------------------------------------------------
    def _zero_accumulators(self) -> None:
        """Initialise all gridded accumulator arrays to zero."""
        shape = (self.im, self.jm)
        self.area_land    = np.zeros(shape)
        self.area_water   = np.zeros(shape)
        self.area_cloud   = np.zeros(shape)
        self.area_unknown = np.zeros(shape)
        # Use plain string keys to match the worker process output.
        self.frp = _make_frp_dict(self.im, self.jm)

    def _accumulate(self, partial: dict) -> None:
        """
        Merge one granule's partial arrays into the class accumulators.
        Called in the main process only — no locking needed.
        Both self.frp and partial['frp'] use plain string keys so there
        is no object-identity ambiguity across process boundaries.
        """
        self.area_land    += partial['area_land']
        self.area_water   += partial['area_water']
        self.area_cloud   += partial['area_cloud']
        self.area_unknown += partial['area_unknown']
        for key, arr in partial['frp'].items():
            self.frp[key] += arr

    # ------------------------------------------------------------------
    def ingest(self, t_start, t_end, max_workers: int | None = None) -> None:
        """
        Ingest all granules for [t_start, t_end), processing them in
        parallel worker processes.

        Each worker process has its own HDF5/NetCDF4 library state and
        its own reader instances, completely eliminating thread-safety
        and shared-state issues. IGBP and watermask are loaded from
        file paths inside each worker and cached after first use.

        The main process accumulates results serially as futures
        complete, so accumulator arrays need no locking.

        Exceptions raised in worker processes propagate through
        future.result() and are logged with full tracebacks, making
        failures clearly distinguishable from legitimate skips.
        """
        global _SHARED_IGBP
        global _SHARED_WATERMASK
        
        # Load the IGBP data into memory ONCE in the main process
        if _SHARED_IGBP is None:
            logging.info(f"Pre-loading IGBP data into shared memory from {self._igbp_file}")
            from qfed.vegetation import IGBPNetCDF
            _SHARED_IGBP = IGBPNetCDF(self._igbp_file, peat_file=self._peat_file)
            
        # Load the watermask data into memory ONCE in the main process
        if _SHARED_WATERMASK is None:
            logging.info(f"Pre-loading watermask data into shared memory from {self._watermask_file}")
            f = nc.Dataset(self._watermask_file)
            _SHARED_WATERMASK = f.variables['watermask'][...]
            f.close()

        self.im        = self._grid.dimensions()['x']
        self.jm        = self._grid.dimensions()['y']
        self.glon      = self._grid.lon()
        self.glat      = self._grid.lat()
        self.grid_type = self._grid.type

        self._zero_accumulators()

        input_data         = self._finder.find(t_start, t_end)
        self.n_input_files = len(input_data)

        if self.n_input_files == 0:
            logging.warning("No input files found for this time interval.")
            return

        # calculate number of workers
        if self._max_workers is not None:
            smart_workers = self._max_workers
        else:
            try:
                cpu_limit = len(os.sched_getaffinity(0))
            except AttributeError:
                cpu_limit = os.cpu_count() or 6

            try:
                mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
                mem_gb = mem_bytes / (1024 ** 3)
            except ValueError:
                mem_gb = 16.0 # Safe fallback

            # Heuristic: Main process = 4GB. Each worker = ~2.5GB
            available_mem = max(mem_gb - 4.0, 0)
            mem_limit = max(1, int(available_mem // 2.5))
            smart_workers = min(cpu_limit, mem_limit, self.n_input_files)        

        logging.info(
            f"Processing {self.n_input_files} granule(s) with "
            f"up to {smart_workers} parallel worker process(es)."
        )

        with ProcessPoolExecutor(max_workers=smart_workers) as executor:
            futures = {
                executor.submit(
                    _process_granule,
                    self.sat,
                    item.geolocation,
                    item.fire,
                    self.im,
                    self.jm,
                    self.grid_type,
                ): item.fire
                for item in input_data
            }

            for future in as_completed(futures):
                fp_file = futures[future]
                fp_filename = os.path.basename(fp_file)
                try:
                    result = future.result()
                    if result is not None:
                        self._accumulate(result)
                    # None means legitimately skipped (no geo file or no
                    # fires) — already logged at WARNING/INFO in the worker.
                except Exception:
                    # Real processing failure — log with full traceback so
                    # the cause is visible in the main process log.
                    logging.error(
                        f"Failed to process granule '{fp_filename}'.",
                        exc_info=True,
                    )

    # ------------------------------------------------------------------
    def save(
        self,
        file,
        timestamp,
        satellite='',
        source='',
        qc=True,
        compress=False,
        fill_value=1e15,
        diskless=False,
    ):
        """Save gridded areas and FRP to a NetCDF4 file."""
        if qc:
            self._apply_qc_cap()
        self._save_as_netcdf4(
            file, timestamp, satellite, source, compress, fill_value, diskless
        )

    # ------------------------------------------------------------------
    def _apply_qc_cap(self) -> None:
        """
        Cap FRP where the implied OC AOD exceeds max_aod_oc.
        All constants are unchanged from the original implementation.
        The original three separate biome loops are merged into one.
        self.frp is keyed by plain strings (bb.type.value).
        Automatically handles any biomes present in fire.BIOMASS_BURNING,
        including peat.
        """
        Alpha             = 1.37e-6
        units_factor      = 1.0e-3
        f_phys            = 6
        max_aod           = 10
        oc_mass_ext_coeff = 4.0
        pom_oc_ratio      = 1.8

        qcscaling = _get_qc_scaling()
        S_f = qcscaling[self.sat]['satellitefactor']

        A_l = self.area_land
        A_w = self.area_water
        A_c = self.area_cloud
        A_o = A_l + A_w

        i_land = A_l > 0

        # safe denominators — avoid division by zero without a Python loop
        denom      = np.where((A_o + A_c) > 0, A_o + A_c, 1.0)
        corr_denom = np.where((A_l + A_c) > 0, A_l + A_c, 1.0)
        corr_num   = A_l + 2 * A_c

        E_total = np.zeros((self.im, self.jm))

        # single loop over biomes — replaces the original three loops
        # self.frp is keyed by bb.type.value (plain string)
        for b in fire.BIOMASS_BURNING:
            key = b.type.value
            B_f = qcscaling['oc'][b.description]
            A_f = Alpha * qcscaling[self.sat][b.description]

            E_b = units_factor * A_f * S_f * B_f * self.frp[key]

            # sequential-b0 normalisation (only where land area > 0)
            E_b[i_land] = (
                E_b[i_land]
                / denom[i_land]
                * (corr_num[i_land] / corr_denom[i_land])
            )

            E_total += E_b

        # OC column density → AOD
        M          = (1e3 * E_total) * (24 * 3600)
        aod_oc     = oc_mass_ext_coeff * pom_oc_ratio * M
        max_aod_oc = f_phys * max_aod

        # vectorised capping factor — no Python loop over grid cells
        q        = np.ones_like(aod_oc)
        i_cap    = aod_oc > max_aod_oc
        q[i_cap] = max_aod_oc / aod_oc[i_cap]

        for b in fire.BIOMASS_BURNING:
            self.frp[b.type.value] *= q

        n_cap = int(np.sum(i_cap))
        if n_cap > 0:
            n_fire = max(int(np.sum(E_total > 0)), 1)
            logging.info(
                f"FRPs in {n_cap} grid cells "
                f"({100.0 * n_cap / n_fire:.1f}% of cells with fires) "
                f"were capped."
            )

    # ------------------------------------------------------------------
    def _save_as_netcdf4(
        self,
        file,
        timestamp,
        satellite='',
        source='',
        compress=False,
        fill_value=1e15,
        diskless=False,
    ):
        """Write gridded areas and FRP to a NetCDF4 file."""
        f = nc.Dataset(file, 'w', format='NETCDF4', diskless=diskless)

        if diskless:
            logging.info(f"Created diskless (in-memory) file '{file}'.")

        # global attributes
        f.Conventions = "COARDS"
        f.institution = "NASA/GSFC, Global Modeling and Assimilation Office"
        f.title       = f"QFED Gridded FRP (Level-3A, v{VERSION})"
        f.contact     = "http://gmao.gsfc.nasa.gov"
        f.version     = VERSION
        f.source      = 'NASA/GSFC/GMAO Aerosol Group'
        f.sensor      = instruments.canonical_instrument[satellite]
        f.processed   = str(datetime.now())
        f.history     = ""

        # dimensions
        f.createDimension('lon',  len(self.glon))
        f.createDimension('lat',  len(self.glat))
        f.createDimension('time', None)

        # coordinate variables
        f.createVariable('lon',  'f8', dimensions='lon')
        f.createVariable('lat',  'f8', dimensions='lat')
        f.createVariable('time', 'i4', dimensions='time')

        # data variables — built dynamically from fire.BIOMASS_BURNING
        # so that any new biome (including peat) is included automatically.
        var_kwargs = dict(zlib=compress, fill_value=fill_value)
        dims3d     = ('time', 'lat', 'lon')

        area_vars = ('land', 'water', 'cloud', 'unknown')
        frp_vars  = [f'frp_{bb.type.value}' for bb in fire.BIOMASS_BURNING]

        for vname in (*area_vars, *frp_vars):
            f.createVariable(vname, 'f4', dimensions=dims3d, **var_kwargs)

        # variable metadata — hardcoded names plus a dynamic frp block
        area_meta = {
            'land':    ('Area of cloud-free land pixels',    'km2'),
            'water':   ('Area of water pixels',              'km2'),
            'cloud':   ('Area of cloud pixels over land',    'km2'),
            'unknown': ('Area of cloud pixels',              'km2'),
        }

        # Human-readable long names per biome type value
        _frp_long_names = {
            'tf': 'Fire Radiative Power (Tropical Forests)',
            'bf': 'Fire Radiative Power (Boreal Forests)',
            'mf': 'Fire Radiative Power (Temperate Forests)',
            'sv': 'Fire Radiative Power (Savanna)',
            'gl': 'Fire Radiative Power (Grasslands)',
            'ag': 'Fire Radiative Power (Agricultural)',
            'pt': 'Fire Radiative Power (Peat)',
        }

        v_meta = {**area_meta}
        for bb in fire.BIOMASS_BURNING:
            key      = bb.type.value
            vname    = f'frp_{key}'
            # Fall back to a generic name for any future biome not listed above
            longname = _frp_long_names.get(
                key, f'Fire Radiative Power ({bb.description})'
            )
            v_meta[vname] = (longname, 'MW')

        # coordinate metadata
        v = f.variables['lon']
        v.long_name = 'longitude'; v.standard_name = 'longitude'
        v.units = 'degrees_east';  v.comment = 'center_of_cell'

        v = f.variables['lat']
        v.long_name = 'latitude';  v.standard_name = 'latitude'
        v.units = 'degrees_north'; v.comment = 'center_of_cell'

        v = f.variables['time']
        v.long_name     = 'time'
        v.standard_name = 'time'
        v.units         = f'minutes since {timestamp:%Y-%m-%d %H:%M:%S}'
        v.begin_date    = np.int32(f'{timestamp:%Y%m%d}')
        v.begin_time    = np.int32(f'{timestamp:%H%M%S}')

        for vname, (long_name, units) in v_meta.items():
            vv = f.variables[vname]
            vv.long_name      = long_name
            vv.units          = units
            vv.missing_value  = np.float32(fill_value)
            vv.fmissing_value = np.float32(fill_value)

        # coordinate data
        f.variables['time'][:] = np.array((0,), dtype=np.int32)
        f.variables['lon'][:]  = self.glon
        f.variables['lat'][:]  = self.glat

        # area arrays
        f.variables['land'][0]    = self.area_land.T
        f.variables['water'][0]   = self.area_water.T
        f.variables['cloud'][0]   = self.area_cloud.T
        f.variables['unknown'][0] = self.area_unknown.T

        # FRP arrays — written dynamically for all biomes
        for key, frp_arr in self.frp.items():
            f.variables[f'frp_{key}'][0] = frp_arr.T

        # bookkeeping
        f.setncattr("number_of_input_files", int(self.n_input_files))
        f.setncattr(
            "comment",
            '' if self.n_input_files > 0 else 'No Observational Data Available'
        )

        f.close()
        logging.info(f"Saved gridded FRP and areas to '{file}'.\n")
