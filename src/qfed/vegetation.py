"""
Simplified vegetation categories.
"""

import os
from enum import IntEnum, unique
from netCDF4 import Dataset
import numpy as np
import logging

TROPICAL       = 1
EXTRA_TROPICAL = 2
SAVANNA        = 3
GRASSLAND      = 4
AGRICULTURAL   = 5
PEATLAND       = 6
NON_VEGETATION = 0  # internal, will be replaced by nonVeg

STATIC_SOURCE = 21
GASFLARING    = 22
VOLCANO       = 23


@unique
class VegetationCategory(IntEnum):
    """
    Simplified representation of vegetation types.

    The symbolic names and values need to be
    consistent with simplified_vegetation().
    """

    TROPICAL_FOREST      = 1
    EXTRATROPICAL_FOREST = 2
    SAVANNA              = 3
    GRASSLAND            = 4
    AGRICULTURAL         = 5
    PEATLAND             = 6
    # reserve for future
#     STATIC_SOURCE = 21
#     GASFLARING    = 22
#     VOLCANO       = 23


class IGBPNetCDF():

    def __init__(self,
                 file,
                 nonVeg=NON_VEGETATION,
                 drops=[STATIC_SOURCE, GASFLARING, VOLCANO],
                 static_heat=False,
                 gasflaring=False,
                 volcano=False,
                 static_heat_threshold=16,
                 peat_file=None):
        """
        Parameters
        ----------
        file : str
            Path to the IGBP NetCDF file.
        nonVeg : int, optional
            Code used to replace non-vegetation pixels (default NON_VEGETATION=0).
        drops : list of int, optional
            Category codes that will be remapped to nonVeg.
        static_heat : bool, optional
            Whether to read static heat source mask from the IGBP+ file.
        gasflaring : bool, optional
            Whether to read gas-flaring mask from the IGBP+ file.
        volcano : bool, optional
            Whether to read volcano mask from the IGBP+ file.
        static_heat_threshold : int, optional
            Minimum value in static_heat_mask to qualify as a static source.
        peat_file : str or None, optional
            Path to the GL_PEAT_GPA22 NetCDF file.  When None (default),
            peat reclassification is skipped entirely.
        """

        self.file                = file
        self.nonVeg              = nonVeg
        self.drops               = drops
        self.peat_file           = peat_file

        self._open_igbp()
        self._open_igbp_plus(static_heat=static_heat,
                             gasflaring=gasflaring,
                             volcano=volcano,
                             static_heat_threshold=static_heat_threshold)
        self._open_peat()   # no-op when peat_file is None


    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open_igbp(self):

        logging.info(f"Reading IGBP file {self.file}")

        ncid = Dataset(self.file, 'r')
        ncid.set_auto_mask(False)
        self.surface_type = ncid['surface_type'][:]

        self.x_min = np.min(ncid['easting'][:])
        self.dx    = abs(np.mean(np.diff(ncid['easting'][:])))

        self.y_max = np.max(ncid['northing'][:])
        self.dy    = abs(np.mean(np.diff(ncid['northing'][:])))

        ncid.close()


    def _open_igbp_plus(self, static_heat=False, gasflaring=False,
                        volcano=False, static_heat_threshold=16):

        ncid = Dataset(self.file, 'r')
        ncid.set_auto_mask(False)

        try:
            dim_northing = len(ncid['northing_plus'][:])
            dim_easting  = len(ncid['easting_plus'][:])

            self.x_plus_min = np.min(ncid['easting_plus'][:])
            self.dx_plus    = abs(np.mean(np.diff(ncid['easting_plus'][:])))

            self.y_plus_max = np.max(ncid['northing_plus'][:])
            self.dy_plus    = abs(np.mean(np.diff(ncid['northing_plus'][:])))

            self.plus_mask = np.zeros((dim_northing, dim_easting), dtype=np.uint8)

            if static_heat:
                try:
                    field = ncid['static_heat_mask'][:]
                except KeyError:
                    field = None
                if field is not None:
                    idx = np.where((field >= static_heat_threshold) & (field < 255))
                    self.plus_mask[idx] = STATIC_SOURCE

            if gasflaring:
                try:
                    field = ncid['gasflaring_mask'][:]
                except KeyError:
                    field = None
                if field is not None:
                    idx = np.where(field == 1)
                    self.plus_mask[idx] = GASFLARING

            if volcano:
                try:
                    field = ncid['volcano_mask'][:]
                except KeyError:
                    field = None
                if field is not None:
                    idx = np.where(field == 1)
                    self.plus_mask[idx] = VOLCANO

        finally:
            ncid.close()


    def _open_peat(self):
        """
        Read the peat_mask variable from the GL_PEAT_GPA22 NetCDF file.

        Sets the following instance attributes (all None when peat_file
        is not supplied):
            peat_mask      : 2-D uint8 array  (northing × easting)
                             0 = non-peat land
                             1 = peat-dominated (continuous)
                             2 = peat-in-soil-mosaic
                           255 = no data / fill
            x_peat_min     : minimum easting  (metres, sinusoidal)
            dx_peat        : easting  pixel size (metres)
            y_peat_max     : maximum northing (metres, sinusoidal)
            dy_peat        : northing pixel size (metres)
        """

        # Initialise to None so downstream code can test `if self.peat_mask is not None`
        self.peat_mask  = None
        self.x_peat_min = None
        self.dx_peat    = None
        self.y_peat_max = None
        self.dy_peat    = None

        if self.peat_file is None:
            return

        logging.info(f"Reading peat file {self.peat_file}")

        ncid = Dataset(self.peat_file, 'r')
        ncid.set_auto_mask(False)

        try:
            self.peat_mask = ncid['peat_mask'][:]          # ubyte, fill=255

            easting  = ncid['easting'][:]
            northing = ncid['northing'][:]

            self.x_peat_min = np.min(easting)
            self.dx_peat    = abs(np.mean(np.diff(easting)))

            self.y_peat_max = np.max(northing)
            self.dy_peat    = abs(np.mean(np.diff(northing)))

        finally:
            ncid.close()


    @staticmethod
    def _geog_to_sinu(lat, lon):
        """
        Convert geographic coordinates (deg) to MODIS sinusoidal x,y (metres).
        lat, lon can be scalars or arrays.
        """
        R   = 6371007.181000
        rad = np.pi / 180.0

        phi   = lat * rad
        lamda = lon * rad

        y = phi * R
        x = np.cos(phi) * lamda * R

        return x, y


    def _index_from_latlon(self, lat, lon, dx, dy, x_min, y_max):
        """
        Convert lat/lon arrays to (iy, ix) indices for a sinusoidal grid
        described by dx, dy, x_min, y_max.
        """
        x, y = self._geog_to_sinu(lat, lon)

        ix = np.floor((x - x_min + 0.5 * dx) / dx).astype(int)
        iy = np.floor((y_max - y + 0.5 * dy) / dy).astype(int)

        return ix, iy


    # ------------------------------------------------------------------
    # Public query methods
    # ------------------------------------------------------------------

    def getDetailedVeg(self, lat, lon):
        """
        Return raw IGBP classes (1..17, 99, 100, etc.) at given lat/lon.
        lat, lon: numpy arrays or scalars with the same shape.
        """
        ix, iy = self._index_from_latlon(lat, lon,
                                         self.dx, self.dy,
                                         self.x_min, self.y_max)

        ny, nx = self.surface_type.shape
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)

        return self.surface_type[iy, ix]


    def getPlusClassification(self, lat, lon):
        """
        Return the IGBP+ override code at given lat/lon
        (STATIC_SOURCE, GASFLARING, VOLCANO, or 0).
        """
        ix, iy = self._index_from_latlon(lat, lon,
                                         self.dx_plus, self.dy_plus,
                                         self.x_plus_min, self.y_plus_max)

        ny, nx = self.plus_mask.shape
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)

        return self.plus_mask[iy, ix]


    def getPeatClassification(self, lat, lon):
        """
        Return the raw peat_mask value at given lat/lon.

        Returns
        -------
        numpy array of uint8
            0   : non-peat land
            1   : peat-dominated (continuous)
            2   : peat-in-soil-mosaic
            255 : no-data / fill  (treat as non-peat in downstream logic)

        Returns an array of zeros (non-peat) for all points when no
        peat file was loaded.
        """
        lat = np.asarray(lat)
        lon = np.asarray(lon)

        if self.peat_mask is None:
            return np.zeros(lat.shape, dtype=np.uint8)

        ix, iy = self._index_from_latlon(lat, lon,
                                         self.dx_peat, self.dy_peat,
                                         self.x_peat_min, self.y_peat_max)

        ny, nx = self.peat_mask.shape
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)

        return self.peat_mask[iy, ix]


    def getSimpleVeg(self, lat, lon):
        """
        Aggregate IGBP classes into:
          1  Tropical Forests
          2  Extra-tropical Forests
          3  Cerrado/woody savanna
          4  Grassland
          5  Cropland/Agriculture
          6  Peat  (extra-tropical forest / savanna / grassland that are
                    peat-dominated, class 1)

        with 0 (non-vegetation) replaced later by nonVeg.
        """
        lat = np.asarray(lat)
        lon = np.asarray(lon)

        igbp      = np.array(self.getDetailedVeg(lat, lon),        copy=False)
        igbp_plus = np.array(self.getPlusClassification(lat, lon), copy=False)

        # Initialize to zero (maps fill-value 31 and water 17 → 0)
        veg = np.zeros_like(igbp, dtype=np.int16)

        abs_lat = np.abs(lat)

        # --- standard biome masks ---
        mask_trop    = (igbp == 2) & (abs_lat < 30.0)

        mask_extra   = (
            (igbp == 1) |
            ((igbp == 2) & (abs_lat >= 30.0)) |
            (igbp == 3) |
            (igbp == 4) |
            (igbp == 5)
        )

        mask_savanna = (igbp >= 6) & (igbp <= 9)

        mask_grass   = (
            (igbp == 10) |
            (igbp == 11) |
            (igbp == 13) |
            (igbp == 15) |
            (igbp == 16)
        )

        mask_crop    = (igbp == 12) | (igbp == 14)

        # This mask overwrites the veg with plus
        mask_plus    = (
            (igbp_plus == STATIC_SOURCE) |
            (igbp_plus == GASFLARING)    |
            (igbp_plus == VOLCANO)
        )

        veg[mask_trop]    = TROPICAL
        veg[mask_extra]   = EXTRA_TROPICAL
        veg[mask_savanna] = SAVANNA
        veg[mask_grass]   = GRASSLAND
        veg[mask_crop]    = AGRICULTURAL
        veg[mask_plus]    = igbp_plus[mask_plus]

        # --- peat override ---
        # Reclassify extra-tropical forest (2), savanna (3), and grassland (4)
        # pixels to PEAT (6) when the GPA22
        # peat_mask indicates class 1 (peat-dominated / continuous peatland).
        if self.peat_mask is not None:
            peat_raw = self.getPeatClassification(lat, lon)

            mask_peat_eligible = (
                (veg == EXTRA_TROPICAL) |                  # extratropical forest
                (veg == SAVANNA) |                         # savanna
                (veg == GRASSLAND)                         # grassland
            )

            mask_peat = mask_peat_eligible & (peat_raw == 1)

            veg[mask_peat] = PEATLAND

        return veg


    def simplified_vegetation(self, lat, lon):
        """
        Wrapper around getSimpleVeg() that applies the nonVeg substitution
        for any category code listed in self.drops.

        Biome codes returned:
          1  Tropical Forest
          2  Extra-tropical Forest
          3  Cerrado / Woody Savanna
          4  Grassland
          5  Cropland / Agriculture
          6  Peat
        """
        veg = self.getSimpleVeg(lat, lon)

        if self.nonVeg is not None:
            mask = np.zeros_like(veg, dtype=bool)
            for category in self.drops:
                mask |= (veg == category)

            veg = veg.copy()
            veg[mask] = self.nonVeg

        return veg


    def get_category(self, lat, lon, return_codes=False):
        """
        Returns
        -------
        category : dict {VegetationCategory: bool mask}
        veg      : (optional) array of simplified veg codes aligned with lat/lon
        """
        veg = self.simplified_vegetation(lat, lon)

        category = {}
        for c in VegetationCategory:
            category[c] = (veg == c.value)

        return (category, veg) if return_codes else category
