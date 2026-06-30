"""
Simplified vegetation categories.
"""

import os
from enum import IntEnum, unique
from netCDF4 import Dataset
import numpy as np
import logging

TROPICAL       = 1
BOREAL         = 2
SAVANNA        = 3
GRASSLAND      = 4
AGRICULTURAL   = 5
PEATLAND       = 6
TEMPERATE      = 7
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
    BOREAL_FOREST        = 2
    SAVANNA              = 3
    GRASSLAND            = 4
    AGRICULTURAL         = 5
    PEATLAND             = 6
    TEMPERATE_FOREST     = 7
    # reserve for future
#     STATIC_SOURCE = 21
#     GASFLARING    = 22
#     VOLCANO       = 23


class IGBPNetCDF():

    def __init__(self,
                 file,
                 nonVeg = NON_VEGETATION,
                 drops = [STATIC_SOURCE, GASFLARING, VOLCANO],
                 static_heat=False,
                 gasflaring=False,
                 volcano=False,
                 static_heat_threshold=16):
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
        """

        self.file                = file
        self.nonVeg              = nonVeg
        self.drops               = drops

        self._open_igbp()
        self._open_igbp_plus(static_heat=static_heat,
                             gasflaring=gasflaring,
                             volcano=volcano,
                             static_heat_threshold=static_heat_threshold)


    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open_igbp(self):

        logging.info(f"Reading IGBP file {self.file}")

        ncid = Dataset(self.file, 'r')
        ncid.set_auto_mask(False)
        self.surface_type = ncid['surface_type'][:]

        self.x_min = np.min(ncid['easting'][:])
        self.dx = abs(np.mean(np.diff(ncid['easting'][:])))

        self.y_max = np.max(ncid['northing'][:])
        self.dy = abs(np.mean(np.diff(ncid['northing'][:])))

        ncid.close()


    def _open_igbp_plus(self, static_heat=False, gasflaring=False,
                        volcano=False, static_heat_threshold=16):

        ncid = Dataset(self.file, 'r')
        ncid.set_auto_mask(False)

        try:
            dim_northing = len(ncid['northing_plus'][:])
            dim_easting = len(ncid['easting_plus'][:])

            self.x_plus_min = np.min(ncid['easting_plus'][:])
            self.dx_plus = abs(np.mean(np.diff(ncid['easting_plus'][:])))

            self.y_plus_max = np.max(ncid['northing_plus'][:])
            self.dy_plus = abs(np.mean(np.diff(ncid['northing_plus'][:])))

            self.plus_mask = np.zeros( (dim_northing, dim_easting), dtype=np.uint8)

            if static_heat:
                try:
                    field = ncid['static_heat_mask'][:]
                except KeyError:
                    field = None
                if field is not None:
                    idx = np.where((field >=static_heat_threshold) & (field<255))
                    self.plus_mask[idx] = STATIC_SOURCE # e.g, 21

            if gasflaring:
                try:
                    field = ncid['gasflaring_mask'][:]
                except KeyError:
                    field = None
                if field is not None:
                    idx = np.where((field ==1))
                    self.plus_mask[idx] = GASFLARING    # e.g., 22

            if volcano:
                try:
                    field = ncid['volcano_mask'][:]
                except KeyError:
                    field = None
                if field is not None:
                    idx = np.where((field ==1))
                    self.plus_mask[idx] = VOLCANO    # e.g., 23

        finally:
            # Always close, even if something above raises
            ncid.close()


    @staticmethod
    def _geog_to_sinu(lat, lon):
        """
        Convert geographic coordinates (deg) to MODIS sinusoidal x,y (meters).
        lat, lon can be scalars or arrays.
        """
        R = 6371007.181000
        rad = np.pi / 180.0

        phi   = lat * rad
        lamda = lon * rad

        y = phi * R
        x = np.cos(phi) * lamda * R

        return x, y


    def _index_from_latlon(self, lat, lon, dx, dy, x_min, y_max):
        """
        Convert lat/lon arrays to (iy, ix) indices for self.surface_type.
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
        lat, lon: numpy arrays or scalars with same shape.
        """
        #get the index for IGBP
        ix, iy = self._index_from_latlon(lat, lon, self.dx, self.dy, self.x_min, self.y_max)

        # clip indices to the array bounds
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


    def getSimpleVeg(self, lat, lon):
        """
        Aggregate IGBP classes into:
          1  Tropical Forests
          2  Boreal Forests
          3  Cerrado/woody savanna
          4  Grassland
          5  Cropland/Agriculture
          6  Peat
          7  Temperate Forests          

        with 0 (non-vegetation) replaced later by nonVeg.
        """
        lat = np.asarray(lat)
        lon = np.asarray(lon)

        igbp      = np.array(self.getDetailedVeg(lat, lon),        copy=False)
        igbp_plus = np.array(self.getPlusClassification(lat, lon), copy=False)

        # Initialize to zero (maps fill-value 31, urban 13, and water 17 → 0)
        veg = np.zeros_like(igbp, dtype=np.int16)

        abs_lat = np.abs(lat)

        # IGBP classes that are any kind of forest
        is_forest = (igbp == 1) | (igbp == 2) | (igbp == 3) | \
                    (igbp == 4) | (igbp == 5)

        # Tropical: IGBP 2, 4, 5 within ±30°
        mask_trop = (
            ((igbp == 2) | (igbp == 4) | (igbp == 5)) &
            (abs_lat < 30.0)
        )

        # Boreal: IGBP 1, 3, 5 north of 60° (Southern Hemisphere has none)
        mask_boreal = (
            ((igbp == 1) | (igbp == 3) | (igbp == 5)) &
            (lat > 60.0)
        )

        # Temperate: any forest pixel that is not tropical or boreal
        mask_temperate = is_forest & ~mask_trop & ~mask_boreal

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

        veg[mask_trop]     = TROPICAL
        veg[mask_boreal]   = BOREAL
        veg[mask_temperate]= TEMPERATE
        veg[mask_savanna]  = SAVANNA
        veg[mask_grass]    = GRASSLAND
        veg[mask_crop]     = AGRICULTURAL
        veg[mask_plus]     = igbp_plus[mask_plus]

        return veg


    def simplified_vegetation(self, lat, lon):
        """
        Wrapper around getSimpleVeg() that applies the nonVeg substitution
        for any category code listed in self.drops.

        Biome codes returned:
          1  Tropical Forest
          2  Boreal Forest
          3  Cerrado / Woody Savanna
          4  Grassland
          5  Cropland / Agriculture
          6  Peat
          7  Temperate Forest
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
        - category : dict {VegetationCategory: bool mask}
        - veg      : (optional) array of simplified veg codes aligned with lat/lon
        """
        veg = self.simplified_vegetation(lat, lon)

        category = {}
        for c in VegetationCategory:
            category[c] = (veg == c.value)

        return (category, veg) if return_codes else category
