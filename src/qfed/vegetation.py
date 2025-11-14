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

    TROPICAL_FOREST = 1
    EXTRATROPICAL_FOREST = 2
    SAVANNA = 3
    GRASSLAND = 4
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
        
        self.file   = file
        self.nonVeg = nonVeg
        self.drops  = drops
        
        self._open_igbp()
        self._open_igbp_plus(static_heat=static_heat,
                             gasflaring=gasflaring,
                             volcano=volcano,
                             static_heat_threshold=static_heat_threshold)

    
    def _open_igbp(self):
    
        logging.info(f"Reading IGBP file {self.file}")
        
        ncid = Dataset(self.file, 'r')
        ncid.set_auto_mask(False)
        self.surface_type = ncid['surface_type'][:]

        self.x_min = np.min(ncid['easting'][:])
        self.dx = abs(np.mean(np.diff(ncid['easting'][:])))

        self.y_max  = np.max(ncid['northing'][:])
        self.dy = abs(np.mean(np.diff(ncid['northing'][:])))
        
        ncid.close()

    def _open_igbp_plus(self, static_heat=False, gasflaring=False, 
                        volcano=False, static_heat_threshold=16):

        ncid = Dataset(self.file, 'r')
        ncid.set_auto_mask(False)
        
        try:
            # --- coords & dimensions ---
            dim_northing = len(ncid['northing_plus'][:])
            dim_easting = len(ncid['easting_plus'][:])

            self.x_plus_min = np.min(ncid['easting_plus'][:])
            self.dx_plus = abs(np.mean(np.diff(ncid['easting_plus'][:])))

            self.y_plus_max  = np.max(ncid['northing_plus'][:])
            self.dy_plus = abs(np.mean(np.diff(ncid['northing_plus'][:])))

            # Create the plus mask
            self.plus_mask = np.zeros( (dim_northing, dim_easting), dtype=np.uint8)

            if static_heat:
                try:
                    field = ncid['static_heat_mask'][:]
                except KeyError:
                    field = None
                
                if field is not None:
                    idx = np.where((field >=static_heat_threshold) & (field<255))
                    self.plus_mask[idx] = STATIC_SOURCE  # e.g., 21

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
                    self.plus_mask[idx] = VOLCANO       # e.g., 23               
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
        """

        x, y = self._geog_to_sinu(lat, lon)

        ix = np.floor((x - x_min + 0.5 * dx) / dx).astype(int)
        iy = np.floor((y_max - y + 0.5 * dy) / dy).astype(int)

        return ix, iy

    def getDetailedVeg(self, lat, lon):
        """
        Return raw IGBP classes (1..17, 99, 100, etc.) at given lat/lon.
        lat, lon: numpy arrays or scalars with same shape.
        """
        # get the index for IGBP
        ix, iy = self._index_from_latlon(lat, lon, self.dx, self.dy, self.x_min, self.y_max)
        
        # clip indices to the array bounds
        ny, nx = self.surface_type.shape
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        
        return self.surface_type[iy, ix]

    def getPlusClassification(self, lat, lon):
        """
        Return raw IGBP classes (1..17, 99, 100, etc.) at given lat/lon.
        lat, lon: numpy arrays or scalars with same shape.
        """
        # get the index for IGBP
        ix, iy = self._index_from_latlon(lat, lon, self.dx, self.dy, self.x_min, self.y_max)
        
        # clip indices to the array bounds
        ny, nx = self.plus_mask.shape
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        
        return self.plus_mask[iy, ix]
    
    
    def getSimpleVeg(self, lat, lon):
        """
        Aggregate IGBP classes into:
          1  Tropical Forests
          2  Extra-tropical Forests
          3  Cerrado/woody savanna
          4  Grassland/cropland
        with 0 (non-vegetation) replaced by nonVeg.
        """
        igbp = self.getDetailedVeg(lat, lon)
        igbp = np.array(igbp, copy=False)

        igbp_plus = self.getPlusClassification(lat, lon)
        igbp_plus = np.array(igbp_plus, copy=False)

        # use zero array array to initilize the veg array
        # this zero initlization essentially maps the 
        # fill value  (31) to -> zero
        # water pixel (17) to -> zero
        veg = np.zeros_like(igbp, dtype=np.int16)
 
        # Latitude condition for tropical vs extra-tropical broadleaf
        abs_lat = np.abs(lat)

        # Tropical forests: IGBP 2, within |lat| < 30 degree
        mask_trop = (igbp == 2) & (abs_lat < 30.0)

        # Extra-tropical forests:
        #   Evergreen needleleaf (1)
        #   Evergreen broadleaf (2) outside tropics
        #   Deciduous needleleaf (3)
        #   Deciduous broadleaf (4)
        #   Mixed forest (5)
        mask_extra = (
            (igbp == 1) |
            ((igbp == 2) & (abs_lat >= 30.0)) |
            (igbp == 3) |
            (igbp == 4) |
            (igbp == 5)
        )

        # Cerrado / woody savanna: IGBP 6–9
        mask_savanna = (igbp >= 6) & (igbp <= 9)

        # Grassland/cropland: IGBP 10–16
        # old code map water pixel (17) to glassland...
        mask_grass = (igbp >= 10) & (igbp < 17)
        
        # This mask overwrites the veg with plus 
        mask_plus = (igbp_plus == STATIC_SOURCE) | (igbp_plus == GASFLARING) | (igbp_plus == VOLCANO)
        
        veg[mask_trop]    = TROPICAL
        veg[mask_extra]   = EXTRA_TROPICAL
        veg[mask_savanna] = SAVANNA
        veg[mask_grass]   = GRASSLAND
        veg[mask_plus]    = igbp_plus[mask_plus]

        return veg

    def simplified_vegetation(self, lat, lon):
        """
        Helper method - aggregates IGBP vegetation classes
        into a reduced set of vegetation types:

          1 -> Tropical Forest:         IGBP  1, 30S < lat < 30N
          2 -> Extra-tropical Forests:  IGBP  1, 2(lat <=30S or lat >= 30N), 3, 4, 5
          3 -> Cerrado/Woody Savanna:   IGBP  6, 7, 8, 9
          4 -> Grassland/Cropland:      IGBP 10, 11, ..., 16
        """


        veg = self.getSimpleVeg(lat, lon)
        # substitute non vegetation (water, snow/ice) data with
        # another type, e.g. GRASSLAND by default
        if self.nonVeg is not None:

            mask = np.zeros_like(veg, dtype=bool)
            for category in self.drops:
                mask |= (veg == category)

            veg = veg.copy()     # avoid modifying underlying array unexpectedly
            veg[mask] = self.nonVeg   # e.g., map 0/water to GRASSLAND, etc.

        return veg

    
    def get_category(self, lat, lon, return_codes=False):
        """
        Returns:
          - category: dict {VegetationCategory: bool mask}
          - veg (optional): 1D/ND array of simplified veg codes (1..4) aligned with lon/lat
        """
        veg = self.simplified_vegetation(lat, lon)

        category = {}
        for c in VegetationCategory:
            category[c] = (veg == c.value)

        return (category, veg) if return_codes else category


# if __name__ == "__main__":
#     IGBP_FILE = '/Dedicated/jwang-data2/shared_satData/GMAO_QFED/GL_IGBP_MODIS/GL_IGBP_MODIS.2024.nc'
#     IGBP_FILE = '/Dedicated/jwang-data2/mzhou/project/OPNL_FILDA/STATIC_SOURCE/IGBP+/GL_IGBP_PLUS.MODIS.2024.nc'
#     igbp = IGBPNetCDF(IGBP_FILE,
#                       nonVeg= NON_VEGETATION,
#                       drops = [0, 21, 22, 23, 31],
#                       static_heat=True,
#                       gasflaring=True,
#                       volcano=True,
#                       static_heat_threshold = 16)
#     
#     # for 2024, 
#     # Point 1 volvano; Point 2 gas flaring; Point 3 static source; 
#     # point 4 Evergreen Broadleaf Forests in Amazon
#     # point 5, Desert in Sahara; 
#     # point 6, Deciduous Needleleaf Forests in Siberia
#     # point 7, Deciduous Needleleaf Forests in Sahara; 
#     # point 8, Evergreen Broadleaf Forests near Seattle
#     test_lats = np.array([40.73, 71.875, -41.13333333, -6.096703, 20.200317, 60.745892, 46.951504])
#     test_lons = np.array([13.897, 71.8438974, 146.84388815, -63.889643, 8.779006, 118.950086, -121.697682])
#     
#     print( igbp.get_category(test_lats, test_lons))
    