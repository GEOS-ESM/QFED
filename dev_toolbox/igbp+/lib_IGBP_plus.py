import numpy as np

# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -  
# function and class related to grid manipulation for Sinusoidal projection

# - - - - - - -  
class SinusoidalGrid:
    def __init__(self, num_cells=800,
                 hid_max=35, vid_max=17,
                 R=6371007.181000):
        self.num_cells = num_cells
        self.hid_max = hid_max
        self.vid_max = vid_max
        self.R = R

        self._compute_geometry()

    def _compute_geometry(self):
        # use your geog_to_sinu here
        self.halfHoriLength = geog_to_sinu(0, 180)[0]
        self.halfVertLength = geog_to_sinu(90, 0)[1]

        self.resol_h = (geog_to_sinu(0, 180)[0] - geog_to_sinu(0, 170)[0]) / self.num_cells
        self.resol_v = (geog_to_sinu(80, 0)[1] - geog_to_sinu(70, 0)[1]) / self.num_cells

        self.n_zonal      = (self.hid_max + 1) * self.num_cells
        self.n_meridional = (self.vid_max + 1) * self.num_cells

        self.easting  = -self.halfHoriLength + np.arange(self.n_zonal + 1) * self.resol_h
        self.northing =  self.halfVertLength - np.arange(self.n_meridional + 1) * self.resol_v

    def meshgrid(self):
        return np.meshgrid(self.easting, self.northing)

# - - - - - - - 
def geog_to_sinu(lat, lon, R=6371007.181000):
    '''
    Function of converting the geographical coordinates to sinusoidal coordinates

    Parameters
    ----------
        geographical coordinates - list or tuple liked, (latitude, longituede)

    Return
    ----------
        sinusoidal point - (x, y)


    '''

    pi = 180.0 / np.pi


    phi = lat / pi
    lamda = lon / pi
    y = phi * R
    x = np.cos(phi) * lamda * R

    return x, y


def sinu_to_geog(x, y):
	'''
	Function of converting the sinusoidal coordinate to geographical coordinates

	Parameters
	----------
		sinusoidal point - x, y

	Return
	----------
		geographical coordinates - latitude, longituede
	'''
	import numpy as np

	pi = 180.0 / np.pi
	R = 6371007.181000

	phi = y/R
	lamda = x / np.cos(phi) / R

	latitude = phi * pi
	longituede = lamda * pi

	return latitude, longituede


def cal_sinu_xy(tile, numCeil):
    '''

    Function to calculate the geographical coordinates of the sinusoidal grid

    Parameters
    ----------
        tile - str format, example: 'h07v05'
        numCeil - number of the ceils in one tile

    Return
    ---------- 
        geographical coordinates of the tile

    Reference: 1. https://code.env.duke.edu/projects/mget/wiki/SinusoidalMODIS
               2. https://onlinelibrary.wiley.com/doi/pdf/10.1111/0033-0124.00327
               3. https://modis-land.gsfc.nasa.gov/GCTP.html
    
    MODIS use 6371007.181 as the radius of the Earth...
    '''
    import numpy as np
    
    numHoriTile = 37
    numVertTile = 19
    
    halfHoriLenght = geog_to_sinu(0, 180)[0]
    halfVertLenght = geog_to_sinu(90, 0)[1]
    resol_ceil = halfHoriLenght/((numHoriTile-1)/2.)/numCeil
    halfCeilLen = resol_ceil/2.0



    xx = np.linspace(-halfHoriLenght, halfHoriLenght, numHoriTile)
    yy = np.linspace(halfVertLenght, -halfVertLenght, numVertTile) 

    vid = int(float(tile[4:6]))
    hid = int(float(tile[1:3]))

    x = np.linspace(xx[hid], xx[hid+1], numCeil + 1)
    y = np.linspace(yy[vid], yy[vid+1], numCeil + 1)
    
    x = (x[0:-1] + x[1:])/2.
    y = (y[0:-1] + y[1:])/2.
    
    xv, yv = np.meshgrid(x, y)
    
    return xv, yv

# given the index, calculating the geographical coordinates
def get_coordinates(northing, easting, idx):
    x = easting[idx[1]]
    y = northing[idx[0]]
    return sinu_to_geog(x, y)
    
    
# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -  
# function dictionary that related to the MCDQ12...
legend_dict = {}
legend_dict['LC_Type1'] = { "long_name": f"MCD12Q1 International Geosphere-Biosphere Programme (IGBP) legend  and class descriptions",
                            "class_01": "Evergreen Needleleaf Forests",
                            "class_02": "Evergreen Broadleaf Forests",
                            "class_03": "Deciduous Needleleaf Forests",
                            "class_04": "Deciduous Broadleaf Forests",
                            "class_05": "Mixed Forests",
                            "class_06": "Closed Shrublands",
                            "class_07": "Open Shrublands",
                            "class_08": "Woody Savannas",
                            "class_09": "Savannas",
                            "class_10": "Grasslands",
                            "class_11": "Permanent Wetlands",
                            "class_12": "Croplands",
                            "class_13": "Urban and Built-up Lands",
                            "class_14": "Cropland/Natural Vegetation Mosaics",
                            "class_15": "Snow and Ice",
                            "class_16": "Barren",
                            "class_17": "Water Bodies",
                            "valid_range": np.array([0, 255], dtype="uint8"),
                            "grid_mapping": "crs",
                            }

legend_dict['LC_Type2'] = { "long_name": f"University of Maryland (UMD) legend and class definitions and class definitions",
                            "class_00": "Water bodies",
                            "class_01": "Evergreen Needleleaf Forests",
                            "class_02": "Evergreen Broadleaf Forests",
                            "class_03": "Deciduous Needleleaf Forests",
                            "class_04": "Deciduous Broadleaf Forests",
                            "class_05": "Mixed Forests",
                            "class_06": "Closed Shrublands",
                            "class_07": "Open Shrublands",
                            "class_08": "Woody Savannas",
                            "class_09": "Savannas",
                            "class_10": "Grasslands",
                            "class_11": "Permanent Wetlands",
                            "class_12": "Croplands",
                            "class_13": "Urban and Built-up Lands",
                            "class_14": "Cropland/Natural Vegetation Mosaics",
                            "class_15": "Non-Vegetated Lands",
                            "valid_range": np.array([0, 255], dtype="uint8"),
                            "grid_mapping": "crs",
                            }

legend_dict['LC_Type3'] = { "long_name": f"Leaf Area Index (LAI) legend and class definitions",
                            "class_00": "Water Bodies", 
                            "class_01": "Grasslands",
                            "class_02": "Shrublands",
                            "class_03": "Broadleaf Croplands",
                            "class_04": "Savannas",
                            "class_05": "Evergreen Broadleaf Forests ",
                            "class_06": "Deciduous Broadleaf Forests",
                            "class_07": "Evergreen Needleleaf Forests",
                            "class_08": "Deciduous Needleleaf Forests",
                            "class_09": "Non-Vegetated Lands",
                            "class_10": "Urban and Built-up Lands",
                            "class_17": "Water Bodies",
                            "valid_range": np.array([0, 255], dtype="uint8"),
                            "grid_mapping": "crs",
                            }

legend_dict['LC_Type4'] = { "long_name": f"BIOME-Biogeochemical Cycles (BGC) legend and class definitions",
                            "class_00": "Water Bodies", 
                            "class_01": "Evergreen Needleleaf Vegetation",
                            "class_02": "Evergreen Broadleaf Vegetation",
                            "class_03": "Deciduous Needleleaf Vegetation",
                            "class_04": "Deciduous Broadleaf Vegetation",
                            "class_05": "Annual Broadleaf Vegetation",
                            "class_06": "Annual Grass Vegetation",
                            "class_07": "Non-Vegetated Lands",
                            "class_08": "Urban and Built-up Lands ",
                            "valid_range": np.array([0, 255], dtype="uint8"),
                            "grid_mapping": "crs",
                            }

legend_dict['LC_Type5'] = { "long_name": f"Plant Functional Types (PFT) legend and class definitions",
                            "class_00": "Water Bodies", 
                            "class_01": "Evergreen Needleleaf Vegetation",
                            "class_02": "Evergreen Broadleaf Vegetation",
                            "class_03": "Deciduous Needleleaf Vegetation",
                            "class_04": "Deciduous Broadleaf Vegetation",
                            "class_05": "Shrub",
                            "class_06": "Grass",
                            "class_07": "Cereal Croplands",
                            "class_08": "Broadleaf Croplands",
                            "class_09": "Urban and Built-up Lands",
                            "class_10": "Permanent Snow and Ice",
                            "class_11": "Barren",
                            "valid_range": np.array([0, 255], dtype="uint8"),
                            "grid_mapping": "crs",
                            }



param_mapping = {}
param_mapping['LC_Type1'] = 'surface_type'
param_mapping['LC_Type2'] = 'surface_type_UMD'
param_mapping['LC_Type3'] = 'surface_type_LAI'
param_mapping['LC_Type4'] = 'surface_type_BGC'
param_mapping['LC_Type5'] = 'surface_type_PFT'


water_mapping = {}
water_mapping['LC_Type1'] = 17
water_mapping['LC_Type2'] = 0
water_mapping['LC_Type3'] = 0
water_mapping['LC_Type4'] = 0
water_mapping['LC_Type5'] = 0
water_mapping['LW'] = 0
    
# - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -  
# functions related to the VNP14IMG...
def get_bit(x, bit=0):
    '''
    Extracts the value of a single bit.
    '''
    return (x >> bit) & 1
    
def is_fire_residual_bowtie(algorithm_qa):
	'''
	Fire pixels identified as residual bow-tie data.

	These pixels should be excluded to avoid 
	double-counting of redunant FRP.
	'''
	bit = get_bit(algorithm_qa, 22)
	return bit 
    


