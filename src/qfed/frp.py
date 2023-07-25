'''
Gridded FRP products.
'''

import os
import logging
from datetime import datetime, timedelta
from glob import glob

import numpy as np
import netCDF4 as nc

from pyobs import IGBP_
from binObs_ import binareas, binareasnr

from qfed import version
from qfed import grid
from qfed import instruments
from qfed import geolocation_products
from qfed import fire_products
from qfed import classification_products


# TODO: introduce fire types to include gas flares, 
#       burning of crop residue, placeholder for peat fires
BIOMES = ('tf', 'xt', 'sv', 'gl')



class GriddedFRP():
    '''
    Grids FRP, areas of fire pixels and areas of non-fire pixels 
    by aggregating data from multiple granules.
    '''

    def __init__(self, grid, finder, 
                 geolocation_product_reader, 
                 fire_product_reader, 
                 classification_product_reader):
        self._grid = grid
        self._finder = finder
        self._gp_reader = geolocation_product_reader
        self._fp_reader = fire_product_reader
        self._cp_reader = classification_product_reader

    def _get_coordinates(self, geolocation_product_file):
        '''
        Read longitude and latitudes and store them 
        as private fields.
        '''
        (
            self._lon, 
            self._lat, 
            self._valid_coordinates, 
            self._lon_range, 
            self._lat_range 
        ) = self._gp_reader.get_coordinates(geolocation_product_file)

    def _get_pixels(self, classification_product_file):
        '''
        Read fire mask and algorithm QA and store them
        as private fields.
        '''
        self._cp_reader.read(classification_product_file)

        self._is_unclassified = self._cp_reader.get_unclassified()
        self._is_cloud = self._cp_reader.get_cloud()
        self._is_cloud_free = self._cp_reader.get_cloud_free()

        # TODO: not used, remove 
        #surface = self._cp_reader.get_surface_type()
        #self.__is_water = surface['water']
        #self.__is_coast = surface['coast']
        #self.__is_land  = surface['land' ]

        #self._is_fire = self._cp_reader.get_fire()
        self._is_fire_low_confidence = self._cp_reader.get_fire(confidence='low')
        self._is_fire_nominal_confidence = self._cp_reader.get_fire(confidence='nominal')
        self._is_fire_high_confidence = self._cp_reader.get_fire(confidence='high')

        self._area = self._cp_reader.get_area()

    def _process(self, geolocation_product_file, fire_product_file):
        '''
        Read and process paired geolocation and fire product files.
        '''

        fp_filename = os.path.basename(fire_product_file)

        match = glob(geolocation_product_file)
        if match:
            gp_file = match[0]
        else:
            gp_file = None

        if gp_file is None:
            msg = (
                f"Skipping file '{fp_filename}' because "
                f"the required geolocation file "
                f"'{geolocation_product_file}' was not found."
            )

            logging.warning(msg)

            # interrupt further processing of data associated with this granule
            return

        # read and bin data
        n_fires = self._fp_reader.get_num_fire_pixels(fire_product_file)

        if n_fires == 0:
            msg = (
                f"Skipping file '{fp_filename}' because "
                f"it does not contain fires.\n"
            )

            logging.info(msg)

            # TODO: Interrupting the processing 'here' means that  
            #       none of the no-fire pixels (water, land, cloud, etc.)
            #       are accounted for in the corresponding area accumulators,
            #       which will bias the emissions high in cloud free conditions. 
            #       Consider for example an intermittent fire at a location  
            #       that is in two or more granules within the time accumulation
            #       window, but only one of the granules had an active fire at 
            #       this location. If the processing is not interrupted, the 
            #       emissions will be ~ FRP/(2*cell_area), whereas if the 
            #       processing is interrupted the emissions will be 
            #       ~ FRP/cell_area. 
            #       However, if the fire was active, but it was not detected 
            #       because it was obscured by clouds then the emission 
            #       estimates will not be biased because ~(2*FRP)/(2*cell_area) = 
            #       = FRP/cell_area.

            # interrupt further processing of data associated with this granule
            return
        
        logging.info(f"Starting processing of file '{fp_filename}'.")

        self._process_non_fire(gp_file, fire_product_file)
        self._process_fire(fire_product_file)
        self._process_unobserved()

        logging.info(f"Successfully processed file '{fp_filename}'.\n")


    def _process_non_fire(self, geolocation_product_file, classification_product_file):
        '''
        Read and process NON-FIRE pixels.
        '''
        self._get_coordinates(geolocation_product_file)
        self._get_pixels(classification_product_file)

        logging.debug("Processing areas without fires.")
        self._process_cloud_free()
        self._process_cloud()

    def _process_cloud_free(self):
        self._process_cloud_free_land()
        self._process_cloud_free_coast()
        self._process_cloud_free_water()
        self._process_cloud_free_unknown()

    def _process_cloud(self):
        self._process_cloud_land()
        self._process_cloud_coast()
        self._process_cloud_water()
        self._process_cloud_unknown()

    
    def _process_unobserved(self):
        # TODO ...
        pass

    def _select(self, i):
        '''
        Helper method - selects and flattens 
        lon, lat and area data.
        '''
        lon = self._lon[i].ravel()
        lat = self._lat[i].ravel()
        area = self._area[i].ravel()
        return lon, lat, area

    def _process_cloud_free_land(self):
        '''
        Process cloud-free land pixels that do not contain active fires.
        '''
        i = self._is_cloud_free['land'] & self._valid_coordinates
        lon, lat, area = self._select(i)
        self.area_land += _binareas(lon, lat, area, self.im, self.jm, self.grid_type)
        logging.debug(f"Added {len(area)} cloud-free land pixels to land area.")

    def _process_cloud_free_coast(self):
        '''
        Process cloud-free coast pixels that do not contain active fires.
        Note that coast pixels are lumped with water pixels.
        '''
        i = self._is_cloud_free['coast'] & self._valid_coordinates
        lon, lat, area = self._select(i)
        self.area_water += _binareas(lon, lat, area, self.im, self.jm, self.grid_type)
        logging.debug(f"Added {len(area)} cloud-free coast pixels to water area.")

    def _process_cloud_free_water(self):
        '''
        Process cloud-free water pixels that do not contain active fires.
        '''
        i = self._is_cloud_free['water'] & self._valid_coordinates
        lon, lat, area = self._select(i) 
        self.area_water += _binareas(lon, lat, area, self.im, self.jm, self.grid_type)
        logging.debug(f"Added {len(area)} cloud-free water pixels to water area.")

    def _process_cloud_free_unknown(self):
        '''
        Process cloud-free unknown (as in not known if they are land, water or coast) 
        pixels that do not contain active fires.
        '''
        i = self._is_cloud_free.get('unknown', False) & self._valid_coordinates
        lon, lat, area = self._select(i) 
        if len(area) > 0:
            logging.critical(f"Found {len(area)} cloud-free unknown pixels!? Excluding them!") 
        else:
            logging.debug("This granule does not contain cloud-free unknown pixels.")

    def _process_cloud_land(self):
        '''
        Process cloud land pixels that do not contain active fires.
        '''
        i = self._is_cloud['land'] & self._valid_coordinates
        lon, lat, area = self._select(i)
        self.area_cloud += _binareas(lon, lat, area, self.im, self.jm, self.grid_type)
        logging.debug(f"Added {len(area)} cloud land pixels to cloud area.")

    def _process_cloud_coast(self):
        '''
        Process cloud coast pixels that do not contain active fires.
        Note that coast pixels are lumped with land pixels. 
        '''
        i = self._is_cloud['coast'] & self._valid_coordinates
        lon, lat, area = self._select(i)
        self.area_water += _binareas(lon, lat, area, self.im, self.jm, self.grid_type)
        logging.debug(f"Added {len(area)} cloud(coast) pixels to water area.")

    def _process_cloud_water(self):
        '''
        Process cloud water pixels that do not contain active fires.
        Note that we lump these with clear sky water pixels because
        it is unlikely that water bodies, including those obscured by
        cluds, contain fires. 
        '''
        i = self._is_cloud['water'] & self._valid_coordinates
        lon, lat, area = self._select(i) 
        self.area_water += _binareas(lon, lat, area, self.im, self.jm, self.grid_type)
        logging.debug(f"Added {len(area)} cloud(water) pixels to water area.") 

    def _process_cloud_unknown(self):
        '''
        Process cloud unknown pixels (i.e., not known if they are land, 
        coast or water pixels) that do not contain active fires.
        '''
        i = self._is_cloud.get('unknown', False) & self._valid_coordinates
        lon, lat, area = self._select(i) 
        self.area_unknown += _binareas(lon, lat, area, self.im, self.jm, self.grid_type)
        logging.debug(f"Added {len(area)} cloud unknown pixels to unknown area.") 

    def _process_fire(self, fire_product_file):
        '''
        Process fire pixels.
        '''
        lon = self._fp_reader.get_fire_longitude(fire_product_file)
        lat = self._fp_reader.get_fire_latitude(fire_product_file)
        frp = self._fp_reader.get_fire_frp(fire_product_file)

        line   = self._fp_reader.get_fire_line(fire_product_file)
        sample = self._fp_reader.get_fire_sample(fire_product_file)
        area   = self._fp_reader.get_fire_pixel_area(fire_product_file)

        # consistency check
        # TODO: this will fail if the geoloc. files   
        #       went trough lossy compression, i.e. VIIRS  
        #assert np.allclose(lon, self._lon[line, sample])
        #assert np.allclose(lat, self._lat[line, sample])
        #assert np.allclose(area, self._area[line, sample])

        logging.debug("Processing areas with fire.")
        self._process_fire_water(lon, lat, line, sample, frp, area)
        self._process_fire_coast(lon, lat, line, sample, frp, area)
        self._process_fire_land(lon, lat, line, sample, frp, area)

    def _process_fire_water(self, lon, lat, line, sample, frp, area):
        '''
        Fires pixels in areas categorized as water.

        Likely offshore gas-flares and as such not
        considered to contribute to FRP from open
        biomass fires.
        '''
        i_valid = self._valid_coordinates[line, sample]
        i_water = (
            self._is_fire_low_confidence['water']     | \
            self._is_fire_nominal_confidence['water'] | \
            self._is_fire_high_confidence['water']
        )[line, sample]

        i = i_water & i_valid
        
        logging.info(f"Found {len(area[i])} water pixels with active fires.")
        self.area_water += _binareas(lon[i], lat[i], area[i], self.im, self.jm, self.grid_type)
        logging.debug(f"Added {len(area[i])} fire(water) pixels to water area.")

    def _process_fire_coast(self, lon, lat, line, sample, frp, area):
        '''
        Fires pixels in areas categorized as coast.

        Likely offshore gas-flares and as such not
        considered to contribute to FRP from open
        biomass fires.
        '''
        i_valid = self._valid_coordinates[line, sample]
        i_coast = (
            self._is_fire_low_confidence['coast']     | \
            self._is_fire_nominal_confidence['coast'] | \
            self._is_fire_high_confidence['coast']
        )[line, sample]

        i = i_coast & i_valid
        
        logging.info(f"Found {len(area[i])} coast pixels with active fires.")
        self.area_water += _binareas(lon[i], lat[i], area[i], self.im, self.jm, self.grid_type)
        logging.debug(f"Added {len(area[i])} fire(coast) pixels to water area.")

    def _process_fire_land(self, lon, lat, line, sample, frp, area):
        '''
        Fires pixels in areas categorized as land.
        '''
        n_fires_initial = frp.size

        # TODO: special cases of fires that are likely gas flares
        #     a) fires in deserts - oil or gas extraction sites, etc.
        #
        #     b) fires in urban/industrial areas - petroleum refineries, 
        #     chemical plants, natural gas processing plants, landfills, 
        #     etc.
        #
        #     c) peat fires
        #

        i_valid = self._valid_coordinates[line, sample]
        i_land = (
            self._is_fire_low_confidence['land']     | \
            self._is_fire_nominal_confidence['land'] | \
            self._is_fire_high_confidence['land']
        )[line, sample]

        i = i_land & i_valid
        
        n_fires = np.sum(i)
        logging.info(f"Found {n_fires} land pixels with active fires.")

        if n_fires != n_fires_initial:
            logging.info(f"The number of fire pixels was reduced from {n_fires_initial} to {n_fires}.")

        self.area_land += _binareas(lon[i], lat[i], area[i], self.im, self.jm, self.grid_type)
        logging.debug(f"Added {len(area[i])} fire-pixels to land area.")

        if n_fires == 0:
            return

        # bin FRP from fires in each of the considered biomes
        veg = _getSimpleVeg(lon[i], lat[i], self.IgbpDir)
        for b, fire_biome in enumerate(BIOMES):
            j = (b == (veg-1))
            self.frp[b,:,:] += _binareas(lon[i][j], lat[i][j], frp[i][j], self.im, self.jm, self.grid_type)


    def grid(self, date_start, date_end):
        '''
        Grid FRP.
        '''

        self.im = self._grid.dimensions()['x']
        self.jm = self._grid.dimensions()['y']
        self.glon = self._grid.lon()
        self.glat = self._grid.lat()

        self.grid_type = self._grid.type

        # gridded data accumulators
        self.area_land = np.zeros((self.im, self.jm))
        self.area_water = np.zeros((self.im, self.jm))
        self.area_cloud = np.zeros((self.im, self.jm))
        self.area_unknown = np.zeros((self.im, self.jm))
        self.frp = np.zeros((len(BIOMES), self.im, self.jm))

        # find the input files and process the data
        search = self._finder.find(date_start, date_end)
        for item in search:
            self.IgbpDir = item.vegetation
            self._process(item.geolocation, item.fire)


    def save(self, filename=None, timestamp=None, dir={'ana':'.', 'bkg':'.'}, qc=True, bootstrap=False, fill_value=1e15):
       """
       Writes gridded Areas and FRP to file.
       """

       if timestamp is None:
           logging.warning("An output file is not written due to mismatched input file.")
           return

       if qc == True:
           raise NotImplementedError('QA is not implemented.')
           # TODO           
           #self.qualityControl()

           pass
       else:
           logging.info("Skipping modulation of FRP due to QC being disabled.")

       self._save_as_netcdf4(filename=filename, date=timestamp, dir=dir['ana'], bootstrap=bootstrap, fill_value=fill_value)
    

    def _save_as_netcdf4(self, date=None, filename=None, dir='.', bootstrap=False, fill_value=1e15):
       """
       Writes gridded Areas and FRP to a NetCDF4 file.
       """

       nymd = 10000*date.year + 100*date.month + date.day
       nhms = 120000

       if bootstrap:
           logging.info("Prior FRP are initialized to zero.")

           # create a file
           f = nc.Dataset(filename, 'w', format='NETCDF4')

           # global attributes
           f.Conventions = 'COARDS'
           f.Source      = 'NASA/GSFC, Global Modeling and Assimilation Office'
           f.Title       = 'QFED Level3a v{version:s} Gridded FRP Estimates'.format(version=version.__version__)
           f.Contact     = 'Anton Darmenov <anton.s.darmenov@nasa.gov>'
           f.Version     = str(version.__version__)
           f.Processed   = str(datetime.now())
           f.History     = '' 

           # dimensions
           f.createDimension('lon', len(self.glon))
           f.createDimension('lat', len(self.glat))
           f.createDimension('time', None)
 
           # variables
           f.createVariable('lon',  'f8', ('lon'))
           f.createVariable('lat',  'f8', ('lat'))
           f.createVariable('time', 'i4', ('time'))

           for v in ('land', 'water', 'cloud', 'unknown'):
               f.createVariable(v, 'f4', ('time', 'lat', 'lon'), 
                                fill_value=fill_value, zlib=False)

           for v in ('frp_tf', 'frp_xf', 'frp_sv', 'frp_gl'): 
               f.createVariable(v, 'f4', ('time', 'lat', 'lon'), 
                                fill_value=fill_value, zlib=False)

           for v in ('fb_tf', 'fb_xf', 'fb_sv', 'fb_gl'):
               f.createVariable(v,  'f4', ('time', 'lat', 'lon'), 
                                fill_value=fill_value, zlib=False)


           # variables attributes
           v = f.variables['lon']
           v.long_name     = 'longitude'
           v.standard_name = 'longitude'
           v.units         = 'degrees_east'
           v.comment       = 'center_of_cell'

           v = f.variables['lat']
           v.long_name     = 'latitude'
           v.standard_name = 'latitude'
           v.units         = 'degrees_north'
           v.comment       = 'center_of_cell'

           v = f.variables['time']
           begin_date   = int(date.strftime('%Y%m%d'))
           begin_time   = int(date.strftime('%H%M%S'))
           v.long_name  = 'time'
           v.units      = 'minutes since {:%Y-%m-%d %H:%M:%S}'.format(date)
           v.begin_date = np.array(begin_date, dtype=np.int32)
           v.begin_time = np.array(begin_time, dtype=np.int32)
           
           # long name and units
           v_meta_data = {
               'land'   : ('Area of cloud-free land pixels', 'km2'),
               'water'  : ('Area of water pixels', 'km2'),
               'cloud'  : ('Area of cloud pixels over land', 'km2'),
               'unknown': ('Area of cloud pixels', 'km2'),
               'frp_tf' : ('Fire Radiative Power (Tropical Forests)', 'MW'),
               'frp_xf' : ('Fire Radiative Power (Extra-tropical Forests)', 'MW'),
               'frp_sv' : ('Fire Radiative Power (Savanna)', 'MW'),
               'frp_gl' : ('Fire Radiative Power (Grasslands)', 'MW'),
               'fb_tf'  : ('Background FRP Density (Tropical Forests)', 'MW km-2'),
               'fb_xf'  : ('Background FRP Density (Extra-tropical Forests)', 'MW km-2'),
               'fb_sv'  : ('Background FRP Density (Savanna)', 'MW km-2'),
               'fb_gl'  : ('Background FRP Density (Grasslands)', 'MW km-2')}

           for _v, (_l, _u) in v_meta_data.items():
               v = f.variables[_v]
               v.long_name = _l
               v.units = _u
               v.missing_value = np.array(fill_value, np.float32)
               v.fmissing_value = np.array(fill_value, np.float32)
               v.vmin = np.array(fill_value, np.float32)
               v.vmax = np.array(fill_value, np.float32)

           # data
           f.variables['time'][:] = np.array((0,))
           f.variables['lon' ][:] = np.array(self.glon)
           f.variables['lat' ][:] = np.array(self.glat)
       else:
           raise NotImplementedError('Sequential FRP estimate is not implemented')
           # TODO: - filename should correspond to date + 24h in case of daily files
           #       - will need new approach to generlize for arbitrary time periods/steps   
           f = nc.Dataset(filename, 'r+', format='NETCDF4')


       # data
       f.variables['land'   ][0,:,:] = np.transpose(self.area_land)
       f.variables['water'  ][0,:,:] = np.transpose(self.area_water)
       f.variables['cloud'  ][0,:,:] = np.transpose(self.area_cloud)
       f.variables['unknown'][0,:,:] = np.transpose(self.area_unknown)
       f.variables['frp_tf' ][0,:,:] = np.transpose(self.frp[0,:,:])
       f.variables['frp_xf' ][0,:,:] = np.transpose(self.frp[1,:,:])
       f.variables['frp_sv' ][0,:,:] = np.transpose(self.frp[2,:,:])
       f.variables['frp_gl' ][0,:,:] = np.transpose(self.frp[3,:,:])

       if bootstrap:
           f.variables['fb_tf'][0,:,:] = np.zeros_like(np.transpose(self.frp[0,:,:]))
           f.variables['fb_xf'][0,:,:] = np.zeros_like(np.transpose(self.frp[0,:,:]))
           f.variables['fb_sv'][0,:,:] = np.zeros_like(np.transpose(self.frp[0,:,:]))
           f.variables['fb_gl'][0,:,:] = np.zeros_like(np.transpose(self.frp[0,:,:]))

       f.close()

       logging.info(f"Wrote file {filename}.\n\n")


def _binareas(lon, lat, area, im, jm, grid_type):
    '''
    Helper method - bins data such as pixel area or FRP.
    '''

    if len(area) == 0:
        return 0.0

    if grid_type == grid.GridType.LATLON_GEOS:
        result = binareas(lon, lat, area, im, jm)
    elif grid_type == grid.GridType.LATLON_3600x1800:
        result = binareasnr(lon, lat, area, im, jm)
    else:
        result = None

    assert result is not None, "Data binning does not support this type of grid."

    return result


def _getSimpleVeg(lon, lat, Path, nonVeg=None):
    """
    Helper method - aggregates IGBP vegetation classes 
    into a reduced set of vegetation types:

         1  Tropical Forest        = IGBP:  1, 30S < lat < 30N
         2  Extra-tropical Forests = IGBP:  1, 2(lat <=30S or lat >= 30N),
                                            3, 4, 5
         3  cerrado/woody savanna  = IGBP:  6 thru  9
         4  Grassland/cropland     = IGBP: 10 thru 17
    """

    nobs = lon.shape[0]
    veg = IGBP_.getsimpleveg(lon, lat, Path+'/IGBP', nobs) # Fortran

    # substitute non vegetation (water, snow/ice) data with 
    # another type, e.g. GRASSLAND by default
    if nonVeg != None:
        i = (veg == -15) | (veg == -17)
        veg[i] = nonVeg # could be one of the biomes, i.e., 1, 2, 3 or 4

    return veg





def _test_frp():

    ##OPS:
    modis_dir = '/discover/nobackup/dao_ops/intermediate/flk/modis'
    ##DEBUG/TEST
    ##modis_dir = '/discover/nobackup/adarmeno/sandbox/qfed-geos-esm.foo/validation/data/modis/'
    viirs_dir = '/css/viirs/data/'
    igbp_dir  = '/discover/nobackup/projects/gmao/share/gmao_ops/qfed/Emissions/Vegetation/GL_IGBP_INPE/' 

    time = datetime(2021, 2, 1, 12)
    time_window = timedelta(hours=24)

    time_s = time - 0.5*time_window
    time_e = time + 0.5*time_window


    grid_ = grid.Grid('c')
   
    #'''
    # MODIS/Terra
    gp_file = os.path.join(modis_dir, '061', 'MOD03', '{0:%Y}', '{0:%j}', 
        'MOD03.A{0:%Y%j}.{0:%H%M}.061.NRT.hdf')
    fp_file = os.path.join(modis_dir, '006', 'MOD14', '{0:%Y}', '{0:%j}',
        'MOD14.A{0:%Y%j}.{0:%H%M}.006.*.hdf')
    vg_file = igbp_dir

    finder = Finder(gp_file, fp_file, vg_file, time_interval=60.0)
    fp_reader = fire_products.create(Instrument.MODIS, Satellite.TERRA)
    gp_reader = geolocation_products.create(Instrument.MODIS, Satellite.TERRA)
    cp_reader = classification_products.create(Instrument.MODIS, Satellite.TERRA)

    frp = GriddedFRP(grid_, finder, gp_reader, fp_reader, cp_reader)
    frp.grid(time_s, time_e)
    frp.save(filename='qfed3-foo.frp.modis-terra.cpr.nc4', timestamp=time, bootstrap=True, qc=False)

     
    # MODIS/Aqua
    gp_file = os.path.join(modis_dir, '061', 'MYD03', '{0:%Y}', '{0:%j}',
        'MYD03.A{0:%Y%j}.{0:%H%M}.061.NRT.hdf')
    fp_file = os.path.join(modis_dir, '006', 'MYD14', '{0:%Y}', '{0:%j}',
        'MYD14.A{0:%Y%j}.{0:%H%M}.006.*.hdf')
    vg_file = igbp_dir

    finder = Finder(gp_file, fp_file, vg_file, time_interval=300.0)
    fp_reader = fire_products.create(Instrument.MODIS, Satellite.AQUA)
    gp_reader = geolocation_products.create(Instrument.MODIS, Satellite.AQUA)
    cp_reader = classification_products.create(Instrument.MODIS, Satellite.TERRA)

    frp = GriddedFRP(grid_, finder, gp_reader, fp_reader, cp_reader)
    frp.grid(time_s, time_e)
    frp.save(filename='qfed3-foo.frp.modis-aqua.cpr.nc4', timestamp=time, bootstrap=True, qc=False)
    #'''

    # VIIRS-NPP
    #gp_dir = os.path.join(viirs_dir, 'Level1', 'NPP_IMFTS_L1', '{0:%Y}', '{0:%j}')
    gp_file = os.path.join(viirs_dir, 'Level1', 'VNP03IMG.trimmed', '{0:%Y}', '{0:%j}',
        'VNP03IMG.A{0:%Y%j}.{0:%H%M}.002.*.nc')
    fp_file = os.path.join(viirs_dir, 'Level2', 'VNP14IMG', '{0:%Y}', '{0:%j}', 
        'VNP14IMG.A{0:%Y%j}.{0:%H%M}.001.*.nc')
    vg_file = igbp_dir

    finder = Finder(gp_file, fp_file, vg_file, time_interval=360.0)
    fp_reader = fire_products.create(Instrument.VIIRS, Satellite.NPP)
    gp_reader = geolocation_products.create(Instrument.VIIRS, Satellite.NPP)
    cp_reader = classification_products.create(Instrument.VIIRS, Satellite.NPP)

    frp = GriddedFRP(grid_, finder, gp_reader, fp_reader, cp_reader)
    frp.grid(time_s, time_e)
    frp.save(filename='qfed3-foo.frp.viirs-npp.cpr.nc4', timestamp=time, bootstrap=True, qc=False)


    # VIIRS-JPSS1
    gp_file = os.path.join(viirs_dir, 'Level1', 'VJ103IMG.trimmed', '{0:%Y}', '{0:%j}',
        'VJ103IMG.A{0:%Y%j}.{0:%H%M}.002.*.nc')
    fp_file = os.path.join(viirs_dir, 'Level2', 'VJ114IMG', '{0:%Y}', '{0:%j}',
        'VJ114IMG.A{0:%Y%j}.{0:%H%M}.002.*.nc')
    vg_file = igbp_dir

    finder = Finder(gp_file, fp_file, vg_file, time_interval=360.0)
    fp_reader = fire_products.create(Instrument.VIIRS, Satellite.JPSS1)
    gp_reader = geolocation_products.create(Instrument.VIIRS, Satellite.JPSS1)
    cp_reader = classification_products.create(Instrument.VIIRS, Satellite.JPSS1)

    frp = GriddedFRP(grid_, finder, gp_reader, fp_reader, cp_reader)
    frp.grid(time_s, time_e)
    frp.save(filename='qfed3-foo.frp.viirs-jpss1.cpr.nc4', timestamp=time, bootstrap=True, qc=False)
    


if __name__ == '__main__':
    from qfed.inventory import Finder
    from qfed.instruments import Instrument, Satellite

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        #filename='frp.log',
     ) 

    _test_frp()

