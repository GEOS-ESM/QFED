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



# TODO: introduce fire types to include gas flares, 
#       burning of crop residue, placeholder for peat fires
BIOMES = ('tf', 'xt', 'sv', 'gl')



class GriddedFRP():
    '''
    Grids FRP, areas of fire pixels and areas of non-fire pixels 
    by aggregating data from multiple granules.
    '''

    def __init__(self, grid, finder, geolocation_product_reader, fire_product_reader, verbosity=0):
        self._grid = grid
        self._finder = finder
        self._gp_reader = geolocation_product_reader
        self._fp_reader = fire_product_reader
        self.verbosity = verbosity

    def _set_coordinates(self, geolocation_product_file):
        '''
        Read longitude and latitudes and store them 
        as private fields.
        '''

        self._lon, self._lat, self._valid_coordinates, self._lon_range, self._lat_range = \
            self._gp_reader.get_coordinates(geolocation_product_file)

        assert self._lon.shape == self._lat.shape


    def _set_fire_mask(self, fire_product_file):
        '''
        Read fire mask and algorithm QA and store them
        as private fields.
        '''

        self._fire_mask = self._fp_reader.get_fire_mask(fire_product_file)
        self._algorithm_qa = self._fp_reader.get_algorithm_qa(fire_product_file)

        assert self._fire_mask.shape == self._algorithm_qa.shape
  

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
            logging.warning((f"Skipping file '{fp_filename}' because "
                             f"the required geolocation file "
                             f"'{geolocation_product_file}' was not found."))

            # interrupt further processing of data associated with this granule
            return

        # read and bin data 
        n_fires = self._fp_reader.get_num_fire_pixels(fire_product_file)

        if n_fires == 0:
            logging.info(f"Skipping file '{fp_filename}' because it does not contain fires.\n")

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
        else:
            logging.info(f"Starting processing of file '{fp_filename}'.") 

            # non-fire
            self._process_areas(gp_file, fire_product_file)

            # fires            
            self._process_fires(fire_product_file)

            # done
            logging.info(f"Successfully processed file '{fp_filename}'.\n")
    
    
    def _process_areas(self, geolocation_product_file, fire_product_file):
        '''
        Read and process NO-FIRE areas
        '''

        self._set_coordinates(geolocation_product_file)
        
        i_water, i_land_nofire, i_land_cloud, i_water_cloud = self._fp_reader.get_pixel_classes(fire_product_file)


        # calculate pixel area
        n_lines, n_samples = self._lon.shape

        area = np.zeros_like(self._lon)
        # TODO: this follows the original MxD14 code, is it valid for VIIRS?
        area[:] = self._fp_reader.get_pixel_area(1-1+np.arange(n_samples))

        # non-fire land pixel
        i = np.logical_and(i_land_nofire, self._valid_coordinates)

        if np.any(i):
            # condensed 1D arrays of clear-land not burning pixels
            lon_ = self._lon[i].ravel()
            lat_ = self._lat[i].ravel()
            area_ = area[i].ravel()

            logging.debug(f"The number of NON-FIRE pixels is {len(area_)}.")

            # bin areas of no-fires and sum
            self.land += _binareas(lon_, lat_, area_, self.im, self.jm, grid_type=self.grid_type)
        

        # non-fire water or cloud over water (very likely a non-fire)
        i = np.logical_and(np.logical_or(i_water_cloud, i_water), self._valid_coordinates)

        if np.any(i):
            # condensed 1D arrays of water pixels
            lon_ = self._lon[i].ravel()
            lat_ = self._lat[i].ravel()
            area_ = area[i].ravel()

            logging.debug(f"The number of WATER pixels is {len(area_)}.")

            # bin areas of water and sum
            self.water += _binareas(lon_, lat_, area_, self.im, self.jm, grid_type=self.grid_type)


        # cloud over land only
        i = np.logical_and(i_land_cloud, self._valid_coordinates)

        if np.any(i):
            # condensed 1D arrays of cloud pixels
            lon_ = self._lon[i].ravel()
            lat_ = self._lat[i].ravel()
            area_ = area[i].ravel()

            logging.debug(f"The number of CLOUD pixels is {len(area_)}.")

            # bin areas of cloud and sum
            self.cloud += _binareas(lon_, lat_, area_, self.im, self.jm, grid_type=self.grid_type)


    def _process_fires(self, fire_product_file):
        '''
        Read and process active fires
        '''

        fp_lon = self._fp_reader.get_fire_longitude(fire_product_file)
        fp_lat = self._fp_reader.get_fire_latitude(fire_product_file)
        fp_frp = self._fp_reader.get_fire_frp(fire_product_file)

        fp_line   = self._fp_reader.get_fire_line(fire_product_file)
        fp_sample = self._fp_reader.get_fire_sample(fire_product_file)

        fp_area = self._fp_reader.get_fire_pixel_area(fire_product_file)
       
        # TODO
        #
        # special cases:
        #   1. fires in water pixels (likely offshore gas flaring)
        #   2. fires in vegetation free pixels (likely gas flaring in deserts)
        #   ... see _getSimpleVeg()
        if False:
#           Determine if there are fires from water pixels (likely offshore gas flaring) and exclude them
#           ---------------------------------------------------------------------------------------------
            # MxD14 collection 6 algorithm quality assessment bits: land/water state (bits 0-1)
            QA_WATER = 0
            QA_COAST = 1
            QA_LAND  = 2

            n_fires_initial = fp_frp.size
            lws = self._fp_reader.get_land_water_mask(fire_product_file)

            i = [n for n in range(n_fires_initial) if lws[fp_line[n],fp_sample[n]] == QA_WATER]
            #i = [n for n in range(n_fires_initial) if lws[fp_line[n],fp_sample[n]] == 1]
            logging.debug(f"The number of FIRE pixels over water is {len(i)}") 
            if len(i) > 0:
                self.water += _binareas(fp_lon[i], fp_lat[i], fp_area[i], self.im, self.jm, grid_type=self.grid_type)

            i = [n for n in range(n_fires_initial) if lws[fp_line[n],fp_sample[n]] in (QA_COAST, QA_LAND)]
            #i = [n for n in range(n_fires_initial) if lws[fp_line[n],fp_sample[n]] in (0, )]
            logging.debug(f"The number of FIRE pixels over land/coast is {len(i)}")
            if len(i) > 0:
                fp_lon = fp_lon[i]
                fp_lat = fp_lat[i]
                fp_frp = fp_frp[i]
                fp_line = fp_line[i]
                fp_sample = fp_sample[i]
                fp_area = fp_area[i] 

                return
 
            n_fires = fp_frp.size

            if n_fires_initial != n_fires:
                logging.debug(f"The number of FIRE pixels was reduced from {n_fires_initial} to {n_fires}")
            

        # bin area of fire pixels 
        self.land += _binareas(fp_lon, fp_lat, fp_area, self.im, self.jm, grid_type=self.grid_type)

        # bin FRP for each fire type
        veg = _getSimpleVeg(fp_lon, fp_lat, self.IgbpDir)
        for fire_type in BIOMES:
            b = BIOMES.index(fire_type)   
            i = (veg == (b+1))
            if np.any(i):
                blon = fp_lon[i]
                blat = fp_lat[i]
                bfrp = fp_frp[i]
                self.frp[b,:,:] += _binareas(blon, blat, bfrp, self.im, self.jm, grid_type=self.grid_type)

        # print('Grid cells with positive FRP: ', len(self.frp[self.frp >0]))


    def grid(self, date_start, date_end):
        '''
        Grid FRP.
        '''

        self.im = self._grid.dimensions()['x']
        self.jm = self._grid.dimensions()['y']
        self.glon = self._grid.lon()
        self.glat = self._grid.lat()

        # TODO: remove the hardcoded grid type and use the new grid types 
        # self.grid_type = self._grid
        self.grid_type = 'GEOS-5 A-Grid'


        # gridded data accumulators
        self.land  = np.zeros((self.im, self.jm))
        self.water = np.zeros((self.im, self.jm))
        self.cloud = np.zeros((self.im, self.jm))
        self.frp   = np.zeros((len(BIOMES), self.im, self.jm))


        # search for files
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

           for v in ('land', 'water', 'cloud'):
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
           v_meta_data = {'land'  : ('Observed Clear Land Area', 'km2'),
                          'water' : ('Water Area', 'km2'),
                          'cloud' : ('Obscured by Clouds Area', 'km2'),
                          'frp_tf': ('Fire Radiative Power (Tropical Forests)', 'MW'),
                          'frp_xf': ('Fire Radiative Power (Extra-tropical Forests)', 'MW'),
                          'frp_sv': ('Fire Radiative Power (Savanna)', 'MW'),
                          'frp_gl': ('Fire Radiative Power (Grasslands)', 'MW'),
                          'fb_tf' : ('Background FRP Density (Tropical Forests)', 'MW km-2'),
                          'fb_xf' : ('Background FRP Density (Extra-tropical Forests)', 'MW km-2'),
                          'fb_sv' : ('Background FRP Density (Savanna)', 'MW km-2'),
                          'fb_gl' : ('Background FRP Density (Grasslands)', 'MW km-2')}

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
       f.variables['land'  ][0,:,:] = np.transpose(self.land)
       f.variables['water' ][0,:,:] = np.transpose(self.water)
       f.variables['cloud' ][0,:,:] = np.transpose(self.cloud)
       f.variables['frp_tf'][0,:,:] = np.transpose(self.frp[0,:,:])
       f.variables['frp_xf'][0,:,:] = np.transpose(self.frp[1,:,:])
       f.variables['frp_sv'][0,:,:] = np.transpose(self.frp[2,:,:])
       f.variables['frp_gl'][0,:,:] = np.transpose(self.frp[3,:,:])

       if bootstrap:
           f.variables['fb_tf'][0,:,:] = np.zeros_like(np.transpose(self.frp[0,:,:]))
           f.variables['fb_xf'][0,:,:] = np.zeros_like(np.transpose(self.frp[0,:,:]))
           f.variables['fb_sv'][0,:,:] = np.zeros_like(np.transpose(self.frp[0,:,:]))
           f.variables['fb_gl'][0,:,:] = np.zeros_like(np.transpose(self.frp[0,:,:]))

       f.close()

       logging.info(f"Wrote file {filename}.\n\n")


def _binareas(lon, lat, area, im, jm, grid_type='GEOS-5 A-Grid'):

    if grid_type == 'GEOS-5 A-Grid':
        result = binareas(lon,lat,area,im,jm)
    elif grid_type == 'DE_x_PE':
        result = binareasnr(lon,lat,area,im,jm)
    else:
        result = None 

    return result


def _getSimpleVeg(lon, lat, Path, nonVeg=None):
    """
        Given the pathname for the IGBP datasets, returns
    information about the *aggregated* vegetation type:  

         1  Tropical Forest         IGBP  1, 30S < lat < 30N
         2  Extra-tropical Forests  IGBP  1, 2(lat <=30S or lat >= 30N),
                                          3, 4, 5
         3  cerrado/woody savanna   IGBP  6 thru  9
         4  Grassland/cropland      IGBP 10 thru 17

     the new attribute name is "veg". Notice that this module also
     defines the following constants:

         TROPICAL = 1
         EXTRA_TROPICAL = 2
         SAVANNA = 3
         GRASSLAND = 4

     corresponding to each aggregated vegetation type.
    """

    nobs = lon.shape[0]
    veg = IGBP_.getsimpleveg(lon,lat,Path+'/IGBP',nobs) # Fortran

    # substitute non vegetation (water, snow/ice) data with 
    # another type, e.g. GRASSLAND by default
    if nonVeg != None:
        i = np.logical_or(veg == -15, veg == -17)
        veg[i] = nonVeg # could be one of the biomes, i.e., 1, 2, 3 or 4

    return veg





def _test_frp():

    modis_dir = '/discover/nobackup/dao_ops/intermediate/flk/modis'
    viirs_dir = '/discover/nobackup/projects/eis_fire/data/VIIRS'
    igbp_dir  = '/discover/nobackup/projects/gmao/share/gmao_ops/qfed/Emissions/Vegetation/GL_IGBP_INPE/' 

    time = datetime(2020, 10, 26, 12)
    time_window = timedelta(hours=24)

    time_s = time - 0.5*time_window
    time_e = time + 0.5*time_window


    grid_ = grid.Grid('d')
   
    
    # MODIS/Terra
    gp_file = os.path.join(modis_dir, '061', 'MOD03', '{0:%Y}', '{0:%j}', 
        'MOD03.A{0:%Y%j}.{0:%H%M}.061.NRT.hdf')
    fp_file = os.path.join(modis_dir, '006', 'MOD14', '{0:%Y}', '{0:%j}',
        'MOD14.A{0:%Y%j}.{0:%H%M}.006.*.hdf')
    vg_file = igbp_dir

    finder = Finder(gp_file, fp_file, vg_file, time_interval=300.0)
    fp_reader = fire_products.create(Instrument.MODIS, Satellite.TERRA, verbosity=10)
    gp_reader = geolocation_products.create(Instrument.MODIS, Satellite.TERRA, verbosity=10)

    frp = GriddedFRP(grid_, finder, gp_reader, fp_reader, verbosity=1)

    frp.grid(time_s, time_e)
    frp.save(filename='qfed3-foo.frp.modis-terra.nc4', timestamp=time, bootstrap=True, qc=False)

     
    # MODIS/Aqua
    gp_file = os.path.join(modis_dir, '061', 'MYD03', '{0:%Y}', '{0:%j}',
        'MYD03.A{0:%Y%j}.{0:%H%M}.061.NRT.hdf')
    fp_file = os.path.join(modis_dir, '006', 'MYD14', '{0:%Y}', '{0:%j}',
        'MYD14.A{0:%Y%j}.{0:%H%M}.006.*.hdf')
    vg_file = igbp_dir

    finder = Finder(gp_file, fp_file, vg_file, time_interval=300.0)
    fp_reader = fire_products.create(Instrument.MODIS, Satellite.AQUA, verbosity=10)
    gp_reader = geolocation_products.create(Instrument.MODIS, Satellite.AQUA, verbosity=10)

    frp = GriddedFRP(grid_, finder, gp_reader, fp_reader, verbosity=1)
    frp.grid(time_s, time_e)
    frp.save(filename='qfed3-foo.frp.modis-aqua.nc4', timestamp=time, bootstrap=True, qc=False)
    
    
    # VIIRS-NPP
    #gp_dir = os.path.join(viirs_dir, 'Level1', 'NPP_IMFTS_L1', '{0:%Y}', '{0:%j}')
    gp_file = os.path.join(viirs_dir, 'Level1', 'VNP03IMG.trimmed', '{0:%Y}', '{0:%j}',
        'VNP03IMG.A{0:%Y%j}.{0:%H%M}.002.*.nc')
    fp_file = os.path.join(viirs_dir, 'Level2', 'VNP14IMG', '{0:%Y}', '{0:%j}', 
        'VNP14IMG.A{0:%Y%j}.{0:%H%M}.001.*.nc')
    vg_file = igbp_dir

    finder = Finder(gp_file, fp_file, vg_file, time_interval=360.0)
    fp_reader = fire_products.create(Instrument.VIIRS, Satellite.NPP, verbosity=10)
    gp_reader = geolocation_products.create(Instrument.VIIRS, Satellite.NPP, verbosity=10)

    frp = GriddedFRP(grid_, finder, gp_reader, fp_reader, verbosity=1)
    frp.grid(time_s, time_e)
    frp.save(filename='qfed3-foo.frp.viirs-npp.nc4', timestamp=time, bootstrap=True, qc=False)


    # VIIRS-JPSS1
    gp_file = os.path.join(viirs_dir, 'Level1', 'VJ103IMG.trimmed', '{0:%Y}', '{0:%j}',
        'VJ103IMG.A{0:%Y%j}.{0:%H%M}.002.*.nc')
    fp_file = os.path.join(viirs_dir, 'Level2', 'VJ114IMG', '{0:%Y}', '{0:%j}',
        'VJ114IMG.A{0:%Y%j}.{0:%H%M}.002.*.nc')
    vg_file = igbp_dir

    finder = Finder(gp_file, fp_file, vg_file, time_interval=360.0)
    fp_reader = fire_products.create(Instrument.VIIRS, Satellite.JPSS1, verbosity=10)
    gp_reader = geolocation_products.create(Instrument.VIIRS, Satellite.JPSS1, verbosity=10)

    frp = GriddedFRP(grid_, finder, gp_reader, fp_reader, verbosity=1)
    frp.grid(time_s, time_e)
    frp.save(filename='qfed3-foo.frp.viirs-jpss1.nc4', timestamp=time, bootstrap=True, qc=False)
    


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

