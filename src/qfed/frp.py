'''
Gridded FRP products.
'''


import os
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

    def _get_file_paths(self, geolocation_dir, fire_product_dir, fire_product_file):
        '''
        Construct full paths to the fire product file/granule and the 
        associated with it geolocation file.
        '''

        # construct full path to the active fire product file
        fp_filepath = os.path.join(fire_product_dir, fire_product_file)

        # obtain the geolocation filename and construct full path to it
        gp_file = self._fp_reader.get_geolocation_file(fp_filepath)
        gp_path_search = os.path.join(geolocation_dir, gp_file)

        if self.verbosity > 5:
            print(fp_filepath, '\n', gp_file)

        _gp_path = glob(gp_path_search)
        if not _gp_path:
            gp_filepath = None
        else:
            gp_filepath = _gp_path[0]

        return (gp_filepath, fp_filepath)


    def _set_coordinates(self, geolocation_product_path):
        self.lon, self.lat, self.valid, self.lon_range, self.lat_range = \
                  self._gp_reader.get_coordinates(geolocation_product_path)


    def _set_fire_mask(self): 

        self._fire_mask = self._fp_reader.get_fire_mask(fp_path)
        self._algorithm_qa = self._fp_reader.get_algorithm_qa(fp_path)

        assert self.lon.shape == self.lat.shape == self._fire_mask.shape
  


    def _process(self, geolocation_dir, fire_product_dir, fire_product_file):
        '''
        Read and process paired geolocation and fire product files.
        '''

        # file paths to the geolocation and fire files
        gp_path, fp_path = self._get_file_paths(geolocation_dir, fire_product_dir, fire_product_file)

        if gp_path is None:
            if self.verbosity > 0:
                print('[w]    could not find the geolocation file {0:s} ...skipping {1:s}'.format(gp_file, fire_product_file))

            # interrupt further processing of data associated with this granule
            return

        # read and bin data 
        n_fires = self._fp_reader.get_num_fire_pixels(fp_path)

        if n_fires == 0:
            if self.verbosity > 0:
                print('[i]    no fires in {0:s} ...ignoring it'.format(fire_product_file))

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
            if self.verbosity > 0:
                print('[i]    processing {0:s}'.format(fire_product_file))


            self._fire_mask = self._fp_reader.get_fire_mask(fp_path)
            self._algorithm_qa = self._fp_reader.get_algorithm_qa(fp_path)

            # non-fire
            self._process_areas(gp_path, fp_path)

            # fires
            self._process_fires(fp_path)

            # done
            if self.verbosity > 1:
                print('[i]    processed {0:s}'.format(fire_product_file))
    
    
    def _process_areas(self, geolocation_product_path, fire_product_path):
        '''
        Read and process NO-FIRE areas
        '''

        lon, lat, valid, lon_range, lat_range = self._gp_reader.get_coordinates(geolocation_product_path)
        
        assert lon.shape == lat.shape == self._fire_mask.shape
       
        i_water, i_land_nofire, i_land_cloud, i_water_cloud = self._fp_reader.get_pixel_classes(fire_product_path)


        # calculate pixel area
        n_lines, n_samples = lon.shape

        area = np.zeros_like(lon)
        # TODO: this follows the original MxD14 code, is it valid for VIIRS?
        area[:] = self._fp_reader.get_pixel_area(1-1+np.arange(n_samples))

        # non-fire land pixel
        i = np.logical_and(i_land_nofire, valid)

        if np.any(i):
            # condensed 1D arrays of clear-land not burning pixels
            lon_ = lon[i].ravel()
            lat_ = lat[i].ravel()
            area_ = area[i].ravel()

            # bin areas of no-fires and sum
            self.land += _binareas(lon_, lat_, area_, self.im, self.jm, grid_type=self.grid_type)
        else:
            if self.verbosity > 1:
                print('[i]    no NON-FIRE pixel for granule')

        # non-fire water or cloud over water (very likely a non-fire)
        i = np.logical_and(np.logical_or(i_water_cloud, i_water), valid)

        if np.any(i):
            # condensed 1D arrays of water pixels
            lon_ = lon[i].ravel()
            lat_ = lat[i].ravel()
            area_ = area[i].ravel()

            # bin areas of water and sum
            self.water += _binareas(lon_, lat_, area_, self.im, self.jm, grid_type=self.grid_type)
        else:
            if self.verbosity > 1:
                print(' [i]    no WATER pixel for granule')


        # cloud over land only
        i = np.logical_and(i_land_cloud, valid)

        if np.any(i):
            # condensed 1D arrays of cloud pixels
            lon_ = lon[i].ravel()
            lat_ = lat[i].ravel()
            area_ = area[i].ravel()

            # bin areas of cloud and sum
            self.cloud += _binareas(lon_, lat_, area_, self.im, self.jm, grid_type=self.grid_type)
        else:
            if self.verbosity > 1:
                print('[i]    no CLOUD pixel for granule')


    def _process_fires(self, fire_product_path):
        '''
        Read and process active fires
        '''

        fp_lon = self._fp_reader.get_fire_longitude(fire_product_path)
        fp_lat = self._fp_reader.get_fire_latitude(fire_product_path)
        fp_frp = self._fp_reader.get_fire_frp(fire_product_path)

        fp_line   = self._fp_reader.get_fire_line(fire_product_path)
        fp_sample = self._fp_reader.get_fire_sample(fire_product_path)

        fp_area = self._fp_reader.get_fire_pixel_area(fire_product_path)
       
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
            lws = self._fp_reader.get_land_water_mask(fire_product_path)

            i = [n for n in range(n_fires_initial) if lws[fp_line[n],fp_sample[n]] == QA_WATER]
            #i = [n for n in range(n_fires_initial) if lws[fp_line[n],fp_sample[n]] == 1]
            if len(i) > 0:
                if self.verbosity > 0:
                    print('       --> found %d FIRE pixel(s) over water' % len(i))

                self.water += _binareas(fp_lon[i], fp_lat[i], fp_area[i], self.im, self.jm, grid_type=self.grid_type)

            i = [n for n in range(n_fires_initial) if lws[fp_line[n],fp_sample[n]] in (QA_COAST, QA_LAND)]
            #i = [n for n in range(n_fires_initial) if lws[fp_line[n],fp_sample[n]] in (0, )]
            if len(i) > 0:
                fp_lon = fp_lon[i]
                fp_lat = fp_lat[i]
                fp_frp = fp_frp[i]
                fp_line = fp_line[i]
                fp_sample = fp_sample[i]
                fp_area = fp_area[i] 
            else:
                if self.verbosity > 0:
                    print('       --> no FIRE pixels over land/coast')

                return
 
            n_fires = fp_frp.size

            if n_fires_initial != n_fires:
                if self.verbosity > 0:
                    print('       --> reduced the number of FIRE pixels from %d to %d' % (n_fires_initial, n_fires))
            

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

        # TODO: remove the hardcoded IGBP directory
        self.IgbpDir = '/discover/nobackup/projects/gmao/share/gmao_ops/qfed/Emissions/Vegetation/GL_IGBP_INPE/'


        # gridded data accumulators
        self.land  = np.zeros((self.im, self.jm))
        self.water = np.zeros((self.im, self.jm))
        self.cloud = np.zeros((self.im, self.jm))
        self.frp   = np.zeros((len(BIOMES), self.im, self.jm))

        # land/water state - not used, TODO: remove 
        # self._lws = None

        # fire mask and algotihm QA
        self._fire_mask = None
        self._algorithm_qa = None

        # search for files
        search = self._finder.find(date_start, date_end)
        for gp_dir, fp_dir, fp_file in search:
            self._process(gp_dir, fp_dir, fp_file)


    def save(self, filename=None, timestamp=None, dir={'ana':'.', 'bkg':'.'}, qc=True, bootstrap=False, fill_value=1e15):
       """
       Writes gridded Areas and FRP to file.
       """

       if timestamp is None:
           if self.verbosity > 0:
               print('[w]    did not find matching files, skipped writing an output file')
           return

       if qc == True:
           raise NotImplementedError('QA is not implemented.')
           # TODO           
           #self.qualityControl()

           pass
       else:
           if self.verbosity > 0:
               print('[i]    skipping QC procedures')

       self._write_ana(filename=filename, date=timestamp, dir=dir['ana'], bootstrap=bootstrap, fill_value=fill_value)
    

    def _write_ana(self, date=None, filename=None, dir='.', bootstrap=False, fill_value=1e15):
       """
       Writes gridded Areas and FRP to file.
       """

       nymd = 10000*date.year + 100*date.month + date.day
       nhms = 120000

       if bootstrap:
           if self.verbosity > 0:
               print('') 
               print('[i]    bootstrapping FRP forecast!')
               print('')

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
           f.variables['lon' ][:]  = np.array(self.glon)
           f.variables['lat' ][:]  = np.array(self.glat)
       else:
           raise NotImplementedError('Sequential FRP estimate is not implemented')
           # TODO: - filename should correspond to date + 24h in case of daily files
           #       - will need new approach to generlize for arbitrary time periods/steps   
           f = nc.Dataset(filename, 'r+', format='NETCDF4')


       # data
       f.variables['land'  ][0,:,:]  = np.transpose(self.land)
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

       if self.verbosity > 0:
           print('[i]    wrote file {file:s}'.format(file=filename))


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

    time = datetime(2020, 10, 26, 12)
    time_window = timedelta(hours=24)

    time_s = time - 0.5*time_window
    time_e = time + 0.5*time_window


    grid_ = grid.Grid('d')
    
    # MODIS/Terra
    gp_dir = os.path.join(modis_dir, '061', 'MOD03', '{0:%Y}', '{0:%j}')
    fp_dir = os.path.join(modis_dir, '006', 'MOD14', '{0:%Y}', '{0:%j}')
    finder = PathFinder(gp_dir, fp_dir, 'MOD14.A{0:%Y%j}.{0:%H%M}.006.*.hdf')
    
    fp_reader = fire_products.create('modis', 'terra', verbosity=10)
    gp_reader = geolocation_products.create('modis', 'terra', verbosity=10)

    frp = GriddedFRP(grid_, finder, gp_reader, fp_reader, verbosity=1)
    frp.grid(time_s, time_e)
    frp.save(filename='qfed3-foo.frp.modis-terra.nc4', timestamp=time, bootstrap=True, qc=False)

    
    # MODIS/Aqua
    gp_dir = os.path.join(modis_dir, '061', 'MYD03', '{0:%Y}', '{0:%j}')
    fp_dir = os.path.join(modis_dir, '006', 'MYD14', '{0:%Y}', '{0:%j}')
    finder = PathFinder(gp_dir, fp_dir, 'MYD14.A{0:%Y%j}.{0:%H%M}.006.*.hdf')

    fp_reader = fire_products.create('modis', 'aqua', verbosity=10)
    gp_reader = geolocation_products.create('modis', 'aqua', verbosity=10)

    frp = GriddedFRP(grid_, finder, gp_reader, fp_reader, verbosity=1)
    frp.grid(time_s, time_e)
    frp.save(filename='qfed3-foo.frp.modis-aqua.nc4', timestamp=time, bootstrap=True, qc=False)
    

    # VIIRS-NPP
    gp_dir = os.path.join(viirs_dir, 'Level1', 'NPP_IMFTS_L1', '{0:%Y}', '{0:%j}')
    fp_dir = os.path.join(viirs_dir, 'Level2', 'VNP14IMG', '{0:%Y}', '{0:%j}')
    finder = PathFinder(gp_dir, fp_dir, 'VNP14IMG.A{0:%Y%j}.{0:%H%M}.001.*.nc')

    fp_reader = fire_products.create('viirs', 'npp', verbosity=10)
    gp_reader = geolocation_products.create('viirs', 'npp', verbosity=10)

    frp = GriddedFRP(grid_, finder, gp_reader, fp_reader, verbosity=1)
    frp.grid(time_s, time_e)
    frp.save(filename='qfed3-foo.frp.viirs-npp.nc4', timestamp=time, bootstrap=True, qc=False)


    # VIIRS-JPSS1
    gp_dir = os.path.join(viirs_dir, 'Level1', 'VJ103IMG', '{0:%Y}', '{0:%j}')
    fp_dir = os.path.join(viirs_dir, 'Level2', 'VJ114IMG', '{0:%Y}', '{0:%j}')
    finder = PathFinder(gp_dir, fp_dir, 'VJ114IMG.A{0:%Y%j}.{0:%H%M}.002.*.nc')

    fp_reader = fire_products.create('viirs', 'jpss-1', verbosity=10)
    gp_reader = geolocation_products.create('viirs', 'jpss-1', verbosity=10)

    frp = GriddedFRP(grid_, finder, gp_reader, fp_reader, verbosity=1)
    frp.grid(time_s, time_e)
    frp.save(filename='qfed3-foo.frp.viirs-jpss1.nc4', timestamp=time, bootstrap=True, qc=False)



if __name__ == '__main__':
    from qfed.pathfinder import PathFinder
    _test_frp()

