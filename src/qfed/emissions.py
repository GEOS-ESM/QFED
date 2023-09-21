"""
Calculate emissions from Fire Radiative Flux (FRP/Area).
"""


import os
import logging
import yaml
from datetime import date, datetime, timedelta

import numpy as np
import netCDF4 as nc

from qfed import grid
from qfed.instruments import Instrument, Satellite
from qfed import VERSION


# Scaling of C6 based on C5 (based on OC tuning)
# ----------------------------------------------
alpha = np.array([0.96450253, 1.09728882, 1.12014982, 1.22951496, 1.21702972])

# Combustion rate constant
# (ECMWF Tech Memo 596)
# It could be biome-dependent in case we want to tinker
# with the A-M emission factors
# -----------------------------------------------------
Alpha = 1.37e-6 # kg(dry mater)/J
A_f = {}

#                          Tropical  Extratrop.   
#                          Forests      Forests  Savanna  Grasslands
#                          --------  ----------  -------  ----------
enhance_gas     = np.array(( 1.0,          1.0,     1.0,      1.0 ))
enhance_aerosol = np.array(( 2.5,          4.5,     1.8,      1.8 ))

enhance_gas = enhance_gas * alpha[1:]
enhance_aerosol = enhance_aerosol * alpha[1:]

A_f['SO2']  = Alpha * enhance_aerosol
A_f['OC']   = Alpha * enhance_aerosol
A_f['BC']   = Alpha * enhance_aerosol
A_f['NH3']  = Alpha * enhance_aerosol
A_f['PM25'] = Alpha * enhance_aerosol
A_f['TPM']  = Alpha * enhance_aerosol
A_f['CO2']  = Alpha * enhance_gas
A_f['CO']   = Alpha * enhance_gas
A_f['NO']   = Alpha * enhance_gas
A_f['MEK']  = Alpha * enhance_gas
A_f['C3H6'] = Alpha * enhance_gas
A_f['C2H6'] = Alpha * enhance_gas
A_f['C3H8'] = Alpha * enhance_gas
A_f['ALK4'] = Alpha * enhance_gas
A_f['ALD2'] = Alpha * enhance_gas
A_f['CH2O'] = Alpha * enhance_gas
A_f['ACET'] = Alpha * enhance_gas
A_f['CH4']  = Alpha * enhance_gas


# Satellite Factors
# -----------------
S_f = {}
S_f['MODIS_TERRA'] = 1.385 * alpha[0] # C6 scaling based on C5 above
S_f['MODIS_AQUA' ] = 0.473

S_f[(Instrument.MODIS, Satellite.TERRA)] = S_f['MODIS_TERRA']
S_f[(Instrument.MODIS, Satellite.AQUA) ] = S_f['MODIS_AQUA' ]
S_f[(Instrument.VIIRS, Satellite.JPSS1)] = 1.0
S_f[(Instrument.VIIRS, Satellite.NPP)  ] = 1.0

#                     Tropical  Extratrop.    
#                      Forests     Forests   Savanna  Grasslands
#                     --------  ----------   -------  ----------
#
#S_f['MODIS_TERRA'] = ( 1.000,      1.000,    1.000,       1.000 )
#S_f['MODIS_AQUA' ] = ( 1.000,      1.000,    1.000,       1.000 )



def read_emission_factors(file):
    with open(file) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            data = None
            logging.critical(exc)

    return data['emission_factors']


class Emissions():
    """
    Class for computing biomass burning emissions 
    from gridded FRP and .
    """

    def __init__(self, time, FRP, F, area, emission_factors_file):
        """
        Initializes an Emission object. 

          timestamp  ---   datetime object

          FRP   ---   Dictionary keyed by 'instrument/satellite'
                      with each element containing a
                      tuple with gridded Fire Radiative Power
                      (MW); each tuple element corresponds to a
                      different biome type, e.g.,
                      
                      FRP['MODIS_TERRA'] = (frp_tf,frp_xf,frp_sv,frp_gl)

          F     ---   Dictionary keyed by satellite name
                      with each element containing a
                      tuple with gridded forecast of FRP density 
                      (MW km-2); each tuple element corresponds to a
                      different biome type, e.g.,
                      
                      F['MODIS_TERRA'] = (f_tf,f_xf,f_sv,f_gl)
        
          Area  ---   Dictionary keyed by satellite name
                      with each element containing a
                      observed clear-land area [km2] for each gridbox
        """

        self.area  = area
        self.area_land  = {k:area[k]['land' ] for k in area.keys()}
        self.area_water = {k:area[k]['water'] for k in area.keys()}
        self.area_cloud = {k:area[k]['cloud'] for k in area.keys()}
        self.area_unknown = {k:area[k].get('unknown', 0.0) for k in area.keys()}
        self.FRP = FRP
        self.F = F
        self.time = time

        self.platform = list(FRP.keys())
        self.biomass_burning = list(FRP[self.platform[0]].keys())

        self.set_grid()

        self.emission_factors_file = emission_factors_file
        self._ef = read_emission_factors(emission_factors_file)

    def set_grid(self):
        """
        Sets the grid.
        """
        #TODO: should use the grid module
        self.im, self.jm = self.area_land[self.platform[0]].shape

        if (5*self.im - 8*(self.jm - 1)) == 0:
            self.lon  = np.linspace(-180.0, 180.0, self.im, endpoint=False)
            self.lat  = np.linspace(-90.0, 90.0, self.jm)
        else:
            d_lon = 360.0 / self.im
            d_lat = 180.0 / self.jm
            self.lon = np.linspace(-180+d_lon/2, 180-d_lon/2, self.im)
            self.lat = np.linspace( -90+d_lat/2,  90-d_lat/2, self.jm)

    def emission_factor(self, species, fire):
        bb = fire.type.name.lower()
        return self._ef[species][bb]

    def calculate(self, species, method='default'):
        """
        Calculate emissions for each species using built-in
        emission coefficients and fudge factors.

        The default method for computing the emissions is
        'sequential-zero'.
        """
        # factor needed to convert B_f from [g/kg] to [kg/kg]
        units_factor = 1e-3

        A_l = np.zeros((self.im, self.jm))
        A_w = np.zeros((self.im, self.jm))
        A_c = np.zeros((self.im, self.jm))
        A_u = np.zeros((self.im, self.jm))

        for p in self.platform:
            A_l += self.area_land[p]
            A_w += self.area_water[p]
            A_c += self.area_cloud[p]
            A_u += self.area_unknown[p]

        A_o = A_l + A_w

        i = (A_l > 0)
        j = ((A_l + A_c) > 0)

        E = {}
        E_= {}
        for s in species:
            E[s]  = {bb:np.zeros((self.im, self.jm)) for bb in self.biomass_burning}
            E_[s] = {bb:np.zeros((self.im, self.jm)) for bb in self.biomass_burning}

            for p in self.platform:
                FRP = self.FRP[p]
                F   = self.F[p]
                A_  = self.area_cloud[p]

                for bb in self.biomass_burning:
                    # TODO: A_f should be a dict.
                    B_f, eB_f = self.emission_factor(s, bb)
                    ib = ('tf', 'xf', 'sv', 'gl').index(bb.type.value)
                    E[s][bb][:,:]  += units_factor * A_f[s.upper()][ib] * S_f[p] * B_f * FRP[bb]
                    E_[s][bb][:,:] += units_factor * A_f[s.upper()][ib] * S_f[p] * B_f * F[bb] * A_

            for bb in self.biomass_burning:
                E_b  = E[s][bb][:,:]
                E_b_ = E_[s][bb][:,:]

                if (method == 'default') or (method == 'sequential'):
                    E_b[j] = ( (E_b[j]  / (A_o[j] + A_c[j])) * (1 + A_c[j] / (A_l[j] + A_c[j])) +
                               (E_b_[j] / (A_o[j] + A_c[j])) * (    A_c[j] / (A_l[j] + A_c[j])) )

                if method == 'sequential-zero':
                    E_b[i] = E_b[i] / (A_o[i] + A_c[i]) * (1.0 + A_c[i] / (A_l[i] + A_c[i]))

                if method == 'nofires':
                    E_b[i] = E_b[i] / (A_o[i] + A_c[i])

                if method == 'similarity':
                    E_b[i] = E_b[i] / A_l[i] * ((A_l[i] + A_c[i]) / (A_o[i] + A_c[i]))

                if method == 'similarity_qfed-2.2':
                    E_b[i] = E_b[i] / A_o[i]
                   

        self.estimate = E


    def total(self, specie):
        """
        Calculates total emissions from fires in all biomes.
        """
        result = np.zeros((self.im, self.jm))
        for bb in self.biomass_burning:
            result += self.estimate[specie][bb][:,:]

        return result


    def _save_ana(self, dir, filename, fill_value=1e15):
       """
       Writes gridded emissions. You must call method
       calculate() first. Optional input parameters:

       filename  ---  file name; if not specified each
                      species will be written to a separate
                      file, e.g.,
                         qfed2.emis_co.sfc.20030205.nc4
       dir       ---  optional directory name, only used
                      when *filename* is omitted
       """
       

       nymd = 22220101#10000*self.date.year + 100*self.date.month + self.date.day
       nhms = 120000

#      Loop over species
#      -----------------
       
       self.file = {}
       for s in self.estimate.keys():
           file = os.path.join(dir, f'qfed.emis_{s.lower()}.{nymd}.nc4')
           self.file[s] = file

#          Create new file or overwrite existing file
#          ------------------------------------------
           f = nc.Dataset(file, 'w', format='NETCDF4')
    
           # global attributes
           f.Conventions = 'COARDS'
           f.institution = 'NASA/GSFC, Global Modeling and Assimilation Office'
           f.title       = 'QFED Gridded Emissions (Level-3B, v{0:s})'.format(VERSION)
           f.contact     = 'Anton Darmenov <anton.s.darmenov@nasa.gov>'
           f.version     = VERSION
           f.source      = 'TODO'
           f.processed   = str(datetime.now())
           f.history     = ''
           f.platform    = 'TODO'

           # dimensions
           f.createDimension('lon', len(self.lon))
           f.createDimension('lat', len(self.lat))
           f.createDimension('time', None)
 
           # variables
           f.createVariable('lon',  'f8', ('lon'))
           f.createVariable('lat',  'f8', ('lat'))
           f.createVariable('time', 'i4', ('time'))

           for v in ('biomass', 'biomass_tf', 'biomass_xf', 'biomass_sv', 'biomass_gl'):
               f.createVariable(v, 'f4', ('time', 'lat', 'lon'), 
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
           begin_date      = int(self.time.strftime('%Y%m%d'))
           begin_time      = int(self.time.strftime('%H%M%S'))
           v.long_name     = 'time'
           v.standard_name = 'latitude'
           v.units         = 'minutes since {:%Y-%m-%d %H:%M:%S}'.format(self.time)
           v.begin_date    = np.array(begin_date, dtype=np.int32)
           v.begin_time    = np.array(begin_time, dtype=np.int32)

           # long name and units
           v_meta_data = {
               'biomass'   : (f'{s} Biomass Emissions', 'kg s-1 m-2'),
               'biomass_tf': (f'{s} Biomass Emissions from Tropical Forests', 'kg s-1 m-2'),
               'biomass_xf': (f'{s} Biomass Emissions from Extratropical Forests', 'kg s-1 m-2'),
               'biomass_sv': (f'{s} Biomass Emissions from Savanna', 'kg s-1 m-2'),
               'biomass_gl': (f'{s} Biomass Emissions from Grasslands', 'kg s-1 m-2')}

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
           f.variables['lon' ][:] = np.array(self.lon)
           f.variables['lat' ][:] = np.array(self.lat)


           f.variables['biomass'   ][0,:,:] = np.transpose(self.total(s)[:,:])

           for bb in self.biomass_burning:
               v = f.variables[f'biomass_{bb.type.value}']
               v[0,:,:] = np.transpose(self.estimate[s][bb][:,:])

           f.close()

           logging.info(f"Successfully saved gridded emissions to file {filename}")


    def save(self, filename=None, dir='.', forecast=None, expid='qfed2',  
                    ndays=1, uncompressed=False):
       """
       Writes gridded emissions that can persist for a number of days. You must 
       call method calculate() first. Optional input parameters:

       filename      ---  file name; if not specified each
                          species will be written to a separate
                          file, e.g.,
                          qfed2.emis_co.sfc.20030205.nc4
       dir           ---  optional directory name, only used
                          when *filename* is omitted
       expid         ---  optional experiment id, only used
                          when *filename* is omitted
       ndays         ---  persist emissions for a number of days
       uncompressed  ---  use n4zip to compress gridded output file
       
       """

#      Write out the emission files
#      ----------------------------
       # TODO:
       # self._write_fcs(forecast, fill_value=1.0e20)

       for n in range(ndays):

#          Write out the emission files
#          ----------------------------
           self._save_ana(dir, filename, fill_value=1e20)

#          Compress the files by default
#          -----------------------------
           if not uncompressed:
               for s in self.filename.keys():
                   rc = os.system("n4zip %s"%self.filename[s])
                   if rc:
                       warnings.warn('cannot compress output file <%s>'%self.filename[s])

#          Increment date by one day
#          ---------------------------------
           self.date = self.time + timedelta(days=1)

#..............................................................


