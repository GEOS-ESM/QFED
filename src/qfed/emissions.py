"""
  Calculate emissions from Fire Radiative Flux (FRP/Area).
"""


import warnings
warnings.simplefilter('ignore',DeprecationWarning)

import os
from datetime import date, datetime, timedelta

import numpy as np
import netCDF4 as nc

from qfed.version import __version__, __tag__



#                  -------------------
#                  Internal Parameters
#                  -------------------

# Fire emission flux density [kg/s/m^2]
#
#     f_s = S_f(sat) * (FRP/A) * Alpha * B_f(s)
#

# Biome-dependent Emission Factors
# (From Andreae & Merlet 2001)
# Units: g(species) / kg(dry mater)
# --------------------------------
B_f  = {}   
eB_f = {} # errors

#                Tropical  Extratrop.    
#                 Forests     Forests   Savanna  Grasslands
#                --------  ----------   -------  ----------
B_f['CO2']  = (  1580.00,    1569.00,  1613.00,     1613.00  )
B_f['CO']   = (   104.00,     107.00,    65.00,       65.00  )
B_f['SO2']  = (     0.57,       1.00,     0.35,        0.35  )
B_f['OC']   = (     5.20,       9.14,     3.40,        3.40  )
B_f['BC']   = (     0.66,       0.56,     0.48,        0.48  )
B_f['NH3']  = (     1.30,       1.40,     1.05,        1.05  )
B_f['PM25'] = (     9.10,      13.00,     5.40,        5.40  )
B_f['TPM']  = (     8.50,      17.60,     8.30,        8.30  ) # note that TPM < PM2.5 for Tropical Forests
B_f['NO']   = (     1.60,       3.00,     3.90,        3.90  ) # NOx as NO
B_f['MEK']  = (     0.43,       0.45,     0.26,        0.26  ) # Methyl Ethyl Ketone
B_f['C3H6'] = (     0.55,       0.59,     0.26,        0.26  ) # Propene/Propylene
B_f['C2H6'] = (     1.20,       0.60,     0.32,        0.32  ) # Ethane
B_f['C3H8'] = (     0.15,       0.25,     0.09,        0.09  ) # Propane
B_f['ALK4'] = (     0.056,      0.091,    0.025,       0.025 ) # C4,5 alkanes (C4H10): n-butane + i-butane
B_f['ALD2'] = (     0.65,       0.50,     0.50,        0.50  ) # Acetaldehyde (C2H4O)
B_f['CH2O'] = (     1.40,       2.20,     0.26,        0.26  ) # Formaldehyde (HCHO)
B_f['ACET'] = (     0.62,       0.56,     0.43,        0.43  ) # Acetone (C3H6O)
B_f['CH4']  = (     6.80,       4.70,     2.30,        2.30  ) # Methene (CH4)


#                Tropical  Extratrop.    
#                 Forests     Forests   Savanna  Grasslands
#                --------  ----------   -------  ----------
eB_f['CO2']  = (   90.00,     131.00,    95.00,       95.00  )
eB_f['CO']   = (   20.00,      37.00,    20.00,       20.00  )
eB_f['SO2']  = (    0.23,       0.23,     0.16,        0.16  )
eB_f['OC']   = (    1.50,       0.55,     1.40,        1.40  )
eB_f['BC']   = (    0.31,       0.19,     0.18,        0.18  )
eB_f['NH3']  = (    0.80,       0.80,     0.45,        0.45  )
eB_f['PM25'] = (    1.50,       7.00,     1.50,        1.50  )
eB_f['TPM']  = (    2.00,       6.40,     3.20,        3.20  )
eB_f['NO']   = (    0.70,       1.40,     2.40,        2.40  )
eB_f['MEK']  = (    0.22,       0.28,     0.13,        0.13  )
eB_f['C3H6'] = (    0.25,       0.16,     0.14,        0.14  )
eB_f['C2H6'] = (    0.70,       0.15,     0.16,        0.16  )
eB_f['C3H8'] = (    0.10,       0.11,     0.03,        0.03  )
eB_f['ALK4'] = (    0.03,       0.05,     0.09,        0.09  )
eB_f['ALD2'] = (    0.32,       0.02,     0.39,        0.39  )
eB_f['CH2O'] = (    0.70,       0.50,     0.44,        0.44  )
eB_f['ACET'] = (    0.31,       0.04,     0.18,        0.18  )
eB_f['CH4']  = (    2.00,       1.90,     0.90,        0.90  )

# Scaling of C6 based on C5 (based on OC tuning)
# ----------------------------------------------
alpha = np.array([0.96450253,1.09728882,1.12014982,1.22951496,1.21702972])
for s in B_f.keys():
    B_f[s] = list(np.array(B_f[s]) * alpha[1:])
    
# Combustion rate constant
# (ECMWF Tech Memo 596)
# It could be biome-dependent in case we want to tinker
# with the A-M emission factors
# -----------------------------------------------------
Alpha = 1.37e-6 # kg(dry mater)/J
A_f = {}
#                             Tropical  Extratrop.    
#                             Forests     Forests    Savanna  Grasslands
#                             --------  ----------   -------  ----------
A_f['CO2']  = Alpha * np.array(( 1.000,     1.000,     1.000,       1.000 ))
A_f['CO']   = Alpha * np.array(( 1.000,     1.000,     1.000,       1.000 ))
A_f['SO2']  = Alpha * np.array(( 2.500,     4.500,     1.800,       1.800 ))
A_f['OC']   = Alpha * np.array(( 2.500,     4.500,     1.800,       1.800 ))
A_f['BC']   = Alpha * np.array(( 2.500,     4.500,     1.800,       1.800 ))
A_f['NH3']  = Alpha * np.array(( 2.500,     4.500,     1.800,       1.800 ))
A_f['PM25'] = Alpha * np.array(( 2.500,     4.500,     1.800,       1.800 ))
A_f['TPM']  = Alpha * np.array(( 2.500,     4.500,     1.800,       1.800 ))
A_f['NO']   = Alpha * np.array(( 1.000,     1.000,     1.000,       1.000 ))
A_f['MEK']  = Alpha * np.array(( 1.000,     1.000,     1.000,       1.000 ))
A_f['C3H6'] = Alpha * np.array(( 1.000,     1.000,     1.000,       1.000 ))
A_f['C2H6'] = Alpha * np.array(( 1.000,     1.000,     1.000,       1.000 ))
A_f['C3H8'] = Alpha * np.array(( 1.000,     1.000,     1.000,       1.000 ))
A_f['ALK4'] = Alpha * np.array(( 1.000,     1.000,     1.000,       1.000 ))
A_f['ALD2'] = Alpha * np.array(( 1.000,     1.000,     1.000,       1.000 ))
A_f['CH2O'] = Alpha * np.array(( 1.000,     1.000,     1.000,       1.000 ))
A_f['ACET'] = Alpha * np.array(( 1.000,     1.000,     1.000,       1.000 ))
A_f['CH4']  = Alpha * np.array(( 1.000,     1.000,     1.000,       1.000 ))


# Satellite Fudge Factor
# ----------------------
S_f = {}
S_f['MODIS_TERRA'] = 1.385 * alpha[0] # C6 scaling based on C5 above
S_f['MODIS_AQUA' ] = 0.473

#                     Tropical  Extratrop.    
#                      Forests     Forests   Savanna  Grasslands
#                     --------  ----------   -------  ----------
#
#S_f['MODIS_TERRA'] = ( 1.000,      1.000,    1.000,       1.000 )
#S_f['MODIS_AQUA' ] = ( 1.000,      1.000,    1.000,       1.000 )



class Emissions(object):
    """
    Class for computing emissions from pre-gridded FRP
    estimates.
    """

#---
    def __init__(self, date, FRP, F, Land, Water, Cloud, biomes='default', Verb=0):
        """
        Initializes an Emission object. On input,

          date  ---   date object

          FRP   ---   Dictionary keyed by satellite name
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
        
          Land  ---   Dictionary keyed by satellite name
                      with each element containing a
                      observed clear-land area [km2] for each gridbox
          
          Water ---   Dictionary keyed by satellite name
                      with each element containing a
                      water area [km2] for each gridbox

          Cloud ---   Dictionary keyed by satellite name
                      with each element containing a
                      cloud area [km2] for each gridbox            
        """

#       Save relevant information
#       -------------------------
        self.Land  = Land
        self.Water = Water
        self.Cloud = Cloud
        self.FRP   = FRP
        self.F     = F
        self.Sat   = list(Land.keys())
        self.date  = date
        self.verb  = Verb


#       Filter missing data
#       -------------------
        eps = 1.0e-2
        FillValue=1.0e20
        
        missing = []
        for sat in self.Sat:
            m = Land[sat][:,:]  > (1 - eps)*FillValue
            m = np.logical_or(m, Water[sat][:,:] > (1 - eps)*FillValue)
            m = np.logical_or(m, Cloud[sat][:,:] > (1 - eps)*FillValue)

            n_biomes = len(FRP[sat])
            for b in range(n_biomes):
                m = np.logical_or(m, FRP[sat][b][:,:] > (1 - eps)*FillValue)

            if np.any(m): 
                print('[w] Detected missing area or FRP values in %s QFED/L3A file on %s' % (sat, self.date))
           
            Land[sat][m]  = 0.0
            Water[sat][m] = 0.0 
            Cloud[sat][m] = 0.0
            for b in range(n_biomes):
                FRP[sat][b][m] = 0.0

            if np.all(m):
                missing.append(True)
            else:
                missing.append(False)

        assert not np.all(missing), '[x] No valid L3A input data. Please persist emissions from the last known good date.'
                

#       Biomes
#       -----------
        if biomes == 'default':
            self.biomes = ('Tropical Forest', 'Extratropical Forests', 'Savanna', 'Grassland')
        else:
            self.biomes = biomes[:]


#       Record grid
#       -----------
        self.im, self.jm = Land[self.Sat[0]].shape
        if (5*self.im - 8*(self.jm - 1)) == 0:
            self.lon  = np.linspace(-180.,180.,self.im,endpoint=False)
            self.lat  = np.linspace(-90.,90.,self.jm)
        else:
            d_lon = 360.0 / self.im
            d_lat = 180.0 / self.jm
            self.lon = np.linspace(-180+d_lon/2, 180-d_lon/2, self.im)
            self.lat = np.linspace( -90+d_lat/2,  90-d_lat/2, self.jm)

#---
    def calculate(self, Species='all', method='default'):
    
        """
        Calculate emissions for each species using built-in
        emission coefficients and fudge factors.

        The default list of species is: 
        Species = ('CO', 'CO2','SO2','OC','BC','NH3','PM25','NO','MEK', 
                   'C3H6','C2H6','C3H8','ALK4','ALD2','CH2O','ACET','CH4')

        The default method for computing the emissions is:
            method = 'sequential-zero' 
        """


        if (Species == 'all') or (Species == 'default'):
            species = ('CO'  , 'CO2' , 'SO2' , 'OC'  , 'BC'  , 'NH3' , 
                       'PM25', 'NO'  , 'MEK' , 'C3H6', 'C2H6', 'C3H8', 
                       'ALK4', 'ALD2', 'CH2O', 'ACET', 'CH4')
        else:
            species = Species[:]


        # factor needed to convert B_f from [g/kg] to [kg/kg]
        units_factor = 1e-3

        n_biomes = len(self.biomes)

        A_l = np.zeros((self.im, self.jm))
        A_w = np.zeros((self.im, self.jm))
        A_c = np.zeros((self.im, self.jm))

        for sat in self.Sat:
            A_l += self.Land[sat]
            A_w += self.Water[sat]
            A_c += self.Cloud[sat]

        A_o = A_l + A_w

        i = (A_l > 0)
        j = ((A_l + A_c) > 0)

        E = {}
        E_= {}
        for s in species:
            E[s]  = np.zeros((n_biomes, self.im, self.jm))
            E_[s] = np.zeros((n_biomes, self.im, self.jm))

            for sat in self.Sat:
                FRP = self.FRP[sat]
                F   = self.F[sat]
                A_  = self.Cloud[sat]

                for b in range(n_biomes):
                    E[s][b,:,:]  += units_factor * A_f[s][b] * S_f[sat] * B_f[s][b] * FRP[b]
                    E_[s][b,:,:] += units_factor * A_f[s][b] * S_f[sat] * B_f[s][b] * F[b] * A_
           
            for b in range(n_biomes):
                E_b  = E[s][b,:,:]
                E_b_ = E_[s][b,:,:]

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
                   

#       Save Forecast dictionary
#       ------------------------
        dt  = 1.0    # days
        tau = 3.0    # days

        for sat in self.Sat:
            for b in range(n_biomes):
                s = species[0]

                self.F[sat][b][:,:] = (E[s][b,:,:] / (units_factor * A_f[s][b] * S_f[sat] * B_f[s][b])) * np.exp(-dt/tau)
                self.F[sat][b][j] = self.F[sat][b][j] * ((A_o[j] + A_c[j]) / (A_l[j] + A_c[j]))

#       Save Emission dictionary
#       ------------------------
        self.Species = species
        self.Emissions = E


#---
    def total(self, specie):
        """
        Calculates the emissions from all biomes.
        """

        return np.sum(self.Emissions[specie][:,:,:], axis=0)


#---
    def _write_ana(self, filename=None, dir='.', expid='qfed2', col='sfc', tag=None, fill_value=1e15):
       """
       Writes gridded emissions. You must call method
       calculate() first. Optional input parameters:

       filename  ---  file name; if not specified each
                      species will be written to a separate
                      file, e.g.,
                         qfed2.emis_co.sfc.20030205.nc4
       dir       ---  optional directory name, only used
                      when *filename* is omitted
       expid     ---  optional experiment id, only used
                      when *filename* is omitted
       col       ---  collection
       tag       ---  tag name, by default it will be set to
                      the QFED CVS tag name as part of the 
                      installation procedure
       
       """
       

#      Create directory for output file
#      --------------------------------
       dir = os.path.join(dir, 'Y%04d'%self.date.year, 'M%02d'%self.date.month)
       rc = os.system("/bin/mkdir -p %s"%dir)
       if rc:
           raise IOError('cannot create output directory')


       nymd = 10000*self.date.year + 100*self.date.month + self.date.day
       nhms = 120000

#      Loop over species
#      -----------------
       vname_ = ( 'biomass', 'biomass_tf', 'biomass_xf', 'biomass_sv', 'biomass_gl' )

       filename_ = filename
       self.filename = {}
       for s in self.Species:

           if filename_ is None:
               filename = dir+'/%s.emis_%s.%s.%d.nc4'\
                          %(expid,s.lower(),col,nymd)
               vname = vname_ 
           else:
               vname = [ '%s_%s' % (s.lower(), name) for name in vname_]

           vtitle = [ '%s Biomass Emissions' % s, 
                      '%s Biomass Emissions from Tropical Forests' % s, 
                      '%s Biomass Emissions from Extratropical Forests' % s,
                      '%s Biomass Emissions from Savanna' % s,
                      '%s Biomass Emissions from Grasslands' % s ]
                      
           vunits = [ 'kg s-1 m-2', 'kg s-1 m-2', 'kg s-1 m-2', 'kg s-1 m-2', 'kg s-1 m-2' ]

           self.filename[s] = filename

#          Create new file or overwrite existing file
#          ------------------------------------------
           f = nc.Dataset(filename, 'w', format='NETCDF4')
    
           # global attributes
           f.Conventions = 'COARDS'
           f.Source      = 'NASA/GSFC, Global Modeling and Assimilation Office'
           f.Title       = 'QFED Level3b v{version:s} Gridded Emissions'.format(version=__version__)
           f.Contact     = 'Anton Darmenov <anton.s.darmenov@nasa.gov>'
           f.Version     = str(__version__)
           f.Processed   = str(datetime.now())
           f.History     = '' 

           # dimensions
           f.createDimension('lon', len(self.lon))
           f.createDimension('lat', len(self.lat))
           f.createDimension('time', None)
 
           # variables
           v_lon    = f.createVariable('lon',  'f8', ('lon'))
           v_lat    = f.createVariable('lat',  'f8', ('lat'))
           v_time   = f.createVariable('time', 'i4', ('time'))

           v_biomass    = f.createVariable('biomass',    'f4', ('time', 'lat', 'lon'), fill_value=fill_value, zlib=False)
           v_biomass_tf = f.createVariable('biomass_tf', 'f4', ('time', 'lat', 'lon'), fill_value=fill_value, zlib=False)
           v_biomass_xf = f.createVariable('biomass_xf', 'f4', ('time', 'lat', 'lon'), fill_value=fill_value, zlib=False)
           v_biomass_sv = f.createVariable('biomass_sv', 'f4', ('time', 'lat', 'lon'), fill_value=fill_value, zlib=False)
           v_biomass_gl = f.createVariable('biomass_gl', 'f4', ('time', 'lat', 'lon'), fill_value=fill_value, zlib=False)

            # variables attributes
           v_lon.long_name         = 'longitude'
           v_lon.standard_name     = 'longitude'
           v_lon.units             = 'degrees_east'
           v_lon.comment           = 'center_of_cell'

           v_lat.long_name         = 'latitude'
           v_lat.standard_name     = 'latitude'
           v_lat.units             = 'degrees_north'
           v_lat.comment           = 'center_of_cell'

           begin_date        = int(self.date.strftime('%Y%m%d'))
           begin_time        = int(self.date.strftime('%H%M%S'))
           v_time.long_name  = 'time'
           v_time.units      = 'minutes since {:%Y-%m-%d %H:%M:%S}'.format(self.date)
           v_time.begin_date = np.array(begin_date, dtype=np.int32)
           v_time.begin_time = np.array(begin_time, dtype=np.int32)
           v_time.time_increment = np.array(240000, dtype=np.int32)

           v_biomass.long_name = vtitle[0]
           v_biomass.units = 'kg s-1 m-2'
           v_biomass.missing_value = np.array(fill_value, np.float32)
           v_biomass.fmissing_value = np.array(fill_value, np.float32)
           v_biomass.vmin = np.array(fill_value, np.float32)
           v_biomass.vmax = np.array(fill_value, np.float32)
           
           v_biomass_tf.long_name = vtitle[1]
           v_biomass_tf.units = 'kg s-1 m-2'
           v_biomass_tf.missing_value = np.array(fill_value, np.float32)
           v_biomass_tf.fmissing_value = np.array(fill_value, np.float32)
           v_biomass_tf.vmin = np.array(fill_value, np.float32)
           v_biomass_tf.vmax = np.array(fill_value, np.float32)

           v_biomass_xf.long_name = vtitle[2]
           v_biomass_xf.units = 'kg s-1 m-2'
           v_biomass_xf.missing_value = np.array(fill_value, np.float32)
           v_biomass_xf.fmissing_value = np.array(fill_value, np.float32)
           v_biomass_xf.vmin = np.array(fill_value, np.float32)
           v_biomass_xf.vmax = np.array(fill_value, np.float32)

           v_biomass_sv.long_name = vtitle[3]
           v_biomass_sv.units = 'kg s-1 m-2'
           v_biomass_sv.missing_value = np.array(fill_value, np.float32)
           v_biomass_sv.fmissing_value = np.array(fill_value, np.float32)
           v_biomass_sv.vmin = np.array(fill_value, np.float32)
           v_biomass_sv.vmax = np.array(fill_value, np.float32)

           v_biomass_gl.long_name = vtitle[4]
           v_biomass_gl.units = 'kg s-1 m-2'
           v_biomass_gl.missing_value = np.array(fill_value, np.float32)
           v_biomass_gl.fmissing_value = np.array(fill_value, np.float32)
           v_biomass_gl.vmin = np.array(fill_value, np.float32)
           v_biomass_gl.vmax = np.array(fill_value, np.float32)

           
           # data
           v_time[:] = np.array((0,))
           v_lon[:]  = np.array(self.lon)
           v_lat[:]  = np.array(self.lat)

           v_biomass[0,:,:]    = np.transpose(self.total(s)[:,:])
           v_biomass_tf[0,:,:] = np.transpose(self.Emissions[s][0,:,:])
           v_biomass_xf[0,:,:] = np.transpose(self.Emissions[s][1,:,:])
           v_biomass_sv[0,:,:] = np.transpose(self.Emissions[s][2,:,:])
           v_biomass_gl[0,:,:] = np.transpose(self.Emissions[s][3,:,:])

           f.close()

           if self.verb >=1:
               print("[w] Wrote file "+filename)


#---
    def _write_fcs(self, forecast, fill_value=1.0e15):
       """
       Writes gridded emissions. You must call method
       calculate() first. Input parameter(s):

       forecast        ---  L3a file names
       forecast_fields ---  Variable names of FRP density forecast
       """
     
       _date = self.date + timedelta(days=1)
       nymd = 10000*_date.year + 100*_date.month + _date.day
       nhms = 120000

       for sat in self.Sat:

           # create a file
           _filename = forecast[sat]

           f = nc.Dataset(_filename, 'w', format='NETCDF4')
        
           # global attributes
           f.Conventions = 'COARDS'
           f.Source      = 'NASA/GSFC, Global Modeling and Assimilation Office'
           f.Title       = 'QFED Level3a v{version:s} Gridded FRP Estimates'.format(version=__version__)
           f.Contact     = 'Anton Darmenov <anton.s.darmenov@nasa.gov>'
           f.Version     = str(__version__)
           f.Processed   = str(datetime.now())
           f.History     = '' 
    
           # dimensions
           f.createDimension('lon', len(self.lon))
           f.createDimension('lat', len(self.lat))
           f.createDimension('time', None)
     
           # variables
           v_lon    = f.createVariable('lon',  'f8', ('lon'))
           v_lat    = f.createVariable('lat',  'f8', ('lat'))
           v_time   = f.createVariable('time', 'i4', ('time'))
    
           v_land   = f.createVariable('land',   'f4', ('time', 'lat', 'lon'), fill_value=fill_value, zlib=False)
           v_water  = f.createVariable('water',  'f4', ('time', 'lat', 'lon'), fill_value=fill_value, zlib=False)
           v_cloud  = f.createVariable('cloud',  'f4', ('time', 'lat', 'lon'), fill_value=fill_value, zlib=False)
    
           v_frp_tf = f.createVariable('frp_tf', 'f4', ('time', 'lat', 'lon'), fill_value=fill_value, zlib=False)
           v_frp_xf = f.createVariable('frp_xf', 'f4', ('time', 'lat', 'lon'), fill_value=fill_value, zlib=False)
           v_frp_sv = f.createVariable('frp_sv', 'f4', ('time', 'lat', 'lon'), fill_value=fill_value, zlib=False)
           v_frp_gl = f.createVariable('frp_gl', 'f4', ('time', 'lat', 'lon'), fill_value=fill_value, zlib=False)
           
           v_fb_tf  = f.createVariable('fb_tf',  'f4', ('time', 'lat', 'lon'), fill_value=fill_value, zlib=False)
           v_fb_xf  = f.createVariable('fb_xf',  'f4', ('time', 'lat', 'lon'), fill_value=fill_value, zlib=False)
           v_fb_sv  = f.createVariable('fb_sv',  'f4', ('time', 'lat', 'lon'), fill_value=fill_value, zlib=False)
           v_fb_gl  = f.createVariable('fb_gl',  'f4', ('time', 'lat', 'lon'), fill_value=fill_value, zlib=False)
    
    
           # variables attributes
           v_lon.long_name         = 'longitude'
           v_lon.standard_name     = 'longitude'
           v_lon.units             = 'degrees_east'
           v_lon.comment           = 'center_of_cell'
    
           v_lat.long_name         = 'latitude'
           v_lat.standard_name     = 'latitude'
           v_lat.units             = 'degrees_north'
           v_lat.comment           = 'center_of_cell'
    
           begin_date        = int(_date.strftime('%Y%m%d'))
           begin_time        = int(_date.strftime('%H%M%S'))
           v_time.long_name  = 'time'
           v_time.units      = 'minutes since {:%Y-%m-%d %H:%M:%S}'.format(_date)
           v_time.begin_date = np.array(begin_date, dtype=np.int32)
           v_time.begin_time = np.array(begin_time, dtype=np.int32)
           v_time.time_increment = np.array(240000, dtype=np.int32)
    
           v_land.long_name = "Observed Clear Land Area"
           v_land.units = "km2"
           v_land.missing_value = np.array(fill_value, np.float32)
           v_land.fmissing_value = np.array(fill_value, np.float32)
           v_land.vmin = np.array(fill_value, np.float32)
           v_land.vmax = np.array(fill_value, np.float32)
           
           v_water.long_name = "Water Area"
           v_water.units = "km2"
           v_water.missing_value = np.array(fill_value, np.float32)
           v_water.fmissing_value = np.array(fill_value, np.float32)
           v_water.vmin = np.array(fill_value, np.float32)
           v_water.vmax = np.array(fill_value, np.float32)
           
           v_cloud.long_name = "Obscured by Clouds Area"
           v_cloud.units = "km2"
           v_cloud.missing_value = np.array(fill_value, np.float32)
           v_cloud.fmissing_value = np.array(fill_value, np.float32)
           v_cloud.vmin = np.array(fill_value, np.float32)
           v_cloud.vmax = np.array(fill_value, np.float32)
           
           v_frp_tf.long_name = "Fire Radiative Power (Tropical Forests)"
           v_frp_tf.units = "MW"
           v_frp_tf.missing_value = np.array(fill_value, np.float32)
           v_frp_tf.fmissing_value = np.array(fill_value, np.float32)
           v_frp_tf.vmin = np.array(fill_value, np.float32)
           v_frp_tf.vmax = np.array(fill_value, np.float32)
           
           v_frp_xf.long_name = "Fire Radiative Power (Extra-tropical Forests)"
           v_frp_xf.units = "MW"
           v_frp_xf.missing_value = np.array(fill_value, np.float32)
           v_frp_xf.fmissing_value = np.array(fill_value, np.float32)
           v_frp_xf.vmin = np.array(fill_value, np.float32)
           v_frp_xf.vmax = np.array(fill_value, np.float32)
           
           v_frp_sv.long_name = "Fire Radiative Power (Savanna)"
           v_frp_sv.units = "MW"
           v_frp_sv.missing_value = np.array(fill_value, np.float32)
           v_frp_sv.fmissing_value = np.array(fill_value, np.float32)
           v_frp_sv.vmin = np.array(fill_value, np.float32)
           v_frp_sv.vmax = np.array(fill_value, np.float32)
           
           v_frp_gl.long_name = "Fire Radiative Power (Grasslands)"
           v_frp_gl.units = "MW"
           v_frp_gl.missing_value = np.array(fill_value, np.float32)
           v_frp_gl.fmissing_value = np.array(fill_value, np.float32)
           v_frp_gl.vmin = np.array(fill_value, np.float32)
           v_frp_gl.vmax = np.array(fill_value, np.float32)
           
           v_fb_tf.long_name = "Background FRP Density (Tropical Forests)"
           v_fb_tf.units = "MW km-2"
           v_fb_tf.missing_value = np.array(fill_value, np.float32)
           v_fb_tf.fmissing_value = np.array(fill_value, np.float32)
           v_fb_tf.vmin = np.array(fill_value, np.float32)
           v_fb_tf.vmax = np.array(fill_value, np.float32)
           
           v_fb_xf.long_name = "Background FRP Density (Extra-tropical Forests)"
           v_fb_xf.units = "MW km-2"
           v_fb_xf.missing_value = np.array(fill_value, np.float32)
           v_fb_xf.fmissing_value = np.array(fill_value, np.float32)
           v_fb_xf.vmin = np.array(fill_value, np.float32)
           v_fb_xf.vmax = np.array(fill_value, np.float32)
           
           v_fb_sv.long_name = "Background FRP Density (Savanna)"
           v_fb_sv.units = "MW km-2"
           v_fb_sv.missing_value = np.array(fill_value, np.float32)
           v_fb_sv.fmissing_value = np.array(fill_value, np.float32)
           v_fb_sv.vmin = np.array(fill_value, np.float32)
           v_fb_sv.vmax = np.array(fill_value, np.float32)
           
           v_fb_gl.long_name = "Background FRP Density (Grasslands)"
           v_fb_gl.units = "MW km-2"
           v_fb_gl.missing_value = np.array(fill_value, np.float32)
           v_fb_gl.fmissing_value = np.array(fill_value, np.float32)
           v_fb_gl.vmin = np.array(fill_value, np.float32)
           v_fb_gl.vmax = np.array(fill_value, np.float32)
    
           # data
           v_time[:] = np.array((0,))
           v_lon[:]  = np.array(self.lon)
           v_lat[:]  = np.array(self.lat)
    
           missing = np.full(np.transpose(self.F[sat][0]).shape, fill_value)
    
           v_land[0,:,:]   = missing[:,:] 
           v_water[0,:,:]  = missing[:,:]
           v_cloud[0,:,:]  = missing[:,:]
           v_frp_tf[0,:,:] = missing[:,:]
           v_frp_xf[0,:,:] = missing[:,:]
           v_frp_sv[0,:,:] = missing[:,:]
           v_frp_gl[0,:,:] = missing[:,:]

           v_fb_tf[0,:,:]  = np.transpose(self.F[sat][0])
           v_fb_xf[0,:,:]  = np.transpose(self.F[sat][1])
           v_fb_sv[0,:,:]  = np.transpose(self.F[sat][2])
           v_fb_gl[0,:,:]  = np.transpose(self.F[sat][3])
          
           f.close()
    
           if self.verb >=1:
               print('[w] Wrote file {file:s}'.format(file=_filename))



#---
    def write(self, filename=None, dir='.', forecast=None, expid='qfed2', col='sfc', 
                    tag=None, ndays=1, uncompressed=False):
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
       col           ---  collection
       tag           ---  tag name, by default it will be set to
                          the QFED CVS tag name as part of the 
                          installation procedure
       ndays         ---  persist emissions for a number of days
       uncompressed  ---  use n4zip to compress gridded output file
       
       """

#      Write out the emission files
#      ----------------------------
       self._write_fcs(forecast, fill_value=1.0e20)

       for n in range(ndays):

#          Write out the emission files
#          ----------------------------
           self._write_ana(filename=filename,dir=dir,expid=expid,col=col,tag=tag,fill_value=1e20)

#          Compress the files by default
#          -----------------------------
           if not uncompressed:
               for s in self.filename.keys():
                   rc = os.system("n4zip %s"%self.filename[s])
                   if rc:
                       warnings.warn('cannot compress output file <%s>'%self.filename[s])

#          Increment date by one day
#          ---------------------------------
           self.date = self.date + timedelta(days=1)

#..............................................................


