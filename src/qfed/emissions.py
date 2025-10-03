"""
Calculate emissions from Fire Radiative Flux (FRP/Area).
"""


import os
import subprocess
import logging
import yaml
from datetime import date, datetime, timedelta

import numpy as np
import netCDF4 as nc

from qfed import grid
from qfed.instruments import Instrument, Satellite
from qfed.instruments import canonical_satellite, canonical_instrument, sensor_code
from qfed import VERSION
from qfed import cli_utils
import re

def read_emission_factors(file):
    with open(file) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            data = None
            logging.critical(exc)

    return data['emission_factors']['species']


class Emissions:
    """
    Class for computing biomass burning emissions
    from gridded FRP and areas.
    """

    def __init__(self, time, FRP, F, area, emission_factors_file, qc_scaling_factors_file):
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

          area  ---   Dictionary keyed by satellite name
                      with each element containing a
                      observed clear-land area [km2] for each gridbox
        """

        self.area = area
        self.area_land = {k: area[k]['land'] for k in area.keys()}
        self.area_water = {k: area[k]['water'] for k in area.keys()}
        self.area_cloud = {k: area[k]['cloud'] for k in area.keys()}
        self.area_unknown = {k: area[k].get('unknown', 0.0) for k in area.keys()}
        self.FRP = FRP
        self.F = F
        self.time = time

        self.platform = list(FRP.keys())
        self.biomass_burning = list(FRP[self.platform[0]].keys())

        self.set_emission_factors(emission_factors_file)
        self.set_parameters(qc_scaling_factors_file)
        self.set_grid()

    def set_grid(self):
        """
        Sets the grid.
        """
        # TODO: should use the grid module
        self.im, self.jm = self.area_land[self.platform[0]].shape

        if (5 * self.im - 8 * (self.jm - 1)) == 0:
            self.lon = np.linspace(-180.0, 180.0, self.im, endpoint=False)
            self.lat = np.linspace(-90.0, 90.0, self.jm)
        else:
            d_lon = 360.0 / self.im
            d_lat = 180.0 / self.jm

            o_lon = 0.5 * d_lon
            o_lat = 0.5 * d_lat

            self.lon = np.linspace(-180.0 + o_lon, 180.0 - o_lon, self.im)
            self.lat = np.linspace(-90.0 + o_lat, 90.0 - o_lat, self.jm)

    def set_emission_factors(self, emission_factors_file):
        """
        Set the emission factors.
        """
        self.emission_factors_file = emission_factors_file
        self._EF = read_emission_factors(emission_factors_file)

    def set_parameters(self, qc_scaling_factors_file):
        """
        Set parameters used in the calculation of emissions,
        including combustion rate, satellite factors, etc.
        """
        
        qc_scaling = cli_utils.read_config(qc_scaling_factors_file)
        
        # Combustion rate constant (ECMWF Tech Memo 596).
        # It could be biome-dependent in case we want to tinker
        # with the A-M emission factors
        Alpha = qc_scaling['Alpha']  # kg(dry mater)/J
        
        # enhancement factors for species contributing to AOD
        AEROSOL_SPECIES = ('oc', 'bc', 'so2', 'nh3', 'pm25', 'tpm')

        # enhancement factors for species contributing to AOD
        enhance_aerosol = qc_scaling['enhance_aerosol_factor']
        # enhancement factors for non-aerosol species
        enhance_gas = qc_scaling['enhance_gas_factor']        

        # effective combustion rate
        self._A_f = {}
        self._A_f[(Instrument.MODIS, Satellite.TERRA)] = {}
        self._A_f[(Instrument.MODIS, Satellite.AQUA)] = {}
        self._A_f[(Instrument.VIIRS, Satellite.JPSS1)] = {}
        self._A_f[(Instrument.VIIRS, Satellite.JPSS2)] = {}
        self._A_f[(Instrument.VIIRS, Satellite.NPP)] = {}
  
        for s in self._EF.keys():
            if s in AEROSOL_SPECIES:
                enhance_factor = enhance_aerosol
            else:
                enhance_factor = enhance_gas

            self._A_f[(Instrument.MODIS, Satellite.TERRA)][s] = {
                b: (qc_scaling['modis/aqua'][b] * Alpha * v) for (b, v) in enhance_factor.items()
            }

            self._A_f[(Instrument.MODIS, Satellite.AQUA)][s] = {
                b: (qc_scaling['modis/terra'][b] * Alpha * v) for (b, v) in enhance_factor.items()
            }

            self._A_f[(Instrument.VIIRS, Satellite.JPSS2)][s] = {
                b: (qc_scaling['viirs/jpss-2'][b] * Alpha * v) for (b, v) in enhance_factor.items()
            }
            self._A_f[(Instrument.VIIRS, Satellite.JPSS1)][s] = {
                b: (qc_scaling['viirs/jpss-1'][b] * Alpha * v) for (b, v) in enhance_factor.items()
            }
            self._A_f[(Instrument.VIIRS, Satellite.NPP)][s] = {
                b: (qc_scaling['viirs/npp'][b] * Alpha * v ) for (b, v) in enhance_factor.items()
            }

        self._S_f = {}
        self._S_f[(Instrument.MODIS, Satellite.TERRA)] = qc_scaling['satscale']['modis/terra']
        self._S_f[(Instrument.MODIS, Satellite.AQUA)]  = qc_scaling['satscale']['modis/aqua']
        self._S_f[(Instrument.VIIRS, Satellite.JPSS2)] = qc_scaling['satscale']['viirs/jpss-2']
        self._S_f[(Instrument.VIIRS, Satellite.JPSS1)] = qc_scaling['satscale']['viirs/jpss-1']
        self._S_f[(Instrument.VIIRS, Satellite.NPP)]   = qc_scaling['satscale']['viirs/npp']


    def emission_factor(self, species, fire):
        """
        Returns emission factor for species emitted
        from different types of fires.
        """
        bb = fire.type.name.lower()
        return self._EF[species][bb]

    def effective_combustion_rate(self, platform, species, fire):
        """
        Returns the effective combustion rate parameter
        used in the calculation of emissions.
        """
        bb = fire.type.name.lower()
        return self._A_f[platform][species][bb]

    def satellite_factor(self, platform):
        """
        Returns the satellite factor for obs. platform.
        """
        return self._S_f[platform]

    def calculate(self, species=[], method='default', 
                  dt = 1.0, tau = 3.0):
        """
        Calculate emissions for each species using built-in
        emission coefficients and fudge factors.

        The default method for computing the emissions is
        'sequential-zero'.
        """
        
        self.dt = dt
        self.tau = tau
        
        if not species:
            species = self._EF.keys()

        # [g/kg] to [kg/kg] conversion factor
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

        i = A_l > 0
        j = (A_l + A_c) > 0

        E = {}
        E_ = {}
        for s in species:
            E[s] = {bb: np.zeros((self.im, self.jm)) for bb in self.biomass_burning}
            E_[s] = {bb: np.zeros((self.im, self.jm)) for bb in self.biomass_burning}

            for p in self.platform:
                S_f = self.satellite_factor(p)
                FRP = self.FRP[p]
                F = self.F[p]
                A_ = self.area_cloud[p]

                for bb in self.biomass_burning:
                    # TODO: A_f should be a dict.
                    B_f, eB_f = self.emission_factor(s, bb)
                    A_f = self.effective_combustion_rate(p, s, bb)
                    E[s][bb][:, :] += units_factor * A_f * S_f * B_f * FRP[bb]
                    E_[s][bb][:, :] += units_factor * A_f * S_f * B_f * F[bb] * A_

            for bb in self.biomass_burning:
                E_b = E[s][bb][:, :]
                E_b_ = E_[s][bb][:, :]

                if (method == 'default') or (method == 'sequential'):
                    E_b[j] = ( (E_b[j]  / (A_o[j] + A_c[j])) * (1 + A_c[j] / (A_l[j] + A_c[j])) +
                               (E_b_[j] / (A_o[j] + A_c[j])) * (    A_c[j] / (A_l[j] + A_c[j])) )

                if method == 'sequential-zero':
                    E_b[i] = (
                        E_b[i] / (A_o[i] + A_c[i]) * (1.0 + A_c[i] / (A_l[i] + A_c[i]))
                    )

                if method == 'nofires':
                    E_b[i] = E_b[i] / (A_o[i] + A_c[i])

                if method == 'similarity':
                    E_b[i] = E_b[i] / A_l[i] * ((A_l[i] + A_c[i]) / (A_o[i] + A_c[i]))

                if method == 'similarity_qfed-2.2':
                    E_b[i] = E_b[i] / A_o[i]

        self.estimate = E
        
        # Update forecast of FRP density based on current emissions


        # Select first species to use for calculating forecast
        s = list(species)[0]

        for p in self.platform:
            S_f = self.satellite_factor(p)
            
            for bb in self.biomass_burning:
                B_f, eB_f = self.emission_factor(s, bb)
                A_f = self.effective_combustion_rate(p, s, bb)
                
                # Calculate F using the first species as reference (similar to old code)
                self.F[p][bb][:,:] = (self.estimate[s][bb][:,:] / 
                                    (units_factor * A_f * S_f * B_f)) * np.exp(-dt/tau)
                self.F[p][bb][j] = self.F[p][bb][j] * ((A_o[j] + A_c[j]) / (A_l[j] + A_c[j]))

    def total(self, species):
        """
        Calculates total emissions from fires in all biomes.
        """
        result = np.zeros((self.im, self.jm))
        for bb in self.biomass_burning:
            result += self.estimate[species][bb][:, :]

        return result

    # helper function to add the l3a source 
    def _platform_label(self, inst_enum, sat_enum):
        inst_label = canonical_instrument.get(inst_enum, inst_enum.value.lower())
        sat_label  = canonical_satellite.get(sat_enum,  sat_enum.value.lower())
        return inst_label, sat_label, f"{inst_label}/{sat_label}"

    def _save_as_netcdf4(self, file, doi, compress=False, fill_value=1e15, diskless=False):
        """
        Saves gridded emissions to a file.
        """
        
        platforms = list(self.F.keys())
        platform_labels = set()
        for platform in platforms:
            try:
                inst_enum, sat_enum = platform
                inst_label, sat_label, label = self._platform_label(inst_enum, sat_enum)
                platform_labels.add(label)
            except Exception:
                logging.warning(f"Platform key is not (Instrument, Satellite): {platform!r}; skipping.")
                continue

        # map species to output files
        self.file = {
            species: file.format(species=species.lower())
            for species in self.estimate.keys()
        }

        for species, file in self.file.items():
            # construct meta data for the data variables: name, long name and units
            v_meta_data = {
                'total': {
                    'name': 'biomass',
                    'long_name': f'Biomass burning emissions of {species.upper()}',
                    'units': 'kg s-1 m-2',
                }
            }
            for bb in self.biomass_burning:
                fire = bb.type.name.title().replace('_', ' ')
                v_meta_data[bb] = {
                    'name': f'biomass_{bb.type.value}',
                    'long_name': f'Biomass burning emissions of {species.upper()} from {fire}',
                    'units': 'kg s-1 m-2',
                }

            # create a file
            f = nc.Dataset(file, 'w', format='NETCDF4', diskless=diskless)

            if diskless:
                logging.info(
                    f"Successfully created a diskless (in-memory) file '{file}'."
                )

            # global attributes
            f.institution = 'NASA/GSFC, Global Modeling and Assimilation Office'
            f.title = 'Quick Fire Emissions Dataset (QFED) Level 3 Gridded Emissions (v{0:s})'.format(VERSION)
            f.contact = 'qfed@lists.nasa.gov'
            f.VersionID = VERSION
            f.source = ', '.join(sorted(platform_labels))
            f.history = ''
            f.ShortName = 'QFED_EMIS' + '_X' + str(self.im) + 'Y' + str(self.jm)
            f.LongName = 'QFED Daily Level 3 Emissions at ' + str(360/self.im) + 'x' + str(np.round(180/self.jm,3)) + ' Degrees'
            f.GranuleID = os.path.basename(file)
            f.Format = 'NetCDF-4'
            f.RangeBeginningDate = self.time.strftime('%Y-%m-%d')
            f.RangeBeginningTime = self.time.strftime('00:00:00.000000')
            f.RangeEndingDate = self.time.strftime('%Y-%m-%d')
            f.RangeEndingTime = self.time.strftime('23:59:59.000000')
            f.IdentifierProductDOIAuthority = 'https://doi.org/'
            f.IdentifierProductDOI = str(doi)
            now=datetime.now()
            f.ProductionDateTime = now.strftime("%Y-%m-%dT%H:%M:%SZ")
            f.ProcessingLevel = 'Level 3'
            f.Conventions = 'CF-1.8'
            f.DataSetQuality = 'TBD'
            f.SouthernmostLatitude=str(self.lat[0])
            f.NorthernmostLatitude=str(self.lat[len(self.lat)-1])
            f.WesternmostLongitude=str(self.lon[0])
            f.EasternmostLongitude=str(self.lon[len(self.lon)-1])   
            f.RelatedURL = 'https://gmao.gsfc.nasa.gov/GMAO_products/qfed'    
            f.e_folding_time = f"{self.tau} days"
              
            # dimensions
            f.createDimension('lon', len(self.lon))
            f.createDimension('lat', len(self.lat))
            f.createDimension('time', None)

            # coordinate variables
            f.createVariable('lon', 'f8', dimensions='lon')
            f.createVariable('lat', 'f8', dimensions='lat')
            f.createVariable('time', 'i4', dimensions='time')

            # data variables
            for v in v_meta_data.values():
                f.createVariable(
                    v['name'],
                    'f4',
                    dimensions=('time', 'lat', 'lon'),
                    fill_value=fill_value,
                    zlib=compress,
                )

            # coordinate variables - attributes
            v = f.variables['lon']
            v.long_name = 'longitude'
            v.standard_name = 'longitude'
            v.units = 'degrees_east'
            v.comment = 'center_of_cell'

            v = f.variables['lat']
            v.long_name = 'latitude'
            v.standard_name = 'latitude'
            v.units = 'degrees_north'
            v.comment = 'center_of_cell'

            v = f.variables['time']
            begin_date = int(self.time.strftime('%Y%m%d'))
            begin_time = int(self.time.strftime('%H%M%S'))
            v.long_name = 'time'
            v.standard_name = 'time'
            v.units = 'minutes since {:%Y-%m-%d %H:%M:%S}'.format(self.time)
            v.begin_date = np.array(begin_date, dtype=np.int32)
            v.begin_time = np.array(begin_time, dtype=np.int32)
            v.time_increment = np.array(240000, dtype=np.int32)

            # data variables - attributes
            for _v in v_meta_data.values():
                v = f.variables[_v['name']]
                v.long_name = _v['long_name']
                v.units = _v['units']
                #v.missing_value = np.array(fill_value, np.float32)
                #v.fmissing_value = np.array(fill_value, np.float32)
                #v.vmin = np.array(fill_value, np.float32)
                #v.vmax = np.array(fill_value, np.float32)

            # coordinate variables - data
            f.variables['time'][:] = np.array((0,))
            f.variables['lon'][:] = np.array(self.lon)
            f.variables['lat'][:] = np.array(self.lat)

            # data variables - data
            f.variables['biomass'][0, :, :] = np.transpose(self.total(species)[:, :])
            for bb in self.biomass_burning:
                v = f.variables[v_meta_data[bb]['name']]
                v[0, :, :] = np.transpose(self.estimate[species][bb][:, :])

            f.close()

            logging.info(f"Successfully saved gridded emissions to file '{file}'.")
            

    def _materialize_sensor_path(self, tmpl_or_dict, inst_enum, sat_enum):
        # Allow dict mapping or a single template string
        if isinstance(tmpl_or_dict, dict):
            return tmpl_or_dict.get((inst_enum, sat_enum))
        tmpl = str(tmpl_or_dict)
        sat_tag = sensor_code(inst_enum, sat_enum)
        if "{sat}" in tmpl:
            return tmpl.format(sat=sat_tag)
        root, ext = os.path.splitext(tmpl)
        return f"{root}.{sat_tag}{ext or '.nc4'}"

    def _canon_labels(inst_enum, sat_enum):
        return (
            canonical_instrument.get(inst_enum, inst_enum.value.lower()),
            canonical_satellite.get(sat_enum,  sat_enum.value.lower()),
        )

    def _write_one_sensor_file(self, out_file, inst_enum, sat_enum, per_bb, compress, fill_value, diskless):
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        f = nc.Dataset(out_file, "w", format="NETCDF4", diskless=diskless)

        # ---- globals (mirror your style) ----
        f.institution  = 'NASA/GSFC, Global Modeling and Assimilation Office'
        f.title        = f'QFED Level 3 FRP Density Forecast (v{VERSION})'
        f.contact      = 'qfed@lists.nasa.gov'
        f.VersionID    = VERSION
        f.history      = ''
        f.ShortName    = 'QFED_FRP_F' + '_X' + str(self.im) + 'Y' + str(self.jm)
        f.LongName     = 'QFED Daily Level 3 FRP Density Forecast at ' + f"{360/self.im}x{np.round(180/self.jm,3)} Degrees"
        f.GranuleID    = os.path.basename(out_file)
        f.Format       = 'NetCDF-4'
        f.RangeBeginningDate = self.time.strftime('%Y-%m-%d')
        f.RangeBeginningTime = self.time.strftime('00:00:00.000000')
        f.RangeEndingDate    = self.time.strftime('%Y-%m-%d')
        f.RangeEndingTime    = self.time.strftime('23:59:59.000000')
        f.ProcessingLevel    = 'Level 3'
        f.Conventions        = 'CF-1.8'
        f.DataSetQuality     = 'TBD'
        f.SouthernmostLatitude = str(self.lat[0])
        f.NorthernmostLatitude = str(self.lat[-1])
        f.WesternmostLongitude = str(self.lon[0])
        f.EasternmostLongitude = str(self.lon[-1])
        f.RelatedURL         = 'https://gmao.gsfc.nasa.gov/GMAO_products/qfed'
        f.ProductionDateTime = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Optional single-code provenance (you said instrument label not needed)
        f.source = sensor_code(inst_enum, sat_enum)  # e.g., vj2/vj1/vnp/mod14/myd14
        f.e_folding_time = f"{self.tau} days"

        # ---- dims/coords ----
        f.createDimension('lon', len(self.lon))
        f.createDimension('lat', len(self.lat))
        f.createDimension('time', None)

        v = f.createVariable('lon', 'f8', ('lon',)); v.long_name='longitude'; v.standard_name='longitude'; v.units='degrees_east'; v.comment='center_of_cell'
        f.variables['lon'][:] = np.array(self.lon)

        v = f.createVariable('lat', 'f8', ('lat',)); v.long_name='latitude'; v.standard_name='latitude'; v.units='degrees_north'; v.comment='center_of_cell'
        f.variables['lat'][:] = np.array(self.lat)

        v = f.createVariable('time', 'i4', ('time',))
        v.long_name='time'; v.standard_name='time'
        v.units=f"minutes since {self.time:%Y-%m-%d %H:%M:%S}"
        v.begin_date = np.int32(int(self.time.strftime('%Y%m%d')))
        v.begin_time = np.int32(int(self.time.strftime('%H%M%S')))
        v.time_increment = np.int32(240000)
        f.variables['time'][:] = np.array((0,))

        # ---- vars: fb_{biome} ----
        for bb in self.biomass_burning:
            bb_code = bb.type.value
            bb_name = bb.type.name.title().replace('_', ' ')
            var_name = f"fb_{bb_code}"

            vv = f.createVariable(var_name, 'f4', ('time','lat','lon'), fill_value=fill_value, zlib=bool(compress))
            vv.long_name = f"Background FRP Density ({bb_name})"
            vv.units = "MW km-2"

            arr = per_bb.get(bb)
            if arr is not None and np.size(arr) > 0:
                vv[0, :, :] = np.transpose(arr)
            else:
                vv[0, :, :] = fill_value

        f.close()

    def _save_forecast(self, l3a_density_out, compress=False, fill_value=1.0e15, diskless=False):
        """
        Write **one FRP-FCS file per sensor** (Instrument, Satellite).
        l3a_density_out: dict mapping (inst, sat)->path OR a single template string.
                         If template lacks {sat}, a .{sensor_code} suffix is appended.
        """
        F = getattr(self, "F", None)
        if not isinstance(F, dict) or not F:
            logging.warning("No forecast field 'F' found or empty; nothing to save.")
            return

        for platform, per_bb in F.items():
            try:
                inst_enum, sat_enum = platform
            except Exception:
                logging.warning(f"Bad platform key: {platform!r}; skipping.")
                continue

            out_file = self._materialize_sensor_path(l3a_density_out, inst_enum, sat_enum)
            if not out_file:
                logging.warning(f"No output path for {inst_enum.name}/{sat_enum.name}; skipping.")
                continue

            self._write_one_sensor_file(
                out_file, inst_enum, sat_enum, per_bb,
                compress=compress, fill_value=fill_value, diskless=diskless
            )
            logging.info(f"Saved FRP-FCS to {out_file}")

    def compress_n4zip(self):
        """
        Compress output files with n4zip.
        """
        for species, file in self.file.items():
            result = subprocess.run(
                'n4zip',
                file,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            if result.returncode:
                logging.warning(f"Could not compress file '{file}' with n4zip.")
            else:
                logging.info(f"Successfully compressed file '{file}' with n4zip.")

    def save(
        self,
        file,
        doi,
        ndays=1,
        compress=False,
        fill_value=1e20,
        diskless=False,
    ):
        """
        Saves gridded emissions to a file.

        If the argument ndays is larger than 1,
        the estimated emissions will be persisted
        over the specified number of days.
        
        Parameters:
        -----------
        file : str
            Filename template for emission files
        doi : str
            DOI for the dataset
        forecast : dict, optional
            Dictionary mapping platforms to forecast filenames
        ndays : int, default=1
            Number of days to persist the emissions
        compress : bool or str, default=False
            Whether to compress files (True or 'n4zip')
        fill_value : float, default=1e20
            Fill value for missing data
        diskless : bool, default=False
            If True, create NetCDF files in memory first
        """
        # Save emission files for current and future days
        for n in range(ndays):
            self._save_as_netcdf4(file, doi, compress in (True,), fill_value, diskless)

            if compress == 'n4zip':
                self.compress_n4zip()

            # increment time to persist emissions
            self.time = self.time + timedelta(hours=24)