#!/usr/bin/env python3


"""
  A Python script to create QFED Level 3b files.
"""

import os
import warnings
import argparse
from datetime import datetime, timedelta
from glob import glob

import numpy as np
import netCDF4 as nc

from qfed.version import __version__
from qfed.emissions import Emissions

Sat = { 'MOD14': 'MODIS_TERRA', 'MYD14': 'MODIS_AQUA' }



def parse_arguments(default_values):
    '''
    Parse command line arguments
    '''

    parser = argparse.ArgumentParser(prog='QFED', description='Creates QFED Level 3b files')

    parser.add_argument('-V', '--version', action='version', 
                        version='%(prog)s {version:s}'.format(version=__version__))

    parser.add_argument('-i', '--input', dest='level3a_dir', 
                        default=default_values['level3a_dir'],
                        help='directory for input FRP files (default=%s)'\
                            %default_values['level3a_dir'])

    parser.add_argument('-o', '--output', dest='level3b_dir', 
                        default=default_values['level3b_dir'],
                        help='directory for output emissions files (default=%s)'\
                            %default_values['level3b_dir'])

    parser.add_argument('-p', '--products', dest='products', 
                        default=default_values['products'],
                        help='list of MODIS fire products (default=%s)'\
                            %default_values['products'])
    
    parser.add_argument('-x', '--expid', dest='expid', 
                        default=default_values['expid'],
                        help='experiment id (default=%s)'\
                            %default_values['expid'])

    parser.add_argument('-u', '--uncompressed',
                        action='store_true', 
                        help='do not compress output files (default=False)')

    parser.add_argument('-n', '--ndays', dest='ndays', type=int, 
                        default=default_values['fill_days'],
                        help='Number of days to fill in (default=%d)'\
                            %default_values['fill_days'])

    parser.add_argument('-v', '--verbose',
                        action='count', default=0, help='verbose')

    parser.add_argument('year', type=int, 
                        help="year specified in 'yyyy' format")

    parser.add_argument('doy', nargs='+', type=int,
                        help='single DOY, or start and end DOYs')


    args = parser.parse_args()

    # modify the Namespace object to set the 'doy' argument value
    if len(args.doy) == 1:
        doy_beg = args.doy[0]
        doy_end = doy_beg
    elif len(args.doy) == 2:
        doy_beg = min(args.doy[0], args.doy[1])
        doy_end = max(args.doy[0], args.doy[1])
    else:
        parser.error("must have one or two DOY arguments: doy | 'start doy' 'end doy'")

    args.doy = [doy_beg, doy_end]

    return args


def display_banner():
    print('')
    print('QFED Level 3B Processing')
    print('------------------------')
    print('')



if __name__ == "__main__":

    default_values = dict(expid       = 'qfed2',
                          level3a_dir = '/nobackup/2/MODIS/Level3',
                          level3b_dir = '/nobackup/2/MODIS/Level3',
                          products    = 'MOD14,MYD14',
                          fill_days   = 1)

    args = parse_arguments(default_values)

    if args.verbose > 0:
        display_banner()


    Products = args.products.split(',')

#   Grid FRP and observed area
#   --------------------------
    for doy in range(args.doy[0], args.doy[1] + 1):

        d = datetime(args.year,1,1) + timedelta(days=(doy - 1)) + timedelta(hours=12)

#       Read input FRP and Area for each satellite
#       ------------------------------------------
        FRP, Land, Water, Cloud, F = ( {}, {}, {}, {}, {} )
        for MxD14 in Products:

            sat = Sat[MxD14]

#           Open input file
#           ---------------
            l3a_dir  = os.path.join(args.level3a_dir, MxD14, 'Y%04d'%d.year, 'M%02d'%d.month)
            l3a_file = '%s_%s.frp.???.%04d%02d%02d.nc4'%(args.expid, MxD14, d.year, d.month, d.day)
            
            pat = os.path.join(l3a_dir, l3a_file)

            try:
                ifn = glob(pat)[0]
                f = nc.Dataset(ifn, 'r')
            except:
                print("[x] cannot find/read input FRP file for %s, ignoring it"%d)
                continue

            if args.verbose > 0:
                print("[] Reading ", ifn) 

            Land[sat]  = np.transpose(f.variables['land' ][0,:,:])
            Water[sat] = np.transpose(f.variables['water'][0,:,:])
            Cloud[sat] = np.transpose(f.variables['cloud'][0,:,:])
            
            FRP[sat] = [ np.transpose(f.variables['frp_tf'][0,:,:]),
                         np.transpose(f.variables['frp_xf'][0,:,:]),
                         np.transpose(f.variables['frp_sv'][0,:,:]),
                         np.transpose(f.variables['frp_gl'][0,:,:]) ]

            F[sat]   = [ np.transpose(f.variables['fb_tf'][0,:,:]),
                         np.transpose(f.variables['fb_xf'][0,:,:]),
                         np.transpose(f.variables['fb_sv'][0,:,:]),
                         np.transpose(f.variables['fb_gl'][0,:,:]) ]

            col = ifn.split('/')[-1].split('.')[2] # collection

#       FRP density forecast files
#       --------------------------
        d_ = d + timedelta(days=1)
        forecast_files = {}
        for MxD14 in Products:
            sat = Sat[MxD14]
 
            if sat in list(FRP.keys()):
                l3a_dir  = os.path.join(args.level3a_dir, MxD14, 'Y%04d'%d_.year, 'M%02d'%d_.month)
                l3a_file = '%s_%s.frp.%s.%04d%02d%02d.nc4'%(args.expid, MxD14, col, d_.year, d_.month, d_.day)
            
                forecast_files[sat] = os.path.join(l3a_dir, l3a_file)


#       Create the top level directory for output files
#       -----------------------------------------------
        dir = os.path.join(args.level3b_dir, 'QFED')
        rc = os.system("/bin/mkdir -p %s"%dir)
        if rc:
            raise IOError('cannot create output directory')

#       Write output file
#       -----------------
        E = Emissions(d, FRP, F, Land, Water, Cloud, Verb=(args.verbose > 0))
        E.calculate()
        E.write(dir=dir, forecast=forecast_files, expid=args.expid, col=col, ndays=args.ndays, 
                uncompressed=args.uncompressed)
        
