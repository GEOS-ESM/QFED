#!/usr/bin/env python3


"""
  A Python script to create QFED Level 3a files.
"""

import warnings

import os

from datetime      import date, timedelta

import argparse  

from qfed.version import __version__
from qfed.mxd14_l3 import MxD14_L3



def parse_arguments(default_values):
    '''
    Parse command line options
    '''

    parser = argparse.ArgumentParser(prog='QFED', description='Creates QFED Level 3a files')

    parser.add_argument('-V', '--version', action='version', 
                      version='%(prog)s {version:s}'.format(version='3.0.rc1'))

    parser.add_argument('-f', '--mxd14', dest='level2_dir', default=default_values['level2_dir'],
                      help='directory for MOD14/MYD14 fire files (default=%s)'\
                           %default_values['level2_dir'])

    parser.add_argument('-g', '--mxd03', dest='level1_dir', default=default_values['level1_dir'],
                      help='directory for MOD03/MYD03 geolocaltion files (default=%s)'\
                           %default_values['level1_dir'])

    parser.add_argument('-i', '--igbp', dest='igbp_dir', default=default_values['igbp_dir'],
                      help='directory for IGBP vegetation database (default=%s)'\
                           %default_values['igbp_dir'])
    
    parser.add_argument('-o', '--output', dest='level3_dir', default=default_values['level3_dir'],
                      help='directory for output files (default=%s)'\
                           %default_values['level3_dir'])

    parser.add_argument('-p', '--products', dest='products', default=default_values['products'],
                      help='CSV list of MODIS fire products (default=%s)'\
                           %default_values['products'])
    
    parser.add_argument('-r', '--resolution', dest='res', default=default_values['res'],
                      help='horizontal resolution: a for 4x5, b for 2x2.5, etc. (default=%s)'\
                           %default_values['res'])
    
    parser.add_argument('-x', '--expid', dest='expid', default=default_values['expid'],
                      help='experiment id (default=%s)'\
                           %default_values['expid'])

    parser.add_argument('-q', '--disable-qc', 
                      action='store_true', dest='disable_qc',
                      help='disable quality control procedures (default=%s)'\
                           % False)

    parser.add_argument('-b', '--bootstrap', 
                      action='store_true', dest='bootstrap',
                      help='initialize FRP forecast (default=False)')

    parser.add_argument('-u', '--uncompressed',
                      action='store_true', 
                      help='do not compress output files (default=False)')

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


#---------------------------------------------------------------------

if __name__ == "__main__":

    default_values = dict(expid      = 'qfed2',
                          level1_dir = '/nobackup/2/MODIS/Level1',
                          level2_dir = '/nobackup/2/MODIS/Level2',
                          level3_dir = '/nobackup/2/MODIS/Level3',
                          products   = 'MOD14,MYD14',
                          igbp_dir   = '/nobackup/Emissions/Vegetation/GL_IGBP_INPE',
                          res        = 'e')


    args = parse_arguments(default_values)

    if args.verbose > 0:
        print('')
        print('QFED Level 3A Processing')
        print('------------------------')
        print('')

    Products = args.products.split(',')

#   Grid FRP and observed area
#   --------------------------
    for doy in range(args.doy[0], args.doy[1] + 1):
        d = date(args.year, 1, 1) + timedelta(days=(doy - 1))
        d_= d + timedelta(days=1)

#       Loop over products
#       ------------------
        for MxD14 in Products:

            Path = os.path.join(args.level2_dir, MxD14, '%04d'%d.year, '%03d'%doy)

#           Do the gridding for this product
#           --------------------------------
            fires = MxD14_L3(Path,
                             args.level1_dir,
                             args.igbp_dir,
                             res=args.res,
                             Verb=args.verbose)

#           Create directory for output file
#           --------------------------------
            dir_a = os.path.join(args.level3_dir, MxD14, 'Y%04d'%d.year, 'M%02d'%d.month)
            dir_f = os.path.join(args.level3_dir, MxD14, 'Y%04d'%d_.year, 'M%02d'%d_.month)

            dir = {'ana': dir_a, 'bkg': dir_f}
            for k in list(dir.keys()):
                rc  = os.system("/bin/mkdir -p %s"%dir[k])
                if rc:
                    raise(IOError, "Cannot create output directory '%s'" % dir[k])


#           Quality Control
#           ---------------
            qc = not args.disable_qc

#           Write output file
#           -----------------
            fires.write(dir=dir, expid=args.expid+'_'+MxD14, qc=qc, bootstrap=args.bootstrap)

            print ('****DEBUG****')

#           Compress it unless the user disabled it
#           ---------------------------------------
            if not args.uncompressed and fires.doy != None:
                rc = os.system("n4zip %s"%fires.filename)
                if rc:
                    warnings.warn('cannot compress output file <%s>'%fires.filename)

