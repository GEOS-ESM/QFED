#!/usr/bin/env python3


"""
  A Python script to create QFED Level 3a files.
"""

import os
from datetime import datetime, timedelta
import yaml
import argparse  

from qfed import version
from qfed import grid
from qfed import geolocation_products
from qfed import fire_products
from qfed import frp


def parse_arguments(default_values):
    '''
    Parse command line options
    '''

    parser = argparse.ArgumentParser(prog='QFED', description='Creates QFED Level 3a files')

    parser.add_argument('-c', '--config', dest='config', default=default_values['config'],
                        help='config file (default=%s)'\
                            %default_values['config'])

    parser.add_argument('-o', '--output', dest='frp_dir', default=default_values['frp_dir'],
                        help='directory for output files (default=%s)'\
                            %default_values['frp_dir'])

    parser.add_argument('-p', '--products', dest='products', default=default_values['products'],
                        help='list of fire products (default=%s)'\
                            %default_values['products'])
    
    parser.add_argument('-r', '--resolution', dest='res', default=default_values['res'],
                        help='horizontal resolution (default=%s)'\
                            %default_values['res'])

    parser.add_argument('-v', '--verbose',
                        action='count', default=0, help='verbose')

     
    args = parser.parse_args()

    return args


def read_config(config):
    with open(config) as file:
        try:
            data = yaml.safe_load(file)   
        except yaml.YAMLError as exc:
            data = None
            print(exc)

    return data


def display_banner():
    print('')
    print('QFED Level 3A Processing')
    print('------------------------')
    print('')



if __name__ == "__main__":

    default_values = dict(products = 'modis/aqua,modis/terra,viirs/npp,viirs/jpss-1',
                          res      = 'e',
                          frp_dir  = './',
                          config   = 'config.yaml')

    time = datetime(2021, 2, 1, 12)
    time_window = timedelta(hours=24)

    time_s = time - 0.5*time_window
    time_e = time + 0.5*time_window

    args = parse_arguments(default_values)
    config = read_config(args.config)
    products = args.products.split(',')

    if args.verbose > 0: 
        display_banner()

    for p in products:
        instrument, satellite = p.split('/')

        gp_dir = config[p]['geolocation']
        fp_dir, fp_template = config[p]['fires']

        grid_ = grid.Grid(args.res)
    
        fs = frp.FileSelector(os.path.join(gp_dir, '{0:%Y}', '{0:%j}'),
                              os.path.join(fp_dir, '{0:%Y}', '{0:%j}'), 
                              fp_template)
    
        fp_reader = fire_products.create(instrument, satellite, verbosity=args.verbose)
        gp_reader = geolocation_products.create(instrument, satellite, verbosity=args.verbose)


        l3a = frp.GriddedFRP(grid_, fs, gp_reader, fp_reader, verbosity=args.verbose)
        l3a.grid(time_s, time_e)

        filename = os.path.join(args.frp_dir, 'qfed3.frp.{0:s}-{1:s}.nc4'.format(instrument, satellite))
        l3a.save(filename=filename, timestamp=time, bootstrap=True, qc=False, fill_value=1e20)


