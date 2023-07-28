#!/usr/bin/env python3


"""
  A Python script to create QFED Level 3a files.
"""

import os
import logging
from datetime import datetime, timedelta
import yaml
import argparse

from qfed import version
from qfed import grid
from qfed import geolocation_products
from qfed import classification_products
from qfed import fire_products
from qfed.inventory import Finder
from qfed.instruments import Instrument, Satellite
from qfed.frp import GriddedFRP


def parse_arguments(default):
    '''
    Parse command line options
    '''

    parser = argparse.ArgumentParser(
        prog='QFED', 
        description='Creates QFED Level 3a files')

    parser.add_argument('-c', '--config', 
        dest='config', default=default['config'],
        help='config file (default={0:s})'.format(default['config']))

    parser.add_argument('-o', '--output', 
        dest='output_dir', default=default['output_dir'],
        help='directory for output files (default={0:s})'.format(default['output_dir']))

    parser.add_argument('-p', '--products', 
        dest='products', default=default['products'],
        help='list of fire products (default={0:s})'.format(default['products']))
    
    parser.add_argument('-r', '--resolution', 
        dest='resolution', default=default['resolution'],
        help='horizontal resolution (default={0:s})'.format(default['resolution']))

    args = parser.parse_args()

    return args


def read_config(config):
    '''
    Parses the QFED config into a dict. 
    '''
    with open(config) as file:
        try:
            data = yaml.safe_load(file)   
        except yaml.YAMLError as exc:
            data = None
            logging.critical(exc)

    return data


def display_banner():
    logging.info('')
    logging.info('QFED Level 3A Processing')
    logging.info('------------------------')
    logging.info('')



if __name__ == "__main__":

    defaults = dict(
        products   = 'modis/aqua,modis/terra,viirs/npp,viirs/jpss-1',
        resolution = 'e',
        output_dir = './',
        config     = 'config.yaml'
    )

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        #filename='qfed_l3a.log',
    )


    time = datetime(2021, 2, 1, 12)
    time_window = timedelta(hours=24)

    time_s = time - 0.5*time_window
    time_e = time + 0.5*time_window

    args = parse_arguments(defaults)
    config = read_config(args.config)

    display_banner()

    products = args.products.split(',')

    for p in products:
        instrument, satellite = p.split('/')
        platform = Instrument(instrument), Satellite(satellite)

        output_grid = grid.Grid(args.resolution)

        gp_dir, gp_template = config[p]['geolocation']
        fp_dir, fp_template = config[p]['fires']
        vg_dir = config['igbp']

        gp_file = os.path.join(gp_dir, '{0:%Y}', '{0:%j}', gp_template)
        fp_file = os.path.join(fp_dir, '{0:%Y}', '{0:%j}', fp_template)

        finder = Finder(gp_file, fp_file, vg_dir)

        gp_reader = geolocation_products.create(*platform)
        fp_reader = fire_products.create(*platform)
        cp_reader = classification_products.create(*platform)

        l3a = GriddedFRP(output_grid, finder, gp_reader, fp_reader, cp_reader)

        l3a.grid(time_s, time_e)

        output_template = config[p]['frp']
        output_file = os.path.join(args.output_dir, output_template.format(time))
        l3a.save(filename=output_file, timestamp=time, bootstrap=True, qc=False, fill_value=1e20)

