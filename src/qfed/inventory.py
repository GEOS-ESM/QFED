'''
Inventory of data sets used in QFED: fire observations, vegetation, etc.
'''

from dataclasses import dataclass

import os
import logging
from datetime import datetime, timedelta
from glob import glob

from qfed.instruments import Instrument, Satellite


@dataclass
class Item:
    '''
    A container holding a date&time, and a group of files 
    containing geolocation data, fire data and vegetation data.
    '''
    time: datetime
    geolocation: str
    fire: str
    vegetation: str


class Finder():
    '''
    Search for files in a time-templated directory structure. 
    '''

    def __init__(self, gp_file, fp_file, vegetation_file, time_interval=60.0):
        self._gp_file = gp_file
        self._fp_file = fp_file
        self._vegetation_file = vegetation_file
        self._time_interval = time_interval

    def find(self, t_start, t_end):
        '''
        Returns a list of Item-s. Each element contains a time label, a geolocation file, 
        a fire product file, and a vegetation file. The list includes files with
        observations in the time window specified by the two arguments.
        '''

        logging.info(f"Starting search for input files containing data between {t_start} and {t_end}.")

        result = []

        t = t_start
        while t < t_end:
            search_path = self._fp_file.format(t)
            
            logging.debug((
                f"Searching for fire product files " 
                f"matching pattern '{os.path.basename(search_path)}' "
                f"in directory '{os.path.dirname(search_path)}'."))

            match = glob(search_path)
            if match:
                fp_file = match[0]

                gp_file = self._gp_file.format(t)

                _item = Item(time=t, geolocation=gp_file, fire=fp_file,
                    vegetation=self._vegetation_file)
                
                logging.debug(f"Found a match: {_item}.")
                
                result.append(_item)
                
                logging.debug(f"Added {_item} to queue for processing.")
            else:
                logging.debug(f"Did not find files matching the pattern '{search_path}'.")

            t = t + timedelta(seconds=self._time_interval)

        if result:
            logging.info((
                f"Search for input files completed. "
                f"{len(result)} fire product files were identified "
                f"and added to the processing queue.\n"))
        else:
            logging.warning((
                f"Search for input files completed. "
                f"However, no fire product files were identified "
                f"that meet the search criteria.\n")) 

        return result

