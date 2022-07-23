'''
Search for files in a time-templated directory structure.
'''

import os
from datetime import timedelta
from glob import glob


class PathFinder():
    '''
    Search for files in a time-templated directory structure. 
    '''

    def __init__(self, path_geolocation_product, path_fire_product, template_fire_product, verbosity=0):
        self._path_geolocation_product = path_geolocation_product
        self._path_fire_product = path_fire_product
        self._template_fire_product = template_fire_product
        self.verbosity = verbosity

    def find(self, datetime_start, datetime_end, step=60.0):
        '''
        Returns list of tuples containing geolocation directory
        and the corresponding fire product directory and file name.
        The list includes all time stamped directories and file names
        found to match the specified time window.
        '''
        result = []

        t = datetime_start
        while t < datetime_end:
            fp_path = os.path.join(self._path_fire_product.format(t),
                                   self._template_fire_product.format(t))

            gp_dir = self._path_geolocation_product.format(t)

            if self.verbosity > 1:
                print(gp_dir, fp_path)

            match = glob(fp_path)
            if match:
                fp_dir  = os.path.dirname(match[0])
                fp_file = os.path.basename(match[0])
                result.append((gp_dir, fp_dir, fp_file))

                if self.verbosity > 1:
                    print('[i]    found: ', gp_dir, fp_dir, fp_file)

            t = t + timedelta(seconds=step)

        return result


