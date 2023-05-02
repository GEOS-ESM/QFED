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

    def __init__(self, gp_dir, gp_filename, fp_dir, fp_filename, time_interval=60.0, verbosity=0):
        self.gp_dir = gp_dir
        self.gp_filename = gp_filename
        self.fp_dir = fp_dir
        self.fp_filename = fp_filename
        self.time_interval = time_interval
        self.verbosity = verbosity

    def find(self, datetime_start, datetime_end):
        '''
        Returns a list of tuples containing full paths to paired
        geolocation and fire product files, and the corresponding
        time of observations. The list includes all files with
        observations in the specified time window.
        '''
        result = []

        t = datetime_start
        while t < datetime_end:
            path = os.path.join(
                self.fp_dir.format(t),
                self.fp_filename.format(t))

            if self.verbosity > 1:
                print(path)

            match = glob(path)
            if match:
                fp_path = match[0]

                gp_path = os.path.join(
                    self.gp_dir.format(t),
                    self.gp_filename.format(t))

                result.append((gp_path, fp_path, t))

                if self.verbosity > 1:
                    print('[i]    found: ', t, gp_path, fp_path)

            t = t + timedelta(seconds=self.time_interval)

        return result

