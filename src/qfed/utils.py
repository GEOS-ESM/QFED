'''
Engines that provide uniform interface for acessing data
saved in commonly used formats such as HDF, NetCDF4, etc. 
'''


import sys
import os
import logging
import re
import abc
from enum import Enum


import numpy as np
from pyhdf import SD
import netCDF4 as nc


if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class DatasetAccessEngine(ABC):

    def __init__(self, msg="Could not open file <{file}> - ignoring it."):
        self._msg = msg

    def _message_on_file_error(self, file):
        logging.warning(self._msg.format(file=file))

    @abc.abstractmethod
    def get_variable(self, file, variable):
        '''
        Read variable from file.
        '''

    @abc.abstractmethod
    def get_attribute(self, file, attribute):
        '''
        Reads attribute from file.
        '''


class DatasetAccessEngine_HDF4(DatasetAccessEngine):

    def _open(self, file):
        try:
            f = SD.SD(file)
        except SD.HDF4Error:
            self._message_on_file_error(file)
            f = None

        return f

    def get_variable(self, file, variable):
        f = self._open(file)
        
        if f is not None:
            sds = f.select(variable)
            if sds.checkempty():
                data = []
            else:    
                data = f.select(variable).get()
        else: 
            data = None

        return data

    def get_attribute(self, file, attribute):
        f = self._open(file)
        
        if f is not None:
            attr = f.attributes()[attribute]
        else:
            attr = None

        return attr


class DatasetAccessEngine_NetCDF4(DatasetAccessEngine):

    def _open(self, file):
        try:
           f = nc.Dataset(file)
        except IOError:
           self._message_on_file_error(file)
           f = None

        return f

    def get_variable(self, file, variable):
        f = self._open(file)

        if f is not None:
            data = f.variables[variable][...]
        else:
            data = None

        return data

    def get_attribute(self, file, attribute):
        f = self._open(file)

        if f is not None:
            attr = f.__dict__[attribute]
        else:
            attr = None
        
        return attr

