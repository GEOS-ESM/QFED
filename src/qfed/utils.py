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

    def __init__(self, verbosity=0):
        self.verbosity = verbosity

    def message_on_file_error(self, file):
        logging.warning(f"Cannot open the fire product file <{file}> - excluding it.")

    @abc.abstractmethod
    def get_handle(self, file):
        return

    @abc.abstractmethod
    def get_variable(self, file, variable):
        return

    @abc.abstractmethod
    def get_attribute(self, file, attribute):
        return


class DatasetAccessEngine_HDF4(DatasetAccessEngine):

    def get_handle(self, file):
        try:
            f = SD.SD(file)
        except SD.HDF4Error:
            self.message_on_file_error(file)
            f = None

        return f

    def get_variable(self, file, variable):
        f = self.get_handle(file)
        
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
        f = self.get_handle(file)
        
        if f is not None:
            attr = f.attributes()[attribute]
        else:
            attr = None

        return attr


class DatasetAccessEngine_NetCDF4(DatasetAccessEngine):

    def get_handle(self, file):
        try:
           f = nc.Dataset(file)
        except IOError:
           self.message_on_file_error(file)
           f = None

        return f

    def get_variable(self, file, variable):
        f = self.get_handle(file)

        if f is not None:
            data = f.variables[variable][...]
        else:
            data = None

        return data

    def get_attribute(self, file, attribute):
        f = self.get_handle(file)

        if f is not None:
            attr = f.__dict__[attribute]
        else:
            attr = None
        
        return attr

