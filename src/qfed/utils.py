"""
This module contains readers for retrieving data and attributes 
from files using formats such as HDF, NetCDF4, etc. 
"""

import sys
import os
import logging
import abc

from pyhdf import SD
import netCDF4 as nc


if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


class DatasetAccessEngine(ABC):
    """
    Abstract engine to access data.
    """

    def __init__(self, msg="Could not open file '{file}' - ignoring it."):
        self._msg = msg

    def _message_on_open_file_error(self, file):
        logging.warning(self._msg.format(file=file))

    @abc.abstractmethod
    def get_variable(self, file, variable):
        """
        Read variable from file.
        """

    @abc.abstractmethod
    def get_attribute(self, file, attribute):
        """
        Reads attribute from file.
        """


class DatasetAccessEngine_HDF4(DatasetAccessEngine):
    """
    Access HDF4 data.
    """

    def _open(self, file):
        try:
            f = SD.SD(file)
        except SD.HDF4Error:
            self._message_on_open_file_error(file)
            f = None

        return f

    def get_variable(self, file, variable):
        f = self._open(file)
        try:
            if f is None:
                return None
            sds = f.select(variable)
            try:
                data = [] if sds.checkempty() else sds.get()
            finally:
                # always end SDS access for HDF4
                sds.endaccess()
            return data
        finally:
            if f is not None:
                f.end()

    def get_attribute(self, file, attribute):
        f = self._open(file)
        try:
            if f is None:
                return None
            return f.attributes().get(attribute)
        
        finally:
            if f is not None:
                f.end()


class DatasetAccessEngine_NetCDF4(DatasetAccessEngine):
    """
    Access NetCDF4 data.
    """

    def _open(self, file):
        try:
            f = nc.Dataset(file)
        except IOError:
            self._message_on_open_file_error(file)
            f = None

        return f

    def get_variable(self, file, variable):
        f = self._open(file)
        try:
            if f is not None:
                return f.variables[variable][...]
            else:
                return None
        finally:
            if f is not None:
                f.close()
        
    def get_attribute(self, file, attribute):
        f = self._open(file)
        try:
            if f is not None:
                return f.__dict__[attribute]
            else:
                return None
        finally:
            if f is not None:
                f.close()
                
        return attr
