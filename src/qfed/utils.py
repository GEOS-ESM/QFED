'''
This module contains:
    - Engines that provide uniform interface for acessing data
      saved in commonly used formats such as HDF, NetCDF4, etc. 
    - Functions that are used in the L3A and L3B scripts/CLI.
'''

import sys
import os
import yaml
import logging
import abc

from pyhdf import SD
import netCDF4 as nc


if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta('ABC', (), {})


def read_config(config):
    """
    Parses the QFED config file into a dictionary.
    """
    with open(config) as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as err:
            data = None
            logging.critical(err)

    return data


def get_path(path, timestamp=None):
    """
    Generate an optionally timestamped path.
    """
    if isinstance(path, list):
        result = path
    else:
        result = [path]

    if timestamp is not None:
        result = [atom.format(timestamp) for atom in result]

    return os.path.join(*result)


def display_description(version, text):
    """
    Displays the QFED version and a text.
    """
    logging.info('')
    logging.info(f'QFED {version}')
    logging.info('')
    logging.info(text)
    logging.info('')


class DatasetAccessEngine(ABC):
    """
    Abstract engine to access data.
    """

    def __init__(self, msg="Could not open file <{file}> - ignoring it."):
        self._msg = msg

    def _message_on_file_error(self, file):
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
    """
    Access NetCDF4 data.
    """

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
