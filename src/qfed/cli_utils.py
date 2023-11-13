"""
This module contains functions without side effects that 
are used in the L3A and L3B CLI scripts.  
"""

import os
import yaml
import logging


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
    Generate a path and optionally evaluate a templated path
    at a specific time.
    """
    if isinstance(path, list):
        result = path
    else:
        result = [path]

    if timestamp is not None:
        result = [atom.format(timestamp) for atom in result]

    return os.path.join(*result)


def get_entire_time_interval(args):
    """
    Parses args and returns the start and end
    of the entire time interval that needs to be
    processed.
    """
    time_start = args.date_start
    time_end = args.date_end

    if time_end is None:
        time_end = time_start

    return time_start, time_end


def get_timestamped_time_intervals(time_start, time_end, time_window):
    """
    Returns a list of timestamped time intervals.

    Use with caution. This code is very basic... sub-intervals may
    end up outside of the complete time interval.
    """
    result = []

    t = time_start
    while t <= time_end:
        t_s = t
        t_e = t + time_window
        t_stamp = t + 0.5 * time_window

        result.append((t, t_e, t_stamp))
        t = t + time_window

    return result



def display_description(version, text):
    """
    Display the QFED version and a text.
    """
    logging.info('')
    logging.info(f'QFED {version}')
    logging.info('')
    logging.info(text)
    logging.info('')

