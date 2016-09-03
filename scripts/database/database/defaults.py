
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import pandas as pd

import clblast
import bests


def set_default_device(database_entry):
    """Sets the device name and parameters to some default values"""
    database_entry["device"] = clblast.DEVICE_NAME_DEFAULT
    database_entry["device_compute_units"] = 0
    database_entry["device_core_clock"] = 0
    return database_entry


def set_default_time(database_entry):
    """Sets the execution time to some default value"""
    database_entry["time"] = 0.0
    return database_entry


def calculate_defaults(database, verbose, calculate_common_best=True):
    """Sets defaults for devices of the same type/vendor. An option determines how to compute the defaults."""
    database_defaults = pd.DataFrame()

    # Defaults per combination of device vendors and device types (e.g. AMD GPU)
    database_type_vendor = database.groupby(clblast.DEVICE_TYPE_ATTRIBUTES + clblast.KERNEL_ATTRIBUTES + ["kernel"] +
                                            clblast.ARGUMENT_ATTRIBUTES)
    for group_name, database_group in database_type_vendor:
        if calculate_common_best:
            default_values = get_common_best(database_group, group_name, verbose)
        else:
            default_values = get_smallest_best(database_group)
        default_values = set_default_device(default_values)
        default_values = set_default_time(default_values)
        database_defaults = database_defaults.append(default_values, ignore_index=True)

    # Checks for mis-matched arguments
    groups = database_defaults.groupby(clblast.DEVICE_TYPE_ATTRIBUTES + clblast.KERNEL_ATTRIBUTES + ["kernel"])
    for group_name, database_group in groups:
        if len(database_group) != 1:
            print("[WARNING] Entries for a single kernel with multiple argument values: " + str(group_name))

    # Defaults over all device types and vendors
    groups = database.groupby(clblast.KERNEL_ATTRIBUTES + ["kernel"] + clblast.ARGUMENT_ATTRIBUTES)
    for group_name, database_group in groups:
        if calculate_common_best:
            default_values = get_common_best(database_group, group_name, verbose)
        else:
            default_values = get_smallest_best(database_group)
        default_values["device_vendor"] = clblast.VENDOR_DEFAULT
        default_values["device_type"] = clblast.DEVICE_TYPE_DEFAULT
        default_values = set_default_device(default_values)
        default_values = set_default_time(default_values)
        database_defaults = database_defaults.append(default_values, ignore_index=True)

    # Database with both types of defaults only
    return database_defaults


def get_smallest_best(database):
    """Sets defaults based on the smallest values of all known entries. The average might be better for performance but
    some parameters might not be supported on other devices."""
    database_best_results = bests.get_best_results(database)
    return database_best_results.min(axis=0)


def get_common_best(database, group_name, verbose):
    """Sets defaults based on the best values of entries supported by all devices. This might cause a problem in case
    not every device was tuned with the same parameters. In that case it falls back to the above method to retrieve
    the smallest best execution time"""

    # Counts the number of devices in this group
    num_devices = len(database.groupby(clblast.DEVICE_ATTRIBUTES))

    # Removes columns without any values
    database = database.dropna(axis=1, how='all')
    database = database.reset_index()

    # In case multiple runs for the exact same configuration where made: take just the best performing one into account
    other_column_names = list(database.columns.values)
    other_column_names.remove("time")
    database_by_time = database.groupby(other_column_names)
    if len(database_by_time) != len(database):
        if verbose:
            print("[database] " + str(group_name) + " keeping only entries with the lowest execution time")
        database = database_by_time.apply(lambda x: x[x["time"] == x["time"].min()])

    # Inserts the relative execution times into the database
    def relative_performance(x):
        x["relative_performance"] = x["time"].min() / x["time"]
        return x
    database = database.groupby(clblast.ATTRIBUTES + ["kernel"]).apply(relative_performance)

    # Retrieves the parameter names for this kernel
    all_column_names = list(database.columns.values)
    parameter_column_names = [c for c in all_column_names if "parameters." in c]

    # Removes entries which are not available for all devices
    database_by_parameters = database.groupby(parameter_column_names)
    database_common = database_by_parameters.filter(lambda x: len(x) == num_devices)

    # Fall back to another method in case there are no shared entries at all across devices
    if len(database_common) == 0:
        if verbose:
            print("[database] No common kernels for: " + str(group_name) + " with devices: %d " % num_devices)
        return get_smallest_best(database)

    # Retrieves the entries with the highest relative performance
    return bests.get_relative_bests(database_common, parameter_column_names, group_name, verbose)
