
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import pandas as pd
import clblast


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


def calculate_defaults(df):
    """# Sets defaults for devices of the same type/vendor based on the smallest values of all known entries. The average
    might be better for performance but some parameters might not be supported on other devices."""
    database_defaults = pd.DataFrame()

    # Defaults per combination of device vendors and device types (e.g. AMD GPU)
    database_type_vendor = df.groupby(clblast.DEVICE_TYPE_ATTRIBUTES + clblast.KERNEL_ATTRIBUTES + ["kernel"] +
                                      clblast.ARGUMENT_ATTRIBUTES)
    for group_name, database_group in database_type_vendor:
        default_values = database_group.min(axis=0)
        default_values = set_default_device(default_values)
        default_values = set_default_time(default_values)
        database_defaults = database_defaults.append(default_values, ignore_index=True)

    # Checks for mis-matched arguments
    groups = database_defaults.groupby(clblast.DEVICE_TYPE_ATTRIBUTES + clblast.KERNEL_ATTRIBUTES + ["kernel"])
    for group_name, database_group in groups:
        if len(database_group) != 1:
            description = database_group["kernel"].min() + " " + database_group["device_vendor"].min()
            print("[WARNING] Entries for a single kernel with multiple argument values: " + description)

    # Defaults over all device types and vendors
    groups = df.groupby(clblast.KERNEL_ATTRIBUTES + ["kernel"] + clblast.ARGUMENT_ATTRIBUTES)
    for group_name, database_group in groups:
        default_values = database_group.min(axis=0)
        default_values["device_vendor"] = clblast.VENDOR_DEFAULT
        default_values["device_type"] = clblast.DEVICE_TYPE_DEFAULT
        default_values = set_default_device(default_values)
        default_values = set_default_time(default_values)
        database_defaults = database_defaults.append(default_values, ignore_index=True)

    # Database with both types of defaults only
    return database_defaults
