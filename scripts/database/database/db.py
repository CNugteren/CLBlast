
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import pandas as pd


def get_entries_by_field(database, field, value):
    """Retrieves entries from the database with a specific value for a given field"""
    return database[database[field] == value]


def concatenate_database(database1, database2):
    """Concatenates two databases row-wise and returns the result"""
    return pd.concat([database1, database2])


def remove_duplicates(database):
    """Removes duplicates from a database"""
    return database.drop_duplicates()


def find_and_replace(database, dictionary):
    """Finds and replaces entries in a database based on a dictionary. Example:
    dictionary = { "key_to_edit": { find1: replace1, find2, replace2 } }"""
    return database.replace(dictionary)


def remove_entries_by_key_value(database, key, value):
    """Removes entries in the databased which have a specific value for a given key"""
    return database[database[key] != value]


def remove_entries_by_device(database, device_name):
    """Shorthand for the above, specifically removes entries for a given device"""
    return remove_entries_by_key_value(database, "device", device_name)


def remove_entries_by_kernel_family(database, kernel_family_name):
    """Shorthand for the above, specifically removes entries for a given kernel family"""
    return remove_entries_by_key_value(database, "kernel_family", kernel_family_name)


def update_database(database, condition, field, value):
    """Updates the database by writing a specific value to a given field, given certain conditions"""
    database.loc[condition, field] = value
    return database
