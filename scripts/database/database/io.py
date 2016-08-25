
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import re
import json

try:
    from urllib.request import urlopen  # Python 3
except ImportError:
    from urllib2 import urlopen  # Python 2

import pandas as pd

import clblast


def download_database(filename, database_url):
    """Downloads a database and saves it to disk"""
    print("[database] Downloading database from '" + database_url + "'...")
    database = urlopen(database_url)
    with open(filename, 'wb') as f:
        f.write(database.read())


def load_database(filename):
    """Loads a database from disk"""
    print("[database] Loading database from '" + filename + "'")
    return pd.read_pickle(filename)


def save_database(database, filename):
    """Saves a database to disk"""
    print("[database] Saving database to '" + filename + "'")
    database.to_pickle(filename)


def load_json_to_pandas(filename):
    """Loads JSON data from file and converts it to a pandas database"""
    with open(filename) as f:
        json_data = json.load(f)

    # Gathers all results and stores them in a new database
    json_database = pd.DataFrame(json_data)
    new_database = pd.io.json.json_normalize(json_database["results"])

    # Sets the common attributes to each entry in the results
    for attribute in clblast.ATTRIBUTES:
        if attribute == "kernel_family":
            new_database[attribute] = re.sub(r'_\d+', '', json_data[attribute])
        elif attribute in json_data:
            new_database[attribute] = json_data[attribute]
        else:
            new_database[attribute] = 0  # For example a parameters that was not used by this kernel
    return new_database
