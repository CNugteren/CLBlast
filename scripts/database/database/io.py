
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


def download_database(filename, database_url):
    """Downloads a database and saves it to disk"""
    print("[database] Downloading database from '" + database_url + "'...")
    database = urlopen(database_url)
    with open(filename, "wb") as f:
        f.write(database.read())


def load_database(filename):
    """Loads a database from disk"""
    print("[database] Loading database from '" + filename + "'")
    with open(filename) as f:
        return json.load(f)


def save_database(database, filename):
    """Saves a database to disk"""
    print("[database] Saving database to '" + filename + "'")
    with open(filename, "w") as f:
        json.dump(database, f, sort_keys=True, indent=4)


def load_tuning_results(filename):
    """Loads JSON data from file and pre-processes it"""
    with open(filename) as f:
        json_data = json.load(f)

    # Removes the numbering following the kernel family name
    json_data["kernel_family"] = re.sub(r'_\d+', '', json_data["kernel_family"])

    # Adds the kernel name to the section instead of to the individual results
    assert len(json_data["results"]) > 0
    json_data["kernel"] = json_data["results"][0]["kernel"]
    for result in json_data["results"]:
        assert json_data["kernel"] == result["kernel"]
        result.pop("kernel", None)

    # Removes the 'PRECISION' parameter from the individual results: it is redundant
    for result in json_data["results"]:
        assert json_data["precision"] == str(result["parameters"]["PRECISION"])
        result["parameters"].pop("PRECISION", None)

    # Fixes the scalar argument values
    for value, replacement in zip(["2.00", "2.00+0.50i"], ["2.000000", "2+0.5i"]):
        for field in ["arg_alpha", "arg_beta"]:
            if field in json_data.keys() and json_data[field] == value:
                json_data[field] = replacement

    # All done
    return json_data
