
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
        database = json.load(f)
    return decompress_database(database)


def save_database(database, filename):
    """Saves a database to disk"""
    compressed_db = compress_database(database)
    print("[database] Saving database to '" + filename + "'")
    with open(filename, "w") as f:
        json.dump(compressed_db, f, sort_keys=True, indent=2, separators=(',', ': '))


def compress_database(database):
    """Moves certain common fields up in the hierarchy, transforms dicts into lists"""
    new_sections = []
    for section in database["sections"]:
        new_section = {}
        for field in section:
            if field == "results":
                parameter_names = [sorted(result["parameters"].keys()) for result in section["results"]]
                assert len(list(set([" ".join(p) for p in parameter_names]))) == 1
                new_section["parameter_names"] = parameter_names[0]  # they are all the same
                new_results = [[",".join([str(result["parameters"][p]) for p in new_section["parameter_names"]]),
                                result["time"]]
                               for result in section["results"]]
                new_section[field] = new_results
            elif field != "parameter_names":
                new_section[field] = section[field]
        new_sections.append(new_section)
    return {"sections": new_sections}


def decompress_database(database):
    """Undo the above compression"""
    for section in database["sections"]:
        new_results = []
        for result in section["results"]:
            parameters = {}
            for name, value in zip(section["parameter_names"], result[0].split(",")):
                parameters[name] = int(value)
            new_result = {
                "parameters": parameters,
                "time": result[1]
            }
            new_results.append(new_result)
        section["results"] = new_results
    return database


def load_tuning_results(filename):
    """Loads JSON data from file and pre-processes it"""
    with open(filename) as f:
        json_data = json.load(f)

    # Removes the numbering following the kernel family name
    json_data["kernel_family"] = re.sub(r'_\d+', '', json_data["kernel_family"])

    # Removes unnecessary data
    if json_data["best_kernel"]:
        del json_data["best_kernel"]
    if json_data["best_time"]:
        del json_data["best_time"]
    if json_data["best_parameters"]:
        del json_data["best_parameters"]

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
