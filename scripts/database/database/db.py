
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import itertools
from operator import itemgetter


def length(database):
    """Computes the total number of tuning entries"""
    num_tuning_entries = 0
    for section in database["sections"]:
        num_tuning_entries += len(section["results"])
    return num_tuning_entries


def add_section(database, new_section):
    """Adds a new section to the database"""
    for old_section in database["sections"]:

        # Verify whether the sections match
        equal = True
        for attribute in new_section.keys():
            if attribute != "results":
                if attribute not in old_section or new_section[attribute] != old_section[attribute]:
                    equal = False
                    break

        # They match: append the new section's results to the corresponding entry in the database and return
        if equal:
            old_section["results"] = combine_results(old_section["results"], new_section["results"])
            return database

    # No match found: append the whole new section to the database
    database["sections"].append(new_section)
    return database


def combine_results(old_results, new_results):
    """Adds new results to the results JSON list"""
    for new_result in new_results:
        old_results = combine_result(old_results, new_result)
    return old_results


def combine_result(old_results, new_result):
    """Adds a new result to the results JSON list; filters for duplicate entries and saves the best performing one"""

    # Loops over all existing results to test for already existing entries with these parameters
    for old_result in old_results:

        # Verify whether the results match
        equal = new_result["parameters"] == old_result["parameters"]

        # They match: keep only the one with the minimum execution time
        if equal:
            old_result["time"] = min(old_result["time"], new_result["time"])
            return old_results

    # No match found: append a new result
    old_results.append(new_result)
    return old_results


def group_by(database, attributes):
    """Returns an list with the name of the group and the corresponding entries in the database"""
    assert len(database) > 0
    attributes = [a for a in attributes if a in database[0]]
    database.sort(key=itemgetter(*attributes))
    result = []
    for key, data in itertools.groupby(database, key=itemgetter(*attributes)):
        result.append((key, list(data)))
    return result
