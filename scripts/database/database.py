#!/usr/bin/env python

# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import sys
import os.path
import glob
import argparse

import database.io as io
import database.db as db
import database.clblast as clblast
import database.bests as bests
import database.defaults as defaults

# Server storing a copy of the database
DATABASE_SERVER_URL = "https://raw.githubusercontent.com/CNugteren/CLBlast-database/master/database.json"


def remove_mismatched_arguments(database):
    """Checks for tuning results with mis-matched entries and removes them according to user preferences"""
    kernel_attributes = clblast.DEVICE_TYPE_ATTRIBUTES + clblast.KERNEL_ATTRIBUTES + ["kernel"]

    # For Python 2 and 3 compatibility
    try:
        user_input = raw_input
    except NameError:
        user_input = input
        pass

    # Check for mis-matched entries
    for kernel_group_name, kernel_group in db.group_by(database["sections"], kernel_attributes):
        group_by_arguments = db.group_by(kernel_group, clblast.ARGUMENT_ATTRIBUTES)
        if len(group_by_arguments) != 1:
            print("[database] WARNING: entries for a single kernel with multiple argument values " +
                  str(kernel_group_name))
            print("[database] Either quit or remove all but one of the argument combinations below:")
            for index, (attribute_group_name, mismatching_entries) in enumerate(group_by_arguments):
                print("[database]     %d: %s" % (index, attribute_group_name))
            for attribute_group_name, mismatching_entries in group_by_arguments:
                response = user_input("[database] Remove entries corresponding to %s, [y/n]? " %
                                      str(attribute_group_name))
                if response == "y":
                    for entry in mismatching_entries:
                        database["sections"].remove(entry)
                    print("[database] Removed %d entry/entries" % len(mismatching_entries))

    # Sanity-check: all mis-matched entries should be removed
    for kernel_group_name, kernel_group in db.group_by(database["sections"], kernel_attributes):
        group_by_arguments = db.group_by(kernel_group, clblast.ARGUMENT_ATTRIBUTES)
        if len(group_by_arguments) != 1:
            print("[database] ERROR: entries for a single kernel with multiple argument values " +
                  str(kernel_group_name))
        assert len(group_by_arguments) == 1


def remove_database_entries(database, remove_if_matches_fields):
    assert len(remove_if_matches_fields.keys()) > 0

    def remove_this_entry(section):
        for key in remove_if_matches_fields.keys():
            if section[key] != remove_if_matches_fields[key]:
                return False
        return True

    old_length = len(database["sections"])
    database["sections"] = [x for x in database["sections"] if not remove_this_entry(x)]
    new_length = len(database["sections"])
    print("[database] Removed %d entries from the database" % (old_length - new_length))


def add_tuning_parameter(database, parameter_name, kernel, value):
    num_changes = 0
    for section in database["sections"]:
        if section["kernel"] == kernel:
            for result in section["results"]:
                if parameter_name not in result["parameters"]:
                    result["parameters"][parameter_name] = value
            section["parameter_names"].append(parameter_name)
            num_changes += 1
    print("[database] Made %d addition(s) of %s" % (num_changes, parameter_name))


def main(argv):

    # Parses the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("source_folder", help="The folder with JSON files to parse to add to the database")
    parser.add_argument("clblast_root", help="Root of the CLBlast sources")
    parser.add_argument("-r", "--remove_device", type=str, default=None, help="Removes all entries for a specific device")
    parser.add_argument("--add_tuning_parameter", type=str, default=None, help="Adds this parameter to existing entries")
    parser.add_argument("--add_tuning_parameter_for_kernel", type=str, default=None, help="Adds the above parameter for this kernel")
    parser.add_argument("--add_tuning_parameter_value", type=int, default=0, help="Set this value as the default for the above parameter")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase verbosity of the script")
    cl_args = parser.parse_args(argv)

    # Parses the path arguments
    database_filename = os.path.join(cl_args.clblast_root, "scripts", "database", "database.json")
    database_best_filename = os.path.join(cl_args.clblast_root, "scripts", "database", "database_best.json")
    json_files = os.path.join(cl_args.source_folder, "*.json")
    cpp_database_path = os.path.join(cl_args.clblast_root, "src", "database", "kernels")

    # Checks whether the command-line arguments are valid
    clblast_header = os.path.join(cl_args.clblast_root, "include", "clblast.h")  # Not used but just for validation
    if not os.path.isfile(clblast_header):
        raise RuntimeError("The path '" + cl_args.clblast_root +
                           "' does not point to the root of the CLBlast library")
    if len(glob.glob(json_files)) < 1:
        print("[database] The path '" + cl_args.source_folder + "' does not contain any JSON files")

    # Downloads the database if a local copy is not present
    if not os.path.isfile(database_filename):
        io.download_database(database_filename, DATABASE_SERVER_URL)

    # Loads the database from disk
    database = io.load_database(database_filename)

    # Loops over all JSON files in the supplied folder
    for file_json in glob.glob(json_files):
        sys.stdout.write("[database] Processing '" + file_json + "' ")  # No newline printed

        try:
            # Loads the newly imported data
            imported_data = io.load_tuning_results(file_json)

            # Adds the new data to the database
            old_size = db.length(database)
            database = db.add_section(database, imported_data)
            new_size = db.length(database)
            print("with " + str(new_size - old_size) + " new items")  # Newline printed here

        except ValueError:
            print("--- WARNING: invalid file, skipping")

    # Checks for tuning results with mis-matched entries
    remove_mismatched_arguments(database)

    # Stores the modified database back to disk
    if len(glob.glob(json_files)) >= 1:
        io.save_database(database, database_filename)

    # Removes database entries before continuing
    if cl_args.remove_device is not None:
        print("[database] Removing all results for device '%s'" % cl_args.remove_device)
        remove_database_entries(database, {"clblast_device_name": cl_args.remove_device})
                                           #, "kernel_family": "xgemm"})
        io.save_database(database, database_filename)

    # Adds new tuning parameters to existing database entries
    if cl_args.add_tuning_parameter is not None and\
       cl_args.add_tuning_parameter_for_kernel is not None:
        print("[database] Adding tuning parameter: '%s' for kernel '%s' with default %d" %
              (cl_args.add_tuning_parameter, cl_args.add_tuning_parameter_for_kernel,
               cl_args.add_tuning_parameter_value))
        add_tuning_parameter(database, cl_args.add_tuning_parameter,
                             cl_args.add_tuning_parameter_for_kernel,
                             cl_args.add_tuning_parameter_value)
        io.save_database(database, database_filename)

    # Retrieves the best performing results
    print("[database] Calculating the best results per device/kernel...")
    database_best_results = bests.get_best_results(database)

    # Determines the defaults for other vendors and per vendor
    print("[database] Calculating the default values...")
    database_defaults = defaults.calculate_defaults(database, cl_args.verbose)
    database_best_results["sections"].extend(database_defaults["sections"])

    # Optionally outputs the database to disk
    if cl_args.verbose:
        io.save_database(database_best_results, database_best_filename)

    # Outputs the database as a C++ database
    print("[database] Producing a C++ database in '" + cpp_database_path + "'...")
    clblast.print_cpp_database(database_best_results, cpp_database_path)

    print("[database] All done")


if __name__ == '__main__':
    main(sys.argv[1:])
