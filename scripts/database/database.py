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

# OpenCL vendor names and their short name
VENDOR_TRANSLATION_TABLE = {
  "GenuineIntel": "Intel",
  "Intel(R) Corporation": "Intel",
  "Advanced Micro Devices, Inc.": "AMD",
  "NVIDIA Corporation": "NVIDIA",
}


def main(argv):

    # Parses the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("source_folder", help="The folder with JSON files to parse to add to the database")
    parser.add_argument("clblast_root", help="Root of the CLBlast sources")
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
        raise RuntimeError("The path '" + cl_args.clblast_root + "' does not point to the root of the CLBlast library")
    if len(glob.glob(json_files)) < 1:
        print("[database] The path '" + cl_args.source_folder + "' does not contain any JSON files")

    # Downloads the database if a local copy is not present
    if not os.path.isfile(database_filename):
        io.download_database(database_filename, DATABASE_SERVER_URL)

    # Loads the database from disk
    database = io.load_database(database_filename)

    # Loops over all JSON files in the supplied folder
    for file_json in glob.glob(json_files):

        # Loads the newly imported data
        sys.stdout.write("[database] Processing '" + file_json + "' ")  # No newline printed
        imported_data = io.load_tuning_results(file_json)

        # Fixes the problem that some vendors use multiple different names
        for target in VENDOR_TRANSLATION_TABLE:
            if imported_data["device_vendor"] == target:
                imported_data["device_vendor"] = VENDOR_TRANSLATION_TABLE[target]

        # Adds the new data to the database
        old_size = db.length(database)
        database = db.add_section(database, imported_data)
        new_size = db.length(database)
        print("with " + str(new_size - old_size) + " new items")  # Newline printed here

    # Stores the modified database back to disk
    if len(glob.glob(json_files)) >= 1:
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
