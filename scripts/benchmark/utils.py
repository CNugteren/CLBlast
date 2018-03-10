# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import csv
import subprocess


def k(value):
    return value * 1024


def m(value):
    return value * 1024 * 1024


def float_to_kilo_mega(value):
    if value % 1024 or value <= 1024:
        return "%.0f" % value
    elif value % (1024 * 1024) or value <= (1024 * 1024):
        return "%.0fK" % (value / 1024.0)
    else:
        return "%.0fM" % (value / (1024.0 * 1024.0))


def powers_of_2(start, stop):
    while start <= stop:
        yield start
        start *= 2


def precision_to_letter(precision):
    if precision == 16:
        return "H"
    elif precision == 32:
        return "S"
    elif precision == 64:
        return "D"
    elif precision == 3232:
        return "C"
    elif precision == 6464:
        return "Z"
    else:
        return "X"


def run_binary(command, arguments):
    full_command = command + " " + " ".join(arguments)
    print("[benchmark] Calling binary: %s" % str(full_command))
    try:
        return subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE).stdout.read()
    except OSError as e:
        print("[benchmark] Error while running the binary, got exception: %s" + str(e))
        return False


def parse_results(csv_data):
    csv_data = csv_data.split("\n")
    results = csv.DictReader(csv_data, delimiter=";", skipinitialspace=True)
    results = [r for r in results]
    for result in results:
        for key in result:
            if "i" in result[key]:
                continue
            else:
                result[key] = float(result[key]) if "." in result[key] else int(result[key])
    return results
