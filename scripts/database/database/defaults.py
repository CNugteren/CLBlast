
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import ast
from collections import defaultdict

import database.bests as bests
import database.clblast as clblast


def set_identifiers(database, group_by_attributes, identifier_name):
    """Sets a group-identifier based on a given set of attributes. Modifies the database but also returns a list of
    unique identifiers."""
    identifiers = []
    for section in database["sections"]:
        identifier = []
        for attribute in group_by_attributes:
            if attribute in section:
                identifier.append(section[attribute])
        section[identifier_name] = ";".join(identifier)
        identifiers.append(section[identifier_name])
    return sorted(set(identifiers))


def remove_identifiers(database, identifier_name):
    """Removes an identifier from all sections in the database"""
    for section in database["sections"]:
        section.pop(identifier_name, None)


def get_groups_by_identifier(database, group_identifiers, identifier_name):
    """Returns a list of (group, group_identifier) tuples based a previously made grouping"""
    groups = []
    for group_identifier in group_identifiers:

        # Get all sections in this group
        group = []
        for section in database["sections"]:
            if section[identifier_name] == group_identifier:
                group.append(section)

        groups.append((group, group_identifier))
    return groups


def add_default_sections(database, grouping, verbose, values_dict, condition, enable_warning):
    default_sections = []

    # Groups the database by a certain grouping
    group_identifiers = set_identifiers(database, grouping, "group_identifier")
    groups = get_groups_by_identifier(database, group_identifiers, "group_identifier")

    # Loops over all groups
    for group, group_identifier in groups:

        # Computes the best parameters
        default_parameters = get_common_best_parameters(group, group_identifier, verbose, enable_warning)
        assert len(group) > 0
        if condition(group[0]):

            # Stores all the section's data
            default_section = {}
            for attribute in group[0].keys():
                if attribute != "results" and attribute != "group_identifier":
                    default_section[attribute] = group[0][attribute]
            default_section["clblast_device_compute_units"] = 0
            default_section["clblast_device_core_clock"] = 0
            for key in values_dict.keys():
                default_section[key] = values_dict[key]
            default_section["results"] = [{"time": 0.0, "parameters": default_parameters}]
            default_sections.append(default_section)
    return default_sections


def calculate_defaults(database, verbose):
    """Sets defaults for devices of the same type/vendor"""
    default_sections = {"sections": []}

    # Groups the database by kernel, vendor and device architecture (e.g. AMD GPU "Fiji")
    architecture_group = clblast.GROUP_ATTRIBUTES + ["clblast_device_architecture"]
    architecture_defaults = add_default_sections(database, architecture_group, verbose,
                                                 {"clblast_device_name": clblast.DEVICE_NAME_DEFAULT},
                                                 lambda entry: True, enable_warning=False)

    # Groups the database by kernel, vendor and device type (e.g. AMD GPU)
    device_defaults = add_default_sections(database, clblast.GROUP_ATTRIBUTES, verbose,
                                           {"clblast_device_name": clblast.DEVICE_NAME_DEFAULT,
                                            "clblast_device_architecture": clblast.DEVICE_ARCHITECTURE_DEFAULT},
                                           lambda entry: entry["clblast_device_architecture"] != "",
                                           enable_warning=True)
    default_sections["sections"].extend(device_defaults)

    # Groups the database by kernel, vendor and device type (e.g. AMD GPU) - but not by arguments!
    # This is to check for mis-matched arguments in the database. Note: this is not a check on the
    # architecture defaults
    attributes = clblast.DEVICE_TYPE_ATTRIBUTES + clblast.KERNEL_ATTRIBUTES + ["kernel"]
    group_identifiers = set_identifiers(default_sections, attributes, "temp_identifier")
    groups = get_groups_by_identifier(default_sections, group_identifiers, "temp_identifier")
    for group, group_identifier in groups:
        if len(group) != 1:
            print("[ERROR] Entries for a single kernel with multiple argument values: " + str(group_identifier))
        assert len(group) == 1
    remove_identifiers(default_sections, "temp_identifier")

    # Adds the architecture defaults only after running the above check
    default_sections["sections"].extend(architecture_defaults)

    # Groups the database by kernel only
    group_identifiers = set_identifiers(database, clblast.KERNEL_ATTRIBUTES + ["kernel"], "group_identifier")
    groups = get_groups_by_identifier(database, group_identifiers, "group_identifier")

    # Loops over all groups
    for group, group_identifier in groups:

        # Computes the best parameters
        default_parameters = get_common_best_parameters(group, group_identifier, verbose,
                                                        enable_warning=True)

        # Stores all the section's data
        assert len(group) > 0
        default_section = {}
        for attribute in group[0].keys():
            if attribute != "results" and attribute != "group_identifier":
                default_section[attribute] = group[0][attribute]
        default_section["clblast_device_name"] = clblast.DEVICE_NAME_DEFAULT
        default_section["clblast_device_architecture"] = clblast.DEVICE_ARCHITECTURE_DEFAULT
        default_section["clblast_device_vendor"] = clblast.VENDOR_DEFAULT
        default_section["clblast_device_type"] = clblast.DEVICE_TYPE_DEFAULT
        default_section["clblast_device_compute_units"] = 0
        default_section["clblast_device_core_clock"] = 0
        default_section["results"] = [{"time": 0.0, "parameters": default_parameters}]
        default_sections["sections"].append(default_section)

    # Database with both types of defaults only
    return default_sections


def get_smallest_best_parameters(group):
    """Sets defaults based on the smallest values of all known entries. The average might be better for performance but
    some parameters might not be supported on other devices."""

    # Counts the number of devices in this group
    assert len(group) > 0

    # Find the smallest values of the parameters
    min_parameters = {}
    for section in group:
        assert len(section["results"]) > 0
        minimum_time = min([result["time"] for result in section["results"]])
        for result in section["results"]:
            if result["time"] == minimum_time:
                for parameter in result["parameters"]:
                    if parameter in min_parameters:
                        min_parameters[parameter] = min(min_parameters[parameter], result["parameters"][parameter])
                    else:
                        min_parameters[parameter] = result["parameters"][parameter]

    return min_parameters


def get_parameter_names(section):
    return [result["parameters"] for result in section["results"]]


def get_common_best_parameters(group, group_identifier, verbose, enable_warning):
    """Sets defaults based on the best values of entries supported by all devices. This might cause a problem in case
    not every device was tuned with the same parameters. In that case it falls back to the above method to retrieve
    the smallest best execution time"""

    # Counts the number of devices in this group
    num_devices = len(group)
    assert num_devices > 0

    # Inserts the relative execution times into the database
    for section in group:
        assert len(section["results"]) > 0
        minimum_time = min([result["time"] for result in section["results"]])
        for result in section["results"]:
            base_line = minimum_time if section["kernel"] != "gemm_kernel_selection" else 1.0
            result["relative_time"] = result["time"] / base_line

    # Determine which parameters are available for all devices
    common_parameters = get_parameter_names(group[0])  # Parameters of the first section
    for i in range(1, num_devices):
        section_parameters = get_parameter_names(group[i])
        common_parameters = [p for p in section_parameters if p in common_parameters]  # Intersection of the parameters

    # Fall back to another method in case there are no shared entries at all across devices
    if len(common_parameters) == 0:
        if verbose:
            print("[database] No common kernels for: " + str(group_identifier) + " across all %d devices " % num_devices)

        # Computes the amount of devices with shared parameters
        parameters_count = defaultdict(int)
        for i in range(0, num_devices):
            for parameters in get_parameter_names(group[i]):
                parameters_count[str(parameters)] += 1
        num_devices_common = max(parameters_count.values())

        # Fall back method in case there are no shared entries at all across devices
        if num_devices_common == 1:
            if enable_warning:
                print("[database] Warning: No common kernels for: " + str(group_identifier) + " at all")
            smallest_best_parameters = get_smallest_best_parameters(group)
            if verbose:
                print("[database] " + str(group_identifier))
            return smallest_best_parameters

        # Checks if perhaps there are many more shared parameters with a bit fewer devices
        num_parameters_common = defaultdict(int)
        for count in parameters_count.values():
            if count != 1:
                num_parameters_common[str(count)] += 1
        if num_parameters_common[str(num_devices_common - 1)] > num_parameters_common[str(num_devices_common)]:
            num_devices_common -= 1
        if verbose:
            print("[database] Found %d common kernels for: " % num_parameters_common[str(num_devices_common)] +
                  str(group_identifier) + " across %d out of %d devices " % (num_devices_common, num_devices))

        # Populates the common parameters
        for parameters_string in parameters_count.keys():
            count = parameters_count[parameters_string]
            if count == num_devices_common:
                parameters = ast.literal_eval(parameters_string)
                common_parameters.append(parameters)

    # Removes entries with parameters which are not common
    common_results = []
    for section in group:
        for result in section["results"]:
            if result["parameters"] in common_parameters:
                common_results.append(result)

    # Retrieves the entries with the highest relative performance
    relative_best_parameters = bests.get_relative_bests(group_identifier, common_results, common_parameters, verbose)
    return relative_best_parameters
