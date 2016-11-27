
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import os

# Constants from the C++ code
VENDOR_DEFAULT = "default"
DEVICE_TYPE_DEFAULT = "All"
DEVICE_NAME_DEFAULT = "default"

# List of attributes
DEVICE_TYPE_ATTRIBUTES = ["device_vendor", "device_type"]
DEVICE_ATTRIBUTES = ["device", "device_core_clock", "device_compute_units"]
KERNEL_ATTRIBUTES = ["precision", "kernel_family"]
ARGUMENT_ATTRIBUTES = ["arg_m", "arg_n", "arg_k", "arg_alpha", "arg_beta"]
ATTRIBUTES = DEVICE_ATTRIBUTES + DEVICE_TYPE_ATTRIBUTES + KERNEL_ATTRIBUTES + ARGUMENT_ATTRIBUTES
GROUP_ATTRIBUTES = DEVICE_TYPE_ATTRIBUTES + KERNEL_ATTRIBUTES + ["kernel"] + ARGUMENT_ATTRIBUTES


def precision_to_string(precision):
    """Translates a precision number (represented as Python string) into a descriptive string"""
    if precision == "16":
        return "Half"
    elif precision == "32":
        return "Single"
    elif precision == "64":
        return "Double"
    elif precision == "3232":
        return "ComplexSingle"
    elif precision == "6464":
        return "ComplexDouble"
    else:
        raise("Unknown precision: " + precision)


def get_cpp_separator():
    """Retrieves a C++ comment separator"""
    return "// ================================================================================================="


def get_cpp_header(family):
    """Retrieves the C++ header"""
    return ("\n" + get_cpp_separator() + """
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the '%s' kernels.
//\n"""
            % family.title() + get_cpp_separator() + \
             "\n\nnamespace clblast {\n" + "namespace database {\n" + get_cpp_separator())


def get_cpp_footer():
    """Retrieves the C++ footer"""
    return "\n} // namespace database\n" + "} // namespace clblast\n"


def get_cpp_precision(family, precision):
    """Retrieves the C++ code for the start of a new precision"""
    precision_string = precision_to_string(precision)
    camelcase_name = family.title().replace("_", "")
    return("\n\nconst Database::DatabaseEntry %s%s = {\n  \"%s\", Precision::k%s, {\n"
           % (camelcase_name, precision_string, camelcase_name, precision_string))


def get_cpp_device_vendor(vendor, device_type):
    """Retrieves the C++ code for the (default) vendor and device type"""
    if vendor == VENDOR_DEFAULT and device_type == DEVICE_TYPE_DEFAULT:
        return "    { // Default\n      kDeviceType%s, \"%s\", {\n" % (device_type, vendor)
    device_type_caps = device_type[0].upper() + device_type[1:]
    return "    { // %s %ss\n      kDeviceType%s, \"%s\", {\n" % (vendor, device_type, device_type_caps, vendor)


def print_cpp_database(database, output_dir):
    """Outputs the database as C++ code"""

    # Iterates over the kernel families
    kernel_families = sorted(set([s["kernel_family"] for s in database["sections"]]))
    for family_name in kernel_families:
        family_database = [s for s in database["sections"] if s["kernel_family"] == family_name]

        # Opens a new file for each kernel family
        full_path = os.path.join(output_dir, family_name + ".hpp")
        with open(full_path, 'w+') as f:
            f.write(get_cpp_header(family_name))

            # Loops over the different precision (e.g. 16, 32, 3232, 64, 6464)
            precisions = sorted(set([s["precision"] for s in database["sections"]]))  # Based on full database
            for precision in precisions:
                precision_database = [s for s in family_database if s["precision"] == precision]
                f.write(get_cpp_precision(family_name, precision))

                # In case there is nothing found at all (e.g. 16-bit): continue as if this was a precision of 32 but
                # with the defaults only
                if len(precision_database) == 0:
                    print("[database] No results found for %s:%s, retrieving defaults from %s:32" %
                          (family_name, precision, family_name))
                    precision_database = [s for s in family_database if s["precision"] == "32"
                                          and s["device_vendor"] == VENDOR_DEFAULT
                                          and s["device_type"] == DEVICE_TYPE_DEFAULT
                                          and s["device"] == DEVICE_NAME_DEFAULT]

                # Loops over device vendors (e.g. AMD)
                device_vendors = sorted(set([s["device_vendor"] for s in precision_database]))
                for vendor in device_vendors:
                    vendor_database = [s for s in precision_database if s["device_vendor"] == vendor]

                    # Loops over device types (e.g. GPU)
                    device_types = sorted(set([s["device_type"] for s in vendor_database]))
                    for device_type in device_types:
                        type_database = [s for s in vendor_database if s["device_type"] == device_type]
                        f.write(get_cpp_device_vendor(vendor, device_type))

                        # Loops over every device of this vendor-type combination
                        devices = sorted(set([s["device"] for s in type_database]))
                        for device_name in devices:
                            device_database = [s for s in type_database if s["device"] == device_name]
                            device_name_quoted = "\"%s\"," % device_name
                            device_name_cpp = "        { %-50s { " % device_name_quoted
                            f.write(device_name_cpp)

                            # Collects the parameters for this entry
                            parameters = []
                            kernels = sorted(set([s["kernel"] for s in device_database]))
                            for kernel in kernels:
                                kernel_database = [s for s in device_database if s["kernel"] == kernel]

                                assert len(kernel_database) == 1
                                results = kernel_database[0]["results"]

                                assert len(results) == 1
                                new_parameters = results[0]["parameters"]
                                for parameter_name in sorted(new_parameters):
                                    parameter_value = new_parameters[parameter_name]
                                    parameters.append("{\"" + parameter_name + "\"," + str(parameter_value) + "}")

                            # Prints the entry
                            f.write(", ".join(parameters))
                            f.write(" } },\n")

                        # Prints the vendor-type combination footer
                        f.write("      }\n    },\n")

                # Prints the precision footer
                f.write("  }\n};\n\n" + get_cpp_separator())

            # Prints the file footer
            f.write(get_cpp_footer())
