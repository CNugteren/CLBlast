
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
            % family.title() + get_cpp_separator() + "\n\nnamespace clblast {\n" + get_cpp_separator())


def get_cpp_footer():
    """Retrieves the C++ footer"""
    return "\n} // namespace clblast\n"


def get_cpp_precision(family, precision):
    """Retrieves the C++ code for the start of a new precision"""
    precision_string = precision_to_string(precision)
    return("\n\nconst Database::DatabaseEntry Database::%s%s = {\n  \"%s\", Precision::k%s, {\n"
           % (family.title(), precision_string, family.title(), precision_string))


def get_cpp_device_vendor(vendor, device_type):
    """Retrieves the C++ code for the (default) vendor and device type"""
    if vendor == VENDOR_DEFAULT and device_type == DEVICE_TYPE_DEFAULT:
        return "    { // Default\n      kDeviceType%s, \"%s\", {\n" % (device_type, vendor)
    device_type_caps = device_type[0].upper() + device_type[1:]
    return "    { // %s %ss\n      kDeviceType%s, \"%s\", {\n" % (vendor, device_type, device_type_caps, vendor)


def print_cpp_database(database, output_dir):
    """Outputs the database as C++ code"""

    # Iterates over the kernel families
    for family_name, family_database in database.groupby(["kernel_family"]):
        family_database = family_database.dropna(axis=1, how='all')

        # Opens a new file for each kernel family
        full_path = os.path.join(output_dir, family_name+'.hpp')
        with open(full_path, 'w+') as f:
            f.write(get_cpp_header(family_name))

            # Loops over the different precision (e.g. 16, 32, 3232, 64, 6464)
            for precision, precision_database in family_database.groupby(["precision"]):
                f.write(get_cpp_precision(family_name, precision))

                # Loops over a combination of device vendors and device types (e.g. AMD GPU)
                for vendor, vendor_database in precision_database.groupby(["device_vendor"]):
                    for device_type, device_type_database in vendor_database.groupby(["device_type"]):
                        f.write(get_cpp_device_vendor(vendor, device_type))

                        # Loops over every device of this vendor-type combination
                        for device_name, device_database in device_type_database.groupby(["device"]):
                            device_name_quoted = "\"%s\"," % device_name
                            device_name_cpp = "        { %-50s { " % device_name_quoted
                            f.write(device_name_cpp)

                            # Collects the parameters for this entry
                            parameters = []
                            for kernel, kernel_database in device_database.groupby(["kernel"]):
                                kernel_database = kernel_database.dropna(axis=1)

                                # Only consider the actual parameters, not the precision
                                def is_parameter(column):
                                    return column.startswith('parameters.') and column != "parameters.PRECISION"
                                column_names = [col for col in list(kernel_database) if is_parameter(col)]

                                for p in column_names:
                                    parameter_name = p.replace("parameters.", "")
                                    parameter_value = int(kernel_database[p].iloc[0])
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
