#!/usr/bin/env python

# ==================================================================================================
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
# project loosely follows the Google C++ styleguide and uses a max-width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# ==================================================================================================

# System modules
import sys
import os.path
import glob
import re
import json
try:
	from urllib.request import urlopen # Python 3
except ImportError:
	from urllib2 import urlopen # Python 2

# Additional modules
import pandas as pd

# Server storing a copy of the database
DATABASE_SERVER_URL = "http://www.cedricnugteren.nl/tuning/clblast.db"

# Constants
VENDOR_DEFAULT = "default"
DEVICETYPE_DEFAULT = "All"
DEVICENAME_DEFAULT = "default"

# Attributes
DEVICETYPE_ATTRIBUTES = ["device_vendor", "device_type"]
DEVICE_ATTRIBUTES = ["device", "device_core_clock", "device_compute_units"]
KERNEL_ATTRIBUTES = ["precision", "kernel_family"]
ARGUMENT_ATTRIBUTES = ["arg_m", "arg_n", "arg_k", "arg_alpha", "arg_beta"]
ATTRIBUTES = DEVICE_ATTRIBUTES + DEVICETYPE_ATTRIBUTES + KERNEL_ATTRIBUTES + ARGUMENT_ATTRIBUTES

# OpenCL vendor names and their short name
VENDOR_NAMES = { "device_vendor": {
  "GenuineIntel": "Intel",
  "Intel(R) Corporation": "Intel",
  "Advanced Micro Devices, Inc.": "AMD",
  "NVIDIA Corporation": "NVIDIA",
}}

# Pandas options
pd.set_option('display.width', 1000)

# ==================================================================================================
# Database operations
# ==================================================================================================

# Downloads the database and save it to disk
def DownloadDatabase(filename):
	print("## Downloading database from '"+DATABASE_SERVER_URL+"'...")
	df = urlopen(DATABASE_SERVER_URL)
	output = open(file_db,'wb')
	output.write(df.read())
	output.close()

# Loads the database from disk
def LoadDatabase(filename):
	return pd.read_pickle(filename)

# Saves the database to disk
def SaveDatabase(df, filename):
	df.to_pickle(filename)

# Loads JSON data from file
def ImportDataFromFile(filename):
	with open(filename) as f:
		data = json.load(f)
	json_data = pd.DataFrame(data)
	df = pd.io.json.json_normalize(json_data["results"])
	for attribute in ATTRIBUTES:
		if attribute == "kernel_family":
			df[attribute] = re.sub(r'_\d+', '', data[attribute])
		elif attribute in data:
			df[attribute] = data[attribute]
		else:
			df[attribute] = 0
	return df

# Returns the row-wise concatenation of two dataframes
def ConcatenateData(df1, df2):
	return pd.concat([df1, df2])

# Removes duplicates from a dataframe
def RemoveDuplicates(df):
	return df.drop_duplicates()

# database = database[(database["device"] != "AMD Radeon R9 M370X Compute Engine") | (database["kernel_family"] != "xgemm") | (database["precision"] != "32")]
def RemoveEntriesByDevice(df, devicename):
	return df[df["device"] != devicename]

def RemoveEntriesByKernelFamily(df, familyname):
	return df[df["kernel_family"] != familyname]

def GetEntriesByField(df, field, value):
	return df[df[field] == value]

# Example usage:
# df = UpdateDatabase(df, (df["kernel_family"] == "xdot") & (df["arg_n"] == "67108864"), "arg_n", "2097152")
def UpdateDatabase(df, condition, field, value):
	df.loc[condition, field] = value
	return df

# Fixes the problem that some vendors use multiple different names
def SanitizeVendorNames(df):
	df = df.replace(VENDOR_NAMES)
	return df

# Retrieves the results with the lowest execution times
def GetBestResults(df):
	dfbest = pd.DataFrame()
	grouped = df.groupby(ATTRIBUTES+["kernel"])
	for name, dfgroup in grouped:
		besttime = dfgroup["time"].min()
		bestcase = dfgroup[dfgroup["time"] == besttime].iloc[0]
		dfbest = dfbest.append(bestcase, ignore_index=True)
	return dfbest

# Sets defaults for devices of the same type/vendor based on the smallest values of all know
# entries. The average might be better for performance but some parameters might not be supported
# on other devices.
def CalculateDefaults(df):
	dfdefault = pd.DataFrame()

	# Defaults per type/vendor
	groups = df.groupby(DEVICETYPE_ATTRIBUTES+KERNEL_ATTRIBUTES+ARGUMENT_ATTRIBUTES+["kernel"])
	for name, dfgroup in groups:
		default_values = dfgroup.min(axis=0)
		default_values["device"] = DEVICENAME_DEFAULT
		default_values["device_compute_units"] = 0
		default_values["device_core_clock"] = 0
		default_values["time"] = 0.0
		dfdefault = dfdefault.append(default_values, ignore_index=True)
	
	# Checks for mis-matched arguments
	groups = dfdefault.groupby(DEVICETYPE_ATTRIBUTES+KERNEL_ATTRIBUTES+["kernel"])
	for name, dfgroup in groups:
		if len(dfgroup) != 1:
			print("[WARNING] Entries for a single kernel with multiple argument values")
			
	# Defaults in general
	groups = df.groupby(KERNEL_ATTRIBUTES+ARGUMENT_ATTRIBUTES+["kernel"])
	for name, dfgroup in groups:
		default_values = dfgroup.min(axis=0)
		default_values["device_vendor"] = VENDOR_DEFAULT
		default_values["device_type"] = DEVICETYPE_DEFAULT
		default_values["device"] = DEVICENAME_DEFAULT
		default_values["device_compute_units"] = 0
		default_values["device_core_clock"] = 0
		default_values["time"] = 0.0
		dfdefault = dfdefault.append(default_values, ignore_index=True)
	
	# Database with both types of defaults only
	return dfdefault

# ==================================================================================================
# C++ header generation
# ==================================================================================================

# The C++ header
def GetHeader(family):
	return("""
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the '%s' kernels.
//
// =================================================================================================

namespace clblast {
// ================================================================================================="""
	% family.title())

# The C++ footer
def GetFooter():
	return("\n} // namespace clblast\n")

# The start of a new C++ precision entry
def GetPrecision(family, precision):
	precisionstring = ""
	if precision == "16":
		precisionstring = "Half"
	elif precision == "32":
		precisionstring = "Single"
	elif precision == "64":
		precisionstring = "Double"
	elif precision == "3232":
		precisionstring = "ComplexSingle"
	elif precision == "6464":
		precisionstring = "ComplexDouble"
	else:
		print("[ERROR] Unknown precision")
		sys.exit()
	return("\n\nconst Database::DatabaseEntry Database::%s%s = {\n  \"%s\", Precision::k%s, {\n"
	       % (family.title(), precisionstring, family.title(), precisionstring))

# The C++ device type and vendor
def GetDeviceVendor(vendor, devtype):
	if vendor == VENDOR_DEFAULT and devtype == DEVICETYPE_DEFAULT:
		return("    { // Default\n      kDeviceType%s, \"%s\", {\n" % (devtype, vendor))
	return("    { // %s %ss\n      kDeviceType%s, \"%s\", {\n" % (vendor, devtype, devtype[0].upper() + devtype[1:], vendor))

# Prints the data to a C++ database
def PrintData(df, outputdir):

	# Iterates over the kernel families: creates a new file per family
	for family, dffamily in df.groupby(["kernel_family"]):
		dffamily = dffamily.dropna(axis=1, how='all')
		f = open(os.path.join(outputdir, family+'.h'), 'w+')
		f.write(GetHeader(family))

		# Loops over the different entries for this family and prints their headers
		for precision, dfprecision in dffamily.groupby(["precision"]):
			f.write(GetPrecision(family, precision))
			for vendor, dfvendor in dfprecision.groupby(["device_vendor"]):
				for devtype, dfdevtype in dfvendor.groupby(["device_type"]):
					f.write(GetDeviceVendor(vendor, devtype))
					for device, dfdevice in dfdevtype.groupby(["device"]):
						devicename = "\"%s\"," % device
						f.write("        { %-50s { " % devicename)

						# Collects the paramaters for this case and prints them
						parameters = []
						for kernel, dfkernel in dfdevice.groupby(["kernel"]):
							dfkernel = dfkernel.dropna(axis=1)
							col_names = [col for col in list(dfkernel) if col.startswith('parameters.') and col != "parameters.PRECISION"]
							parameters += ["{\"%s\",%d}" % (p.replace("parameters.",""), dfkernel[p].iloc[0]) for p in col_names]
						f.write(", ".join(parameters))
						f.write(" } },\n")

					# Prints the footers
					f.write("      }\n    },\n")
			f.write("  }\n};\n\n// =================================================================================================")
		f.write(GetFooter())

# ==================================================================================================
# Command-line arguments parsing and verification
# ==================================================================================================

# Checks for the number of command-line arguments
if len(sys.argv) != 3:
	print("[ERROR] Usage: database.py <folder_with_json_files> <root_of_clblast>")
	sys.exit()

# Parses the command-line arguments
path_json = sys.argv[1]
path_clblast = sys.argv[2]
file_db = os.path.join(path_clblast, "scripts", "database", "database.db")
glob_json = os.path.join(path_json, "*.json")

# Checks whether the command-line arguments are valid; exists otherwise
clblast_h = os.path.join(path_clblast, "include", "clblast.h") # Not used but just for validation
if not os.path.isfile(clblast_h):
	print("[ERROR] The path '"+path_clblast+"' does not point to the root of the CLBlast library")
	sys.exit()
if len(glob.glob(glob_json)) < 1:
	print("## The path '"+path_json+"' does not contain any JSON files")

# ==================================================================================================
# The main body of the script
# ==================================================================================================

# Downloads the database if a local copy is not present
db_exists = os.path.isfile(file_db)
if not db_exists:
	DownloadDatabase(file_db)

# Loads the database from disk
print("## Loading the database from disk...")
database = LoadDatabase(file_db)

# Loops over all JSON files in the supplied folder
for file_json in glob.glob(glob_json):

	# Loads the newly imported data
	sys.stdout.write("## Processing '"+file_json+"' ")
	imported_data = ImportDataFromFile(file_json)
	imported_data = SanitizeVendorNames(imported_data)

	# Adds the new data to the database
	old_size = len(database.index)
	database = ConcatenateData(database, imported_data)
	database = RemoveDuplicates(database)
	new_size = len(database.index)
	print("with "+str(new_size-old_size)+" new items")

# Stores the modified database back to disk
if len(glob.glob(glob_json)) >= 1:
	print("## Storing the database to disk...")
	SaveDatabase(database, file_db)

# Retrieves the best performing results
print("## Calculating the best results per device/kernel...")
bests = GetBestResults(database)

# Determines the defaults for other vendors and per vendor
defaults = CalculateDefaults(bests)
bests = ConcatenateData(bests, defaults)

# Outputs the data as a C++ database
path_cpp_database = os.path.join(path_clblast, "include", "internal", "database")
print("## Producing a C++ database in '"+path_cpp_database+"'...")
PrintData(bests, path_cpp_database)

print("## All done")

# ==================================================================================================
