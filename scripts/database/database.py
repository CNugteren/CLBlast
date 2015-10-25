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

# Additional modules
import pandas as pd

# Constants
ATTRIBUTES = ["device", "type", "vendor", "precision", "kernel_family", "arg_m", "arg_n", "arg_k"]

# Pandas options
pd.set_option('display.width', 1000)

# ==================================================================================================
# Database operations
# ==================================================================================================

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

# Bests
def GetBestResults(df):
	dfbest = pd.DataFrame()
	grouped = df.groupby(ATTRIBUTES+["kernel"])
	for name, dfgroup in grouped:
		bestcase = dfgroup.loc[[dfgroup["time"].idxmin()]]
		dfbest = ConcatenateData(dfbest, bestcase)
	return dfbest

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
	precisionstring = "Single"
	if precision == "64":
		precisionstring = "Double"
	elif precision == "3232":
		precisionstring = "ComplexSingle"
	elif precision == "6464":
		precisionstring = "ComplexDouble"
	return("\n\nconst Database::DatabaseEntry Database::%s%s = {\n  \"%s\", Precision::k%s, {\n"
	       % (family.title(), precisionstring, family.title(), precisionstring))

# The C++ device type and vendor
def GetDeviceVendor(vendor, devtype):
	return("    { // %s %ss\n      kDeviceType%s, kDeviceVendor%s, {\n"
	       % (vendor, devtype, devtype, vendor))

# Prints the data to a C++ database
def PrintData(df):

	# Iterates over the kernel families: creates a new file per family
	for family, dffamily in df.groupby(["kernel_family"]):
		dffamily = dffamily.dropna(axis=1, how='all')
		f = open(family+'.h', 'w+')
		f.write(GetHeader(family))

		# Loops over the different entries for this family and prints their headers
		for precision, dfprecision in dffamily.groupby(["precision"]):
			f.write(GetPrecision(family, precision))
			for vendor, dfvendor in dfprecision.groupby(["vendor"]):
				for devtype, dfdevtype in dfvendor.groupby(["type"]):
					f.write(GetDeviceVendor(vendor, devtype))
					for device, dfdevice in dfdevtype.groupby(["device"]):
						devicename = "\"%s\"," % device
						f.write("        { %-20s { " % devicename)

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
	print "[ERROR] Usage: database.py <folder_with_json_files> <root_of_clblast>"
	sys.exit()

# Parses the command-line arguments
path_json = sys.argv[1]
path_clblast = sys.argv[2]
file_db = path_clblast+"/src/database.db"
glob_json = path_json+"/*.json"

# Checks whether the command-line arguments are valid; exists otherwise
clblast_h = path_clblast+"/include/clblast.h" # Not used but just for validation
if not os.path.isfile(clblast_h):
	print "[ERROR] The path '"+path_clblast+"' does not point to the root of the CLBlast library"
	sys.exit()
if len(glob.glob(glob_json)) < 1:
	print "[ERROR] The path '"+path_json+"' does not contain any JSON files"
	sys.exit()

# ==================================================================================================
# The main body of the script
# ==================================================================================================

# Loads the database if it exists. If not, a new database is initialized
db_exists = os.path.isfile(file_db)
database = LoadDatabase(file_db) if db_exists else pd.DataFrame()

# Loops over all JSON files in the supplied folder
for file_json in glob.glob(glob_json):

	# Loads the newly imported data
	print "## Processing '"+file_json+"'",
	imported_data = ImportDataFromFile(file_json)

	# Adds the new data to the database
	old_size = len(database.index)
	database = ConcatenateData(database, imported_data)
	database = RemoveDuplicates(database)
	new_size = len(database.index)
	print "with "+str(new_size-old_size)+" new items"

# Stores the new database back to disk
SaveDatabase(database, file_db)

# Retrieves the best performing results
bests = GetBestResults(database)

# TODO: Determines the defaults for other vendors and per vendor
#defaults = CalculateDefaults(bests)
#bests = ConcatenateData(bests, defaults)

# Outputs the data as a C++ database
PrintData(bests)

# ==================================================================================================
