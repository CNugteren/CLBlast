
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import pandas as pd
import clblast


def get_best_results(df):
    """Retrieves the results with the lowests execution times"""
    database_bests = pd.DataFrame()
    database_entries = df.groupby(clblast.ATTRIBUTES + ["kernel"])
    for name, database_entry in database_entries:
        best_time = database_entry["time"].min()
        best_parameters = database_entry[database_entry["time"] == best_time].iloc[0]
        database_bests = database_bests.append(best_parameters, ignore_index=True)
    return database_bests
