
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


def get_relative_bests(df, parameter_column_names, name, verbose=False):
    """Retrieves the relative best execution time over different devices"""

    # Computes the sum of the execution times over the different devices
    def sum_performance(x):
        x["group_performance"] = x["relative_performance"].sum()
        return x
    df = df.groupby(parameter_column_names).apply(sum_performance)

    # Retrieves the entries with the highest performance
    best_performance = df["group_performance"].max()
    df_bests = df[df["group_performance"] == best_performance]

    # Retrieves one example only (the parameters are the same anyway)
    df_bests = df_bests.drop_duplicates(["group_performance"])

    # Completed, report and return the results
    if verbose:
        print("[database] " + str(name) + " with performance " + str(best_performance) + " " + str(df_bests.shape))
    assert len(df_bests) == 1
    return df_bests
