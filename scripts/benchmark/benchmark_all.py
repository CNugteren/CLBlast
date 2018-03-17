#!/usr/bin/env python

# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import argparse
import os
import sys

from benchmark import benchmark_single, COMPARISONS


BENCHMARKS = ["axpy", "gemv", "gemm", "summary", "axpybatched", "gemmbatched", "gemmstridedbatched"]


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Runs all (main) benchmarks in one go for a given device")
    parser.add_argument("-c", "--comparisons", default=[], nargs='+', help="The library(s) to compare against (choose from %s)" % COMPARISONS)
    parser.add_argument("-p", "--platform", required=True, type=int, help="The ID of the OpenCL platform to test on")
    parser.add_argument("-d", "--device", required=True, type=int, help="The ID of the OpenCL device to test on")
    parser.add_argument("-x", "--precision", type=int, default=32, help="The precision to test for (choose from 16, 32, 64, 3232, 6464")
    parser.add_argument("-l", "--load_from_disk", action="store_true", help="Increase verbosity of the script")
    parser.add_argument("-t", "--plot_title", default="", help="The title for the plots, defaults to benchmark name")
    parser.add_argument("-o", "--output_folder", default=os.getcwd(), help="Sets the folder for output plots (defaults to current folder)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase verbosity of the script")
    cl_args = parser.parse_args(argv)
    return vars(cl_args)


def benchmark_all(comparisons, platform, device, precision, load_from_disk,
                  plot_title, output_folder, verbose):
    for bench in BENCHMARKS:
        from_disk = load_from_disk
        for tight_plot in [True, False]:  # two plots for a single benchmark
            benchmark_single(bench, comparisons, platform, device, None, precision, from_disk,
                             plot_title, tight_plot, output_folder, verbose)
            from_disk = True  # for the next plot of the same data


if __name__ == '__main__':
    parsed_arguments = parse_arguments(sys.argv[1:])
    benchmark_all(**parsed_arguments)
