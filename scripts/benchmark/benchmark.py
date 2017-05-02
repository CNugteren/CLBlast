#!/usr/bin/env python

# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This file follows the
# PEP8 Python style guide and uses a max-width of 120 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>

import argparse
import json
import os
import sys

import settings
import plot
import utils

EXPERIMENTS = {
    "axpy": settings.AXPY,
    "axpybatched": settings.AXPYBATCHED,
    "gemv": settings.GEMV,
    "gemm": settings.GEMM,
    "gemm_small": settings.GEMM_SMALL,
    "gemmbatched": settings.GEMMBATCHED,
    "symm": settings.SYMM,
    "syrk": settings.SYRK,
    "summary": settings.SUMMARY,
}


def run_benchmark(name, arguments_list, precision, num_runs, platform, device):
    binary = "./clblast_client_x" + name

    # Loops over sub-benchmarks per benchmark
    results = []
    for arguments in arguments_list:

        # Sets the arguments
        constant_arguments = ["-warm_up", "-q", "-no_abbrv", "-cblas 0", "-cublas 0"]
        common_arguments = ["-precision %d" % precision, "-runs %d" % num_runs]
        opencl_arguments = ["-platform %d" % platform, "-device %d" % device]
        all_arguments = opencl_arguments + common_arguments + constant_arguments
        for name, value in arguments.items():
            all_arguments.append("-" + name + " " + str(value))

        # Calls the binary and parses the results
        benchmark_output = utils.run_binary(binary, all_arguments)
        result = utils.parse_results(benchmark_output)

        # For half-precision: also runs single-precision for comparison
        if precision == 16:
            all_arguments = [arg if arg != "-precision 16" else "-precision 32" for arg in all_arguments]
            benchmark_output = utils.run_binary(binary, all_arguments)
            result_extra = utils.parse_results(benchmark_output)
            for index in range(len(min(result, result_extra))):
                result[index]["GBs_1_FP32"] = result_extra[index]["GBs_1"]
                result[index]["GBs_2"] = result_extra[index]["GBs_2"]
                result[index]["GFLOPS_1_FP32"] = result_extra[index]["GFLOPS_1"]
                result[index]["GFLOPS_2"] = result_extra[index]["GFLOPS_2"]

        results.extend(result)
    return results


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="Runs a full benchmark for a specific routine on a specific device")
    parser.add_argument("-b", "--benchmark", required=True, help="The benchmark to perform (choose from %s)" % sorted(EXPERIMENTS.keys()))
    parser.add_argument("-p", "--platform", required=True, type=int, help="The ID of the OpenCL platform to test on")
    parser.add_argument("-d", "--device", required=True, type=int, help="The ID of the OpenCL device to test on")
    parser.add_argument("-n", "--num_runs", type=int, default=None, help="Overrides the default number of benchmark repeats for averaging")
    parser.add_argument("-x", "--precision", type=int, default=32, help="The precision to test for (choose from 16, 32, 64, 3232, 6464")
    parser.add_argument("-l", "--load_from_disk", action="store_true", help="Increase verbosity of the script")
    parser.add_argument("-t", "--plot_title", default="", help="The title for the plots, defaults to benchmark name")
    parser.add_argument("-z", "--tight_plot", action="store_true", help="Enables tight plot layout for in paper or presentation")
    parser.add_argument("-o", "--output_folder", default=os.getcwd(), help="Sets the folder for output plots (defaults to current folder)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase verbosity of the script")
    cl_args = parser.parse_args(argv)
    return vars(cl_args)


def benchmark_single(benchmark, platform, device, num_runs, precision, load_from_disk,
                     plot_title, tight_plot, output_folder, verbose):

    # Sanity check
    if not os.path.isdir(output_folder):
        print("[benchmark] Error: folder '%s' doesn't exist" % output_folder)
        return

    # The benchmark name and plot title
    benchmark_name = utils.precision_to_letter(precision) + benchmark.upper()
    if benchmark.upper() != "SUMMARY":
        plot_title = benchmark_name if plot_title is "" else benchmark_name + ": " + plot_title

    # Retrieves the benchmark settings
    if benchmark not in EXPERIMENTS.keys():
        print("[benchmark] Invalid benchmark '%s', choose from %s" % (benchmark, EXPERIMENTS.keys()))
        return
    experiment = EXPERIMENTS[benchmark]
    benchmarks = experiment["benchmarks"]

    # Either run the benchmarks for this experiment or load old results from disk
    json_file_name = os.path.join(output_folder, benchmark_name.lower() + "_benchmarks.json")
    if load_from_disk and os.path.isfile(json_file_name):
        print("[benchmark] Loading previous benchmark results from '" + json_file_name + "'")
        with open(json_file_name) as f:
            results = json.load(f)
    else:

        # Runs all the individual benchmarks
        print("[benchmark] Running on platform %d, device %d" % (platform, device))
        print("[benchmark] Running %d benchmarks for settings '%s'" % (len(benchmarks), benchmark))
        results = {"label_names": experiment["label_names"], "num_rows": experiment["num_rows"],
                   "num_cols": experiment["num_cols"], "benchmarks": []}
        for bench in benchmarks:
            num_runs_benchmark = bench["num_runs"] if num_runs is None else num_runs
            print("[benchmark] Running benchmark '%s:%s'" % (bench["name"], bench["title"]))
            result = run_benchmark(bench["name"], bench["arguments"], precision, num_runs_benchmark,
                                   platform, device)
            results["benchmarks"].append(result)

        # Stores the results to disk
        print("[benchmark] Saving benchmark results to '" + json_file_name + "'")
        with open(json_file_name, "wb") as f:
            json.dump(results, f, sort_keys=True, indent=4)

    # Retrieves the data from the benchmark settings
    file_name_suffix = "_tight" if tight_plot else ""
    pdf_file_name = os.path.join(output_folder, benchmark_name.lower() + "_plot" + file_name_suffix + ".pdf")
    titles = [utils.precision_to_letter(precision) + b["name"].upper() + " " + b["title"] for b in benchmarks]
    x_keys = [b["x_keys"] for b in benchmarks]
    y_keys = [b["y_keys"] for b in benchmarks]
    x_labels = [b["x_label"] for b in benchmarks]
    y_labels = [b["y_label"] for b in benchmarks]
    label_names = results["label_names"]

    # For half-precision: also adds single-precision results for comparison
    if precision == 16:
        label_names = ["CLBlast FP16", "clBLAS FP32", "CLBlast FP32"]
        y_keys = [y_key + [y_key[0] + "_FP32"] for y_key in y_keys]

    # Plots the graphs
    plot.plot_graphs(results["benchmarks"], pdf_file_name, results["num_rows"], results["num_cols"],
                     x_keys, y_keys, titles, x_labels, y_labels,
                     label_names, plot_title, tight_plot, verbose)

    print("[benchmark] All done")


if __name__ == '__main__':
    parsed_arguments = parse_arguments(sys.argv[1:])
    benchmark_single(**parsed_arguments)
