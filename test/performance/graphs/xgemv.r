
# ==================================================================================================
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
# project uses a tab-size of two spaces and a max-width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# This file implements the performance script for the Xgemv routine
#
# ==================================================================================================

# Includes the common functions
args <- commandArgs(trailingOnly = FALSE)
thisfile <- (normalizePath(sub("--file=", "", args[grep("--file=", args)])))
source(file.path(dirname(thisfile), "common.r"))

# ==================================================================================================

# Settings
routine_name <- "xgemv"
parameters <- c("-n","-m","-incx","-incy","-layout",
                "-num_steps","-step","-runs","-precision")
precision <- 32

# Sets the names of the test-cases
test_names <- list(
  "multiples of 256",
  "multiples of 256 (+1)",
  "around n=m=2K",
  "multiples of 256 [rotated]",
  "multiples of 256 (+1) [rotated]",
  "strides (n=2K)"
)

# Defines the test-cases
test_values <- list(
  list(c(256, 256, 1, 1, 1, 16, 256, num_runs, precision)),
  list(c(256+1, 256+1, 1, 1, 1, 16, 256, num_runs, precision)),
  list(c(2*kilo, 2*kilo, 1, 1, 1, 16, 1, num_runs, precision)),
  list(c(256, 256, 1, 1, 0, 16, 256, num_runs, precision)),
  list(c(256+1, 256+1, 1, 1, 0, 16, 256, num_runs, precision)),
  list(
    c(2*kilo, 2*kilo, 1, 1, 1, 1, 0, num_runs, precision),
    c(2*kilo, 2*kilo, 2, 1, 1, 1, 0, num_runs, precision),
    c(2*kilo, 2*kilo, 4, 1, 1, 1, 0, num_runs, precision),
    c(2*kilo, 2*kilo, 8, 1, 1, 1, 0, num_runs, precision),
    c(2*kilo, 2*kilo, 1, 2, 1, 1, 0, num_runs, precision),
    c(2*kilo, 2*kilo, 1, 4, 1, 1, 0, num_runs, precision),
    c(2*kilo, 2*kilo, 1, 8, 1, 1, 0, num_runs, precision),
    c(2*kilo, 2*kilo, 2, 2, 1, 1, 0, num_runs, precision),
    c(2*kilo, 2*kilo, 4, 4, 1, 1, 0, num_runs, precision),
    c(2*kilo, 2*kilo, 8, 8, 1, 1, 0, num_runs, precision)
  )
)

# Defines the x-labels corresponding to the test-cases
test_xlabels <- list(
  "vector sizes (n)",
  "vector sizes (n)",
  "vector sizes (n)",
  "vector sizes (n)",
  "vector sizes (n)",
  "increments/strides for x and y"
)

# Defines the x-axis of the test-cases
test_xaxis <- list(
  c("n", ""),
  c("n", ""),
  c("n", ""),
  c("n", ""),
  c("n", ""),
  list(1:10, c("x1y1", "x2y1", "x4y1", "x8y1", "x1y2", "x1y4", "x1y8", "x2y2", "x4y4", "x8y8"))
)

# ==================================================================================================

# Start the script
main(routine_name=routine_name, precision=precision, test_names=test_names, test_values=test_values,
     test_xlabels=test_xlabels, test_xaxis=test_xaxis, metric_gflops=FALSE)

# ==================================================================================================