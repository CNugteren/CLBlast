
# ==================================================================================================
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
# project uses a tab-size of two spaces and a max-width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# This file implements the performance script for the Xsymm routine
#
# ==================================================================================================

# Includes the common functions
args <- commandArgs(trailingOnly = FALSE)
thisfile <- (normalizePath(sub("--file=", "", args[grep("--file=", args)])))
source(file.path(dirname(thisfile), "common.r"))

# ==================================================================================================

# Settings
routine_name <- "xsymm"
parameters <- c("-m","-n","-layout","-side","-triangle",
                "-num_steps","-step","-runs","-precision")
precision <- 32

# Sets the names of the test-cases
test_names <- list(
  "multiples of 128",
  "multiples of 128 (+1)",
  "around m=n=512",
  "around m=n=2048",
  "layouts and side/triangle (m=n=1024)",
  "powers of 2"
)

# Defines the test-cases
test_values <- list(
  list(c( 128,  128, 102, 111, 111, 16, 128, num_runs, precision)),
  list(c( 129,  129, 102, 111, 111, 16, 128, num_runs, precision)),
  list(c( 512,  512, 102, 111, 111, 16, 1, num_runs, precision)),
  list(c(2048, 2048, 102, 111, 111, 16, 1, num_runs, precision)),
  list(
    c(1024, 1024, 101, 111, 111, 1, 0, num_runs, precision),
    c(1024, 1024, 101, 111, 112, 1, 0, num_runs, precision),
    c(1024, 1024, 101, 112, 111, 1, 0, num_runs, precision),
    c(1024, 1024, 101, 112, 112, 1, 0, num_runs, precision),
    c(1024, 1024, 102, 111, 111, 1, 0, num_runs, precision),
    c(1024, 1024, 102, 111, 112, 1, 0, num_runs, precision),
    c(1024, 1024, 102, 112, 111, 1, 0, num_runs, precision),
    c(1024, 1024, 102, 112, 112, 1, 0, num_runs, precision)
  ),
  list(
    c(   8,    8, 102, 111, 111, 1, 0, num_runs, precision),
    c(  16,   16, 102, 111, 111, 1, 0, num_runs, precision),
    c(  32,   32, 102, 111, 111, 1, 0, num_runs, precision),
    c(  64,   64, 102, 111, 111, 1, 0, num_runs, precision),
    c( 128,  128, 102, 111, 111, 1, 0, num_runs, precision),
    c( 256,  256, 102, 111, 111, 1, 0, num_runs, precision),
    c( 512,  512, 102, 111, 111, 1, 0, num_runs, precision),
    c(1024, 1024, 102, 111, 111, 1, 0, num_runs, precision),
    c(2048, 2048, 102, 111, 111, 1, 0, num_runs, precision),
    c(4096, 4096, 102, 111, 111, 1, 0, num_runs, precision),
    c(8192, 8192, 102, 111, 111, 1, 0, num_runs, precision)
  )
)

# Defines the x-labels corresponding to the test-cases
test_xlabels <- list(
  "matrix sizes (m=n)",
  "matrix sizes (m=n)",
  "matrix sizes (m=n)",
  "matrix sizes (m=n)",
  "layout (row/col), side (l/r), triangle (up/lo)",
  "matrix sizes (m=n)"
)

# Defines the x-axis of the test-cases
test_xaxis <- list(
  c("m", ""),
  c("m", ""),
  c("m", ""),
  c("m", ""),
  list(1:8, c("row,l,up", "row,r,up", "row,l,lo", "row,r,lo",
              "col,l,up", "col,r,up", "col,l,lo", "col,r,lo")),
  c("m", "x")
)

# ==================================================================================================

# Start the script
main(routine_name=routine_name, precision=precision, test_names=test_names, test_values=test_values,
     test_xlabels=test_xlabels, test_xaxis=test_xaxis, metric_gflops=TRUE)

# ==================================================================================================