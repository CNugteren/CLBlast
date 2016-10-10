
# ==================================================================================================
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
# project uses a tab-size of two spaces and a max-width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# This file implements the performance script for small sizes of Xgemm, testing the direct kernel
#
# ==================================================================================================

# Includes the common functions
args <- commandArgs(trailingOnly = FALSE)
thisfile <- (normalizePath(sub("--file=", "", args[grep("--file=", args)])))
source(file.path(dirname(thisfile), "common.r"))

# ==================================================================================================

# Settings
routine_name <- "xgemm"
parameters <- c("-m","-n","-k","-layout","-transA","-transB",
                "-num_steps","-step","-runs","-precision")
precision <- 32

# Sets the names of the test-cases
test_names <- list(
  "small matrices in steps of 16",
  "small matrices in steps of 1"
)

# Defines the test-cases
test_values <- list(
  list(c( 128,  128,  128, 102, 111, 111, 57, 16, num_runs_short, precision)),
  list(c( 128,  128,  128, 102, 111, 111, 385, 1, num_runs_short, precision))
)

# Defines the x-labels corresponding to the test-cases
test_xlabels <- list(
  "matrix sizes (m=n=k)",
  "matrix sizes (m=n=k)"
)

# Defines the x-axis of the test-cases
test_xaxis <- list(
  c("m", ""),
  c("m", "")
)

# ==================================================================================================

# Start the script
main(routine_name=routine_name, precision=precision, test_names=test_names, test_values=test_values,
     test_xlabels=test_xlabels, test_xaxis=test_xaxis, metric_gflops=TRUE)

# ==================================================================================================