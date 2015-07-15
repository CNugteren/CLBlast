
# ==================================================================================================
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
# project uses a tab-size of two spaces and a max-width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# This file implements the performance script for the Xgemm routine
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
  "multiples of 128",
  "multiples of 128 (+1)",
  "around m=n=k=512",
  "around m=n=k=2048",
  "layouts and transposing (m=n=k=1024)",
  "powers of 2"
)

# Defines the test-cases
test_values <- list(
  list(c( 128,  128,  128, 1, 0, 0, 16, 128, num_runs, precision)),
  list(c( 129,  129,  129, 1, 0, 0, 16, 128, num_runs, precision)),
  list(c( 512,  512,  512, 1, 0, 0, 16, 1, num_runs, precision)),
  list(c(2048, 2048, 2048, 1, 0, 0, 16, 1, num_runs, precision)),
  list(
    c(1024, 1024, 1024, 0, 0, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 1024, 0, 0, 1, 1, 0, num_runs, precision),
    c(1024, 1024, 1024, 0, 1, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 1024, 0, 1, 1, 1, 0, num_runs, precision),
    c(1024, 1024, 1024, 1, 0, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 1024, 1, 0, 1, 1, 0, num_runs, precision),
    c(1024, 1024, 1024, 1, 1, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 1024, 1, 1, 1, 1, 0, num_runs, precision)
  ),
  list(
    c(   8,    8,    8, 1, 0, 0, 1, 0, num_runs, precision),
    c(  16,   16,   16, 1, 0, 0, 1, 0, num_runs, precision),
    c(  32,   32,   32, 1, 0, 0, 1, 0, num_runs, precision),
    c(  64,   64,   64, 1, 0, 0, 1, 0, num_runs, precision),
    c( 128,  128,  128, 1, 0, 0, 1, 0, num_runs, precision),
    c( 256,  256,  256, 1, 0, 0, 1, 0, num_runs, precision),
    c( 512,  512,  512, 1, 0, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 1024, 1, 0, 0, 1, 0, num_runs, precision),
    c(2048, 2048, 2048, 1, 0, 0, 1, 0, num_runs, precision),
    c(4096, 4096, 4096, 1, 0, 0, 1, 0, num_runs, precision),
    c(8192, 8192, 8192, 1, 0, 0, 1, 0, num_runs, precision)
  )
)

# Defines the x-labels corresponding to the test-cases
test_xlabels <- list(
  "matrix sizes (m=n=k)",
  "matrix sizes (m=n=k)",
  "matrix sizes (m=n=k)",
  "matrix sizes (m=n=k)",
  "layout (row/col), transA (n/y), transB (n/y)",
  "matrix sizes (m=n=k)"
)

# Defines the x-axis of the test-cases
test_xaxis <- list(
  c("m", ""),
  c("m", ""),
  c("m", ""),
  c("m", ""),
  list(1:8, c("row,n,n", "row,n,y", "row,y,n", "row,y,y",
              "col,n,n", "col,n,y", "col,y,n", "col,y,y")),
  c("m", "x")
)

# ==================================================================================================

# Start the script
main(routine_name=routine_name, precision=precision, test_names=test_names, test_values=test_values,
     test_xlabels=test_xlabels, test_xaxis=test_xaxis, metric_gflops=TRUE)

# ==================================================================================================