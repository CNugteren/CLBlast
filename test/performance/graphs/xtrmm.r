
# ==================================================================================================
# This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
# project uses a tab-size of two spaces and a max-width of 100 characters per line.
#
# Author(s):
#   Cedric Nugteren <www.cedricnugteren.nl>
#
# This file implements the performance script for the Xtrmm routine
#
# ==================================================================================================

# Includes the common functions
args <- commandArgs(trailingOnly = FALSE)
thisfile <- (normalizePath(sub("--file=", "", args[grep("--file=", args)])))
source(file.path(dirname(thisfile), "common.r"))

# ==================================================================================================

# Settings
routine_name <- "xtrmm"
parameters <- c("-m","-n","-layout","-side","-triangle","-transA","-diagonal",
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
  list(c(128, 128, 0, 0, 0, 0, 0, 16, 128, num_runs, precision)),
  list(c(129, 129, 0, 0, 0, 0, 0, 16, 128, num_runs, precision)),
  list(c(512, 512, 0, 0, 0, 0, 0, 16, 1, num_runs, precision)),
  list(c(2048, 2048, 0, 0, 0, 0, 0, 16, 1, num_runs, precision)),
  list(
    c(1024, 1024, 0, 0, 0, 0, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 0, 0, 0, 1, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 0, 0, 1, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 0, 0, 1, 1, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 0, 1, 0, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 0, 1, 0, 1, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 0, 1, 1, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 0, 1, 1, 1, 1, 0, num_runs, precision),

    c(1024, 1024, 0, 1, 0, 0, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 1, 0, 0, 1, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 1, 0, 1, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 1, 0, 1, 1, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 1, 1, 0, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 1, 1, 0, 1, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 1, 1, 1, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 1, 1, 1, 1, 1, 0, num_runs, precision),

    c(1024, 1024, 0, 0, 0, 0, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 0, 0, 0, 1, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 0, 0, 1, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 0, 0, 1, 1, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 0, 1, 0, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 0, 1, 0, 1, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 0, 1, 1, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 0, 1, 1, 1, 1, 0, num_runs, precision),

    c(1024, 1024, 1, 1, 0, 0, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 1, 1, 0, 0, 1, 1, 0, num_runs, precision),
    c(1024, 1024, 1, 1, 0, 1, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 1, 1, 0, 1, 1, 1, 0, num_runs, precision),
    c(1024, 1024, 1, 1, 1, 0, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 1, 1, 1, 0, 1, 1, 0, num_runs, precision),
    c(1024, 1024, 1, 1, 1, 1, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 1, 1, 1, 1, 1, 1, 0, num_runs, precision)
  ),
  list(
    c(8, 8, 0, 0, 0, 0, 0, 1, 0, num_runs, precision),
    c(16, 16, 0, 0, 0, 0, 0, 1, 0, num_runs, precision),
    c(32, 32, 0, 0, 0, 0, 0, 1, 0, num_runs, precision),
    c(64, 64, 0, 0, 0, 0, 0, 1, 0, num_runs, precision),
    c(128, 128, 0, 0, 0, 0, 0, 1, 0, num_runs, precision),
    c(256, 256, 0, 0, 0, 0, 0, 1, 0, num_runs, precision),
    c(512, 512, 0, 0, 0, 0, 0, 1, 0, num_runs, precision),
    c(1024, 1024, 0, 0, 0, 0, 0, 1, 0, num_runs, precision),
    c(2048, 2048, 0, 0, 0, 0, 0, 1, 0, num_runs, precision),
    c(4096, 4096, 0, 0, 0, 0, 0, 1, 0, num_runs, precision),
    c(8192, 8192, 0, 0, 0, 0, 0, 1, 0, num_runs, precision)
  )
)

# Defines the x-labels corresponding to the test-cases
test_xlabels <- list(
  "matrix sizes (m=n)",
  "matrix sizes (m=n)",
  "matrix sizes (m=n)",
  "matrix sizes (m=n)",
  "layout (row/col), side (l/r), triangle (up/lo), transA (n/y), diag (u/nu)",
  "matrix sizes (m=n)"
)

# Defines the x-axis of the test-cases
test_xaxis <- list(
  c("m", ""),
  c("m", ""),
  c("m", ""),
  c("m", ""),
  list(1:32, c("row,l,up,n,u", "row,l,up,n,nu", "row,l,up,y,u", "row,l,up,y,nu",
               "row,r,up,n,u", "row,r,up,n,nu", "row,r,up,y,u", "row,r,up,y,nu",
               "row,l,lo,n,u", "row,l,lo,n,nu", "row,l,lo,y,u", "row,l,lo,y,nu",
               "row,r,lo,n,u", "row,r,lo,n,nu", "row,r,lo,y,u", "row,r,lo,y,nu",
               "col,l,up,n,u", "col,l,up,n,nu", "col,l,up,y,u", "col,l,up,y,nu",
               "col,r,up,n,u", "col,r,up,n,nu", "col,r,up,y,u", "col,r,up,y,nu",
               "col,l,lo,n,u", "col,l,lo,n,nu", "col,l,lo,y,u", "col,l,lo,y,nu",
               "col,r,lo,n,u", "col,r,lo,n,nu", "col,r,lo,y,u", "col,r,lo,y,nu")),
  c("m", "x")
)

# ==================================================================================================

# Start the script
main(routine_name=routine_name, precision=precision, test_names=test_names, test_values=test_values,
     test_xlabels=test_xlabels, test_xaxis=test_xaxis, metric_gflops=TRUE)

# ==================================================================================================