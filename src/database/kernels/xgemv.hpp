
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xgemv' kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::XgemvHalf = {
  "Xgemv", Precision::kHalf, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"WGS1",128}, {"WPT1",1}, {"VW2",2}, {"WGS2",128}, {"WPT2",2}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"WGS1",128}, {"WPT1",1}, {"VW2",2}, {"WGS2",128}, {"WPT2",2}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",128}, {"WPT1",1}, {"VW2",2}, {"WGS2",128}, {"WPT2",2}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemvSingle = {
  "Xgemv", Precision::kSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",128}, {"WPT1",1}, {"VW2",1}, {"WGS2",128}, {"WPT2",1}, {"VW3",8}, {"WGS3",16}, {"WPT3",16} } },
        { "Hawaii",                                          { {"WGS1",128}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Oland",                                           { {"WGS1",128}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",4}, {"WGS3",256}, {"WPT3",4} } },
        { "Pitcairn",                                        { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Tahiti",                                          { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"WGS1",128}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",64}, {"WPT1",1}, {"VW2",4}, {"WGS2",128}, {"WPT2",4}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"WGS1",64}, {"WPT1",4}, {"VW2",1}, {"WGS2",64}, {"WPT2",4}, {"VW3",2}, {"WGS3",64}, {"WPT3",4} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",4}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"WGS1",256}, {"WPT1",1}, {"VW2",4}, {"WGS2",128}, {"WPT2",4}, {"VW3",4}, {"WGS3",256}, {"WPT3",4} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",4}, {"WGS3",64}, {"WPT3",4} } },
        { "Iris",                                            { {"WGS1",64}, {"WPT1",2}, {"VW2",1}, {"WGS2",128}, {"WPT2",2}, {"VW3",4}, {"WGS3",64}, {"WPT3",8} } },
        { "Iris Pro",                                        { {"WGS1",256}, {"WPT1",2}, {"VW2",1}, {"WGS2",128}, {"WPT2",2}, {"VW3",4}, {"WGS3",64}, {"WPT3",4} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",128}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"WGS1",256}, {"WPT1",1}, {"VW2",2}, {"WGS2",256}, {"WPT2",2}, {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "GeForce GTX 1070",                                { {"WGS1",128}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "GeForce GTX 480",                                 { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",128}, {"WPT2",1}, {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "GeForce GTX 670",                                 { {"WGS1",64}, {"WPT1",1}, {"VW2",2}, {"WGS2",256}, {"WPT2",2}, {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "GeForce GTX 680",                                 { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",128}, {"WPT2",1}, {"VW3",2}, {"WGS3",128}, {"WPT3",2} } },
        { "GeForce GTX 750",                                 { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",2}, {"WGS3",128}, {"WPT3",2} } },
        { "GeForce GTX 750 Ti",                              { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",4}, {"WGS3",128}, {"WPT3",4} } },
        { "GeForce GTX 980",                                 { {"WGS1",128}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
        { "GeForce GTX TITAN",                               { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
        { "GeForce GTX TITAN X",                             { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
        { "Tesla K20m",                                      { {"WGS1",128}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
        { "Tesla K40m",                                      { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemvComplexSingle = {
  "Xgemv", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",64}, {"WPT1",1}, {"VW2",2}, {"WGS2",256}, {"WPT2",2}, {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
        { "Hawaii",                                          { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Oland",                                           { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",4}, {"WGS3",64}, {"WPT3",4} } },
        { "Pitcairn",                                        { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "Tahiti",                                          { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",128}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",128}, {"WPT1",1}, {"VW2",1}, {"WGS2",128}, {"WPT2",2}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"WGS1",64}, {"WPT1",4}, {"VW2",4}, {"WGS2",64}, {"WPT2",4}, {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",2}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { {"WGS1",64}, {"WPT1",1}, {"VW2",2}, {"WGS2",128}, {"WPT2",2}, {"VW3",4}, {"WGS3",64}, {"WPT3",4} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"WGS1",128}, {"WPT1",1}, {"VW2",2}, {"WGS2",128}, {"WPT2",2}, {"VW3",4}, {"WGS3",128}, {"WPT3",4} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",4}, {"WGS3",64}, {"WPT3",4} } },
        { "Iris",                                            { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Iris Pro",                                        { {"WGS1",64}, {"WPT1",1}, {"VW2",4}, {"WGS2",128}, {"WPT2",4}, {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "GeForce GTX 1070",                                { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "GeForce GTX 480",                                 { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "GeForce GTX 670",                                 { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "GeForce GTX 680",                                 { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "GeForce GTX 750",                                 { {"WGS1",128}, {"WPT1",1} } },
        { "GeForce GTX 750 Ti",                              { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1} } },
        { "GeForce GTX TITAN",                               { {"WGS1",256}, {"WPT1",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemvDouble = {
  "Xgemv", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
        { "Hawaii",                                          { {"WGS1",128}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Oland",                                           { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",4}, {"WGS3",256}, {"WPT3",4} } },
        { "Pitcairn",                                        { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "Tahiti",                                          { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",64}, {"WPT1",2}, {"VW2",4}, {"WGS2",128}, {"WPT2",4}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"WGS1",64}, {"WPT1",4}, {"VW2",1}, {"WGS2",64}, {"WPT2",4}, {"VW3",1}, {"WGS3",64}, {"WPT3",2} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",2}, {"VW2",1}, {"WGS2",64}, {"WPT2",4}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"WGS1",128}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "GeForce GTX 1070",                                { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
        { "GeForce GTX 480",                                 { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "GeForce GTX 670",                                 { {"WGS1",128}, {"WPT1",1}, {"VW2",1}, {"WGS2",128}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "GeForce GTX 680",                                 { {"WGS1",128}, {"WPT1",1}, {"VW2",1}, {"WGS2",128}, {"WPT2",1}, {"VW3",2}, {"WGS3",128}, {"WPT3",2} } },
        { "GeForce GTX 750",                                 { {"WGS1",64}, {"WPT1",1}, {"VW2",2}, {"WGS2",256}, {"WPT2",2}, {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "GeForce GTX 750 Ti",                              { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",2}, {"WGS3",256}, {"WPT3",2} } },
        { "GeForce GTX 980",                                 { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "GeForce GTX TITAN",                               { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
        { "GeForce GTX TITAN X",                             { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",128}, {"WPT2",1}, {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
        { "Tesla K20m",                                      { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",128}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Tesla K40m",                                      { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemvComplexDouble = {
  "Xgemv", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
        { "Hawaii",                                          { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Oland",                                           { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
        { "Pitcairn",                                        { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Tahiti",                                          { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",64}, {"WPT1",1}, {"VW2",2}, {"WGS2",64}, {"WPT2",4}, {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"WGS1",64}, {"WPT1",4}, {"VW2",4}, {"WGS2",64}, {"WPT2",4}, {"VW3",2}, {"WGS3",256}, {"WPT3",2} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",2}, {"WGS2",64}, {"WPT2",4}, {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"WGS1",128}, {"WPT1",1}, {"VW2",1}, {"WGS2",128}, {"WPT2",1}, {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
        { "GeForce GTX 480",                                 { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "GeForce GTX 670",                                 { {"WGS1",128}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
