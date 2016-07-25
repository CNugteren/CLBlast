
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xgemv_Fast_Rot' kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::XgemvFastRotHalf = {
  "XgemvFastRot", Precision::kHalf, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemvFastRotSingle = {
  "XgemvFastRot", Precision::kSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Hawaii",                                          { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Oland",                                           { {"VW3",4}, {"WGS3",256}, {"WPT3",4} } },
        { "Pitcairn",                                        { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Tahiti",                                          { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"VW3",2}, {"WGS3",64}, {"WPT3",4} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"VW3",4}, {"WGS3",256}, {"WPT3",4} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"VW3",4}, {"WGS3",64}, {"WPT3",4} } },
        { "Iris",                                            { {"VW3",4}, {"WGS3",64}, {"WPT3",8} } },
        { "Iris Pro",                                        { {"VW3",4}, {"WGS3",64}, {"WPT3",4} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "GeForce GTX 1070",                                { {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "GeForce GTX 480",                                 { {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "GeForce GTX 670",                                 { {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "GeForce GTX 680",                                 { {"VW3",2}, {"WGS3",128}, {"WPT3",2} } },
        { "GeForce GTX 750",                                 { {"VW3",2}, {"WGS3",128}, {"WPT3",2} } },
        { "GeForce GTX 750 Ti",                              { {"VW3",4}, {"WGS3",128}, {"WPT3",4} } },
        { "GeForce GTX 980",                                 { {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
        { "GeForce GTX TITAN",                               { {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
        { "GeForce GTX TITAN X",                             { {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
        { "Tesla K20m",                                      { {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
        { "Tesla K40m",                                      { {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemvFastRotComplexSingle = {
  "XgemvFastRot", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
        { "Hawaii",                                          { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Oland",                                           { {"VW3",4}, {"WGS3",64}, {"WPT3",4} } },
        { "Pitcairn",                                        { {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "Tahiti",                                          { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics 530",                        { {"VW3",4}, {"WGS3",64}, {"WPT3",4} } },
        { "Intel(R) HD Graphics Haswell Ultrabook GT2 Mobile", { {"VW3",4}, {"WGS3",128}, {"WPT3",4} } },
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"VW3",4}, {"WGS3",64}, {"WPT3",4} } },
        { "Iris",                                            { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Iris Pro",                                        { {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "GeForce GTX 480",                                 { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "GeForce GTX 670",                                 { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "GeForce GTX 680",                                 { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemvFastRotDouble = {
  "XgemvFastRot", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
        { "Hawaii",                                          { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Oland",                                           { {"VW3",4}, {"WGS3",256}, {"WPT3",4} } },
        { "Pitcairn",                                        { {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "Tahiti",                                          { {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"VW3",1}, {"WGS3",64}, {"WPT3",2} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "GeForce GTX 1070",                                { {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
        { "GeForce GTX 480",                                 { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "GeForce GTX 670",                                 { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "GeForce GTX 680",                                 { {"VW3",2}, {"WGS3",128}, {"WPT3",2} } },
        { "GeForce GTX 750",                                 { {"VW3",2}, {"WGS3",64}, {"WPT3",2} } },
        { "GeForce GTX 750 Ti",                              { {"VW3",2}, {"WGS3",256}, {"WPT3",2} } },
        { "GeForce GTX 980",                                 { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "GeForce GTX TITAN",                               { {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
        { "GeForce GTX TITAN X",                             { {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
        { "Tesla K20m",                                      { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Tesla K40m",                                      { {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemvFastRotComplexDouble = {
  "XgemvFastRot", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
        { "Hawaii",                                          { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Oland",                                           { {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
        { "Pitcairn",                                        { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "Tahiti",                                          { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
        { "Intel(R) Core(TM) i7-3770 CPU @ 3.40GHz",         { {"VW3",2}, {"WGS3",256}, {"WPT3",2} } },
        { "default",                                         { {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "Intel(R) Many Integrated Core Acceleration Card", { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GRID K520",                                       { {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
        { "GeForce GTX 480",                                 { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "GeForce GTX 670",                                 { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
