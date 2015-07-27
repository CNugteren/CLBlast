
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file populates the database with best-found tuning parameters for the Copy kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::CopySingle = {
  "Copy", Precision::kSingle, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
        { "GeForce GTX 480",  { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_WPT",1}, {"COPY_VW",2} } },
        { "Tesla K20m",       { {"COPY_DIMX",8}, {"COPY_DIMY",16}, {"COPY_WPT",2}, {"COPY_VW",4} } },
        { "Tesla K40m",       { {"COPY_DIMX",16}, {"COPY_DIMY",16}, {"COPY_WPT",4}, {"COPY_VW",4} } },
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
        { "Tahiti",           { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_WPT",4}, {"COPY_VW",2} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
        { "Iris",             { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_WPT",1}, {"COPY_VW",4} } },
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_WPT",1}, {"COPY_VW",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::CopyDouble = {
  "Copy", Precision::kDouble, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
        { "GeForce GTX 480",  { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_WPT",1}, {"COPY_VW",1} } },
        { "Tesla K20m",       { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_WPT",1}, {"COPY_VW",2} } },
        { "Tesla K40m",       { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_WPT",1}, {"COPY_VW",2} } },
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
        { "Tahiti",           { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_WPT",2}, {"COPY_VW",4} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_WPT",1}, {"COPY_VW",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::CopyComplexSingle = {
  "Copy", Precision::kComplexSingle, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
        { "GeForce GTX 480",  { {"COPY_DIMX",16}, {"COPY_DIMY",16}, {"COPY_WPT",1}, {"COPY_VW",1} } },
        { "Tesla K20m",       { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_WPT",2}, {"COPY_VW",1} } },
        { "Tesla K40m",       { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_WPT",1}, {"COPY_VW",1} } },
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
        { "Tahiti",           { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_WPT",1}, {"COPY_VW",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
        { "Iris",             { {"COPY_DIMX",32}, {"COPY_DIMY",8}, {"COPY_WPT",1}, {"COPY_VW",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_WPT",1}, {"COPY_VW",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::CopyComplexDouble = {
  "Copy", Precision::kComplexDouble, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
        { "GeForce GTX 480",  { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_WPT",1}, {"COPY_VW",1} } },
        { "Tesla K20m",       { {"COPY_DIMX",8}, {"COPY_DIMY",32}, {"COPY_WPT",1}, {"COPY_VW",1} } },
        { "Tesla K40m",       { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_WPT",1}, {"COPY_VW",1} } },
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
        { "Tahiti",           { {"COPY_DIMX",8}, {"COPY_DIMY",32}, {"COPY_WPT",4}, {"COPY_VW",2} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_WPT",1}, {"COPY_VW",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
