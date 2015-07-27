
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file populates the database with best-found tuning parameters for the Pad kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::PadSingle = {
  "Pad", Precision::kSingle, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
        { "GeForce GTX 480",  { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",4} } },
        { "Tesla K20m",       { {"PAD_DIMX",16}, {"PAD_DIMY",32}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
        { "Tesla K40m",       { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
        { "Tahiti",           { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
        { "Iris",             { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::PadDouble = {
  "Pad", Precision::kDouble, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
        { "GeForce GTX 480",  { {"PAD_DIMX",16}, {"PAD_DIMY",16}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Tesla K20m",       { {"PAD_DIMX",16}, {"PAD_DIMY",16}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Tesla K40m",       { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
        { "Tahiti",           { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::PadComplexSingle = {
  "Pad", Precision::kComplexSingle, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
        { "GeForce GTX 480",  { {"PAD_DIMX",16}, {"PAD_DIMY",16}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Tesla K20m",       { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "Tesla K40m",       { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
        { "Tahiti",           { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
        { "Iris",             { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::PadComplexDouble = {
  "Pad", Precision::kComplexDouble, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
        { "GeForce GTX 480",  { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Tesla K20m",       { {"PAD_DIMX",32}, {"PAD_DIMY",16}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "Tesla K40m",       { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
        { "Tahiti",           { {"PAD_DIMX",8}, {"PAD_DIMY",16}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
