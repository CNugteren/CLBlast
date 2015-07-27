
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file populates the database with best-found tuning parameters for the Xaxpy kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::XaxpySingle = {
  "Xaxpy", Precision::kSingle, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
        { "GeForce GTX 480",  { {"WGS",128}, {"WPT",1}, {"VW",2} } },
        { "Tesla K20m",       { {"WGS",128}, {"WPT",2}, {"VW",2} } },
        { "Tesla K40m",       { {"WGS",128}, {"WPT",1}, {"VW",4} } },
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
        { "Tahiti",           { {"WGS",64}, {"WPT",1}, {"VW",2} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
        { "Iris",             { {"WGS",512}, {"WPT",1}, {"VW",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"WGS",128}, {"WPT",1}, {"VW",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XaxpyDouble = {
  "Xaxpy", Precision::kDouble, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
        { "GeForce GTX 480",  { {"WGS",128}, {"WPT",1}, {"VW",1} } },
        { "Tesla K20m",       { {"WGS",512}, {"WPT",1}, {"VW",2} } },
        { "Tesla K40m",       { {"WGS",64}, {"WPT",1}, {"VW",2} } },
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
        { "Tahiti",           { {"WGS",256}, {"WPT",1}, {"VW",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"WGS",128}, {"WPT",1}, {"VW",1} } },
      }
    },
  }
};
// =================================================================================================

const Database::DatabaseEntry Database::XaxpyComplexSingle = {
  "Xaxpy", Precision::kComplexSingle, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
        { "GeForce GTX 480",  { {"WGS",256}, {"WPT",1}, {"VW",1} } },
        { "Tesla K20m",       { {"WGS",128}, {"WPT",1}, {"VW",1} } },
        { "Tesla K40m",       { {"WGS",128}, {"WPT",2}, {"VW",1} } },
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
        { "Tahiti",           { {"WGS",64}, {"WPT",1}, {"VW",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
        { "Iris",             { {"WGS",256}, {"WPT",1}, {"VW",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"WGS",128}, {"WPT",1}, {"VW",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XaxpyComplexDouble = {
  "Xaxpy", Precision::kComplexDouble, {
    { // NVIDIA GPUs
      kDeviceTypeGPU, kDeviceVendorNVIDIA, {
        { "GeForce GTX 480",  { {"WGS",128}, {"WPT",2}, {"VW",1} } },
        { "Tesla K20m",       { {"WGS",256}, {"WPT",1}, {"VW",1} } },
        { "Tesla K40m",       { {"WGS",64}, {"WPT",2}, {"VW",1} } },
      }
    },
    { // AMD GPUs
      kDeviceTypeGPU, kDeviceVendorAMD, {
        { "Tahiti",           { {"WGS",64}, {"WPT",1}, {"VW",1} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, kDeviceVendorIntel, {
      }
    },
    { // Default
      kDeviceTypeAll, kDeviceVendorAll, {
        { kDefaultDevice,     { {"WGS",128}, {"WPT",1}, {"VW",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
