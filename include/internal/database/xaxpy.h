
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xaxpy' kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::XaxpySingle = {
  "Xaxpy", Precision::kSingle, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                          { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "default",                                       { {"VW",1}, {"WGS",64}, {"WPT",1} } },
      }
    },
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"VW",4}, {"WGS",128}, {"WPT",1} } },
        { "default",                                       { {"VW",4}, {"WGS",128}, {"WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"VW",1}, {"WGS",64}, {"WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XaxpyComplexSingle = {
  "Xaxpy", Precision::kComplexSingle, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                          { {"VW",2}, {"WGS",128}, {"WPT",1} } },
        { "default",                                       { {"VW",2}, {"WGS",128}, {"WPT",1} } },
      }
    },
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"VW",1}, {"WGS",128}, {"WPT",1} } },
        { "default",                                       { {"VW",1}, {"WGS",128}, {"WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"VW",1}, {"WGS",128}, {"WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XaxpyDouble = {
  "Xaxpy", Precision::kDouble, {
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"VW",2}, {"WGS",128}, {"WPT",1} } },
        { "default",                                       { {"VW",2}, {"WGS",128}, {"WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"VW",2}, {"WGS",128}, {"WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XaxpyComplexDouble = {
  "Xaxpy", Precision::kComplexDouble, {
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"VW",1}, {"WGS",64}, {"WPT",1} } },
        { "default",                                       { {"VW",1}, {"WGS",64}, {"WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"VW",1}, {"WGS",64}, {"WPT",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
