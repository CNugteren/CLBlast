
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xdot' kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::XdotSingle = {
  "Xdot", Precision::kSingle, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                          { {"VW",1}, {"WGS1",512}, {"WGS2",32} } },
        { "default",                                       { {"VW",1}, {"WGS1",512}, {"WGS2",32} } },
      }
    },
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
        { "default",                                       { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"VW",1}, {"WGS1",128}, {"WGS2",32} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XdotComplexSingle = {
  "Xdot", Precision::kComplexSingle, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                          { {"VW",1}, {"WGS1",512}, {"WGS2",1024} } },
        { "default",                                       { {"VW",1}, {"WGS1",512}, {"WGS2",1024} } },
      }
    },
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
        { "default",                                       { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XdotDouble = {
  "Xdot", Precision::kDouble, {
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"VW",1}, {"WGS1",256}, {"WGS2",1024} } },
        { "default",                                       { {"VW",1}, {"WGS1",256}, {"WGS2",1024} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"VW",1}, {"WGS1",256}, {"WGS2",1024} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XdotComplexDouble = {
  "Xdot", Precision::kComplexDouble, {
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
        { "default",                                       { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"VW",1}, {"WGS1",128}, {"WGS2",1024} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
