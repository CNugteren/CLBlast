
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

const Database::DatabaseEntry Database::XgemvSingle = {
  "Xgemv", Precision::kSingle, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                          { {"WGS1",64}, {"WPT1",2}, {"VW2",1}, {"WGS2",128}, {"WPT2",2}, {"VW3",4}, {"WGS3",64}, {"WPT3",8} } },
        { "default",                                       { {"WGS1",64}, {"WPT1",2}, {"VW2",1}, {"WGS2",128}, {"WPT2",2}, {"VW3",4}, {"WGS3",64}, {"WPT3",8} } },
      }
    },
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
        { "default",                                       { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"WGS1",64}, {"WPT1",1}, {"VW2",1}, {"WGS2",128}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemvComplexSingle = {
  "Xgemv", Precision::kComplexSingle, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                          { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
        { "default",                                       { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",64}, {"WPT2",1}, {"VW3",1}, {"WGS3",64}, {"WPT3",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemvDouble = {
  "Xgemv", Precision::kDouble, {
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
        { "default",                                       { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"WGS1",256}, {"WPT1",1}, {"VW2",1}, {"WGS2",256}, {"WPT2",1}, {"VW3",1}, {"WGS3",256}, {"WPT3",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemvComplexDouble = {
  "Xgemv", Precision::kComplexDouble, {
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"WGS1",128}, {"WPT1",1}, {"VW2",1}, {"WGS2",128}, {"WPT2",1}, {"VW3",1}, {"WGS3",128}, {"WPT3",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
