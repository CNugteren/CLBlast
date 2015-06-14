
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file populates the database with best-found tuning parameters for the Xgemv kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::XgemvSingle = {
  "Xgemv", Precision::kSingle, {
    { // NVIDIA GPUs
      CL_DEVICE_TYPE_GPU, "NVIDIA Corporation", {
        { "GeForce GTX 480",  { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
        { "Tesla K20m",       { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
        { "Tesla K40m",       { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
      }
    },
    { // AMD GPUs
      CL_DEVICE_TYPE_GPU, "AMD", {
        { "Tahiti",           { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
      }
    },
    { // Intel GPUs
      CL_DEVICE_TYPE_GPU, "Intel", {
        { "Iris",             { {"WGS1",256}, {"WPT1",2}, {"WGS2",64}, {"WPT2",4}, {"VW2",4}, {"WGS3",256}, {"WPT3",2}, {"VW3",8} } },
      }
    },
    { // Default
      CL_DEVICE_TYPE_ALL, kDefault, {
        { kDefault,           { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemvDouble = {
  "Xgemv", Precision::kDouble, {
    { // NVIDIA GPUs
      CL_DEVICE_TYPE_GPU, "NVIDIA Corporation", {
        { "GeForce GTX 480",  { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
        { "Tesla K20m",       { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
        { "Tesla K40m",       { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
      }
    },
    { // AMD GPUs
      CL_DEVICE_TYPE_GPU, "AMD", {
        { "Tahiti",           { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
      }
    },
    { // Intel GPUs
      CL_DEVICE_TYPE_GPU, "Intel", {
      }
    },
    { // Default
      CL_DEVICE_TYPE_ALL, kDefault, {
        { kDefault,           { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
      }
    },
  }
};
// =================================================================================================

const Database::DatabaseEntry Database::XgemvComplexSingle = {
  "Xgemv", Precision::kComplexSingle, {
    { // NVIDIA GPUs
      CL_DEVICE_TYPE_GPU, "NVIDIA Corporation", {
        { "GeForce GTX 480",  { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
        { "Tesla K20m",       { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
        { "Tesla K40m",       { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
      }
    },
    { // AMD GPUs
      CL_DEVICE_TYPE_GPU, "AMD", {
        { "Tahiti",           { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
      }
    },
    { // Intel GPUs
      CL_DEVICE_TYPE_GPU, "Intel", {
        { "Iris",             { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
      }
    },
    { // Default
      CL_DEVICE_TYPE_ALL, kDefault, {
        { kDefault,           { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgemvComplexDouble = {
  "Xgemv", Precision::kComplexDouble, {
    { // NVIDIA GPUs
      CL_DEVICE_TYPE_GPU, "NVIDIA Corporation", {
        { "GeForce GTX 480",  { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
        { "Tesla K20m",       { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
        { "Tesla K40m",       { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
      }
    },
    { // AMD GPUs
      CL_DEVICE_TYPE_GPU, "AMD", {
        { "Tahiti",           { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
      }
    },
    { // Intel GPUs
      CL_DEVICE_TYPE_GPU, "Intel", {
      }
    },
    { // Default
      CL_DEVICE_TYPE_ALL, kDefault, {
        { kDefault,           { {"WGS1",64}, {"WPT1",1}, {"WGS2",64}, {"WPT2",1}, {"VW2",1}, {"WGS3",64}, {"WPT3",1}, {"VW3",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
