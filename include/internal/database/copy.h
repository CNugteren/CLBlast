
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Copy' kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::CopySingle = {
  "Copy", Precision::kSingle, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                          { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",2} } },
        { "default",                                       { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",2} } },
      }
    },
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",4}, {"COPY_WPT",2} } },
        { "default",                                       { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",4}, {"COPY_WPT",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",2} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::CopyComplexSingle = {
  "Copy", Precision::kComplexSingle, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                          { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",2} } },
        { "default",                                       { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",2} } },
      }
    },
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "default",                                       { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"COPY_DIMX",16}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::CopyDouble = {
  "Copy", Precision::kDouble, {
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",2} } },
        { "default",                                       { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",2}, {"COPY_WPT",2} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::CopyComplexDouble = {
  "Copy", Precision::kComplexDouble, {
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
        { "default",                                       { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"COPY_DIMX",8}, {"COPY_DIMY",8}, {"COPY_VW",1}, {"COPY_WPT",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
