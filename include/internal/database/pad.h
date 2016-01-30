
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Pad' kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::PadSingle = {
  "Pad", Precision::kSingle, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                          { {"PAD_DIMX",32}, {"PAD_DIMY",16}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
        { "default",                                       { {"PAD_DIMX",32}, {"PAD_DIMY",16}, {"PAD_WPTX",2}, {"PAD_WPTY",1} } },
      }
    },
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "default",                                       { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"PAD_DIMX",32}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::PadComplexSingle = {
  "Pad", Precision::kComplexSingle, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                          { {"PAD_DIMX",32}, {"PAD_DIMY",16}, {"PAD_WPTX",2}, {"PAD_WPTY",4} } },
        { "default",                                       { {"PAD_DIMX",32}, {"PAD_DIMY",16}, {"PAD_WPTX",2}, {"PAD_WPTY",4} } },
      }
    },
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "default",                                       { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::PadDouble = {
  "Pad", Precision::kDouble, {
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
        { "default",                                       { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"PAD_DIMX",16}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",2} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::PadComplexDouble = {
  "Pad", Precision::kComplexDouble, {
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
        { "default",                                       { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"PAD_DIMX",8}, {"PAD_DIMY",8}, {"PAD_WPTX",1}, {"PAD_WPTY",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
