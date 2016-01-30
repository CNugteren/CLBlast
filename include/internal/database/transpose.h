
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Transpose' kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::TransposeSingle = {
  "Transpose", Precision::kSingle, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                          { {"TRA_DIM",8}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
        { "default",                                       { {"TRA_DIM",8}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
      }
    },
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"TRA_DIM",8}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
        { "default",                                       { {"TRA_DIM",8}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"TRA_DIM",8}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",4} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::TransposeComplexSingle = {
  "Transpose", Precision::kComplexSingle, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                          { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
        { "default",                                       { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",2} } },
      }
    },
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "default",                                       { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"TRA_DIM",8}, {"TRA_PAD",0}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::TransposeDouble = {
  "Transpose", Precision::kDouble, {
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
        { "default",                                       { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",1}, {"TRA_WPT",2} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::TransposeComplexDouble = {
  "Transpose", Precision::kComplexDouble, {
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
        { "default",                                       { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"TRA_DIM",16}, {"TRA_PAD",1}, {"TRA_SHUFFLE",0}, {"TRA_WPT",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
