
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Padtranspose' kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::PadtransposeSingle = {
  "Padtranspose", Precision::kSingle, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                          { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "default",                                       { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
      }
    },
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",2} } },
        { "default",                                       { {"PADTRA_PAD",1}, {"PADTRA_TILE",32}, {"PADTRA_WPT",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::PadtransposeComplexSingle = {
  "Padtranspose", Precision::kComplexSingle, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris",                                          { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
        { "default",                                       { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",2} } },
      }
    },
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "default",                                       { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"PADTRA_PAD",0}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::PadtransposeDouble = {
  "Padtranspose", Precision::kDouble, {
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "default",                                       { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::PadtransposeComplexDouble = {
  "Padtranspose", Precision::kComplexDouble, {
    { // NVIDIA Corporation GPUs
      kDeviceTypeGPU, "NVIDIA Corporation", {
        { "Tesla K40m",                                    { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
        { "default",                                       { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                       { {"PADTRA_PAD",1}, {"PADTRA_TILE",16}, {"PADTRA_WPT",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
