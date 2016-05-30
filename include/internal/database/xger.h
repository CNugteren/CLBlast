
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Database generator <database.py>
//
// This file populates the database with best-found tuning parameters for the 'Xger' kernels.
//
// =================================================================================================

namespace clblast {
// =================================================================================================

const Database::DatabaseEntry Database::XgerHalf = {
  "Xger", Precision::kHalf, {
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Intel(R) HD Graphics Skylake ULT GT2",            { {"WGS1",64}, {"WGS2",1}, {"WPT",1} } },
        { "default",                                         { {"WGS1",64}, {"WGS2",1}, {"WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",64}, {"WGS2",1}, {"WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgerSingle = {
  "Xger", Precision::kSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",256}, {"WGS2",1}, {"WPT",1} } },
        { "Hawaii",                                          { {"WGS1",64}, {"WGS2",2}, {"WPT",1} } },
        { "Pitcairn",                                        { {"WGS1",64}, {"WGS2",1}, {"WPT",1} } },
        { "Tahiti",                                          { {"WGS1",256}, {"WGS2",1}, {"WPT",1} } },
        { "default",                                         { {"WGS1",64}, {"WGS2",1}, {"WPT",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"WGS1",64}, {"WGS2",4}, {"WPT",4} } },
        { "default",                                         { {"WGS1",64}, {"WGS2",4}, {"WPT",4} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",128}, {"WGS2",2}, {"WPT",4} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"WGS1",128}, {"WGS2",1}, {"WPT",4} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",1}, {"WPT",4} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris Pro",                                        { {"WGS1",64}, {"WGS2",1}, {"WPT",4} } },
        { "default",                                         { {"WGS1",64}, {"WGS2",1}, {"WPT",4} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 480",                                 { {"WGS1",256}, {"WGS2",1}, {"WPT",4} } },
        { "GeForce GTX 680",                                 { {"WGS1",128}, {"WGS2",1}, {"WPT",4} } },
        { "GeForce GTX TITAN",                               { {"WGS1",32}, {"WGS2",4}, {"WPT",2} } },
        { "default",                                         { {"WGS1",32}, {"WGS2",1}, {"WPT",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",32}, {"WGS2",1}, {"WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgerComplexSingle = {
  "Xger", Precision::kComplexSingle, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",64}, {"WGS2",4}, {"WPT",1} } },
        { "Hawaii",                                          { {"WGS1",64}, {"WGS2",1}, {"WPT",2} } },
        { "Pitcairn",                                        { {"WGS1",128}, {"WGS2",2}, {"WPT",1} } },
        { "Tahiti",                                          { {"WGS1",64}, {"WGS2",2}, {"WPT",1} } },
        { "default",                                         { {"WGS1",64}, {"WGS2",1}, {"WPT",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"WGS1",128}, {"WGS2",1}, {"WPT",1} } },
        { "default",                                         { {"WGS1",128}, {"WGS2",1}, {"WPT",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",256}, {"WGS2",1}, {"WPT",4} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"WGS1",512}, {"WGS2",4}, {"WPT",2} } },
        { "default",                                         { {"WGS1",256}, {"WGS2",1}, {"WPT",2} } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "Iris Pro",                                        { {"WGS1",16}, {"WGS2",2}, {"WPT",4} } },
        { "default",                                         { {"WGS1",16}, {"WGS2",2}, {"WPT",4} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 480",                                 { {"WGS1",128}, {"WGS2",2}, {"WPT",2} } },
        { "GeForce GTX 680",                                 { {"WGS1",32}, {"WGS2",4}, {"WPT",2} } },
        { "GeForce GTX TITAN",                               { {"WGS1",16}, {"WGS2",16}, {"WPT",2} } },
        { "default",                                         { {"WGS1",16}, {"WGS2",2}, {"WPT",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",16}, {"WGS2",1}, {"WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgerDouble = {
  "Xger", Precision::kDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",32}, {"WGS2",4}, {"WPT",1} } },
        { "Hawaii",                                          { {"WGS1",32}, {"WGS2",4}, {"WPT",2} } },
        { "Pitcairn",                                        { {"WGS1",64}, {"WGS2",1}, {"WPT",1} } },
        { "Tahiti",                                          { {"WGS1",64}, {"WGS2",2}, {"WPT",1} } },
        { "default",                                         { {"WGS1",32}, {"WGS2",1}, {"WPT",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"WGS1",64}, {"WGS2",4}, {"WPT",1} } },
        { "default",                                         { {"WGS1",64}, {"WGS2",4}, {"WPT",1} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",512}, {"WGS2",16}, {"WPT",1} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"WGS1",512}, {"WGS2",8}, {"WPT",2} } },
        { "default",                                         { {"WGS1",512}, {"WGS2",8}, {"WPT",1} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 480",                                 { {"WGS1",32}, {"WGS2",4}, {"WPT",2} } },
        { "GeForce GTX 680",                                 { {"WGS1",128}, {"WGS2",4}, {"WPT",2} } },
        { "GeForce GTX TITAN",                               { {"WGS1",16}, {"WGS2",8}, {"WPT",2} } },
        { "default",                                         { {"WGS1",16}, {"WGS2",4}, {"WPT",2} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",16}, {"WGS2",1}, {"WPT",1} } },
      }
    },
  }
};

// =================================================================================================

const Database::DatabaseEntry Database::XgerComplexDouble = {
  "Xger", Precision::kComplexDouble, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "AMD Radeon R9 M370X Compute Engine",              { {"WGS1",64}, {"WGS2",1}, {"WPT",1} } },
        { "Hawaii",                                          { {"WGS1",128}, {"WGS2",1}, {"WPT",1} } },
        { "Pitcairn",                                        { {"WGS1",64}, {"WGS2",4}, {"WPT",1} } },
        { "Tahiti",                                          { {"WGS1",32}, {"WGS2",4}, {"WPT",1} } },
        { "default",                                         { {"WGS1",32}, {"WGS2",1}, {"WPT",1} } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "Mali-T628",                                       { {"WGS1",64}, {"WGS2",2}, {"WPT",4} } },
        { "default",                                         { {"WGS1",64}, {"WGS2",2}, {"WPT",4} } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz",        { {"WGS1",512}, {"WGS2",4}, {"WPT",2} } },
        { "Intel(R) Core(TM) i7-5930K CPU @ 3.50GHz",        { {"WGS1",256}, {"WGS2",1}, {"WPT",2} } },
        { "default",                                         { {"WGS1",256}, {"WGS2",1}, {"WPT",2} } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "GeForce GTX 480",                                 { {"WGS1",64}, {"WGS2",2}, {"WPT",2} } },
        { "GeForce GTX 680",                                 { {"WGS1",8}, {"WGS2",16}, {"WPT",1} } },
        { "GeForce GTX TITAN",                               { {"WGS1",32}, {"WGS2",4}, {"WPT",2} } },
        { "default",                                         { {"WGS1",8}, {"WGS2",2}, {"WPT",1} } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default",                                         { {"WGS1",8}, {"WGS2",1}, {"WPT",1} } },
      }
    },
  }
};

// =================================================================================================
} // namespace clblast
