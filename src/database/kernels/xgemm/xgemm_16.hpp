
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. It
// is auto-generated by the 'scripts/database/database.py' Python script.
//
// This file populates the database with best-found tuning parameters for the 'Xgemm16' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {

const DatabaseEntry XgemmHalf = {
  "Xgemm", Precision::kHalf, {"GEMMK", "KREG", "KWG", "KWI", "MDIMA", "MDIMC", "MWG", "NDIMB", "NDIMC", "NWG", "SA", "SB", "STRM", "STRN", "VWM", "VWN"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "Ellesmere", {
          { Name{"AMD Radeon RX 480                                 "}, Params{ 0, 1, 32, 2, 8, 8, 64, 16, 16, 64, 1, 1, 0, 0, 4, 4 } },
          { kDeviceNameDefault                                        , Params{ 0, 1, 32, 2, 8, 8, 64, 16, 16, 64, 1, 1, 0, 0, 4, 4 } },
        } },
        { "default", {
          { kDeviceNameDefault                                        , Params{ 0, 1, 32, 2, 8, 8, 64, 16, 16, 64, 1, 1, 0, 0, 4, 4 } },
        } },
        { "gfx1010:xnack-", {
          { Name{"AMD Radeon RX 5700 XT                             "}, Params{ 0, 1, 32, 2, 8, 16, 128, 16, 16, 128, 1, 1, 1, 0, 8, 8 } },
          { kDeviceNameDefault                                        , Params{ 0, 1, 32, 2, 8, 16, 128, 16, 16, 128, 1, 1, 1, 0, 8, 8 } },
        } },
        { "gfx1030", {
          { Name{"AMD Radeon RX 6800 XT                             "}, Params{ 0, 1, 32, 2, 8, 16, 128, 16, 16, 128, 1, 1, 1, 0, 8, 8 } },
          { kDeviceNameDefault                                        , Params{ 0, 1, 32, 2, 8, 16, 128, 16, 16, 128, 1, 1, 1, 0, 8, 8 } },
        } },
        { "gfx1032", {
          { Name{"AMD Radeon RX 6600 XT                             "}, Params{ 0, 1, 32, 2, 8, 16, 128, 16, 16, 128, 1, 1, 1, 0, 8, 8 } },
          { kDeviceNameDefault                                        , Params{ 0, 1, 32, 2, 8, 16, 128, 16, 16, 128, 1, 1, 1, 0, 8, 8 } },
        } },
        { "gfx1034", {
          { Name{"AMD Radeon RX 6500 XT                             "}, Params{ 0, 1, 32, 2, 16, 16, 128, 32, 8, 128, 1, 1, 0, 0, 2, 4 } },
          { kDeviceNameDefault                                        , Params{ 0, 1, 32, 2, 16, 16, 128, 32, 8, 128, 1, 1, 0, 0, 2, 4 } },
        } },
      }
    },
    { // ARM GPUs
      kDeviceTypeGPU, "ARM", {
        { "default", {
          { Name{"Mali-T628                                         "}, Params{ 0, 1, 32, 2, 8, 16, 128, 8, 8, 32, 0, 1, 0, 1, 8, 4 } },
          { Name{"Mali-T760                                         "}, Params{ 0, 1, 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4 } },
          { kDeviceNameDefault                                        , Params{ 0, 1, 32, 2, 16, 16, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4 } },
        } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "default", {
          { Name{"Intel(R) HD Graphics 620                          "}, Params{ 0, 1, 32, 2, 8, 8, 64, 8, 8, 64, 0, 0, 0, 0, 1, 1 } },
          { Name{"Intel(R) HD Graphics Skylake ULT GT2              "}, Params{ 0, 1, 16, 2, 8, 8, 32, 16, 16, 128, 0, 1, 1, 0, 4, 8 } },
          { Name{"Intel(R) UHD Graphics 770                         "}, Params{ 0, 1, 32, 2, 8, 16, 64, 8, 8, 128, 1, 1, 0, 1, 1, 8 } },
          { kDeviceNameDefault                                        , Params{ 0, 1, 32, 2, 8, 8, 64, 16, 16, 64, 1, 1, 0, 0, 2, 2 } },
        } },
      }
    },
    { // QUALCOMM GPUs
      kDeviceTypeGPU, "QUALCOMM", {
        { "default", {
          { Name{"QUALCOMM Adreno(TM)                               "}, Params{ 0, 1, 32, 2, 8, 8, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4 } },
          { kDeviceNameDefault                                        , Params{ 0, 1, 32, 2, 8, 8, 64, 8, 8, 64, 1, 1, 0, 0, 4, 4 } },
        } },
        { "OpenCL C 3.0 Adreno(TM) 730", {
          { Name{"QUALCOMM Adreno(TM)                               "}, Params{ 0, 1, 32, 2, 32, 32, 128, 8, 8, 128, 0, 0, 0, 1, 2, 8 } },
          { kDeviceNameDefault                                        , Params{ 0, 1, 32, 2, 32, 32, 128, 8, 8, 128, 0, 0, 0, 1, 2, 8 } },
        } },
        { "OpenCL C 3.0 Adreno(TM) 740", {
          { Name{"QUALCOMM Adreno(TM)                               "}, Params{ 0, 1, 16, 2, 16, 16, 64, 8, 8, 128, 1, 0, 1, 1, 2, 8 } },
          { kDeviceNameDefault                                        , Params{ 0, 1, 16, 2, 16, 16, 64, 8, 8, 128, 1, 0, 1, 1, 2, 8 } },
        } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default", {
          { kDeviceNameDefault                                        , Params{ 0, 1, 32, 2, 8, 8, 64, 16, 16, 64, 1, 1, 0, 0, 4, 4 } },
        } },
      }
    },
  }
};

} // namespace database
} // namespace clblast
