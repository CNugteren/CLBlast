
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. It
// is auto-generated by the 'scripts/database/database.py' Python script.
//
// This file populates the database with best-found tuning parameters for the 'Xconvgemm32' kernels.
//
// =================================================================================================

namespace clblast {
namespace database {

const DatabaseEntry XconvgemmSingle = {
  "Xconvgemm", Precision::kSingle, {"KWID", "MDIMAD", "MDIMCD", "NDIMBD", "NDIMCD", "PADA", "PADB", "VWMD", "VWND", "WGD"}, {
    { // AMD GPUs
      kDeviceTypeGPU, "AMD", {
        { "default", {
          { Name{"AMD Radeon Pro 450 Compute Engine                 "}, Params{ 1, 8, 16, 16, 8, 0, 0, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 1, 8, 16, 8, 8, 0, 0, 2, 2, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1010:xnack-", {
          { Name{"AMD Radeon RX 5700 XT                             "}, Params{ 1, 8, 8, 16, 8, 0, 0, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 1, 8, 8, 16, 8, 0, 0, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1030", {
          { Name{"AMD Radeon RX 6800 XT                             "}, Params{ 1, 8, 16, 32, 8, 0, 0, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 1, 8, 16, 32, 8, 0, 0, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "gfx1034", {
          { Name{"AMD Radeon RX 6500 XT                             "}, Params{ 1, 8, 8, 8, 8, 0, 0, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 1, 8, 8, 8, 8, 0, 0, 2, 1, 32, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // Intel CPUs
      kDeviceTypeCPU, "Intel", {
        { "default", {
          { Name{"Intel(R) Core(TM) i9-9980HK CPU @ 2.40GHz         "}, Params{ 1, 16, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 1, 16, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // Intel GPUs
      kDeviceTypeGPU, "Intel", {
        { "default", {
          { Name{"Intel(R) Gen9 HD Graphics NEO                     "}, Params{ 1, 16, 32, 8, 8, 0, 0, 1, 4, 32, 0, 0, 0, 0, 0, 0 } },
          { Name{"Intel(R) HD Graphics Skylake ULT GT2              "}, Params{ 1, 16, 8, 8, 16, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 1, 8, 16, 16, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // Intel accelerators
      kDeviceTypeAccelerator, "Intel", {
        { "default", {
          { Name{"Intel(R) FPGA Emulation Device                    "}, Params{ 1, 8, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 1, 8, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // NVIDIA GPUs
      kDeviceTypeGPU, "NVIDIA", {
        { "SM7.0", {
          { Name{"Quadro GV100                                      "}, Params{ 1, 8, 32, 16, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { Name{"Tesla V100-PCIE-16GB                              "}, Params{ 1, 8, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 1, 8, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "SM7.5", {
          { Name{"Quadro T2000                                      "}, Params{ 1, 32, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { Name{"Tesla T4                                          "}, Params{ 1, 32, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 1, 32, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "SM8.0", {
          { Name{"A100-PCIE-40GB                                    "}, Params{ 1, 8, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
          { kDeviceNameDefault                                        , Params{ 1, 8, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
        } },
        { "default", {
          { kDeviceNameDefault                                        , Params{ 1, 8, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
    { // Default
      kDeviceTypeAll, "default", {
        { "default", {
          { kDeviceNameDefault                                        , Params{ 1, 8, 32, 32, 8, 0, 0, 1, 1, 32, 0, 0, 0, 0, 0, 0 } },
        } },
      }
    },
  }
};

} // namespace database
} // namespace clblast
