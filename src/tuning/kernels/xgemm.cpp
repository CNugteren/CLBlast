
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file uses the CLTune auto-tuner to tune the xgemm OpenCL kernels. There are two variations:
// - V==1: This tests some limited set of tuning parameters exhaustively.
// - V==2: This tests a much larger set of tuning parameters by randomly sampling a subset.
//
// =================================================================================================

#include <string>
#include <vector>

#include "utilities/utilities.hpp"
#include "tuning/tuning.hpp"


namespace clblast {
// =================================================================================================

// See comment at top of file for a description of the class
template <typename T, int V>
class TuneXgemm {
 public:

  // The representative kernel and the source code
  // static std::string KernelFamily() { return (V==1) ? "xgemm_1" : "xgemm_2"; }
  static std::string KernelFamily() { switch(V){ case 1: return "xgemm_1"; case 2: case 3:return "xgemm_2";  }}
  static std::string KernelName() { return "Xgemm"; }
  static std::string GetSources() {
    return
      #include "../src/kernels/common.opencl"
      #include "../src/kernels/level3/xgemm_part1.opencl"
      #include "../src/kernels/level3/xgemm_part2.opencl"
      #include "../src/kernels/level3/xgemm_part3.opencl"
    ;
  }

  // The list of arguments relevant for this routine
  static std::vector<std::string> GetOptions() {
    return {kArgM, kArgN, kArgK, kArgAlpha, kArgBeta, kArgFraction,
     kArgHeuristicSelection, kArgMultiSearchStrategy, kArgPsoSwarmSize, 
     kArgPsoInfGlobal, kArgPsoInfLocal, kArgPsoInfRandom};
  }

  // Tests for valid arguments
  static void TestValidArguments(const Arguments<T> &) { }

  // Sets the default values for the arguments
  static size_t DefaultM() { return 1024; }
  static size_t DefaultN() { return 1024; }
  static size_t DefaultK() { return 1024; }
  static size_t DefaultBatchCount() { return 1; } // N/A for this kernel
  static double DefaultFraction() { return (V==1) ? 1.0 : 512.0; } // test all or sample randomly
  static size_t DefaultNumRuns() { return 2; } // run every kernel this many times for averaging
  static size_t DefaultNumSearchStragegy() { return 2; } // Full search and Random/PSO
  static size_t DefaultSwarmSizePSO() { return 8; } 
  static double DefaultInfluenceGlobalPSO(){ return 0.1; }
  static double DefaultInfluenceLocalPSO(){ return 0.3; }
  static double DefaultInfluenceRandomPSO(){ return 0.6; }
  static size_t DefaultHeuristic(){ return size_t{3};} // PSO
  static double DefaultMaxTempAnn(){ return 1.0;}
  
  // Describes how to obtain the sizes of the buffers
  static size_t GetSizeX(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeY(const Arguments<T> &) { return 1; } // N/A for this kernel
  static size_t GetSizeA(const Arguments<T> &args) { return args.m * args.k; }
  static size_t GetSizeB(const Arguments<T> &args) { return args.n * args.k; }
  static size_t GetSizeC(const Arguments<T> &args) { return args.m * args.n; }
  static size_t GetSizeTemp(const Arguments<T> &) { return 1; } // N/A for this kernel

  // Sets the tuning parameters and their possible values
  static void SetParameters(cltune::Tuner &tuner, const size_t id) {
    if (V==1) { // limited subset of tuning parameters - but explorable exhaustively
      tuner.AddParameter(id, "MWG", {16, 32, 64});
      tuner.AddParameter(id, "NWG", {16, 32, 64});
      tuner.AddParameter(id, "KWG", {32});
      tuner.AddParameter(id, "MDIMC", {8, 16, 32});
      tuner.AddParameter(id, "NDIMC", {8, 16, 32});
      tuner.AddParameter(id, "MDIMA", {8, 16, 32});
      tuner.AddParameter(id, "NDIMB", {8, 16, 32});
      tuner.AddParameter(id, "KWI", {2});
      tuner.AddParameter(id, "VWM", {1, 2, 4});
      tuner.AddParameter(id, "VWN", {1, 2, 4});
      tuner.AddParameter(id, "STRM", {0});
      tuner.AddParameter(id, "STRN", {0});
      tuner.AddParameter(id, "SA", {0, 1});
      tuner.AddParameter(id, "SB", {0, 1});
    } // a lot more tuning parameters - has to be sampled randomly, too much to test all
    else {
      //RANDOM_SEARCH & PSO
      tuner.AddParameter(id, "MWG", {16, 32, 64, 128});
      tuner.AddParameter(id, "NWG", {16, 32, 64, 128});
      tuner.AddParameter(id, "KWG", {16, 32});
      tuner.AddParameter(id, "MDIMC", {8, 16, 32});
      tuner.AddParameter(id, "NDIMC", {8, 16, 32});
      tuner.AddParameter(id, "MDIMA", {8, 16, 32});
      tuner.AddParameter(id, "NDIMB", {8, 16, 32});
      tuner.AddParameter(id, "KWI", {2});
      tuner.AddParameter(id, "VWM", {1, 2, 4, 8});
      tuner.AddParameter(id, "VWN", {1, 2, 4, 8});
      tuner.AddParameter(id, "STRM", {0, 1});
      tuner.AddParameter(id, "STRN", {0, 1});
      tuner.AddParameter(id, "SA", {0, 1});
      tuner.AddParameter(id, "SB", {0, 1});
    }

  }

  // Sets the constraints
  static void SetConstraints(cltune::Tuner &tuner, const size_t id) {
    auto MultipleOfX = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]); };
    auto MultipleOfXMulY = [] (std::vector<size_t> v) { return IsMultiple(v[0], v[1]*v[2]); };
    auto MultipleOfXMulYDivZ = [] (std::vector<size_t> v) { return IsMultiple(v[0], (v[1]*v[2])/v[3]); };
    // Requirement for unrolling the KWG loop
    tuner.AddConstraint(id, MultipleOfX, {"KWG", "KWI"});
    // Required for integer MWI and NWI
    tuner.AddConstraint(id, MultipleOfXMulY, {"MWG", "MDIMC", "VWM"});
    tuner.AddConstraint(id, MultipleOfXMulY, {"NWG", "NDIMC", "VWN"});
    // Required for integer MWIA and NWIB
    tuner.AddConstraint(id, MultipleOfXMulY, {"MWG", "MDIMA", "VWM"});
    tuner.AddConstraint(id, MultipleOfXMulY, {"NWG", "NDIMB", "VWN"});
    // KWG has to be a multiple of KDIMA = ((MDIMC*NDIMC)/(MDIMA)) and KDIMB = (...)
    tuner.AddConstraint(id, MultipleOfXMulYDivZ, {"KWG", "MDIMC", "NDIMC", "MDIMA"});
    tuner.AddConstraint(id, MultipleOfXMulYDivZ, {"KWG", "MDIMC", "NDIMC", "NDIMB"});

    // Extra constraints for variation 1 to limit the set of options significantly
    if (V==1) {
      auto IsEqual = [] (std::vector<size_t> v) { return v[0] == v[1]; };
      tuner.AddConstraint(id, IsEqual, {"MDIMC", "MDIMA"});
      tuner.AddConstraint(id, IsEqual, {"NDIMC", "NDIMB"});
      tuner.AddConstraint(id, IsEqual, {"SA", "SB"});
    }
  }

  // Sets the local memory size
  static void SetLocalMemorySize(cltune::Tuner &tuner, const size_t id, const Arguments<T> &args) {
    auto LocalMemorySize = [args] (std::vector<size_t> v) {
      return (((v[0]*v[1]*v[2]) + (v[3]*v[4]*v[5]))*GetBytes(args.precision));
    };
    tuner.SetLocalMemoryUsage(id, LocalMemorySize, {"SA", "KWG", "MWG",
                                                    "SB", "KWG", "NWG"});
  }

  // Sets the base thread configuration
  static std::vector<size_t> GlobalSize(const Arguments<T> &args) { return {args.m, args.n}; }
  static std::vector<size_t> GlobalSizeRef(const Arguments<T> &args) { return GlobalSize(args); }
  static std::vector<size_t> LocalSize() { return {1, 1}; }
  static std::vector<size_t> LocalSizeRef() { return {8, 8}; }

  // Transforms the thread configuration based on the parameters
  using TransformVector = std::vector<std::vector<std::string>>;
  static TransformVector MulLocal() { return {{"MDIMC", "NDIMC"}}; }
  static TransformVector DivLocal() { return {}; }
  static TransformVector MulGlobal() { return {{"MDIMC", "NDIMC"}}; }
  static TransformVector DivGlobal() { return {{"MWG", "NWG"}}; }

  // Sets the kernel's arguments
  static void SetArguments(cltune::Tuner &tuner, const Arguments<T> &args,
                           std::vector<T> &, std::vector<T> &,
                           std::vector<T> &a_mat, std::vector<T> &b_mat, std::vector<T> &c_mat,
                           std::vector<T> &) {
    tuner.AddArgumentScalar(static_cast<int>(args.m));
    tuner.AddArgumentScalar(static_cast<int>(args.n));
    tuner.AddArgumentScalar(static_cast<int>(args.k));
    tuner.AddArgumentScalar(GetRealArg(args.alpha));
    tuner.AddArgumentScalar(GetRealArg(args.beta));
    tuner.AddArgumentInput(a_mat);
    tuner.AddArgumentInput(b_mat);
    tuner.AddArgumentOutput(c_mat);
  }

  // Describes how to compute the performance metrics
  static size_t GetMetric(const Arguments<T> &args) {
    return 2 * args.m * args.n * args.k;
  }
  static std::string PerformanceUnit() { return "GFLOPS"; }
  
  // Returns which Heuristic to run 
  static size_t GetCurrentHeuristic(const Arguments<T> &args){

     //  Multi Search Strategy is enable
    if( args.multi_search_strategy){
      if( V == 1){ 
        return size_t{0};
      }
      else{
        return args.heuristic_selection;
      }
    }

    // Use full-search to explore all parameter combinations or random-search to search only a part of
    // the parameter values. The fraction is set as a command-line argument.
    if (args.fraction == 1.0 || args.fraction == 0.0) {
      return size_t{0}; // Full search
    }
    else {
      return args.heuristic_selection;
    }
  }
};

// =================================================================================================
} // namespace clblast

// Shortcuts to the clblast namespace
using half = clblast::half;
using float2 = clblast::float2;
using double2 = clblast::double2;

// Function to tune a specific variation V (not within the clblast namespace)
template <int V>
void StartVariation(int argc, char *argv[]) {
  const auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  switch(clblast::GetPrecision(command_line_args)) {
    case clblast::Precision::kHalf: clblast::Tuner<clblast::TuneXgemm<half,V>, half>(argc, argv); break;
    case clblast::Precision::kSingle: clblast::Tuner<clblast::TuneXgemm<float,V>, float>(argc, argv); break;
    case clblast::Precision::kDouble: clblast::Tuner<clblast::TuneXgemm<double,V>, double>(argc, argv); break;
    case clblast::Precision::kComplexSingle: clblast::Tuner<clblast::TuneXgemm<float2,V>, float2>(argc, argv); break;
    case clblast::Precision::kComplexDouble: clblast::Tuner<clblast::TuneXgemm<double2,V>, double2>(argc, argv); break;
  }
}

// Test multiple heuristics if kArgMultiSearchStrategy is enabled
// Otherwise, run the heuristic specified in kArgMultiSearchStrategy
void TestHeuristic(int argc, char *argv[]){

  auto command_line_args = clblast::RetrieveCommandLineArguments(argc, argv);
  auto help = std::string{""};
  auto heuristic_selected = clblast::GetArgument(command_line_args, help, clblast::kArgMultiSearchStrategy, 0); 
  auto multi_search_strategy = clblast::GetArgument(command_line_args, help, clblast::kArgMultiSearchStrategy, 0);

  if(multi_search_strategy){
    StartVariation<1>(argc, argv);
    StartVariation<2>(argc, argv);
  }
  else
  { 
    switch(heuristic_selected){
      case 1:
      case 2:
      case 3:
        StartVariation<2>(argc, argv);
        break;
      case 0:
      default:
        StartVariation<1>(argc, argv);
        break;
    }
  }
}

// Main function (not within the clblast namespace)
int main(int argc, char *argv[]) {

  TestHeuristic(argc, argv);
  return 0;
}

// =================================================================================================
