
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the parameter configurations for the CLBlast auto-tuner (taken from CLTune).
// This is only used for the optional tuner binaries and not part of the core of CLBlast.
//
// =================================================================================================

#ifndef CLBLAST_TUNING_CONFIGURATIONS_H_
#define CLBLAST_TUNING_CONFIGURATIONS_H_

#include <vector>
#include <string>
#include <map>

#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

using Configuration = std::map<std::string, size_t>;
using Parameter = std::pair<std::string, std::vector<size_t>>;
using TransformVector = std::vector<std::vector<std::string>>;

// Helper structure holding a constraint on parameters. This constraint consists of a constraint
// function object and a vector of parameter names represented as strings.
using ConstraintFunction = std::function<bool(std::vector<size_t>)>;
struct Constraint {
  ConstraintFunction valid_if;
  std::vector<std::string> parameters;
};
using Constraints = std::vector<Constraint>;

// As above, but for local memory size
using LocalMemSizeFunction = std::function<size_t(std::vector<size_t>)>;
struct LocalMemSizeInfo {
  LocalMemSizeFunction local_mem_size;
  std::vector<std::string> parameters;
};

// =================================================================================================

// Initializes an empty configuration (vector of name/value pairs) and kicks-off the recursive
// function to find all configurations. It also applies the user-defined constraints within.
std::vector<Configuration> SetConfigurations(const Device& device,
                                             const std::vector<Parameter> parameters,
                                             const std::vector<size_t>& local_size_base,
                                             const TransformVector& mul_local_config,
                                             const TransformVector& div_local_config,
                                             const Constraints& constraints,
                                             const LocalMemSizeInfo& local_mem_size_info);

// Iterates recursively over all permutations of the user-defined parameters. This code creates
// multiple chains, in which each chain selects a unique combination of values for all parameters.
// At the end of each chain (when all parameters are considered), the function stores the result
// into the configuration list.
void PopulateConfigurations(const std::vector<Parameter> &parameters,
                            const std::vector<size_t> local_size_base,
                            const TransformVector& mul_local_config,
                            const TransformVector& div_local_config,
                            const size_t index, const Configuration &config,
                            std::vector<Configuration> &configuration,
                            const size_t local_mem_max,
                            const Constraints& constraints,
                            const LocalMemSizeInfo& local_mem_size_info,
                            const std::vector<size_t>& max_work_item_sizes,
                            const size_t max_work_group_size);

// Loops over all user-defined constraints to check whether or not the configuration is valid.
// Assumes initially all configurations are valid, then returns false if one of the constraints has
// not been met. Constraints consist of a user-defined function and a list of parameter names, which
// are replaced by parameter values in this function.
bool ValidConfiguration(const Configuration &config,
                        const size_t local_mem_max,
                        const Constraints& constraints,
                        const LocalMemSizeInfo& local_mem_size_info,
                        const std::vector<size_t> local_size_base,
                        const TransformVector& mul_local_config,
                        const TransformVector& div_local_config,
                        const std::vector<size_t>& max_work_item_sizes,
                        const size_t max_work_group_size);

// Processes multipliers and dividers to obtain the final thread configuration
std::vector<size_t> SetThreadConfiguration(const Configuration& config,
                                           const std::vector<size_t> base,
                                           const TransformVector& mul_config,
                                           const TransformVector& div_config);

// =================================================================================================
} // namespace clblast

// CLBLAST_TUNING_CONFIGURATIONS_H_
#endif
