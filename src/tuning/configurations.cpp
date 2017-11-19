
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

#include <vector>
#include <string>

#include "tuning/configurations.hpp"

namespace clblast {
// =================================================================================================

// Finds all configurations. It also applies the user-defined constraints within.
std::vector<Configuration> SetConfigurations(const std::vector<Parameter> parameters,
                                             const Constraints& constraints) {
  auto config = Configuration();
  auto configurations = std::vector<Configuration>();
  PopulateConfigurations(parameters, 0, config, configurations, constraints);
  return configurations;
}

// Iterates recursively over all permutations of the user-defined parameters
void PopulateConfigurations(const std::vector<Parameter> &parameters,
                            const size_t index, const Configuration &config,
                            std::vector<Configuration> &configuration,
                            const Constraints& constraints) {

  // End of the chain: all parameters are considered, store the resulting configuration if it is a
  // valid one according to the constraints
  if (index == parameters.size()) {
    if (ValidConfiguration(config, constraints)) {
      configuration.push_back(config);
    }
    return;
  }

  // This loop iterates over all values of the current parameter and calls this function recursively
  Parameter parameter = parameters[index];
  for (auto &value: parameter.second) {
    auto config_copy = config;
    config_copy[parameter.first] = value;
    PopulateConfigurations(parameters, index+1, config_copy, configuration, constraints);
  }
}

// Loops over all user-defined constraints to check whether or not the configuration is valid
bool ValidConfiguration(const Configuration &config,
                        const Constraints& constraints) {

  // Iterates over all constraints
  for (auto &constraint: constraints) {

    // Finds the values of the parameters
    auto values = std::vector<size_t>(constraint.parameters.size());
    for (auto i=size_t{0}; i<constraint.parameters.size(); ++i) {
      values[i] = config.at(constraint.parameters[i]);
    }

    // Checks this constraint for these values
    if (!constraint.valid_if(values)) {
      return false;
    }
  }

  // Everything was OK: this configuration is valid
  return true;
}

// Multiplies and/or dividers a thread configuration (local/global)
std::vector<size_t> SetThreadConfiguration(const Configuration& config,
                                           const std::vector<size_t> base,
                                           const TransformVector& mul_config,
                                           const TransformVector& div_config) {
  auto result = base;
  for (const auto &multipliers: mul_config) {
    for (auto i = size_t{0}; i < multipliers.size(); ++i) {
      result[i] *= config.at(multipliers[i]);
    }
  }
  for (const auto &dividers: div_config) {
    for (auto i = size_t{0}; i < dividers.size(); ++i) {
      result[i] /= config.at(dividers[i]);
    }
  }
  return result;
}

// =================================================================================================
} // namespace clblast
