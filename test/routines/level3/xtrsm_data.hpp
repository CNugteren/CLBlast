
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements data-prepration routines for proper input for the TRSM routine. Note: The
// data-preparation routines are taken from clBLAS
//
// =================================================================================================

#ifndef CLBLAST_TEST_ROUTINES_XTRSM_DATA_H_
#define CLBLAST_TEST_ROUTINES_XTRSM_DATA_H_

#include <vector>
#include <string>
#include <random>

#include "utilities/utilities.hpp"

namespace clblast {
// =================================================================================================

// Limits to prepare proper input data
template <typename T> double TrsmLimitMatA();
template <> double TrsmLimitMatA<float>() { return pow(2.0, 7); }
template <> double TrsmLimitMatA<double>() { return pow(2.0, 5); }
template <> double TrsmLimitMatA<float2>() { return TrsmLimitMatA<float>(); }
template <> double TrsmLimitMatA<double2>() { return TrsmLimitMatA<double>(); }
template <typename T> double TrsmLimitMatB();
template <> double TrsmLimitMatB<float>() { return pow(2.0, 16); }
template <> double TrsmLimitMatB<double>() { return pow(2.0, 47); }
template <> double TrsmLimitMatB<float2>() { return TrsmLimitMatB<float>(); }
template <> double TrsmLimitMatB<double2>() { return TrsmLimitMatB<double>(); }

// Matrix element setter
template <typename T>
void SetElement(const clblast::Layout layout,
                const size_t row, const size_t column, T *mat, const size_t ld, const T value)
{
  if (layout == clblast::Layout::kRowMajor) { mat[column + ld * row] = value; }
  else { mat[row + ld * column] = value; }
}

// Matrix element getter
template <typename T>
T GetElement(const clblast::Layout layout,
             const size_t row, const size_t column, const T *mat, const size_t ld)
{
  if (layout == clblast::Layout::kRowMajor) { return mat[column + ld * row]; }
  else { return mat[row + ld * column]; }
}

// Bounds a value between 'left' and 'right'. The random value is assumed to be between -1 and +1.
template<typename T>
T BoundRandom(const double rand_val, const double left, const double right)
{
  const auto value = Constant<T>(rand_val * (right - left));
  if (AbsoluteValue<T>(value) < 0.0) {
    return value - Constant<T>(left);
  }
  else {
    return value + Constant<T>(left);
  }
}

// The clBLAS function to generate proper input matrices for matrices A & B. Note that this routine
// should remain deterministic. Random values are therefore taken from the existing input, which
// is scaled between -1 and +1.
template <typename T>
void GenerateProperTrsmMatrices(const Arguments<T> &args, const int seed, T *mat_a, T *mat_b)
{
  // Random number generator
  std::mt19937 mt(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  const auto k = (args.side == Side::kLeft) ? args.m : args.n;

  // Determines: max(|a_{ii}|) and  min(|a_{ii}|)
  // Generates: a_{ii} which are constrainted by min/max
  auto min = ConstantZero<T>();
  if (args.diagonal ==  clblast::Diagonal::kUnit) {
    for (auto i = size_t{0}; i < k; ++i) {
      SetElement<T>(args.layout, i, i, mat_a, args.a_ld, ConstantOne<T>()); // must not be accessed
    }
  }
  else {
    auto max = Constant<T>(dist(mt) * TrsmLimitMatA<T>());
    if (AbsoluteValue(max) < 1.0) { max += Constant<T>(3.0); } // no zero's on the diagonal
    min = max / Constant<T>(100.0);
    SetElement<T>(args.layout, 0, 0, mat_a, args.a_ld, max);
    for (auto i = size_t{1}; i < k; ++i) {
      auto value = BoundRandom<T>(dist(mt), AbsoluteValue(min), AbsoluteValue(max));
      if (AbsoluteValue(value) == 0) {
        value = max;
      }
      SetElement<T>(args.layout, i, i, mat_a, args.a_ld, value);
    }
  }

  // Generates a_{ij} for all j <> i.
  for (auto i = size_t{0}; i < k; ++i) {
    auto sum = (args.diagonal == clblast::Diagonal::kUnit) ?
                                 AbsoluteValue(ConstantOne<T>()) :
                                 AbsoluteValue(GetElement<T>(args.layout, i, i, mat_a, args.a_ld));
    for (auto j = size_t{0}; j < k; ++j) {
      if (j == i) { continue; }
      auto value = ConstantZero<T>();
      if (((args.triangle == clblast::Triangle::kUpper) && (j > i)) ||
          ((args.triangle == clblast::Triangle::kLower) && (j < i))) {
        if (sum >= 1.0) {
          const auto limit = sum / std::sqrt(static_cast<double>(k) - static_cast<double>(j));
          value = Constant<T>(dist(mt) * limit);
          sum -= AbsoluteValue(value);
        }
      }
      SetElement<T>(args.layout, i, j, mat_a, args.a_ld, value);
    }
  }

  // Generate matrix B
  if (args.side == clblast::Side::kLeft) {
    for (auto j = size_t{0}; j < args.n; ++j) {
      auto sum = TrsmLimitMatB<T>();
      for (auto i = size_t{0}; i < args.m; ++i) {
        const auto a_value = GetElement<T>(args.layout, i, i, mat_a, args.a_ld);
        auto value = ConstantZero<T>();
        if (sum >= 0.0) {
          const auto limit = sum * AbsoluteValue(a_value) / std::sqrt(static_cast<double>(args.m) - static_cast<double>(i));
          value = Constant<T>(dist(mt) * limit);
          sum -= AbsoluteValue(value) / AbsoluteValue(a_value);
        }
        SetElement<T>(args.layout, i, j, mat_b, args.b_ld, value);
        if ((i == 0 && j == 0) || (AbsoluteValue(value) < AbsoluteValue(min))) {
          min = value;
        }
      }
    }
  }
  else {
    for (auto i = size_t{0}; i < args.m; ++i) {
      auto sum = TrsmLimitMatB<T>();
      for (auto j = size_t{0}; j < args.n; ++j) {
        const auto a_value = GetElement<T>(args.layout, j, j, mat_a, args.a_ld);
        auto value = ConstantZero<T>();
        if (sum >= 0.0) {
          const auto limit = sum * AbsoluteValue(a_value) / std::sqrt(static_cast<double>(args.n) - static_cast<double>(j));
          value = Constant<T>(dist(mt) * limit);
          sum -= AbsoluteValue(value) / AbsoluteValue(a_value);
        }
        SetElement<T>(args.layout, i, j, mat_b, args.b_ld, value);
        if ((i == 0 && j == 0) || (AbsoluteValue(value) < AbsoluteValue(min))) {
          min = value;
        }
      }
    }
  }
  if (args.diagonal == clblast::Diagonal::kUnit) {
    for (auto i = size_t{0}; i < k; ++i) {
      SetElement<T>(args.layout, i, i, mat_a, args.a_ld, ConstantOne<T>()); // must not be accessed
    }
  }

  // Calculate a proper alpha
  if (AbsoluteValue(min) > AbsoluteValue(args.alpha)) {
    // Not implemented
  }

  // Adjust matrix B according to the value of alpha
  if (AbsoluteValue(args.alpha) != 1.0 && AbsoluteValue(args.alpha) != 0.0) {
    for (auto i = size_t{0}; i < args.m; ++i) {
      for (auto j = size_t{0}; j < args.n; ++j) {
        auto value = GetElement<T>(args.layout, i, j, mat_b, args.b_ld);
        value /= args.alpha;
        SetElement<T>(args.layout, i, j, mat_b, args.b_ld, value);
      }
    }
  }
}

// =================================================================================================
} // namespace clblast

// CLBLAST_TEST_ROUTINES_XTRSM_DATA_H_
#endif
