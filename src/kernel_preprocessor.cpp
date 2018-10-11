
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the OpenCL kernel preprocessor (see the header for more information).
//
// Restrictions:
// - Use comments only single-line "//" style, not "/*" and "*/"
// - Don't use strings with characters parsed (e.g. '//', '}', '#ifdef')
// - Supports conditionals: #if #ifdef #ifndef #else #elif #endif
// - ...with the operators: ==
// - "#pragma unroll" requires next loop in the form "for (int w = 0; w < 4; w += 1) {"
//   The above also requires the spaces in that exact form
// - The loop variable should be a unique string within the code in the for-loop body (e.g. don't
//   use 'i' or 'w' but rather '_w' or a longer name.
// - The pragma "#pragma promote_to_registers" unrolls an array into multiple scalar values. The
//   name of this scalar should be unique (see above).
//
// =================================================================================================

#include <string>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <map>
#include <vector>

#include "kernel_preprocessor.hpp"

namespace clblast {
// =================================================================================================

struct compare_longer_string {
  bool operator() (const std::string &lhs, const std::string &rhs) const {
    if (lhs.size() > rhs.size()) { return true; }
    if (lhs.size() < rhs.size()) { return false; }
    return lhs < rhs;
  }
};

using DefinesIntMap = std::map<std::string, int, compare_longer_string>;
using DefinesStringMap = std::map<std::string, std::string, std::greater<std::string>>;

void RaiseError(const std::string& source_line, const std::string& exception_message) {
  printf("[OpenCL pre-processor] Error in source line: %s\n", source_line.c_str());
  throw Error<std::runtime_error>(exception_message);
}

// =================================================================================================

bool HasOnlyDigits(const std::string& str) {
  if (str == "") { return false; }
  return str.find_first_not_of(" 0123456789") == std::string::npos;
}

// Simple unsigned integer math parser
int ParseMath(const std::string& str) {

  // Handles brackets
  if (str.find(")") != std::string::npos) {
    const auto split_close = split(str, ')');
    const auto split_end = split(split_close[0], '(');
    if (split_end.size() < 2) { RaiseError(str, "Mismatching brackets #0"); }
    const auto bracket_contents = ParseMath(split_end[split_end.size() - 1]);
    auto before = std::string{};
    for (auto i = size_t{0}; i < split_end.size() - 1; ++i) {
      before += split_end[i];
      if (i != split_end.size() - 2) { before += "("; }
    }
    auto after = std::string{};
    for (auto i = size_t{1}; i < split_close.size(); ++i) {
      after += split_close[i];
      if (i != split_close.size() - 1) { after += ")"; }
    }
    return ParseMath(before + ToString(bracket_contents) + after);
  }

  // Handles addition
  const auto split_add = split(str, '+');
  if (split_add.size() == 2) {
    const auto lhs = ParseMath(split_add[0]);
    const auto rhs = ParseMath(split_add[1]);
    if (lhs == -1 || rhs == -1) { return -1; }
    return lhs + rhs;
  }

  // Handles multiplication
  const auto split_mul = split(str, '*');
  if (split_mul.size() == 2) {
    const auto lhs = ParseMath(split_mul[0]);
    const auto rhs = ParseMath(split_mul[1]);
    if (lhs == -1 || rhs == -1) { return -1; }
    return lhs * rhs;
  }

  // Handles division
  const auto split_div = split(str, '/');
  if (split_div.size() == 2) {
    const auto lhs = ParseMath(split_div[0]);
    const auto rhs = ParseMath(split_div[1]);
    if (lhs == -1 || rhs == -1) { return -1; }
    return lhs / rhs;
  }

  // Handles the digits
  if (HasOnlyDigits(str)) {
    return std::stoi(str);
  }
  return -1; // error value
}


// Converts a string to an integer. The source line is printed in case an exception is raised.
size_t StringToDigit(const std::string& str, const std::string& source_line) {
  const auto result = ParseMath(str);
  if (result == -1) { RaiseError(source_line, "Not a digit: " + str); }
  return static_cast<size_t>(result);
}


// =================================================================================================

void FindReplace(std::string &subject, const std::string &search, const std::string &replace)
{
  auto pos = size_t{0};
  while ((pos = subject.find(search, pos)) != std::string::npos) {
    subject.replace(pos, search.length(), replace);
    pos += replace.length();
  }
}

void SubstituteDefines(const DefinesIntMap& defines,
                       std::string& source_string) {
  for (const auto &define : defines) {
    FindReplace(source_string, define.first, std::to_string(define.second));
  }
}

bool EvaluateCondition(std::string condition,
                       const DefinesIntMap &defines,
                       const DefinesStringMap &defines_string) {

  // Replace macros in the string
  SubstituteDefines(defines, condition);

  // Process the or sign
  const auto or_pos = condition.find(" || ");
  if (or_pos != std::string::npos) {
    const auto left = condition.substr(0, or_pos);
    const auto right = condition.substr(or_pos + 4);
    return EvaluateCondition(left, defines, defines_string) ||
        EvaluateCondition(right, defines, defines_string);
  }

  // Process the and sign
  const auto and_pos = condition.find(" && ");
  if (and_pos != std::string::npos) {
    const auto left = condition.substr(0, and_pos);
    const auto right = condition.substr(and_pos + 4);
    return EvaluateCondition(left, defines, defines_string) &&
        EvaluateCondition(right, defines, defines_string);
  }

  // Process the defined() construct
  const auto defined_pos = condition.find("defined(");
  if (defined_pos != std::string::npos) {
    const auto contents = condition.substr(defined_pos + 8);
    const auto defined_split = split(contents, ')');
    const auto defined_val = defined_split[0];
    return (defines_string.find(defined_val) != defines_string.end());
  }

  // Process the equality sign
  const auto equal_pos = condition.find(" == ");
  if (equal_pos != std::string::npos) {
    const auto left = condition.substr(0, equal_pos);
    const auto right = condition.substr(equal_pos + 4);
    return (left == right);
  }

  // Process the not equal sign
  const auto not_equal_pos = condition.find(" != ");
  if (not_equal_pos != std::string::npos) {
    const auto left = condition.substr(0, not_equal_pos);
    const auto right = condition.substr(not_equal_pos + 4);
    return (left != right);
  }

  // Process the smaller than sign
  const auto smaller_than_pos = condition.find(" < ");
  if (smaller_than_pos != std::string::npos) {
    const auto left = condition.substr(0, smaller_than_pos);
    const auto right = condition.substr(smaller_than_pos + 3);
    return (left < right);
  }

  // Process the larger than sign
  const auto larger_than_pos = condition.find(" > ");
  if (larger_than_pos != std::string::npos) {
    const auto left = condition.substr(0, larger_than_pos);
    const auto right = condition.substr(larger_than_pos + 3);
    return (left > right);
  }

  printf("Warning unknown condition: %s\n", condition.c_str());
  return false; // unknown error
}

// =================================================================================================

// Array to register promotion, e.g. arr[w] to {arr_0, arr_1}
void ArrayToRegister(std::string &source_line, const DefinesIntMap& defines,
                     const std::unordered_map<std::string, size_t>& arrays_to_registers,
                     const size_t num_brackets) {

  for (const auto array_name_map : arrays_to_registers) {  // only if marked to be promoted

    // Outside of a function
    if (num_brackets == 0) {

      // Case 1: argument in a function declaration (e.g. 'void func(const float arr[2])')
      const auto array_pos = source_line.find(array_name_map.first + "[");
      if (array_pos != std::string::npos) {
        SubstituteDefines(defines, source_line);

        // Finds the full array declaration (e.g. 'const float arr[2]')
        const auto left_split = split(source_line, '(');
        auto arguments = left_split.size() >= 2 ? left_split[1] : source_line;
        const auto right_split = split(arguments, ')');
        arguments = right_split.size() >= 1 ? right_split[0] : arguments;
        const auto comma_split = split(arguments, ',');
        for (auto j = size_t{0}; j < comma_split.size(); ++j) {
          if (comma_split[j].find(array_name_map.first + "[") != std::string::npos) {

            // Retrieves the array index
            const auto left_square_split = split(comma_split[j], '[');
            if (left_square_split.size() < 2) { RaiseError(source_line, "Mis-formatted array declaration #A"); }
            const auto right_square_split = split(left_square_split[1], ']');
            if (right_square_split.size() < 1) { RaiseError(source_line, "Mis-formatted array declaration #B"); }
            auto array_index_string = right_square_split[0];
            const auto array_index = StringToDigit(array_index_string, source_line);

            // Creates the new string
            auto replacement = std::string{};
            for (auto index = size_t{0}; index < array_index; ++index) {
              replacement += left_square_split[0] + "_" + ToString(index);
              if (index != array_index - 1) { replacement += ","; }
            }

            // Performs the actual replacement
            FindReplace(source_line, comma_split[j], replacement);
          }
        }
      }
    }

    // Inside a function
    else {
      auto array_pos = source_line.find(array_name_map.first + "[");

      // Case 2: passed to another function (e.g. 'func(arr)')
      if (array_pos == std::string::npos) { // assumes case 2 and case 3 (below) cannot occur in one line
        auto bracket_split = split(source_line, '(');
        if (bracket_split.size() >= 2) {
          auto replacement = std::string{};
          for (auto i = size_t{0}; i < array_name_map.second; ++i) {
            replacement += array_name_map.first + "_" + ToString(i);
            if (i != array_name_map.second - 1) { replacement += ", "; }
          }
          FindReplace(source_line, array_name_map.first, replacement);
        }
      }

      // Case 2: used as an array (e.g. 'arr[w]')
      while (array_pos != std::string::npos) {

        // Retrieves the array index
        const auto loop_remainder = source_line.substr(array_pos);
        const auto loop_split = split(split(loop_remainder, '[')[1], ']');
        if (loop_split.size() < 2) { RaiseError(source_line, "Mis-formatted array declaration #C"); }
        auto array_index_string = loop_split[0];

        // Replaces the array with a register value
        SubstituteDefines(defines, array_index_string);
        const auto array_index = StringToDigit(array_index_string, source_line);
        FindReplace(source_line, array_name_map.first + "[" + loop_split[0] + "]",
                    array_name_map.first + "_" + ToString(array_index));

        // Performs an extra substitution if this array occurs another time in this line
        array_pos = source_line.find(array_name_map.first + "[");
      }
    }
  }
}

// =================================================================================================

// First pass: detect defines and comments
std::vector<std::string> PreprocessDefinesAndComments(const std::string& source,
                                                      DefinesIntMap& defines_int) {
  auto lines = std::vector<std::string>();
  auto defines_string = DefinesStringMap();

  // Parse the input string into a vector of lines
  const auto max_depth_defines = 30;
  auto disabled = std::vector<unsigned int>(max_depth_defines, 0);
  auto depth = size_t{0};
  std::stringstream source_stream(source);
  auto line = std::string{""};
  while (std::getline(source_stream, line)) {
    //printf("[@%zu] disabled=%d '%s'\n", depth, disabled[depth], line.c_str());

    // Decide whether or not to remain in 'disabled' mode
    // {0 => enabled, 1 => disabled, but could become enabled again later, 2 => disabled until #endif
    if (line.find("#endif") != std::string::npos) {
      disabled[depth] = 0;
    }
    if (line.find("#elif") != std::string::npos || line.find("#else") != std::string::npos) {
      if (disabled[depth] == 0) { disabled[depth] = 2; } // was enabled, now disabled until #endif
      if (disabled[depth] == 1) { disabled[depth] = 0; } // was disabled, now potentially enabled again
    }

    // Measures the depth of pre-processor defines
    if ((line.find("#ifndef ") != std::string::npos) ||
        (line.find("#ifdef ") != std::string::npos) ||
        (line.find("#if ") != std::string::npos)) {
      depth++;
      if (depth >= max_depth_defines) { throw Error<std::runtime_error>("too deep define nest"); }
    }
    if (line.find("#endif") != std::string::npos) {
      if (depth == 0) { throw Error<std::runtime_error>("incorrect define nest"); }
      depth--;
    }

    // Verifies whether this level or any level below is disabled
    auto is_disabled = false;
    for (auto d = size_t{0}; d <= depth; ++d) {
      if (disabled[d] >= 1) { is_disabled = true; }
    }

    // Not in a disabled-block
    if (!is_disabled) {

      // Skip empty lines
      if (line == "") { continue; }

      // Single line comments
      const auto comment_pos = line.find("//");
      if (comment_pos != std::string::npos) {
        if (comment_pos == 0) { continue; }
        line.erase(comment_pos);
      }

      // Detect #define macros
      const auto define_pos = line.find("#define ");
      if (define_pos != std::string::npos) {
        const auto define = line.substr(define_pos + 8); // length of "#define "
        const auto value_pos = define.find(" ");
        auto value = define.substr(value_pos + 1);
        const auto name = define.substr(0, value_pos);
        SubstituteDefines(defines_int, value);
        const auto value_int = ParseMath(value);
        if (value_int != -1) {
          defines_int.emplace(name, value_int);
        }
        defines_string.emplace(name, value);
      }

      // Detect #ifndef blocks
      const auto ifndef_pos = line.find("#ifndef ");
      if (ifndef_pos != std::string::npos) {
        const auto define = line.substr(ifndef_pos + 8); // length of "#ifndef "
        if (defines_string.find(define) != defines_string.end()) { disabled[depth] = 1; }
        continue;
      }

      // Detect #ifdef blocks
      const auto ifdef_pos = line.find("#ifdef ");
      if (ifdef_pos != std::string::npos) {
        const auto define = line.substr(ifdef_pos + 7); // length of "#ifdef "
        if (defines_string.find(define) == defines_string.end()) { disabled[depth] = 1; }
        continue;
      }

      // Detect #if blocks
      const auto if_pos = line.find("#if ");
      if (if_pos != std::string::npos) {
        const auto condition = line.substr(if_pos + 4); // length of "#if "
        if (!EvaluateCondition(condition, defines_int, defines_string)) { disabled[depth] = 1; }
        continue;
      }

      // Detect #elif blocks
      const auto elif_pos = line.find("#elif ");
      if (elif_pos != std::string::npos) {
        const auto condition = line.substr(elif_pos + 6); // length of "#elif "
        if (!EvaluateCondition(condition, defines_int, defines_string)) { disabled[depth] = 1; }
        continue;
      }

      // Discard #else and #endif statements
      if (line.find("#endif") != std::string::npos || line.find("#else") != std::string::npos) {
        continue;
      }

      lines.push_back(line);
    }
  }
  return lines;
}

// =================================================================================================

// Second pass: detect array-to-register promotion pragma's and replace declarations & function calls
std::vector<std::string> PreprocessUnrollLoops(const std::vector<std::string>& source_lines,
                                               const DefinesIntMap& defines,
                                               std::unordered_map<std::string, size_t>& arrays_to_registers) {
  auto lines = std::vector<std::string>();

  auto brackets = size_t{0};
  auto promote_next_array_to_registers = false;

  for (auto line_id = size_t{0}; line_id < source_lines.size(); ++line_id) {
    auto line = source_lines[line_id];

    // Detect #pragma promote_to_registers directives (unofficial pragma)
    if (line.find("#pragma promote_to_registers") != std::string::npos) {
      promote_next_array_to_registers = true;
      continue;
    }

    // Brackets
    brackets += std::count(line.begin(), line.end(), '{');
    brackets -= std::count(line.begin(), line.end(), '}');

    // Promote array declarations to registers
    if (promote_next_array_to_registers) {
      promote_next_array_to_registers = false;
      const auto line_split1 = split(line, '[');
      if (line_split1.size() != 2) { RaiseError(line, "Mis-formatted array declaration #0"); }
      const auto line_split2 = split(line_split1[1], ']');
      if (line_split2.size() != 2) { RaiseError(line, "Mis-formatted array declaration #1"); }
      auto array_size_string = line_split2[0];
      SubstituteDefines(defines, array_size_string);
      const auto array_size = StringToDigit(array_size_string, line);
      for (auto loop_iter = size_t{0}; loop_iter < array_size; ++loop_iter) {
        lines.emplace_back(line_split1[0] + "_" + ToString(loop_iter) + line_split2[1]);
      }

      // Stores the array name
      const auto array_name_split = split(line_split1[0], ' ');
      if (array_name_split.size() < 2) { RaiseError(line, "Mis-formatted array declaration #2"); }
      const auto array_name = array_name_split[array_name_split.size() - 1];
      arrays_to_registers[array_name] = array_size;
      // TODO: bracket count not used currently for scope checking
      continue;
    }

    // Regular line
    lines.emplace_back(line);
  }
  return lines;
}

// =================================================================================================

// Third pass: unroll loops and perform actual array-to-register promotion
std::vector<std::string> PreprocessUnrollLoops(const std::vector<std::string>& source_lines,
                                               const DefinesIntMap& defines,
                                               std::unordered_map<std::string, size_t>& arrays_to_registers,
                                               const bool array_to_register_promotion) {
  auto lines = std::vector<std::string>();

  auto brackets = size_t{0};
  auto unroll_next_loop = false;

  for (auto line_id = size_t{0}; line_id < source_lines.size(); ++line_id) {
    auto line = source_lines[line_id];

    // Detect #pragma unroll directives
    if (line.find("#pragma unroll") != std::string::npos) {
      unroll_next_loop = true;
      continue;
    }

    // Brackets
    const auto num_brackets_before = brackets;
    brackets += std::count(line.begin(), line.end(), '{');
    brackets -= std::count(line.begin(), line.end(), '}');

    // Loop unrolling assuming it to be in the form "for (int w = 0; w < 4; w += 1) {"
    if (unroll_next_loop) {
      unroll_next_loop = false;

      // Parses loop structure
      const auto for_pos = line.find("for (");
      if (for_pos == std::string::npos) { RaiseError(line, "Mis-formatted for-loop #0"); }
      const auto remainder = line.substr(for_pos + 5); // length of "for ("
      const auto line_split = split(remainder, ' ');
      if (line_split.size() != 11) { RaiseError(line, "Mis-formatted for-loop #1"); }

      // Retrieves loop information (and checks for assumptions)
      const auto variable_type = line_split[0];
      const auto variable_name = line_split[1];
      if (variable_name != line_split[4]) { RaiseError(line, "Mis-formatted for-loop #2"); }
      if (variable_name != line_split[7]) { RaiseError(line, "Mis-formatted for-loop #3"); }
      auto loop_start_string = line_split[3];
      auto loop_end_string = line_split[6];
      auto loop_increment_string = line_split[9];
      remove_character(loop_start_string, ';');
      remove_character(loop_end_string, ';');
      remove_character(loop_increment_string, ')');

      // Parses loop information
      SubstituteDefines(defines, loop_start_string);
      SubstituteDefines(defines, loop_end_string);
      SubstituteDefines(defines, loop_increment_string);
      const auto loop_start = StringToDigit(loop_start_string, line);
      const auto loop_end = StringToDigit(loop_end_string, line);
      const auto loop_increment = StringToDigit(loop_increment_string, line);
      auto indent = std::string{""};
      for (auto i = size_t{0}; i < for_pos; ++i) { indent += " "; }

      // Start of the loop
      line_id++;
      const auto loop_num_brackets = brackets;
      const auto line_id_start = line_id;
      for (auto loop_iter = loop_start; loop_iter < loop_end; loop_iter += loop_increment) {
        line_id = line_id_start;
        brackets = loop_num_brackets;
        lines.emplace_back(indent + "{");

        // Body of the loop
        //lines.emplace_back(indent + "  " + variable_type + " " + variable_name + " = " + ToString(loop_iter) + ";");
        while (brackets >= loop_num_brackets) {
          auto loop_line = source_lines[line_id];
          brackets += std::count(loop_line.begin(), loop_line.end(), '{');
          brackets -= std::count(loop_line.begin(), loop_line.end(), '}');

          // Regular variable substitution
          FindReplace(loop_line, variable_name, ToString(loop_iter));

          // Array to register promotion
          if (array_to_register_promotion) {
            ArrayToRegister(loop_line, defines, arrays_to_registers, num_brackets_before);
          }

          lines.emplace_back(loop_line);
          line_id++;
        }
        line_id--;
      }
    }
    else {

      // Array to register promotion
      if (array_to_register_promotion) {
        ArrayToRegister(line, defines, arrays_to_registers, num_brackets_before);
      }

      lines.emplace_back(line);
    }
  }
  return lines;
}

// =================================================================================================

std::string PreprocessKernelSource(const std::string& kernel_source) {

  // Retrieves the defines and removes comments from the source lines
  auto defines = DefinesIntMap();
  auto lines = PreprocessDefinesAndComments(kernel_source, defines);

  // Unrolls loops (single level each call)
  auto arrays_to_registers = std::unordered_map<std::string, size_t>();
  lines = PreprocessUnrollLoops(lines, defines, arrays_to_registers);
  lines = PreprocessUnrollLoops(lines, defines, arrays_to_registers, false);
  lines = PreprocessUnrollLoops(lines, defines, arrays_to_registers, false);
  lines = PreprocessUnrollLoops(lines, defines, arrays_to_registers, false);
  lines = PreprocessUnrollLoops(lines, defines, arrays_to_registers, false);
  lines = PreprocessUnrollLoops(lines, defines, arrays_to_registers, true);

  // Gather the results
  auto processed_kernel = std::string{""};
  for (const auto& line : lines) {
    processed_kernel += line + "\n";
  }

  // Debugging
  if (false) {
    for (auto i = size_t{0}; i < lines.size(); ++i) {
      printf("[%zu] %s\n", i, lines[i].c_str());
    }
  }
  return processed_kernel;
}

// =================================================================================================
} // namespace clblast
