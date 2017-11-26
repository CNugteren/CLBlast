
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
//
// =================================================================================================

#include <string>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <vector>

#include "kernel_preprocessor.hpp"

namespace clblast {
// =================================================================================================

void FindReplace(std::string &subject, const std::string &search, const std::string &replace)
{
  auto pos = size_t{0};
  while ((pos = subject.find(search, pos)) != std::string::npos) {
    subject.replace(pos, search.length(), replace);
    pos += replace.length();
  }
}

bool EvaluateCondition(std::string condition,
                       const std::unordered_map<std::string, int> &defines) {

  // Replace macros in the string
  for (const auto &define : defines) {
    FindReplace(condition, define.first, std::to_string(define.second));
  }

  // Process the equality sign
  const auto equal_pos = condition.find(" == ");
  if (equal_pos != std::string::npos) {
    const auto left = condition.substr(0, equal_pos);
    const auto right = condition.substr(equal_pos + 4);
    return (left == right);
  }
  return false; // unknown error
}

// =================================================================================================

// First pass: detect defines and comments
std::vector<std::string> PreprocessDefinesAndComments(const std::string& source,
                                                      std::unordered_map<std::string, int>& defines) {
  auto lines = std::vector<std::string>();

  // Parse the input string into a vector of lines
  auto disabled = false;
  auto source_stream = std::stringstream(source);
  auto line = std::string{""};
  while (std::getline(source_stream, line)) {

    // Decide whether or not to remain in 'disabled' mode
    if (line.find("#endif") != std::string::npos ||
        line.find("#elif") != std::string::npos) {
      disabled = false;
    }
    if (line.find("#else") != std::string::npos) {
      disabled = !disabled;
    }

    // Not in a disabled-block
    if (!disabled) {

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
        const auto value = define.substr(value_pos + 1);
        const auto name = define.substr(0, value_pos);
        defines.emplace(name, std::stoi(value));
        //continue;
      }

      // Detect #ifndef blocks
      const auto ifndef_pos = line.find("#ifndef ");
      if (ifndef_pos != std::string::npos) {
        const auto define = line.substr(ifndef_pos + 8); // length of "#ifndef "
        if (defines.find(define) != defines.end()) { disabled = true; }
        continue;
      }

      // Detect #ifdef blocks
      const auto ifdef_pos = line.find("#ifdef ");
      if (ifdef_pos != std::string::npos) {
        const auto define = line.substr(ifdef_pos + 7); // length of "#ifdef "
        if (defines.find(define) == defines.end()) { disabled = true; }
        continue;
      }

      // Detect #if blocks
      const auto if_pos = line.find("#if ");
      if (if_pos != std::string::npos) {
        const auto condition = line.substr(if_pos + 4); // length of "#if "
        if (!EvaluateCondition(condition, defines)) { disabled = true; }
        continue;
      }

      // Detect #elif blocks
      const auto elif_pos = line.find("#elif ");
      if (elif_pos != std::string::npos) {
        const auto condition = line.substr(elif_pos + 6); // length of "#elif "
        if (!EvaluateCondition(condition, defines)) { disabled = true; }
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

inline void SubstituteDefines(const std::unordered_map<std::string, int>& defines,
                              std::string& source_string) {
  if (defines.count(source_string) == 1) {
    source_string = ToString(defines.at(source_string));
  }
}

// Second pass: unroll loops
std::vector<std::string> PreprocessUnrollLoops(const std::vector<std::string>& source_lines,
                                               const std::unordered_map<std::string, int>& defines) {
  auto lines = std::vector<std::string>();

  auto brackets = 0;
  auto unroll_next_loop = false;

  for (auto line_id = size_t{0}; line_id < source_lines.size(); ++line_id) {
    const auto line = source_lines[line_id];

    // Detect #pragma unroll directives
    if (line.find("#pragma unroll") != std::string::npos) {
      unroll_next_loop = true;
      continue;
    }

    // Brackets
    brackets += std::count(line.begin(), line.end(), '{');
    brackets -= std::count(line.begin(), line.end(), '}');

    // Loop unrolling assuming it to be in the form "for (int w = 0; w < 4; w += 1) {"
    if (unroll_next_loop) {
      unroll_next_loop = false;

      // Parses loop structure
      const auto for_pos = line.find("for (");
      if (for_pos == std::string::npos) { throw Error<std::runtime_error>("Mis-formatted for-loop #0"); }
      const auto remainder = line.substr(for_pos + 5); // length of "for ("
      const auto line_split = split(remainder, ' ');
      if (line_split.size() != 11) { throw Error<std::runtime_error>("Mis-formatted for-loop #1"); }

      // Retrieves loop information (and checks for assumptions)
      const auto variable_type = line_split[0];
      const auto variable_name = line_split[1];
      if (variable_name != line_split[4]) { throw Error<std::runtime_error>("Mis-formatted for-loop #2"); }
      if (variable_name != line_split[7]) { throw Error<std::runtime_error>("Mis-formatted for-loop #3"); }
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
      const auto loop_start = std::stoi(loop_start_string);
      const auto loop_end = std::stoi(loop_end_string);
      const auto loop_increment = std::stoi(loop_increment_string);
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
          FindReplace(loop_line, variable_name, ToString(loop_iter));
          lines.emplace_back(loop_line);
          line_id++;
        }
        line_id--;
      }
    }
    else {
      lines.emplace_back(line);
    }
  }
  return lines;
}

// =================================================================================================

std::string PreprocessKernelSource(const std::string& kernel_source) {

  // Retrieves the defines and removes comments from the source lines
  auto defines = std::unordered_map<std::string, int>();
  auto lines = PreprocessDefinesAndComments(kernel_source, defines);

  // Unrolls loops (single level each call)
  lines = PreprocessUnrollLoops(lines, defines);
  lines = PreprocessUnrollLoops(lines, defines);

  // Gather the results
  auto processed_kernel = std::string{""};
  for (const auto& line : lines) {
    processed_kernel += line + "\n";
  }
  return processed_kernel;
}

// =================================================================================================
} // namespace clblast
