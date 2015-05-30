/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/


#include <iostream>
#include <fstream>
#include <vector>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>

#include "config.h"

using namespace clMath;
namespace po = boost::program_options;

bool
Config::isSane()
{
    if (!hasFuncID_) {
        std::cerr << "Missing required options 'function'" << std::endl;
        return false;
    }

    return true;
}

void
Config::setOptDesc(
    po::options_description& opts,
    bool useDefaults)
{
    po::options_description genOpts("Generator Arguments");
    genOpts.add_options()
        ("cpp",
            (useDefaults ? po::value<std::string>()->default_value(cpp())
                         : po::value<std::string>()),
            "Output file name for C++ generated source")
        ("cl", po::value<std::string>(),
            "Output file name for OpenCL generated source")
        ("data",
            (useDefaults ? po::value<std::string>()->default_value("random")
                         : po::value<std::string>()),
            "Data generation pattern\n"
            "Format: {random | unit | sawtooth}")
        ( "skip-accuracy",
          "Don't generate code for accuracy check. Applicable if the program "
          "is needed only for performance measurement")
    ;

    po::options_description openclOpts("OpenCL Arguments");
    openclOpts.add_options()
        ("platform",
            (useDefaults ? po::value<std::string>()->default_value(platform())
                         : po::value<std::string>()),
            "Platform name")
        ("device",
            (useDefaults ? po::value<std::string>()->default_value(device())
                         : po::value<std::string>()),
            "Device name")
        ("build-options", po::value<std::string>(),
            "Build options")
    ;

    po::options_description kargsOpts("BLAS Arguments");
    kargsOpts.add_options()
        ("function,f", po::value<std::string>(),
            "Function name, mandatory\n"
            "Format: {s | d | c | z}{BLAS function}")
        ("order",
            (useDefaults ? po::value<clblasOrder>()->default_value(clblasRowMajor)
                         : po::value<clblasOrder>()),
            "Data ordering\n"
            "Format: {column | row}")
        ("side",
            (useDefaults ? po::value<clblasSide>()->default_value(clblasLeft)
                         : po::value<clblasSide>()),
            "The side matrix A is located relative to matrix B\n"
            "Format: {left | right}")
        ("uplo",
            (useDefaults ? po::value<clblasUplo>()->default_value(clblasUpper)
                         : po::value<clblasUplo>()),
            "Upper or lower triangle of matrix is being referenced\n"
            "Format: {upper | lower}")
        ("transA",
            (useDefaults ? po::value<clblasTranspose>()->default_value(clblasNoTrans)
                         : po::value<clblasTranspose>()),
            "Matrix A transposition operation\n"
            "Format: {n | t | c}")
        ("transB",
            (useDefaults ? po::value<clblasTranspose>()->default_value(clblasNoTrans)
                         : po::value<clblasTranspose>()),
            "Matrix B transposition operation\n"
            "Format: {n | t | c}")
        ("diag",
            (useDefaults ? po::value<clblasDiag>()->default_value(clblasNonUnit)
                         : po::value<clblasDiag>()),
            "Whether the matrix is unit triangular\n"
            "Format: {unit | nonunit}")
        ("M,M",
            (useDefaults ? po::value<size_t>()->default_value(256)
                         : po::value<size_t>()->default_value(256))
        )
        ("N,N",
            (useDefaults ? po::value<size_t>()->default_value(256)
                         : po::value<size_t>())
        )
        ("K,K",
            (useDefaults ? po::value<size_t>()->default_value(256)
                         : po::value<size_t>())
        )
        ("alpha",
            (useDefaults ? po::value<std::string>()->default_value("1")
                         : po::value<std::string>()),
            "Alpha multiplier\n"
            "Format: real[,imag]")
        ("beta",
            (useDefaults ? po::value<std::string>()->default_value("1")
                         : po::value<std::string>()),
            "Beta multiplier\n"
            "Format: real[,imag]")
        ("lda", po::value<size_t>(),
            "Leading dimension of the matrix A")
        ("ldb", po::value<size_t>(),
            "Leading dimension of the matrix B")
        ("ldc", po::value<size_t>(),
            "Leading dimension of the matrix C")
        ("offA",
            (useDefaults ? po::value<size_t>()->default_value(0)
                         : po::value<size_t>()),
            "Start offset in buffer of matrix A")
        ("offBX",
            (useDefaults ? po::value<size_t>()->default_value(0)
                         : po::value<size_t>()),
            "Start offset in buffer of matrix B or vector X")
        ("offCY",
            (useDefaults ? po::value<size_t>()->default_value(0)
                         : po::value<size_t>()),
            "Start offset in buffer of matrix C or vector Y")
        ("incx",
            (useDefaults ? po::value<int>()->default_value(1)
                         : po::value<int>()),
            "Increment in the array X")
        ("incy",
            (useDefaults ? po::value<int>()->default_value(1)
                         : po::value<int>()),
            "Increment in the array Y")
    ;

    po::options_description decompositionOpts("Decomposition Options");
    decompositionOpts.add_options()
        ("decomposition,d", po::value<std::string>(),
            "SubproblemDim\n"
            "Format: {subdims[0].x},{subdims[0].y},\n"
            "        {subdims[0].bwidth},\n"
            "        {subdims[1].x},{subdims[1].y},\n"
            "        {subdims[1].bwidth}")
        ("multikernel", useDefaults ? po::value<bool>()->default_value(false)
                                    : po::value<bool>(),
            "Allow division of one BLAS function between several kernels")
    ;

    opts.add(genOpts).add(openclOpts).add(kargsOpts).add(decompositionOpts);
}

bool
Config::loadConfig(const char* filename)
{
    po::options_description cfgOpts;
    setOptDesc(cfgOpts, false);

    if ((filename == NULL) || (*filename == '\0')) {
        return false;
    }

    try {
        std::ifstream in(filename);
        po::store(po::parse_config_file<char>(in, cfgOpts), vm);
        po::notify(vm);
    }
    catch (const po::invalid_command_line_syntax &err) {
#if BOOST_VERSION >= 104200
        switch (err.kind()) {
        case po::invalid_syntax::missing_parameter:
            std::cerr << "Missing argument for option `" << err.tokens()
                << "'" << std::endl;
            break;
        default:
            std::cerr << "Syntax error, kind " << int(err.kind())
                << std::endl;
            break;
        }
#else
        std::cerr << err.msg;
#endif
        return false;
    }
    catch (const po::validation_error &err) {
        std::cerr << err.what() << std::endl;
        return false;
    }
#if BOOST_VERSION >= 104200
    catch (const po::reading_file &err) {
        std::cerr << err.what() << std::endl;
        return false;
    }
#endif
    catch (const po::unknown_option &err) {
        std::cerr << err.what() << std::endl;
    }

    return applyOptions(vm, false);
}

bool
Config::parseCommandLine(int argc, char *argv[])
{

    po::options_description helpOpts("Application Arguments");
    helpOpts.add_options()
        ("config", po::value<std::string>()->default_value(defaultConfig_),
            "Configuration file")
        ("help,h", "Show this help message");
    po::options_description visibleOpts;
    visibleOpts.add(helpOpts);
    setOptDesc(visibleOpts, true);

    try {
        po::store(po::parse_command_line(argc, argv, visibleOpts), vm);
        po::notify(vm);
    }
    catch (const po::invalid_command_line_syntax &err) {
#if BOOST_VERSION >= 104200
        switch (err.kind()) {
        case po::invalid_syntax::missing_parameter:
            std::cerr << "Missing argument for option `" << err.tokens()
                << "'" << std::endl;
            break;
        default:
            std::cerr << "Syntax error, kind " << int(err.kind())
                << std::endl;
            break;
        };
#else
        std::cerr << err.msg;
#endif
        return false;
    }
    catch (const po::validation_error &err) {
        std::cerr << err.what() << std::endl;
        return false;
    }
    catch (const po::unknown_option &err) {
        std::cerr << err.what() << std::endl;
    }

    if (vm.count("help")) {
        std::cout << visibleOpts << std::endl;
        return false;
    }
    if (vm.count("config")) {
        loadConfig(vm["config"].as<std::string>().c_str());
    }

    return applyOptions(vm);
}

bool
Config::applyOptions(
    const po::variables_map& vm,
    bool stopOnError)
{
    bool rc;
    ArgMultiplier v;

    rc = true;

    if (vm.count("function")) {
        if (!setFunction(vm["function"].as<std::string>())) {
            std::cerr << "Invalid function name: " <<
                vm["function"].as<std::string>() << std::endl;
            return false;
        }
    }

    if (vm.count("cpp")) {
        setCpp(vm["cpp"].as<std::string>());
    }
    if (vm.count("cl")) {
        setCl(vm["cl"].as<std::string>());
    }
    if (vm.count("data")) {
        if (!setDataPattern(vm["data"].as<std::string>())) {
            std::cerr << "Invalid data pattern name" << std::endl;
            rc = false;
            if (stopOnError) {
                return false;
            }
        }
    }
    if (vm.count("skip-accuracy")) {
        setSkipAccuracy();
    }

    if (vm.count("platform")) {
        if (!setPlatform(vm["platform"].as<std::string>())) {
            std::cerr << "Invalid platform name" << std::endl;
            rc = false;
            if (stopOnError) {
                return false;
            }
        }
    }
    if (vm.count("device")) {
        if (!setDevice(vm["device"].as<std::string>())) {
            std::cerr << "Invalid device name" << std::endl;
            rc = false;
            if (stopOnError) {
                return false;
            }
        }
    }
    if (vm.count("build-options")) {
        setBuildOptions(vm["build-options"].as<std::string>());
    }

    if (vm.count("order")) {
        setOrder(vm["order"].as<clblasOrder>());
    }
    if (vm.count("side")) {
        setSide(vm["side"].as<clblasSide>());
    }
    if (vm.count("uplo")) {
        setUplo(vm["uplo"].as<clblasUplo>());
    }
    if (vm.count("transA")) {
        setTransA(vm["transA"].as<clblasTranspose>());
    }
    if (vm.count("transB")) {
        setTransB(vm["transB"].as<clblasTranspose>());
    }
    if (vm.count("diag")) {
        setDiag(vm["diag"].as<clblasDiag>());
    }
    if (vm.count("M")) {
        setM(vm["M"].as<size_t>());
    }
    if (vm.count("N")) {
        setN(vm["N"].as<size_t>());
    }
    if (vm.count("K")) {
        setK(vm["K"].as<size_t>());
    }
    if (vm.count("alpha")) {
        if (!parseArgMultiplier(vm["alpha"].as<std::string>(), v)) {
            std::cerr << "in option 'alpha': invalid option value" << std::endl;
            rc = false;
            if (stopOnError) {
                return false;
            }
        }
        setAlpha(v);
    }
    if (vm.count("beta")) {
        if (!parseArgMultiplier(vm["beta"].as<std::string>(), v)) {
            std::cerr << "in option 'beta': invalid option value" << std::endl;
            rc = false;
            if (stopOnError) {
                return false;
            }
        }
        setBeta(v);
    }
    if (vm.count("lda")) {
        setLDA(vm["lda"].as<size_t>());
    }
    if (vm.count("ldb")) {
        setLDB(vm["ldb"].as<size_t>());
    }
    if (vm.count("ldc")) {
        setLDC(vm["ldc"].as<size_t>());
    }
    if (vm.count("offA")) {
        setOffA(vm["offA"].as<size_t>());
    }
    if (vm.count("offBX")) {
        setOffBX(vm["offBX"].as<size_t>());
    }
    if (vm.count("offCY")) {
        setOffCY(vm["offCY"].as<size_t>());
    }
    if (vm.count("incx")) {
        setIncX(vm["incx"].as<int>());
    }
    if (vm.count("incy")) {
        setIncY(vm["incy"].as<int>());
    }

    if (vm.count("decomposition")) {
        if (!parseDecompositionOpt(vm["decomposition"].as<std::string>())) {
            std::cerr << "in option 'decomposition': invalid option value" << std::endl;
            rc = false;
            if (stopOnError) {
                return false;
            }
        }
    }

    if (vm.count("multikernel")) {
        setMultiKernel(vm["multikernel"].as<bool>());
    }

    return rc;
}

std::istream& operator>>(std::istream& in, clblasOrder& order)
{
    std::string token;

    in >> token;
    if (token == "row") {
        order = clblasRowMajor;
    }
    else if (token == "column") {
        order = clblasColumnMajor;
    }
    else {
#if BOOST_VERSION >= 104200
        throw po::validation_error(po::validation_error::invalid_option_value);
#else
        throw po::validation_error("invalid option value");
#endif
    }

    return in;
}

std::ostream& operator<<(std::ostream& out, const clblasOrder& order)
{
    switch (order) {
    case clblasRowMajor:
        out << "row";
        break;
    case clblasColumnMajor:
        out << "column";
        break;
    }

    return out;
}

std::istream& operator>>(std::istream& in, clblasSide& side)
{
    std::string token;

    in >> token;
    if (token == "left") {
        side = clblasLeft;
    }
    else if (token == "right") {
        side = clblasRight;
    }
    else {
#if BOOST_VERSION >= 104200
        throw po::validation_error(po::validation_error::invalid_option_value);
#else
        throw po::validation_error("invalid option value");
#endif
    }

    return in;
}

std::ostream& operator<<(std::ostream& out, const clblasSide& side)
{
    switch (side) {
    case clblasLeft:
        out << "left";
        break;
    case clblasRight:
        out << "right";
        break;
    }

    return out;
}

std::istream& operator>>(std::istream& in, clblasUplo& uplo)
{
    std::string token;

    in >> token;
    if (token == "upper") {
        uplo = clblasUpper;
    }
    else if (token == "lower") {
        uplo = clblasLower;
    }
    else {
#if BOOST_VERSION >= 104200
        throw po::validation_error(po::validation_error::invalid_option_value);
#else
        throw po::validation_error("invalid option value");
#endif
    }

    return in;
}

std::ostream& operator<<(std::ostream& out, const clblasUplo& uplo)
{
    switch (uplo) {
    case clblasUpper:
        out << "upper";
        break;
    case clblasLower:
        out << "lower";
        break;
    }

    return out;
}

std::istream& operator>>(std::istream& in, clblasTranspose& trans)
{
    std::string token;

    in >> token;
    if (token == "n") {
        trans = clblasNoTrans;
    }
    else if (token == "t") {
        trans = clblasTrans;
    }
    else if (token == "c") {
        trans = clblasConjTrans;
    }
    else {
#if BOOST_VERSION >= 104200
        throw po::validation_error(po::validation_error::invalid_option_value);
#else
        throw po::validation_error("invalid option value");
#endif
    }

    return in;
}

std::ostream& operator<<(std::ostream& out, const clblasTranspose& trans)
{
    switch (trans) {
    case clblasNoTrans:
        out << "n";
        break;
    case clblasTrans:
        out << "t";
        break;
    case clblasConjTrans:
        out << "c";
        break;
    }

    return out;
}

std::istream& operator>>(std::istream& in, clblasDiag& diag)
{
    std::string token;

    in >> token;
    if (token == "unit") {
        diag = clblasUnit;
    }
    else if (token == "nonunit") {
        diag = clblasNonUnit;
    }
    else {
#if BOOST_VERSION >= 104200
        throw po::validation_error(po::validation_error::invalid_option_value);
#else
        throw po::validation_error("invalid option value");
#endif
    }

    return in;
}

std::ostream& operator<<(std::ostream& out, const clblasDiag& diag)
{
    switch (diag) {
    case clblasUnit:
        out << "unit";
        break;
    case clblasNonUnit:
        out << "nonunit";
        break;
    }

    return out;
}

bool
Config::parseDecompositionOpt(const std::string& opt)
{
    size_t v[6];    // x0, y0, bwidth0, x1, y1, bwidth1

    boost::tokenizer<> tok(opt);
    boost::tokenizer<>::iterator it = tok.begin();

    for (int i = 0; i < 6; i++) {
        if (it == tok.end()) {
            return false;
        }
        try {
            v[i] = boost::lexical_cast<size_t>(*it);
        }
        catch (boost::bad_lexical_cast&) {
            return false;
        }
        ++it;
    }
    if (it != tok.end()) {
        return false;
    }

    setDecomposition(v[0], v[1], v[2], v[3], v[4], v[5]);
    return true;
}

bool
Config::parseArgMultiplier(
    const std::string& opt,
    ArgMultiplier& v)
{
    boost::char_separator<char> sep(",");
    boost::tokenizer< boost::char_separator<char> > tok(opt, sep);
    boost::tokenizer< boost::char_separator<char> >::iterator it = tok.begin();

    try {
        switch (kargs_.dtype) {
        case TYPE_FLOAT:
            v.argFloat = boost::lexical_cast<float>(*it);
            ++it;
            break;
        case TYPE_DOUBLE:
            v.argDouble = boost::lexical_cast<double>(*it);
            ++it;
            break;
        case TYPE_COMPLEX_FLOAT:
            v.argFloatComplex.s[0] = boost::lexical_cast<float>(*it);
            ++it;
            if (it == tok.end()) {
                v.argFloatComplex.s[1] = 0;
            }
            else {
                v.argFloatComplex.s[1] = boost::lexical_cast<float>(*it);
                ++it;
            }
            break;
        case TYPE_COMPLEX_DOUBLE:
            v.argDoubleComplex.s[0] = boost::lexical_cast<double>(*it);
            ++it;
            if (it == tok.end()) {
                v.argDoubleComplex.s[1] = 0;
            }
            else {
                v.argDoubleComplex.s[1] = boost::lexical_cast<double>(*it);
                ++it;
            }
            break;
        }
    }
    catch (boost::bad_lexical_cast&) {
        return false;
    }

    return (it == tok.end());
}
