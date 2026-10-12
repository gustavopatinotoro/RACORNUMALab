//
//  BuildInfo.cpp
//  RACORNUMALab
//
//  Implementación del diagnóstico reproducible del entorno de compilación.
//

#include "core/BuildInfo.hpp"

#include <sstream>
#include <string>

namespace racorn_uma::core {

std::string BuildInfo::compiler() {

#if defined(__clang__)
    std::ostringstream stream;

    stream
        << "Apple/LLVM Clang "
        << __clang_major__
        << '.'
        << __clang_minor__
        << '.'
        << __clang_patchlevel__;

    return stream.str();
#else
    return "Compilador no identificado";
#endif
}

std::string BuildInfo::architecture() {

#if defined(__aarch64__) || defined(__arm64__)
    return "arm64";
#elif defined(__x86_64__)
    return "x86_64";
#else
    return "arquitectura-desconocida";
#endif
}

std::string BuildInfo::cpp_standard() {

#if __cplusplus >= 202302L
    return "C++23";
#elif __cplusplus >= 202002L
    return "C++20";
#elif __cplusplus >= 201703L
    return "C++17";
#else
    return "C++ anterior a C++17";
#endif
}

std::string BuildInfo::build_mode() {

#if defined(NDEBUG)
    return "Release";
#else
    return "Debug";
#endif
}

} // namespace racorn_uma::core
