//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Punto de entrada de la aplicación de terminal RACORNUMALab.
//
//  Gate actual:
//  G1 — Fundación estructural y diagnóstico de plataforma.
//
//  Responsabilidades actuales:
//  - Verificar el entorno de compilación.
//  - Verificar ejecución arm64.
//  - Verificar propiedades fundamentales del Apple Silicon.
//  - Confirmar el presupuesto físico de memoria.
//
//  RACORN y USEARCH todavía NO forman parte de este Gate.
//

#include "core/BuildInfo.hpp"
#include "platform/apple/SystemInfo.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>

namespace {

double bytes_to_gib(std::uint64_t bytes) {

    constexpr double bytes_per_gib =
        1024.0 * 1024.0 * 1024.0;

    return static_cast<double>(bytes) / bytes_per_gib;
}

} // namespace

int main() {

    using racorn_uma::core::BuildInfo;
    using racorn_uma::platform::apple::SystemInfo;

    const std::uint64_t physical_memory =
        SystemInfo::physical_memory_bytes();

    std::cout
        << "========================================\n"
        << " RACORNUMALab\n"
        << " Gate G1 - Diagnostico de plataforma\n"
        << "========================================\n\n";

    std::cout
        << "Compilador        : "
        << BuildInfo::compiler()
        << '\n';

    std::cout
        << "Estandar C++      : "
        << BuildInfo::cpp_standard()
        << '\n';

    std::cout
        << "Arquitectura build: "
        << BuildInfo::architecture()
        << '\n';

    std::cout
        << "Modo build        : "
        << BuildInfo::build_mode()
        << '\n';

    std::cout
        << "Modelo hardware   : "
        << SystemInfo::hardware_model()
        << '\n';

    std::cout
        << "hw.machine        : "
        << SystemInfo::machine()
        << '\n';

    std::cout
        << "CPU logicas       : "
        << SystemInfo::logical_cpu_count()
        << '\n';

    std::cout
        << "Memoria fisica    : "
        << physical_memory
        << " bytes";

    if (physical_memory > 0) {
        std::cout
            << " ("
            << std::fixed
            << std::setprecision(2)
            << bytes_to_gib(physical_memory)
            << " GiB)";
    }

    std::cout << "\n\n";

    const bool architecture_ok =
        BuildInfo::architecture() == "arm64";

    const bool memory_ok =
        physical_memory > 0;

    const bool cpp_ok =
        BuildInfo::cpp_standard() == "C++20" ||
        BuildInfo::cpp_standard() == "C++23";

    if (!architecture_ok ||
        !memory_ok ||
        !cpp_ok) {

        std::cerr
            << "[G1] FAIL\n";

        return 1;
    }

    std::cout
        << "[G1] PASS\n";

    return 0;
}
