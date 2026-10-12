//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Punto de entrada de la aplicación de terminal RACORNUMALab.
//
//  Gate actual:
//  G2 — Integración reproducible del artefacto RACORN-1.
//
//  Responsabilidades:
//  - Verificar plataforma Apple Silicon.
//  - Verificar toolchain.
//  - Verificar disponibilidad compilable de RACORN.
//  - Exponer la procedencia exacta del artefacto científico.
//

#include "core/BuildInfo.hpp"
#include "integration/RacornProbe.hpp"
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
    using racorn_uma::integration::RacornProbe;
    using racorn_uma::platform::apple::SystemInfo;

    const std::uint64_t physical_memory =
        SystemInfo::physical_memory_bytes();

    std::cout
        << "========================================\n"
        << " RACORNUMALab\n"
        << " Gate G2 - Integracion RACORN\n"
        << "========================================\n\n";

    std::cout
        << "[Plataforma]\n";

    std::cout
        << "Compilador        : "
        << BuildInfo::compiler()
        << '\n';

    std::cout
        << "Estandar C++      : "
        << BuildInfo::cpp_standard()
        << '\n';

    std::cout
        << "Arquitectura      : "
        << BuildInfo::architecture()
        << '\n';

    std::cout
        << "Modelo hardware   : "
        << SystemInfo::hardware_model()
        << '\n';

    std::cout
        << "CPU logicas       : "
        << SystemInfo::logical_cpu_count()
        << '\n';

    std::cout
        << "Memoria fisica    : "
        << std::fixed
        << std::setprecision(2)
        << bytes_to_gib(physical_memory)
        << " GiB\n\n";

    std::cout
        << "[RACORN]\n";

    std::cout
        << "Repositorio        : "
        << RacornProbe::repository()
        << '\n';

    std::cout
        << "Version            : "
        << RacornProbe::version()
        << '\n';

    std::cout
        << "Commit             : "
        << RacornProbe::commit()
        << '\n';

    std::cout
        << "Headers            : "
        << (
            RacornProbe::headers_available()
                ? "OK"
                : "FAIL"
        )
        << "\n\n";

    const bool platform_ok =
        BuildInfo::architecture() == "arm64";

    const bool memory_ok =
        physical_memory > 0;

    const bool racorn_ok =
        RacornProbe::headers_available();

    if (!platform_ok ||
        !memory_ok ||
        !racorn_ok) {

        std::cerr
            << "[G2] FAIL\n";

        return 1;
    }

    std::cout
        << "[G2] PASS\n";

    return 0;
}
