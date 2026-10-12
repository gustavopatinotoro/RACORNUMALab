//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Punto de entrada de RACORNUMALab.
//
//  Gate actual:
//  G3A — Baseline funcional HNSW.
//
//  Responsabilidades:
//  - Mantener la identificación reproducible de plataforma.
//  - Mantener la procedencia del artefacto RACORN.
//  - Ejecutar el primer HNSW real.
//  - Comparar búsqueda HNSW contra un oráculo exacto.
//

#include "baseline/HnswBaseline.hpp"
#include "core/BuildInfo.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/SystemInfo.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>

namespace {

double bytes_to_gib(
    std::uint64_t bytes
) {

    constexpr double bytes_per_gib =
        1024.0 *
        1024.0 *
        1024.0;

    return
        static_cast<double>(bytes) /
        bytes_per_gib;
}

void print_keys(
    char const* label,
    std::vector<std::size_t> const& keys
) {

    std::cout
        << label
        << " :";

    for (
        std::size_t key :
        keys
    ) {

        std::cout
            << ' '
            << key;
    }

    std::cout << '\n';
}

} // namespace

int main() {

    using racorn_uma::baseline::HnswBaseline;
    using racorn_uma::core::BuildInfo;
    using racorn_uma::integration::RacornProbe;
    using racorn_uma::platform::apple::SystemInfo;

    const std::uint64_t physical_memory =
        SystemInfo::physical_memory_bytes();

    std::cout
        << "========================================\n"
        << " RACORNUMALab\n"
        << " Gate G3A - HNSW funcional\n"
        << "========================================\n\n";

    std::cout
        << "[Plataforma]\n"
        << "Compilador        : "
        << BuildInfo::compiler()
        << '\n'
        << "Estandar C++      : "
        << BuildInfo::cpp_standard()
        << '\n'
        << "Arquitectura      : "
        << BuildInfo::architecture()
        << '\n'
        << "Modelo hardware   : "
        << SystemInfo::hardware_model()
        << '\n'
        << "CPU logicas       : "
        << SystemInfo::logical_cpu_count()
        << '\n'
        << "Memoria fisica    : "
        << std::fixed
        << std::setprecision(2)
        << bytes_to_gib(
            physical_memory
        )
        << " GiB\n\n";

    std::cout
        << "[RACORN]\n"
        << "Repositorio        : "
        << RacornProbe::repository()
        << '\n'
        << "Version            : "
        << RacornProbe::version()
        << '\n'
        << "Commit             : "
        << RacornProbe::commit()
        << "\n\n";

    std::cout
        << "[HNSW baseline]\n";

    const auto result =
        HnswBaseline::run();

    std::cout
        << "Vectores           : "
        << result.vector_count
        << '\n';

    std::cout
        << "Dimensiones        : "
        << result.dimensions
        << '\n';

    std::cout
        << "M                   : "
        << result.connectivity
        << '\n';

    std::cout
        << "efConstruction      : "
        << result.ef_construction
        << '\n';

    std::cout
        << "efSearch            : "
        << result.ef_search
        << '\n';

    std::cout
        << "k                   : "
        << result.k
        << '\n';

    print_keys(
        "Exact keys         ",
        result.exact_keys
    );

    print_keys(
        "HNSW keys          ",
        result.hnsw_keys
    );

    std::cout
        << "Recall@"
        << result.k
        << "            : "
        << std::fixed
        << std::setprecision(4)
        << result.recall_at_k
        << "\n\n";

    if (!result.passed) {

        std::cerr
            << "[G3A] FAIL\n";

        return 1;
    }

    std::cout
        << "[G3A] PASS\n";

    return 0;
}
