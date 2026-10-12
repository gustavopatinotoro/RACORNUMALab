//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar un único experimento HNSW parametrizado por N.
//
//  Gate:
//  G3C — Escalado controlado de memoria.
//
//  Uso:
//      RACORNUMALab --n 1000
//
//  Restricción metodológica:
//  Cada N debe ejecutarse en un proceso independiente.
//

#include "baseline/HnswBaseline.hpp"
#include "core/BuildInfo.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/SystemInfo.hpp"

#include <charconv>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

double bytes_to_gib(
    std::uint64_t bytes
) {

    return
        static_cast<double>(bytes) /
        (1024.0 * 1024.0 * 1024.0);
}

double bytes_to_mib(
    std::uint64_t bytes
) {

    return
        static_cast<double>(bytes) /
        (1024.0 * 1024.0);
}

std::size_t parse_vector_count(
    int argc,
    char* argv[]
) {

    constexpr std::size_t default_n =
        1000;

    if (argc == 1) {
        return default_n;
    }

    if (argc != 3) {
        return 0;
    }

    if (
        std::string_view(argv[1]) !=
        "--n"
    ) {

        return 0;
    }

    std::size_t value = 0;

    const std::string_view text(
        argv[2]
    );

    const auto result =
        std::from_chars(
            text.data(),
            text.data() +
                text.size(),
            value
        );

    if (
        result.ec !=
            std::errc{} ||
        result.ptr !=
            text.data() +
            text.size()
    ) {

        return 0;
    }

    return value;
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

void print_memory(
    char const* label,
    racorn_uma::platform::apple::
        ProcessMemorySnapshot const& snapshot,
    racorn_uma::platform::apple::
        ProcessMemorySnapshot const& baseline
) {

    const std::int64_t footprint_delta =
        static_cast<std::int64_t>(
            snapshot.physical_footprint_bytes
        ) -
        static_cast<std::int64_t>(
            baseline.physical_footprint_bytes
        );

    const std::int64_t resident_delta =
        static_cast<std::int64_t>(
            snapshot.resident_bytes
        ) -
        static_cast<std::int64_t>(
            baseline.resident_bytes
        );

    std::cout
        << std::left
        << std::setw(20)
        << label

        << " RSS="
        << std::right
        << std::setw(9)
        << std::fixed
        << std::setprecision(2)
        << bytes_to_mib(
            snapshot.resident_bytes
        )

        << " MiB | dRSS="
        << std::setw(9)
        << (
            static_cast<double>(
                resident_delta
            ) /
            (1024.0 * 1024.0)
        )

        << " MiB | Footprint="
        << std::setw(9)
        << bytes_to_mib(
            snapshot.physical_footprint_bytes
        )

        << " MiB | dFootprint="
        << std::setw(9)
        << (
            static_cast<double>(
                footprint_delta
            ) /
            (1024.0 * 1024.0)
        )

        << " MiB\n";
}

} // namespace

int main(
    int argc,
    char* argv[]
) {

    using racorn_uma::baseline::
        HnswBaseline;

    using racorn_uma::core::
        BuildInfo;

    using racorn_uma::integration::
        RacornProbe;

    using racorn_uma::platform::apple::
        SystemInfo;

    const std::size_t vector_count =
        parse_vector_count(
            argc,
            argv
        );

    if (vector_count < 10) {

        std::cerr
            << "Uso: RACORNUMALab --n <N>\n"
            << "N debe ser >= 10\n";

        return 2;
    }

    const auto result =
        HnswBaseline::run(
            vector_count
        );

    std::cout
        << "========================================\n"
        << " RACORNUMALab\n"
        << " Gate G3C - Escalado HNSW\n"
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
        << "Memoria fisica    : "
        << std::fixed
        << std::setprecision(2)
        << bytes_to_gib(
            SystemInfo::
                physical_memory_bytes()
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
        << "[Configuracion]\n"
        << "N                  : "
        << result.vector_count
        << '\n'
        << "D                  : "
        << result.dimensions
        << '\n'
        << "M                  : "
        << result.connectivity
        << '\n'
        << "efConstruction     : "
        << result.ef_construction
        << '\n'
        << "efSearch           : "
        << result.ef_search
        << '\n'
        << "k                  : "
        << result.k
        << "\n\n";

    print_keys(
        "Exact keys        ",
        result.exact_keys
    );

    print_keys(
        "HNSW keys         ",
        result.hnsw_keys
    );

    std::cout
        << "Recall@"
        << result.k
        << "          : "
        << std::fixed
        << std::setprecision(4)
        << result.recall_at_k
        << "\n\n";

    std::cout
        << "[Memoria del proceso]\n";

    print_memory(
        "Inicio",
        result.memory_start,
        result.memory_start
    );

    print_memory(
        "Vectores",
        result.memory_after_vectors,
        result.memory_start
    );

    print_memory(
        "Reserve",
        result.memory_after_reserve,
        result.memory_start
    );

    print_memory(
        "HNSW construido",
        result.memory_after_build,
        result.memory_start
    );

    print_memory(
        "Despues busqueda",
        result.memory_after_search,
        result.memory_start
    );

    std::cout << '\n';

    if (!result.passed) {

        std::cerr
            << "[G3C] FAIL\n";

        return 1;
    }

    std::cout
        << "[G3C] PASS\n";

    return 0;
}
