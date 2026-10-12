//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar un experimento HNSW parametrizado por cardinalidad
//  y dimensionalidad.
//
//  Gate:
//  G3D — Descomposición de memoria.
//
//  Uso:
//      RACORNUMALab --n 16000 --d 384
//
//  Cada configuración debe ejecutarse en un proceso independiente.
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

struct RunConfig final {
    std::size_t vector_count = 16000;
    std::size_t dimensions = 4;
    bool valid = true;
};

bool parse_size(
    std::string_view text,
    std::size_t& output
) {

    const auto result =
        std::from_chars(
            text.data(),
            text.data() + text.size(),
            output
        );

    return
        result.ec == std::errc{} &&
        result.ptr ==
            text.data() + text.size();
}

RunConfig parse_arguments(
    int argc,
    char* argv[]
) {

    RunConfig config;

    int i = 1;

    while (i < argc) {

        const std::string_view option(
            argv[i]
        );

        if (
            option == "--n" &&
            i + 1 < argc
        ) {

            if (
                !parse_size(
                    argv[i + 1],
                    config.vector_count
                )
            ) {

                config.valid = false;
                return config;
            }

            i += 2;
            continue;
        }

        if (
            option == "--d" &&
            i + 1 < argc
        ) {

            if (
                !parse_size(
                    argv[i + 1],
                    config.dimensions
                )
            ) {

                config.valid = false;
                return config;
            }

            i += 2;
            continue;
        }

        config.valid = false;
        return config;
    }

    if (
        config.vector_count < 10 ||
        config.dimensions == 0 ||
        config.dimensions > 4096
    ) {

        config.valid = false;
    }

    return config;
}

double bytes_to_mib(
    std::uint64_t bytes
) {

    return
        static_cast<double>(bytes) /
        (1024.0 * 1024.0);
}

double bytes_to_gib(
    std::uint64_t bytes
) {

    return
        static_cast<double>(bytes) /
        (1024.0 * 1024.0 * 1024.0);
}

std::int64_t delta_bytes(
    std::uint64_t current,
    std::uint64_t baseline
) {

    return
        static_cast<std::int64_t>(
            current
        ) -
        static_cast<std::int64_t>(
            baseline
        );
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

    const auto rss_delta =
        delta_bytes(
            snapshot.resident_bytes,
            baseline.resident_bytes
        );

    const auto footprint_delta =
        delta_bytes(
            snapshot.physical_footprint_bytes,
            baseline.physical_footprint_bytes
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
                rss_delta
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

    const RunConfig config =
        parse_arguments(
            argc,
            argv
        );

    if (!config.valid) {

        std::cerr
            << "Uso:\n"
            << "  RACORNUMALab --n <N> --d <D>\n\n"
            << "Restricciones:\n"
            << "  N >= 10\n"
            << "  1 <= D <= 4096\n";

        return 2;
    }

    const auto result =
        HnswBaseline::run(
            config.vector_count,
            config.dimensions
        );

    std::cout
        << "========================================\n"
        << " RACORNUMALab\n"
        << " Gate G3D - Dimension vs HNSW\n"
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
        << '\n'
        << "VectorStore teorico: "
        << std::fixed
        << std::setprecision(2)
        << bytes_to_mib(
            result.expected_vector_bytes
        )
        << " MiB\n\n";

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

    const std::int64_t observed_vector_bytes =
        delta_bytes(
            result.memory_after_vectors
                .physical_footprint_bytes,
            result.memory_start
                .physical_footprint_bytes
        );

    const std::int64_t index_over_vectors_bytes =
        delta_bytes(
            result.memory_after_build
                .physical_footprint_bytes,
            result.memory_after_vectors
                .physical_footprint_bytes
        );

    const double index_bytes_per_vector =
        static_cast<double>(
            index_over_vectors_bytes
        ) /
        static_cast<double>(
            result.vector_count
        );

    std::cout
        << "\n[Descomposicion]\n"
        << "Vectores teoricos  : "
        << bytes_to_mib(
            result.expected_vector_bytes
        )
        << " MiB\n"
        << "Vectores observados: "
        << (
            static_cast<double>(
                observed_vector_bytes
            ) /
            (1024.0 * 1024.0)
        )
        << " MiB\n"
        << "Indice sobre vector: "
        << (
            static_cast<double>(
                index_over_vectors_bytes
            ) /
            (1024.0 * 1024.0)
        )
        << " MiB\n"
        << "Indice bytes/vector: "
        << std::fixed
        << std::setprecision(2)
        << index_bytes_per_vector
        << " B/vector\n\n";

    if (!result.passed) {

        std::cerr
            << "[G3D] FAIL\n";

        return 1;
    }

    std::cout
        << "[G3D] PASS\n";

    return 0;
}
