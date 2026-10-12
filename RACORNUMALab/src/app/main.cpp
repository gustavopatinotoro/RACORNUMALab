//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar un único experimento de búsqueda filtrada parametrizado
//  por selectividad.
//
//  Gate:
//  G4B — Barrido controlado de selectividad.
//
//  Uso:
//
//      RACORNUMALab --modulus 10
//
//      RACORNUMALab --modulus 20
//
//      RACORNUMALab --modulus 50
//
//      RACORNUMALab --modulus 100
//
//      RACORNUMALab --modulus 200
//
//      RACORNUMALab --modulus 400
//
//      RACORNUMALab --modulus 1000
//
//  El threshold por defecto es 1.
//
//  También puede especificarse explícitamente:
//
//      RACORNUMALab --modulus 200 --threshold 1
//
//  Regla experimental:
//  Cada selectividad debe ejecutarse en un proceso independiente.
//

#include "core/BuildInfo.hpp"
#include "filtered/FilteredSearchBaseline.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/SystemInfo.hpp"

#include <charconv>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {

struct RunConfiguration final {

    std::size_t modulus = 200;
    std::size_t threshold = 1;

    bool valid = true;
    bool help_requested = false;
};

bool parse_size(
    std::string_view text,
    std::size_t& output
) noexcept {

    if (text.empty()) {
        return false;
    }

    const auto result =
        std::from_chars(
            text.data(),
            text.data() +
                text.size(),
            output
        );

    return
        result.ec ==
            std::errc{} &&
        result.ptr ==
            text.data() +
            text.size();
}

RunConfiguration parse_arguments(
    int argc,
    char* argv[]
) noexcept {

    RunConfiguration configuration;

    int index = 1;

    while (
        index <
        argc
    ) {

        const std::string_view option(
            argv[index]
        );

        if (
            option == "--help" ||
            option == "-h"
        ) {

            configuration
                .help_requested =
                true;

            return configuration;
        }

        if (
            option ==
                "--modulus" &&
            index + 1 <
                argc
        ) {

            if (
                !parse_size(
                    argv[index + 1],
                    configuration.modulus
                )
            ) {

                configuration.valid =
                    false;

                return configuration;
            }

            index += 2;
            continue;
        }

        if (
            option ==
                "--threshold" &&
            index + 1 <
                argc
        ) {

            if (
                !parse_size(
                    argv[index + 1],
                    configuration.threshold
                )
            ) {

                configuration.valid =
                    false;

                return configuration;
            }

            index += 2;
            continue;
        }

        configuration.valid =
            false;

        return configuration;
    }

    if (
        configuration.modulus ==
            0 ||
        configuration.threshold ==
            0 ||
        configuration.threshold >
            configuration.modulus
    ) {

        configuration.valid =
            false;
    }

    return configuration;
}

void print_usage() {

    std::cout
        << "Uso:\n"
        << "  RACORNUMALab "
        << "--modulus <M> "
        << "[--threshold <T>]\n\n"

        << "Predicado:\n"
        << "  (key % M) < T\n\n"

        << "Ejemplos del sweep G4B:\n"
        << "  --modulus 10    "
        << "-> 10.00 %\n"
        << "  --modulus 20    "
        << "->  5.00 %\n"
        << "  --modulus 50    "
        << "->  2.00 %\n"
        << "  --modulus 100   "
        << "->  1.00 %\n"
        << "  --modulus 200   "
        << "->  0.50 %\n"
        << "  --modulus 400   "
        << "->  0.25 %\n"
        << "  --modulus 1000  "
        << "->  0.10 %\n";
}

void print_mode(
    racorn_uma::filtered::
        FilteredModeStats const& mode,
    std::size_t k
) {

    std::cout
        << '\n'
        << '['
        << mode.name
        << "]\n";

    std::cout
        << "Recall@"
        << k
        << " medio"
        << std::setw(7)
        << " "
        << ": "
        << std::fixed
        << std::setprecision(4)
        << mode.mean_recall_at_k
        << '\n';

    std::cout
        << "Resultados medios    : "
        << std::fixed
        << std::setprecision(4)
        << mode.mean_result_count
        << '\n';

    std::cout
        << "Distancias medias    : "
        << std::fixed
        << std::setprecision(4)
        << mode.mean_computed_distances
        << '\n';

    std::cout
        << "Visitados medios     : "
        << std::fixed
        << std::setprecision(4)
        << mode.mean_visited_members
        << '\n';

    std::cout
        << "Predicados medios    : "
        << std::fixed
        << std::setprecision(4)
        << mode.mean_predicate_evaluations
        << '\n';

    std::cout
        << "Fallback medio       : "
        << std::fixed
        << std::setprecision(4)
        << mode.mean_fallback_count
        << '\n';

    std::cout
        << "Queries con fallback : "
        << mode.queries_with_fallback
        << '\n';

    if (
        mode.queries_with_fallback >
        0
    ) {

        std::cout
            << "Sel. fallback media  : "
            << std::fixed
            << std::setprecision(6)
            << mode.mean_fallback_selectivity
            << '\n';
    }

    std::cout
        << "Filtro valido        : "
        << (
            mode
                .all_results_satisfy_filter
                ? "SI"
                : "NO"
        )
        << '\n';
}

} // namespace

int main(
    int argc,
    char* argv[]
) {

    using racorn_uma::core::
        BuildInfo;

    using racorn_uma::filtered::
        FilteredSearchBaseline;

    using racorn_uma::integration::
        RacornProbe;

    using racorn_uma::platform::apple::
        SystemInfo;

    const RunConfiguration configuration =
        parse_arguments(
            argc,
            argv
        );

    if (
        configuration
            .help_requested
    ) {

        print_usage();
        return 0;
    }

    if (
        !configuration.valid
    ) {

        std::cerr
            << "Parametros invalidos.\n\n";

        print_usage();

        return 2;
    }

    const auto result =
        FilteredSearchBaseline::run(
            configuration.modulus,
            configuration.threshold
        );

    std::cout
        << "========================================\n"
        << " RACORNUMALab\n"
        << " Gate G4B - Sweep de selectividad\n"
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
        << "\n\n";

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
        << "[Experimento]\n"
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
        << "Queries            : "
        << result.query_count
        << '\n'
        << "k                  : "
        << result.k
        << '\n'
        << "Filter modulus     : "
        << result.filter_modulus
        << '\n'
        << "Filter threshold   : "
        << result.filter_threshold
        << '\n'
        << "Elegibles          : "
        << result.eligible_count
        << '\n';

    std::cout
        << "Selectividad nominal: "
        << std::fixed
        << std::setprecision(4)
        << (
            result.nominal_selectivity *
            100.0
        )
        << " %\n";

    std::cout
        << "Selectividad real   : "
        << std::fixed
        << std::setprecision(4)
        << (
            result.observed_selectivity *
            100.0
        )
        << " %\n";

    print_mode(
        result.hnsw,
        result.k
    );

    print_mode(
        result.acorn,
        result.k
    );

    print_mode(
        result.racorn,
        result.k
    );

    std::cout
        << "\n[Validacion]\n";

    std::cout
        << "Ground truth exacto : "
        << (
            result
                .exact_ground_truth_complete
                ? "OK"
                : "FAIL"
        )
        << '\n';

    std::cout
        << "ASF RACORN observado: "
        << (
            result
                .racorn_fallback_observed
                ? "SI"
                : "NO"
        )
        << '\n';

    std::cout
        << "Estado experimento  : "
        << (
            result.passed
                ? "PASS"
                : "FAIL"
        )
        << "\n\n";

    if (
        !result.passed
    ) {

        std::cerr
            << "[G4B] FAIL\n";

        return 1;
    }

    std::cout
        << "[G4B] PASS\n";

    return 0;
}
