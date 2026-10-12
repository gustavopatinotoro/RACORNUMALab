//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar G4C: múltiples realizaciones topológicas de un filtro con
//  aproximadamente la misma selectividad sobre un único índice HNSW.
//
//  Gate:
//  G4C — Replicación topológica del filtro.
//
//  Ejemplo:
//
//      RACORNUMALab --modulus 100 --residues 8
//
//  Con threshold = 1:
//
//      modulus = 100
//
//  corresponde aproximadamente a:
//
//      s = 1 %
//
//  y --residues 8 evalúa ocho clases residuales distribuidas a lo
//  largo del espacio [0, modulus).
//
//  Nota metodológica:
//  Las realizaciones se ejecutan deliberadamente dentro del mismo proceso
//  porque este Gate no mide memoria ni latencia. Queremos que todas ellas
//  compartan exactamente la misma topología HNSW.
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

    std::size_t modulus =
        100;

    std::size_t threshold =
        1;

    std::size_t residues =
        8;

    bool valid =
        true;

    bool help_requested =
        false;
};

bool parse_size(
    std::string_view text,
    std::size_t& output
) noexcept {

    if (
        text.empty()
    ) {

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

    int index =
        1;

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

            configuration.help_requested =
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

            index +=
                2;

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

            index +=
                2;

            continue;
        }

        if (
            option ==
                "--residues" &&
            index + 1 <
                argc
        ) {

            if (
                !parse_size(
                    argv[index + 1],
                    configuration.residues
                )
            ) {

                configuration.valid =
                    false;

                return configuration;
            }

            index +=
                2;

            continue;
        }

        configuration.valid =
            false;

        return configuration;
    }

    if (
        configuration.modulus ==
            0 ||
        configuration.modulus >
            1'000'000 ||
        configuration.threshold ==
            0 ||
        configuration.threshold >
            configuration.modulus ||
        configuration.residues ==
            0 ||
        configuration.residues >
            256
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
        << "[--threshold <T>] "
        << "[--residues <R>]\n\n"

        << "Valores por defecto:\n"
        << "  modulus   = 100\n"
        << "  threshold = 1\n"
        << "  residues  = 8\n\n"

        << "Ejemplo:\n"
        << "  RACORNUMALab "
        << "--modulus 100 "
        << "--residues 8\n";
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
        << " global      : "
        << std::fixed
        << std::setprecision(4)
        << mode.mean_recall_at_k
        << '\n';

    std::cout
        << "Recall std residuos : "
        << std::fixed
        << std::setprecision(6)
        << mode.realization_recall_stddev
        << '\n';

    std::cout
        << "Recall min residuo  : "
        << std::fixed
        << std::setprecision(4)
        << mode.min_realization_recall
        << '\n';

    std::cout
        << "Recall max residuo  : "
        << std::fixed
        << std::setprecision(4)
        << mode.max_realization_recall
        << '\n';

    std::cout
        << "Resultados medios   : "
        << std::fixed
        << std::setprecision(4)
        << mode.mean_result_count
        << '\n';

    std::cout
        << "Distancias medias   : "
        << std::fixed
        << std::setprecision(4)
        << mode.mean_computed_distances
        << '\n';

    std::cout
        << "Visitados medios    : "
        << std::fixed
        << std::setprecision(4)
        << mode.mean_visited_members
        << '\n';

    std::cout
        << "Predicados medios   : "
        << std::fixed
        << std::setprecision(4)
        << mode.mean_predicate_evaluations
        << '\n';

    std::cout
        << "Fallback medio      : "
        << std::fixed
        << std::setprecision(4)
        << mode.mean_fallback_count
        << '\n';

    std::cout
        << "Queries evaluadas   : "
        << mode.total_query_evaluations
        << '\n';

    std::cout
        << "Queries con fallback: "
        << mode.queries_with_fallback
        << '\n';

    std::cout
        << "Fraccion fallback   : "
        << std::fixed
        << std::setprecision(4)
        << (
            mode.fallback_query_fraction *
            100.0
        )
        << " %\n";

    if (
        mode.queries_with_fallback >
        0
    ) {

        std::cout
            << "Sel. fallback media : "
            << std::fixed
            << std::setprecision(6)
            << mode.mean_fallback_selectivity
            << '\n';
    }

    std::cout
        << "Filtro valido       : "
        << (
            mode
                .all_results_satisfy_filter
                ? "SI"
                : "NO"
        )
        << '\n';
}

void print_realizations(
    racorn_uma::filtered::
        FilteredSearchResult const& result
) {

    std::cout
        << "\n[Realizaciones]\n";

    std::cout
        << "idx"
        << std::setw(10)
        << "residuo"
        << std::setw(12)
        << "elegibles"
        << std::setw(12)
        << "sel.%"
        << std::setw(14)
        << "HNSW-R"
        << std::setw(14)
        << "ACORN-R"
        << std::setw(14)
        << "RACORN-R"
        << std::setw(16)
        << "RAC-dist"
        << std::setw(16)
        << "RAC-fb"
        << '\n';

    for (
        auto const& realization :
        result.realizations
    ) {

        std::cout
            << std::left
            << std::setw(3)
            << realization
                .realization_index

            << std::right
            << std::setw(10)
            << realization.residue

            << std::setw(12)
            << realization
                .eligible_count

            << std::setw(12)
            << std::fixed
            << std::setprecision(3)
            << (
                realization
                    .observed_selectivity *
                100.0
            )

            << std::setw(14)
            << std::fixed
            << std::setprecision(4)
            << realization
                .hnsw_recall_at_k

            << std::setw(14)
            << realization
                .acorn_recall_at_k

            << std::setw(14)
            << realization
                .racorn_recall_at_k

            << std::setw(16)
            << std::fixed
            << std::setprecision(2)
            << realization
                .racorn_mean_distances

            << std::setw(16)
            << realization
                .racorn_mean_fallback_count

            << '\n';
    }
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
        configuration.help_requested
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
            configuration.threshold,
            configuration.residues
        );

    std::cout
        << "========================================\n"
        << " RACORNUMALab\n"
        << " Gate G4C - Replicacion del filtro\n"
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
        << "Queries/residuo    : "
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
        << "Realizaciones      : "
        << result.realization_count
        << '\n'
        << "Elegibles min/max  : "
        << result.min_eligible_count
        << " / "
        << result.max_eligible_count
        << '\n'
        << "Selectividad nominal: "
        << std::fixed
        << std::setprecision(4)
        << (
            result.nominal_selectivity *
            100.0
        )
        << " %\n";

    print_realizations(
        result
    );

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
        << "Filtros validos     : "
        << (
            result.filters_valid
                ? "SI"
                : "NO"
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
            << "[G4C] FAIL\n";

        return 1;
    }

    std::cout
        << "[G4C] PASS\n";

    return 0;
}
