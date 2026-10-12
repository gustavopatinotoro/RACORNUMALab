//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar la ablación de bridge_budget_ratio de RACORN-1.
//
//  Gate:
//  G4D — Sweep de bridge_budget_ratio.
//
//  Ejemplo:
//
//      RACORNUMALab --modulus 200 --residues 8
//
//  Con:
//
//      modulus = 200
//      threshold = 1
//
//  la selectividad nominal es:
//
//      0.5 %
//
//  El programa evalúa internamente:
//
//      BR = 0.25
//      BR = 0.50
//      BR = 0.75
//      BR = 1.00
//
//  sobre el mismo índice, consultas y realizaciones del filtro.
//

#include "core/BuildInfo.hpp"
#include "filtered/BridgeBudgetSweep.hpp"
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
        200;

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

        << "Ejemplo:\n"
        << "  RACORNUMALab "
        << "--modulus 200 "
        << "--residues 8\n";
}

void print_baseline(
    racorn_uma::filtered::
        BridgeModeStats const& mode,
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
        << "Distancias medias   : "
        << std::fixed
        << std::setprecision(2)
        << mode.mean_computed_distances
        << '\n';

    std::cout
        << "Predicados medios   : "
        << std::fixed
        << std::setprecision(2)
        << mode.mean_predicate_evaluations
        << '\n';
}

void print_racorn_table(
    racorn_uma::filtered::
        BridgeBudgetResult const& result
) {

    std::cout
        << "\n[RACORN-1 - Bridge budget]\n\n";

    std::cout
        << std::left
        << std::setw(8)
        << "BR"

        << std::right
        << std::setw(12)
        << "Recall"

        << std::setw(12)
        << "Std"

        << std::setw(12)
        << "Min"

        << std::setw(12)
        << "Max"

        << std::setw(14)
        << "Dist"

        << std::setw(14)
        << "Visit"

        << std::setw(14)
        << "Pred"

        << std::setw(14)
        << "Fallback"

        << std::setw(14)
        << "FB queries"

        << '\n';

    for (
        auto const& mode :
        result.racorn_modes
    ) {

        std::cout
            << std::left
            << std::setw(8)
            << std::fixed
            << std::setprecision(2)
            << mode.bridge_budget_ratio

            << std::right
            << std::setw(12)
            << std::setprecision(4)
            << mode.mean_recall_at_k

            << std::setw(12)
            << std::setprecision(5)
            << mode.realization_recall_stddev

            << std::setw(12)
            << std::setprecision(4)
            << mode.min_realization_recall

            << std::setw(12)
            << mode.max_realization_recall

            << std::setw(14)
            << std::setprecision(2)
            << mode.mean_computed_distances

            << std::setw(14)
            << mode.mean_visited_members

            << std::setw(14)
            << mode.mean_predicate_evaluations

            << std::setw(14)
            << mode.mean_fallback_count

            << std::setw(13)
            << std::setprecision(1)
            << (
                mode.fallback_query_fraction *
                100.0
            )
            << "%"

            << '\n';
    }
}

void print_efficiency(
    racorn_uma::filtered::
        BridgeBudgetResult const& result
) {

    std::cout
        << "\n[Reduccion de distancias vs HNSW]\n";

    for (
        auto const& mode :
        result.racorn_modes
    ) {

        const double ratio =
            result.hnsw
                .mean_computed_distances /
            mode.mean_computed_distances;

        const double reduction =
            1.0 -
            (
                mode.mean_computed_distances /
                result.hnsw
                    .mean_computed_distances
            );

        std::cout
            << "BR="
            << std::fixed
            << std::setprecision(2)
            << mode.bridge_budget_ratio

            << " -> "
            << std::setprecision(2)
            << ratio
            << "x menos distancias; "

            << std::setprecision(2)
            << (
                reduction *
                100.0
            )
            << " % reduccion\n";
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
        BridgeBudgetSweep;

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
        BridgeBudgetSweep::run(
            configuration.modulus,
            configuration.threshold,
            configuration.residues
        );

    std::cout
        << "========================================\n"
        << " RACORNUMALab\n"
        << " Gate G4D - Bridge Budget Sweep\n"
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

    print_baseline(
        result.hnsw,
        result.k
    );

    print_baseline(
        result.acorn,
        result.k
    );

    print_racorn_table(
        result
    );

    print_efficiency(
        result
    );

    std::cout
        << "\n[Validacion]\n"
        << "Ground truth exacto : "
        << (
            result
                .exact_ground_truth_complete
                ? "OK"
                : "FAIL"
        )
        << '\n'
        << "Filtros validos     : "
        << (
            result.filters_valid
                ? "SI"
                : "NO"
        )
        << '\n'
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
            << "[G4D] FAIL\n";

        return 1;
    }

    std::cout
        << "[G4D] PASS\n";

    return 0;
}
