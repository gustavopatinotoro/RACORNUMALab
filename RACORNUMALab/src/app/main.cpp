//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar G-UMA5E.
//
//  Objetivo:
//  Encontrar el menor bridge_budget_ratio evaluado que satisfaga:
//
//      Recall@10 medio >= 0.99
//
//      AND
//
//      Recall@10 mínimo por query >= 0.99.
//
//  Este Gate no evalúa GPU ni rendimiento temporal.
//
//  Ejecutar en Release.
//

#include "core/BuildInfo.hpp"
#include "core/RacornBridgeBudgetRefinement.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/SystemInfo.hpp"

#include <iomanip>
#include <iostream>

namespace {

char const* yes_no(
    bool value
) noexcept {

    return
        value
            ? "SI"
            : "NO";
}

} // namespace

int main() {

    using racorn_uma::core::
        BuildInfo;

    using racorn_uma::core::
        RacornBridgeBudgetRefinement;

    using racorn_uma::integration::
        RacornProbe;

    using racorn_uma::platform::apple::
        SystemInfo;

    const auto result =
        RacornBridgeBudgetRefinement::run();

    std::cout
        << "\n========================================\n"
        << " RACORNUMALab\n"
        << " Gate G-UMA5E - Bridge Budget Refinement\n"
        << "========================================\n\n";

    std::cout
        << "[Plataforma]\n"
        << "Compilador                 : "
        << BuildInfo::compiler()
        << '\n'
        << "Estandar C++               : "
        << BuildInfo::cpp_standard()
        << '\n'
        << "Arquitectura               : "
        << BuildInfo::architecture()
        << '\n'
        << "Modelo hardware            : "
        << SystemInfo::hardware_model()
        << "\n\n";

    std::cout
        << "[RACORN congelado]\n"
        << "Repositorio                 : "
        << RacornProbe::repository()
        << '\n'
        << "Version                     : "
        << RacornProbe::version()
        << '\n'
        << "Commit                      : "
        << RacornProbe::commit()
        << "\n\n";

    std::cout
        << "[Configuracion]\n"
        << "D                           : "
        << result.dimensions
        << '\n'
        << "M                           : "
        << result.connectivity
        << '\n'
        << "efConstruction              : "
        << result.ef_construction
        << '\n'
        << "efSearch                    : "
        << result.ef_search
        << '\n'
        << "k                           : "
        << result.k
        << '\n'
        << "Queries                     : "
        << result.query_count
        << '\n'
        << "two_hop_threshold           : "
        << std::fixed
        << std::setprecision(2)
        << result.two_hop_threshold
        << '\n'
        << "Mean Recall target          : "
        << std::setprecision(4)
        << result.mean_recall_target
        << '\n'
        << "Min Recall target           : "
        << result.min_recall_target
        << "\n\n";

    std::cout
        << "[Sweep Bridge Budget]\n\n";

    std::cout
        << std::left
        << std::setw(10)
        << "N"

        << std::right
        << std::setw(8)
        << "BR"

        << std::setw(11)
        << "Recall"

        << std::setw(11)
        << "MinRec"

        << std::setw(9)
        << "WorstQ"

        << std::setw(12)
        << "Dist"

        << std::setw(12)
        << "Visited"

        << std::setw(13)
        << "Pred"

        << std::setw(11)
        << "Fallback"

        << std::setw(10)
        << "Robust"

        << '\n';

    for (
        auto const& item :
        result.cases
    ) {

        std::cout
            << std::left
            << std::setw(10)
            << item.vector_count

            << std::right
            << std::fixed
            << std::setprecision(2)

            << std::setw(8)
            << item.bridge_budget_ratio

            << std::setprecision(4)

            << std::setw(11)
            << item.mean_recall_at_k

            << std::setw(11)
            << item.min_recall_at_k

            << std::setw(9)
            << item.worst_query_index

            << std::setw(12)
            << item.mean_computed_distances

            << std::setw(12)
            << item.mean_visited_members

            << std::setw(13)
            << item.mean_predicate_evaluations

            << std::setw(11)
            << item.mean_fallback_count

            << std::setw(10)
            << (
                item.robust_quality_pass
                    ? "PASS"
                    : "FAIL"
            )

            << '\n';
    }

    std::cout
        << "\n[Selectividad y validez]\n";

    for (
        auto const& item :
        result.cases
    ) {

        std::cout
            << "N="
            << item.vector_count
            << " BR="
            << std::fixed
            << std::setprecision(2)
            << item.bridge_budget_ratio

            << " | sel="
            << std::setprecision(4)
            << (
                item.mean_selectivity *
                100.0
            )
            << "%"

            << " | elegibles="
            << item.selected_min
            << ".."
            << item.selected_max

            << " | resultados="
            << item.result_count_min
            << ".."
            << item.result_count_max

            << " | GT="
            << yes_no(
                item.exact_ground_truth_complete
            )

            << " | filtro="
            << yes_no(
                item.returned_results_filter_valid
            )

            << " | valido="
            << yes_no(
                item.valid
            )

            << '\n';
    }

    std::cout
        << "\n[Puntos robustos minimos observados]\n";

    for (
        auto const& point :
        result.optimal_points
    ) {

        std::cout
            << "N="
            << point.vector_count;

        if (
            !point.found
        ) {

            std::cout
                << " | BR*=NO ENCONTRADO\n";

            continue;
        }

        std::cout
            << " | BR*="
            << std::fixed
            << std::setprecision(2)
            << point
                .minimum_robust_bridge_budget_ratio

            << " | Recall="
            << std::setprecision(4)
            << point.mean_recall_at_k

            << " | MinRecall="
            << point.min_recall_at_k

            << " | dist="
            << point.mean_computed_distances

            << " | pred="
            << point.mean_predicate_evaluations

            << '\n';
    }

    std::cout
        << "\n[Invariantes]\n"
        << "RACORN modificado            : "
        << yes_no(
            result.racorn_modified
        )
        << '\n'
        << "Navegacion                  : CPU\n"
        << "Metal                       : NO\n"
        << "GPU                         : NO\n"
        << "Bitset                      : NO\n"
        << "Predicate                   : CPU directo\n"
        << "Ground truth                : independiente\n"
        << "Todos los casos validos     : "
        << yes_no(
            result.all_cases_valid
        )
        << '\n'
        << "BR robusto para cada N      : "
        << yes_no(
            result.robust_point_found_for_all_sizes
        )
        << "\n\n";

    if (
        !result.valid
    ) {

        std::cerr
            << "[G-UMA5E] FAIL\n";

        return 1;
    }

    std::cout
        << "[G-UMA5E] PASS\n";

    return 0;
}
