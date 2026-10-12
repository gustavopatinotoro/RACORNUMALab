//
//  main.cpp
//  RACORNUMALab
//
//  Gate:
//  G4A — HNSW filtrado vs ACORN-1 vs RACORN-1.
//
//  Propósito:
//  Ejecutar la primera comparación funcional de las tres rutas de
//  búsqueda filtrada sobre un único índice HNSW.
//

#include "core/BuildInfo.hpp"
#include "filtered/FilteredSearchBaseline.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/SystemInfo.hpp"

#include <iomanip>
#include <iostream>

namespace {

void print_mode(
    racorn_uma::filtered::
        FilteredModeStats const& mode
) {

    std::cout
        << '\n'
        << '['
        << mode.name
        << "]\n";

    std::cout
        << "Recall@10 medio      : "
        << std::fixed
        << std::setprecision(4)
        << mode.mean_recall_at_k
        << '\n';

    std::cout
        << "Resultados medios    : "
        << mode.mean_result_count
        << '\n';

    std::cout
        << "Distancias medias    : "
        << mode.mean_computed_distances
        << '\n';

    std::cout
        << "Visitados medios     : "
        << mode.mean_visited_members
        << '\n';

    std::cout
        << "Predicados medios    : "
        << mode.mean_predicate_evaluations
        << '\n';

    std::cout
        << "Fallback medio       : "
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

int main() {

    using racorn_uma::core::
        BuildInfo;

    using racorn_uma::filtered::
        FilteredSearchBaseline;

    using racorn_uma::integration::
        RacornProbe;

    using racorn_uma::platform::apple::
        SystemInfo;

    std::cout
        << "========================================\n"
        << " RACORNUMALab\n"
        << " Gate G4A - Busqueda filtrada\n"
        << "========================================\n\n";

    std::cout
        << "[Plataforma]\n"
        << "Compilador        : "
        << BuildInfo::compiler()
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

    const auto result =
        FilteredSearchBaseline::run();

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
        << "Elegibles          : "
        << result.eligible_count
        << '\n'
        << "Selectividad       : "
        << std::fixed
        << std::setprecision(4)
        << (
            result.observed_selectivity *
            100.0
        )
        << " %\n";

    print_mode(
        result.hnsw
    );

    print_mode(
        result.acorn
    );

    print_mode(
        result.racorn
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
        << "ASF RACORN observado: "
        << (
            result
                .racorn_fallback_observed
                ? "SI"
                : "NO"
        )
        << "\n\n";

    if (!result.passed) {

        std::cerr
            << "[G4A] FAIL\n";

        return 1;
    }

    std::cout
        << "[G4A] PASS\n";

    return 0;
}
