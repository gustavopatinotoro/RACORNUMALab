//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar G-REP0 — Repeated Predicate Sanity Baseline.
//
//  Comparación:
//      RACORN-Cached
//      Exact-Materialized
//
//  No existe overlay persistente todavía.
//

#include "core/BuildInfo.hpp"
#include "core/RepeatedPredicateBaseline.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/SystemInfo.hpp"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

char const* yes_no(
    bool value
) noexcept {

    return
        value
            ? "SI"
            : "NO";
}

std::filesystem::path resolve_data_directory(
    int argc,
    char* argv[]
) {

    for (
        int index = 1;
        index < argc;
        ++index
    ) {

        if (
            std::string(
                argv[index]
            ) ==
            "--data-dir"
        ) {

            if (
                index +
                    1 >=
                argc
            ) {

                throw std::runtime_error(
                    "--data-dir requiere una ruta."
                );
            }

            return
                std::filesystem::path(
                    argv[
                        index +
                            1
                    ]
                );
        }
    }

    if (
        char const* value =
            std::getenv(
                "RACORNUMA_SIFT_DIR"
            );
        value != nullptr &&
        value[0] != '\0'
    ) {

        return
            std::filesystem::path(
                value
            );
    }

    throw std::runtime_error(
        "Especifique --data-dir."
    );
}

} // namespace

int main(
    int argc,
    char* argv[]
) {

    using racorn_uma::core::
        BuildInfo;

    using racorn_uma::core::
        RepeatedPredicateBaseline;

    using racorn_uma::integration::
        RacornProbe;

    using racorn_uma::platform::apple::
        SystemInfo;

    try {

        const auto data_directory =
            resolve_data_directory(
                argc,
                argv
            );

        const auto result =
            RepeatedPredicateBaseline::run(
                data_directory
            );

        std::cout
            << "\n========================================\n"
            << " RACORNUMALab\n"
            << " Gate G-REP0 - Repeated Predicate Baseline\n"
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
            << "[Dataset]\n"
            << "Dataset                     : SIFT1M\n"
            << "Subset                      : train[0:100000]\n"
            << "N                           : "
            << result.vector_count
            << '\n'
            << "D                           : "
            << result.dimensions
            << '\n'
            << "Queries                     : "
            << result.query_count
            << '\n'
            << "k                           : "
            << result.k
            << "\n\n";

        std::cout
            << "[Predicado repetido]\n"
            << "tenant                      : "
            << result.tenant_required
            << '\n'
            << "year                        : >= 2022\n"
            << "language                    : >= "
            << result.language_min
            << '\n'
            << "category                    : {3,7,9}\n"
            << "security                    : <= 2\n"
            << "Elegibles                   : "
            << result.eligible_count
            << '\n'
            << "Selectividad                : "
            << std::fixed
            << std::setprecision(4)
            << (
                result.selectivity *
                100.0
            )
            << "%\n\n";

        std::cout
            << "[Representaciones persistidas]\n"
            << "Bitset RACORN               : "
            << result.bitset_logical_bytes
            << " bytes\n"
            << "Lista IDs exact logical     : "
            << result.eligible_logical_bytes
            << " bytes\n"
            << "Lista IDs exact allocated   : "
            << result.eligible_allocated_bytes
            << " bytes\n\n";

        std::cout
            << "[Materializacion one-shot]\n"
            << "Muestras                    : "
            << result.materialization_samples
            << '\n'
            << "RACORN bitset mean/p50      : "
            << std::fixed
            << std::setprecision(4)
            << result.racorn_materialization_mean_ms
            << " / "
            << result.racorn_materialization_p50_ms
            << " ms\n"
            << "Exact IDs mean/p50          : "
            << result.exact_materialization_mean_ms
            << " / "
            << result.exact_materialization_p50_ms
            << " ms\n\n";

        std::cout
            << "[Calidad RACORN]\n"
            << "Mean Recall@10              : "
            << std::fixed
            << std::setprecision(4)
            << result.mean_recall_at_k
            << '\n'
            << "Min Recall@10               : "
            << result.min_recall_at_k
            << '\n'
            << "Perfectas                   : "
            << result.perfect_queries
            << "/"
            << result.query_count
            << '\n'
            << "Imperfectas                 : "
            << result.imperfect_queries
            << '\n'
            << "Underfill                   : "
            << result.underfilled_queries
            << "\n\n";

        std::cout
            << "[Trabajo por query]\n"
            << "RACORN distances            : "
            << result.mean_racorn_computed_distances
            << '\n'
            << "RACORN visited              : "
            << result.mean_racorn_visited_members
            << '\n'
            << "RACORN predicate evals      : "
            << result.mean_racorn_predicate_evaluations
            << '\n'
            << "RACORN fallback             : "
            << result.mean_racorn_fallback_count
            << '\n'
            << "Exact distances             : "
            << result.exact_distance_evaluations_per_query
            << "\n\n";

        std::cout
            << "[Latencia por query]\n"
            << "RACORN-Cached\n"
            << "  mean/p50/p95/p99          : "
            << result.racorn.mean_ms
            << " / "
            << result.racorn.p50_ms
            << " / "
            << result.racorn.p95_ms
            << " / "
            << result.racorn.p99_ms
            << " ms\n"
            << "  QPS mean                  : "
            << std::setprecision(2)
            << result.racorn.qps_from_mean
            << "\n\n";

        std::cout
            << "Exact-Materialized\n"
            << "  mean/p50/p95/p99          : "
            << std::setprecision(4)
            << result.exact.mean_ms
            << " / "
            << result.exact.p50_ms
            << " / "
            << result.exact.p95_ms
            << " / "
            << result.exact.p99_ms
            << " ms\n"
            << "  QPS mean                  : "
            << std::setprecision(2)
            << result.exact.qps_from_mean
            << "\n\n";

        std::cout
            << "[Comparacion]\n"
            << "Exact speedup vs RACORN     : "
            << std::fixed
            << std::setprecision(3)
            << result.exact_speedup_vs_racorn
            << "x\n"
            << "Exact query mas rapido      : "
            << yes_no(
                result.exact_query_faster
            )
            << '\n'
            << "Exact build mas barato      : "
            << yes_no(
                result.exact_build_cheaper
            )
            << '\n';

        if (
            result.exact_break_even_exists
        ) {

            std::cout
                << "Break-even Exact R*         : "
                << std::setprecision(3)
                << result.exact_break_even_repetitions
                << " queries\n";

        } else {

            std::cout
                << "Break-even Exact R*         : NO EXISTE\n";
        }

        std::cout
            << "\n[Coste acumulado estimado]\n"
            << "R       RACORN(ms)      Exact(ms)       Winner\n";

        for (
            auto const& point :
            result.cumulative_costs
        ) {

            std::cout
                << std::left
                << std::setw(8)
                << point.repetitions

                << std::right
                << std::fixed
                << std::setprecision(3)

                << std::setw(15)
                << point.racorn_cumulative_ms

                << std::setw(15)
                << point.exact_cumulative_ms

                << "       "
                << point.winner
                << '\n';
        }

        std::cout
            << "\n[Validacion]\n"
            << "Representation mismatches   : "
            << result.representation_mismatches
            << '\n'
            << "Exact/reference mismatches  : "
            << result.exact_reference_mismatch_queries
            << '\n'
            << "RACORN filter-invalid       : "
            << result.invalid_racorn_result_queries
            << '\n'
            << "Ground truth completo       : "
            << yes_no(
                result.ground_truth_complete
            )
            << '\n'
            << "Exact implementation valida : "
            << yes_no(
                result.exact_implementation_valid
            )
            << '\n'
            << "Materializacion valida      : "
            << yes_no(
                result.predicate_materialization_valid
            )
            << '\n'
            << "RACORN filter valido        : "
            << yes_no(
                result.racorn_results_filter_valid
            )
            << "\n\n";

        std::cout
            << "[Invariantes]\n"
            << "Predicado por query         : NO\n"
            << "Bitset recalculado/query    : NO\n"
            << "IDs recalculados/query      : NO\n"
            << "HNSW dentro del timing      : NO\n"
            << "Metal                       : NO\n"
            << "RACORN modificado           : "
            << yes_no(
                result.racorn_modified
            )
            << "\n\n";

        if (
            !result.valid
        ) {

            std::cerr
                << "[G-REP0] FAIL\n";

            return 1;
        }

        std::cout
            << "[G-REP0] PASS\n";

        return 0;

    } catch (
        std::exception const& error
    ) {

        std::cerr
            << "\n[G-REP0] FAIL: "
            << error.what()
            << '\n';

        return 1;
    }
}
