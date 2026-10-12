//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar:
//
//      G-REPAIR0
//      Persistent Repair vs RACORN Transient.
//
//  El mismo predicado se reutiliza en todas las queries.
//

#include "core/BuildInfo.hpp"
#include "core/PersistentRepairBenchmark.hpp"
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
                    "--data-dir requiere ruta."
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

void print_strategy(
    char const* name,
    racorn_uma::core::
        RepairLatencyStats const& stats
) {

    std::cout
        << name
        << '\n'
        << "  mean/p50/p95/p99          : "
        << std::fixed
        << std::setprecision(4)
        << stats.mean_ms
        << " / "
        << stats.p50_ms
        << " / "
        << stats.p95_ms
        << " / "
        << stats.p99_ms
        << " ms\n"
        << "  QPS mean                  : "
        << std::setprecision(2)
        << stats.qps_mean
        << '\n'
        << "  distances/query           : "
        << std::setprecision(2)
        << stats.mean_computed_distances
        << '\n'
        << "  visited/query             : "
        << stats.mean_visited_members
        << '\n';
}

} // namespace

int main(
    int argc,
    char* argv[]
) {

    using racorn_uma::core::
        BuildInfo;

    using racorn_uma::core::
        PersistentRepairBenchmark;

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
            PersistentRepairBenchmark::run(
                data_directory
            );

        std::cout
            << "\n========================================\n"
            << " RACORNUMALab\n"
            << " Gate G-REPAIR0\n"
            << " Persistent vs Transient Repair\n"
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
            << '\n'
            << "M                           : "
            << result.connectivity
            << '\n'
            << "M0                          : "
            << result.connectivity_base
            << '\n'
            << "efConstruction              : "
            << result.ef_construction
            << '\n'
            << "efSearch RACORN             : "
            << result.ef_search
            << '\n'
            << "efSearch Persistent         : "
            << result.persistent_ef_search
            << '\n'
            << "BR                          : "
            << std::fixed
            << std::setprecision(2)
            << result.bridge_budget_ratio
            << '\n'
            << "two_hop_threshold           : "
            << result.two_hop_threshold
            << '\n'
            << "Exact fallback              : NO\n\n";

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
            << "%\n"
            << "Bitset común                : "
            << result.predicate_bitset_bytes
            << " bytes\n\n";

        std::cout
            << "[Snapshot HNSW]\n"
            << "Serializado temporal        : "
            << result.serialized_index_bytes
            << " bytes\n"
            << "L0 edges parseadas          : "
            << result.parsed_l0_directed_edges
            << '\n'
            << "L0 edges stats(0)           : "
            << result.index_l0_directed_edges
            << '\n'
            << "Snapshot valido             : "
            << yes_no(
                result.topology_snapshot_valid
            )
            << '\n'
            << "Snapshot time               : "
            << std::fixed
            << std::setprecision(4)
            << result.topology_snapshot_ms
            << " ms\n\n";

        std::cout
            << "[Reparacion persistente]\n"
            << "Shortcut proposal budget    : "
            << result.shortcut_proposal_budget
            << '\n'
            << "Componentes inducidos       : "
            << result.components_induced
            << '\n'
            << "Aristas inducidas           : "
            << result.induced_undirected_edges
            << '\n'
            << "Shortcuts 2-hop añadidos    : "
            << result.two_hop_shortcut_edges
            << '\n'
            << "Componentes tras 2-hop      : "
            << result.components_after_two_hop
            << '\n'
            << "Edges completion            : "
            << result.completion_edges
            << '\n'
            << "Componentes finales         : "
            << result.components_final
            << '\n'
            << "Aristas finales             : "
            << result.final_undirected_edges
            << '\n'
            << "Overlay logical             : "
            << result.overlay_logical_bytes
            << " bytes\n"
            << "Overlay allocated           : "
            << result.overlay_allocated_bytes
            << " bytes\n"
            << "Repair build                : "
            << result.repair_build_ms
            << " ms\n"
            << "Snapshot + repair           : "
            << result.total_persistent_build_ms
            << " ms\n\n";

        std::cout
            << "[Calidad]\n"
            << "RACORN-Transient Recall     : "
            << std::fixed
            << std::setprecision(4)
            << result.transient_mean_recall
            << " | min="
            << result.transient_min_recall
            << '\n'
            << "Persistent Recall           : "
            << result.persistent_mean_recall
            << " | min="
            << result.persistent_min_recall
            << '\n'
            << "RACORN perfectas            : "
            << result.transient_perfect_queries
            << "/"
            << result.query_count
            << '\n'
            << "Persistent perfectas        : "
            << result.persistent_perfect_queries
            << "/"
            << result.query_count
            << '\n'
            << "RACORN underfill            : "
            << result.transient_underfilled_queries
            << '\n'
            << "Persistent underfill        : "
            << result.persistent_underfilled_queries
            << '\n'
            << "Persistent quality admisible: "
            << yes_no(
                result.persistent_quality_admissible
            )
            << "\n\n";

        std::cout
            << "[Trabajo RACORN Transient]\n"
            << "predicate evals/query       : "
            << result.transient_mean_predicate_evaluations
            << '\n'
            << "fallback/query              : "
            << result.transient_mean_fallback_count
            << "\n\n";

        std::cout
            << "[Latencia por query]\n";

        print_strategy(
            "RACORN-Transient",
            result.transient
        );

        std::cout
            << '\n';

        print_strategy(
            "Persistent-Repair",
            result.persistent
        );

        std::cout
            << "\n[Amortizacion]\n"
            << "Persistent speedup/query    : "
            << std::fixed
            << std::setprecision(3)
            << result.persistent_speedup_vs_transient
            << "x\n"
            << "Persistent query faster     : "
            << yes_no(
                result.persistent_query_faster
            )
            << '\n';

        if (
            result.break_even_exists
        ) {

            std::cout
                << "R* repair-only              : "
                << result.break_even_repair_only_repetitions
                << " queries\n"
                << "R* snapshot+repair          : "
                << result.break_even_total_repetitions
                << " queries\n";

        } else {

            std::cout
                << "R*                          : NO EXISTE\n";
        }

        std::cout
            << "\n[Coste acumulado]\n"
            << "R       Transient(ms)   Persistent(ms)      Winner\n";

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
                << point.transient_cumulative_ms

                << std::setw(17)
                << point.persistent_cumulative_ms

                << "      "
                << point.winner
                << '\n';
        }

        std::cout
            << "\n[Validez]\n"
            << "Ground truth completo       : "
            << yes_no(
                result.ground_truth_complete
            )
            << '\n'
            << "Overlay conectado           : "
            << yes_no(
                result.overlay_connected
            )
            << '\n'
            << "Transient invalid results   : "
            << result.invalid_transient_results
            << '\n'
            << "Persistent invalid results  : "
            << result.invalid_persistent_results
            << "\n\n";

        std::cout
            << "[Invariantes]\n"
            << "Mismo predicado             : SI\n"
            << "Predicado recalculado/query : NO\n"
            << "RACORN reparación/query     : SI\n"
            << "Persistent repair/query     : NO\n"
            << "HNSW modificado             : NO\n"
            << "Vectores duplicados         : NO\n"
            << "Metal                       : NO\n"
            << "RACORN modificado           : "
            << yes_no(
                result.racorn_modified
            )
            << "\n\n";

        std::cout
            << "========================================\n"
            << " DECISION: "
            << result.decision
            << '\n'
            << "========================================\n\n";

        if (
            !result.valid
        ) {

            std::cerr
                << "[G-REPAIR0] FAIL\n";

            return 1;
        }

        std::cout
            << "[G-REPAIR0] PASS\n";

        return 0;

    } catch (
        std::exception const& error
    ) {

        std::cerr
            << "\n[G-REPAIR0] FAIL: "
            << error.what()
            << '\n';

        return 1;
    }
}
