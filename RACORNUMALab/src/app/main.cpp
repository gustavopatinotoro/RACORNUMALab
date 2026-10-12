//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar G-DATA3A — SIFT Real / CPU vs UMA-Metal.
//
//  Estrategias:
//      CPU-Lazy
//      CPU-Bitset
//      GPU-Bitset
//
//  Ejecutar en Release.
//
//  Dataset:
//      --data-dir <ruta>
//
//  o:
//      RACORNUMA_SIFT_DIR=<ruta>
//

#include "core/BuildInfo.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/SiftUmaPredicateBenchmark.hpp"
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

        const std::string argument =
            argv[index];

        if (
            argument ==
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
        char const* environment =
            std::getenv(
                "RACORNUMA_SIFT_DIR"
            );
        environment != nullptr &&
        environment[0] != '\0'
    ) {

        return
            std::filesystem::path(
                environment
            );
    }

    throw std::runtime_error(
        "Especifique --data-dir o RACORNUMA_SIFT_DIR."
    );
}

void print_latency(
    char const* label,
    racorn_uma::platform::apple::
        SiftUmaLatencyStats const& stats
) {

    std::cout
        << label
        << "\n"
        << "  total p50/p95/p99   : "
        << std::fixed
        << std::setprecision(4)
        << stats.total_p50_ms
        << " / "
        << stats.total_p95_ms
        << " / "
        << stats.total_p99_ms
        << " ms\n"
        << "  prepare p50/p95/p99 : "
        << stats.prepare_p50_ms
        << " / "
        << stats.prepare_p95_ms
        << " / "
        << stats.prepare_p99_ms
        << " ms\n"
        << "  search p50/p95/p99  : "
        << stats.search_p50_ms
        << " / "
        << stats.search_p95_ms
        << " / "
        << stats.search_p99_ms
        << " ms\n"
        << "  QPS desde p50       : "
        << std::setprecision(2)
        << stats.qps_from_p50
        << '\n';

    if (
        stats.gpu_device_timing_available
    ) {

        std::cout
            << "  GPU device p50/p95/p99: "
            << std::setprecision(4)
            << stats.gpu_device_p50_ms
            << " / "
            << stats.gpu_device_p95_ms
            << " / "
            << stats.gpu_device_p99_ms
            << " ms\n";
    }
}

} // namespace

int main(
    int argc,
    char* argv[]
) {

    using racorn_uma::core::
        BuildInfo;

    using racorn_uma::integration::
        RacornProbe;

    using racorn_uma::platform::apple::
        SiftUmaPredicateBenchmark;

    using racorn_uma::platform::apple::
        SystemInfo;

    try {

        const auto data_directory =
            resolve_data_directory(
                argc,
                argv
            );

        const auto result =
            SiftUmaPredicateBenchmark::run(
                data_directory
            );

        std::cout
            << "\n========================================\n"
            << " RACORNUMALab\n"
            << " Gate G-DATA3A - SIFT CPU vs UMA-Metal\n"
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
            << "Directorio externo          : "
            << result.data_directory.string()
            << '\n'
            << "N                           : "
            << result.vector_count
            << '\n'
            << "D                           : "
            << result.dimensions
            << '\n'
            << "Queries semantica           : "
            << result.semantic_query_count
            << '\n'
            << "Queries benchmark           : "
            << result.benchmark_query_count
            << '\n'
            << "Selectividad media          : "
            << std::fixed
            << std::setprecision(4)
            << (
                result.mean_selectivity *
                100.0
            )
            << "%\n"
            << "Elegibles/query             : "
            << result.selected_min
            << ".."
            << result.selected_max
            << "\n\n";

        std::cout
            << "[Configuracion RACORN]\n"
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
            << "BR                          : "
            << std::fixed
            << std::setprecision(2)
            << result.bridge_budget_ratio
            << '\n'
            << "two_hop_threshold           : "
            << result.two_hop_threshold
            << '\n'
            << "fallback_exact_threshold    : 0.0\n\n";

        std::cout
            << "[Metal]\n"
            << "Metal disponible            : "
            << yes_no(
                result.metal_available
            )
            << '\n'
            << "Dispositivo                 : "
            << result.device_name
            << '\n'
            << "Memoria unificada           : "
            << yes_no(
                result.unified_memory
            )
            << '\n'
            << "Kernel                      : "
            << result.kernel_name
            << '\n'
            << "Pipeline                    : "
            << yes_no(
                result.pipeline_available
            )
            << '\n'
            << "Execution width             : "
            << result.thread_execution_width
            << '\n'
            << "Threads/group               : "
            << result.threads_per_threadgroup
            << "\n\n";

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
            << "Queries perfectas           : "
            << result.perfect_queries
            << "/"
            << result.semantic_query_count
            << '\n'
            << "Queries imperfectas         : "
            << result.imperfect_queries
            << '\n'
            << "Underfill                   : "
            << result.underfilled_queries
            << "\n\n";

        std::cout
            << "[Trabajo RACORN]\n"
            << "computed_distances mean     : "
            << result.mean_computed_distances
            << '\n'
            << "visited_members mean        : "
            << result.mean_visited_members
            << '\n'
            << "predicate_evaluations mean  : "
            << result.mean_predicate_evaluations
            << '\n'
            << "fallback_count mean         : "
            << result.mean_fallback_count
            << "\n\n";

        std::cout
            << "[Equivalencia semantica]\n"
            << "Bitset word mismatches      : "
            << result.bitset_word_mismatches
            << '\n'
            << "Predicate mismatches        : "
            << result.predicate_representation_mismatches
            << '\n'
            << "Top-k mismatches            : "
            << result.topk_mismatch_queries
            << '\n'
            << "Result-count mismatches     : "
            << result.result_count_mismatch_queries
            << '\n'
            << "Recall mismatches           : "
            << result.recall_mismatch_queries
            << '\n'
            << "Distance-count mismatches   : "
            << result.computed_distances_mismatch_queries
            << '\n'
            << "Visited mismatches          : "
            << result.visited_members_mismatch_queries
            << '\n'
            << "Predicate-count mismatches  : "
            << result.predicate_evaluations_mismatch_queries
            << '\n'
            << "Fallback mismatches         : "
            << result.fallback_count_mismatch_queries
            << '\n'
            << "Semantica identica          : "
            << yes_no(
                result.navigation_semantics_identical
            )
            << "\n\n";

        std::cout
            << "[Benchmark]\n"
            << "Warm-up rounds              : "
            << result.warmup_rounds
            << '\n'
            << "Measured rounds             : "
            << result.measured_rounds
            << "\n\n";

        print_latency(
            "CPU-Lazy",
            result.cpu_lazy
        );

        std::cout
            << '\n';

        print_latency(
            "CPU-Bitset",
            result.cpu_bitset
        );

        std::cout
            << '\n';

        print_latency(
            "GPU-Bitset",
            result.gpu_bitset
        );

        std::cout
            << "\n[Speedup p50 end-to-end]\n"
            << "CPU-Bitset / Lazy           : "
            << std::fixed
            << std::setprecision(3)
            << result.cpu_bitset_speedup_vs_lazy
            << "x\n"
            << "GPU-Bitset / Lazy           : "
            << result.gpu_bitset_speedup_vs_lazy
            << "x\n"
            << "GPU-Bitset / CPU-Bitset     : "
            << result.gpu_vs_cpu_bitset
            << "x\n"
            << "Ganador p50                 : "
            << result.p50_winner
            << "\n\n";

        std::cout
            << "[Validez]\n"
            << "Base valida                 : "
            << yes_no(
                result.base_file_valid
            )
            << '\n'
            << "Queries validas              : "
            << yes_no(
                result.query_file_valid
            )
            << '\n'
            << "Metadata valida              : "
            << yes_no(
                result.metadata_file_valid
            )
            << '\n'
            << "Ground truth completo        : "
            << yes_no(
                result.exact_ground_truth_complete
            )
            << '\n'
            << "Filtro resultados valido     : "
            << yes_no(
                result.filter_results_valid
            )
            << '\n'
            << "Semantica CPU/GPU identica  : "
            << yes_no(
                result.navigation_semantics_identical
            )
            << "\n\n";

        std::cout
            << "[Invariantes]\n"
            << "RACORN modificado            : "
            << yes_no(
                result.racorn_modified
            )
            << '\n'
            << "Navegacion                  : CPU\n"
            << "Metal                       : solo predicado\n"
            << "Metadata                    : Shared\n"
            << "CPU bitset                  : Shared\n"
            << "GPU bitset                  : Shared\n"
            << "Copias CPU-VRAM             : NO\n"
            << "Exact fallback              : NO\n\n";

        if (
            !result.metal_error.empty()
        ) {

            std::cout
                << "[Metal error]\n"
                << result.metal_error
                << "\n\n";
        }

        if (
            !result.valid
        ) {

            std::cerr
                << "[G-DATA3A] FAIL\n";

            return 1;
        }

        std::cout
            << "[G-DATA3A] PASS\n";

        return 0;

    } catch (
        std::exception const& error
    ) {

        std::cerr
            << "\n[G-DATA3A] FAIL: "
            << error.what()
            << '\n';

        return 1;
    }
}
