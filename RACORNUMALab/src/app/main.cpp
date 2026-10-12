//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar el Gate final de la rama Metal-predicate:
//
//      G-DATA3B — SIFT UMA Pipeline.
//
//  Resultado posible:
//
//      CONTINUE_METAL
//
//  o:
//
//      STOP_METAL_PREDICATE
//
//  Un STOP no implica FAIL experimental.
//

#include "core/BuildInfo.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/SiftUmaPipelineBenchmark.hpp"
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
                argv[
                    index +
                    1
                ];
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

        return value;
    }

    throw std::runtime_error(
        "Especifique --data-dir."
    );
}

void print_stats(
    char const* name,
    racorn_uma::platform::apple::
        PipelineLatencyStats const& stats
) {

    std::cout
        << name
        << '\n'
        << "  batch p50/p95/p99      : "
        << std::fixed
        << std::setprecision(3)
        << stats.batch_p50_ms
        << " / "
        << stats.batch_p95_ms
        << " / "
        << stats.batch_p99_ms
        << " ms\n"
        << "  exposed wait p50/p95   : "
        << stats.exposed_wait_p50_ms
        << " / "
        << stats.exposed_wait_p95_ms
        << " ms\n"
        << "  prepare work p50/p95   : "
        << stats.prepare_work_p50_ms
        << " / "
        << stats.prepare_work_p95_ms
        << " ms\n"
        << "  QPS p50                : "
        << std::setprecision(2)
        << stats.qps_p50
        << "\n";
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
        SiftUmaPipelineBenchmark;

    using racorn_uma::platform::apple::
        SystemInfo;

    try {

        const auto data_directory =
            resolve_data_directory(
                argc,
                argv
            );

        const auto result =
            SiftUmaPipelineBenchmark::run(
                data_directory
            );

        std::cout
            << "\n========================================\n"
            << " RACORNUMALab\n"
            << " Gate G-DATA3B - SIFT UMA Pipeline\n"
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
            << "Queries/batch               : "
            << result.query_count
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
            << "BR                          : "
            << std::fixed
            << std::setprecision(2)
            << result.bridge_budget_ratio
            << '\n'
            << "two_hop_threshold           : "
            << result.two_hop_threshold
            << '\n'
            << "Exact fallback              : NO\n"
            << "Warmup batches              : "
            << result.warmup_batches
            << '\n'
            << "Measured batches            : "
            << result.measured_batches
            << "\n\n";

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
            << "Threads/group               : "
            << result.threads_per_threadgroup
            << "\n\n";

        std::cout
            << "[Semantica]\n"
            << "Bitset mismatches           : "
            << result.bitset_word_mismatches
            << '\n'
            << "CPU top-k mismatches        : "
            << result.cpu_topk_mismatches
            << '\n'
            << "CPU counter mismatches      : "
            << result.cpu_counter_mismatches
            << '\n'
            << "GPU top-k mismatches        : "
            << result.gpu_topk_mismatches
            << '\n'
            << "GPU counter mismatches      : "
            << result.gpu_counter_mismatches
            << '\n'
            << "Identidad semantica         : "
            << yes_no(
                result.semantic_identity
            )
            << "\n\n";

        std::cout
            << "[Calidad - 256 queries]\n"
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
            << "[Pipeline]\n";

        print_stats(
            "CPU-Lazy",
            result.cpu_lazy
        );

        std::cout
            << '\n';

        print_stats(
            "CPU-Bitset-Pipeline",
            result.cpu_pipeline
        );

        std::cout
            << '\n';

        print_stats(
            "GPU-Bitset-Pipeline",
            result.gpu_pipeline
        );

        std::cout
            << "\n[Speedup]\n"
            << "CPU-Pipeline / Lazy         : "
            << std::fixed
            << std::setprecision(3)
            << result.cpu_speedup_vs_lazy
            << "x\n"
            << "GPU-Pipeline / Lazy         : "
            << result.gpu_speedup_vs_lazy
            << "x\n"
            << "GPU-Pipeline / CPU-Pipeline : "
            << result.gpu_speedup_vs_cpu
            << "x\n"
            << "GPU p95 / CPU p95           : "
            << result.gpu_p95_ratio_vs_cpu
            << "x\n\n";

        std::cout
            << "[Criterio de corte predefinido]\n"
            << "GPU p50 >= 1.02x CPU        : "
            << yes_no(
                result.gpu_p50_margin_pass
            )
            << '\n'
            << "GPU p95 <= 1.02x CPU        : "
            << yes_no(
                result.gpu_p95_guard_pass
            )
            << '\n'
            << "Metal merece continuar      : "
            << yes_no(
                result.gpu_pipeline_worthwhile
            )
            << "\n\n";

        std::cout
            << "========================================\n"
            << " DECISION: "
            << result.decision
            << '\n'
            << "========================================\n\n";

        std::cout
            << "[Invariantes]\n"
            << "RACORN modificado           : "
            << yes_no(
                result.racorn_modified
            )
            << '\n'
            << "Navegacion RACORN           : CPU\n"
            << "CPU worker persistente      : SI\n"
            << "GPU preparacion asincrona   : SI\n"
            << "Double buffering CPU        : SI\n"
            << "Double buffering GPU        : SI\n"
            << "Navegacion GPU              : NO\n"
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
                << "[G-DATA3B] FAIL\n";

            return 1;
        }

        std::cout
            << "[G-DATA3B] PASS\n";

        return 0;

    } catch (
        std::exception const& error
    ) {

        std::cerr
            << "\n[G-DATA3B] FAIL: "
            << error.what()
            << '\n';

        return 1;
    }
}
