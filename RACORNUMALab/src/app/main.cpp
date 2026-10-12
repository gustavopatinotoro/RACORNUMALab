//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar G-UMA5C y comparar throughput sostenido:
//
//      CPU-Lazy
//      CPU-Bitset Serial
//      CPU-Bitset Pipeline
//      GPU-Bitset Pipeline
//
//  RACORN siempre navega en CPU.
//
//  Gate:
//  G-UMA5C — Pipeline CPU/GPU.
//
//  Ejecutar exclusivamente en Release.
//

#include "core/BuildInfo.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/RacornPipelineBenchmark.hpp"
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

    using racorn_uma::integration::
        RacornProbe;

    using racorn_uma::platform::apple::
        RacornPipelineBenchmark;

    using racorn_uma::platform::apple::
        SystemInfo;

    const auto result =
        RacornPipelineBenchmark::run();

    std::cout
        << "\n========================================\n"
        << " RACORNUMALab\n"
        << " Gate G-UMA5C - Pipeline CPU/GPU\n"
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
        << "Queries/lote                : "
        << result.query_count
        << '\n'
        << "Warmup batches              : "
        << result.warmup_batches
        << '\n'
        << "Measured batches            : "
        << result.measured_batches
        << '\n'
        << "two_hop_threshold           : "
        << std::fixed
        << std::setprecision(2)
        << result.two_hop_threshold
        << '\n'
        << "bridge_budget_ratio         : "
        << result.bridge_budget_ratio
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
        << "[Throughput - batch p50/p95 ms]\n\n";

    std::cout
        << std::left
        << std::setw(10)
        << "N"

        << std::right
        << std::setw(12)
        << "Lazy50"

        << std::setw(12)
        << "Serial50"

        << std::setw(12)
        << "CPUpipe50"

        << std::setw(12)
        << "GPUpipe50"

        << std::setw(12)
        << "GPUpipe95"

        << std::setw(24)
        << "Winner"

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
            << std::setprecision(4)

            << std::setw(12)
            << item.cpu_lazy.batch_p50_ms

            << std::setw(12)
            << item
                .cpu_bitset_serial
                .batch_p50_ms

            << std::setw(12)
            << item
                .cpu_bitset_pipeline
                .batch_p50_ms

            << std::setw(12)
            << item
                .gpu_bitset_pipeline
                .batch_p50_ms

            << std::setw(12)
            << item
                .gpu_bitset_pipeline
                .batch_p95_ms

            << std::setw(24)
            << item.throughput_winner

            << '\n';
    }

    std::cout
        << "\n[Throughput p50]\n";

    for (
        auto const& item :
        result.cases
    ) {

        std::cout
            << "N="
            << item.vector_count
            << '\n'

            << "  CPU-Lazy             : "
            << std::fixed
            << std::setprecision(2)
            << item.cpu_lazy.qps_p50
            << " QPS\n"

            << "  CPU-Bitset Serial    : "
            << item
                .cpu_bitset_serial
                .qps_p50
            << " QPS\n"

            << "  CPU-Bitset Pipeline  : "
            << item
                .cpu_bitset_pipeline
                .qps_p50
            << " QPS\n"

            << "  GPU-Bitset Pipeline  : "
            << item
                .gpu_bitset_pipeline
                .qps_p50
            << " QPS\n";
    }

    std::cout
        << "\n[Espera expuesta del pipeline p50/p95]\n";

    for (
        auto const& item :
        result.cases
    ) {

        std::cout
            << "N="
            << item.vector_count

            << " | CPU="
            << std::fixed
            << std::setprecision(4)
            << item
                .cpu_bitset_pipeline
                .exposed_wait_p50_ms
            << "/"
            << item
                .cpu_bitset_pipeline
                .exposed_wait_p95_ms
            << " ms"

            << " | GPU="
            << item
                .gpu_bitset_pipeline
                .exposed_wait_p50_ms
            << "/"
            << item
                .gpu_bitset_pipeline
                .exposed_wait_p95_ms
            << " ms\n";
    }

    std::cout
        << "\n[Speedup]\n";

    for (
        auto const& item :
        result.cases
    ) {

        std::cout
            << "N="
            << item.vector_count

            << " | CPU-pipeline vs serial="
            << std::fixed
            << std::setprecision(3)
            << item
                .cpu_pipeline_speedup_vs_serial
            << "x"

            << " | GPU-pipeline vs serial="
            << item
                .gpu_pipeline_speedup_vs_serial
            << "x"

            << " | GPU-pipeline vs lazy="
            << item
                .gpu_pipeline_speedup_vs_lazy
            << "x\n";
    }

    std::cout
        << "\n[Calidad RACORN]\n";

    for (
        auto const& item :
        result.cases
    ) {

        std::cout
            << "N="
            << item.vector_count

            << " | sel="
            << std::fixed
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

            << " | Recall@"
            << result.k
            << "="
            << item.mean_recall_at_k

            << " | dist="
            << item.mean_computed_distances

            << " | visited="
            << item.mean_visited_members

            << " | pred="
            << item.mean_predicate_evaluations

            << " | fallback="
            << item.mean_fallback_count

            << '\n';
    }

    std::cout
        << "\n[Equivalencia semantica]\n";

    for (
        auto const& item :
        result.cases
    ) {

        std::cout
            << "N="
            << item.vector_count
            << '\n'

            << "  Bitset mismatches             : "
            << item.bitset_word_mismatches
            << '\n'

            << "  Predicate mismatches          : "
            << item
                .predicate_representation_mismatches
            << '\n'

            << "  CPU pipeline top-k mismatch   : "
            << item
                .cpu_pipeline_topk_mismatch_queries
            << '\n'

            << "  CPU pipeline counter mismatch : "
            << item
                .cpu_pipeline_counter_mismatch_queries
            << '\n'

            << "  GPU pipeline top-k mismatch   : "
            << item
                .gpu_pipeline_topk_mismatch_queries
            << '\n'

            << "  GPU pipeline counter mismatch : "
            << item
                .gpu_pipeline_counter_mismatch_queries
            << '\n'

            << "  CPU pipeline semantica        : "
            << yes_no(
                item
                    .cpu_pipeline_semantics_identical
            )
            << '\n'

            << "  GPU pipeline semantica        : "
            << yes_no(
                item
                    .gpu_pipeline_semantics_identical
            )
            << "\n\n";
    }

    if (
        !result.metal_error.empty()
    ) {

        std::cout
            << "[Error Metal]\n"
            << result.metal_error
            << "\n\n";
    }

    std::cout
        << "[Invariantes]\n"
        << "RACORN modificado            : "
        << yes_no(
            result.racorn_modified
        )
        << '\n'
        << "Navegacion RACORN            : CPU\n"
        << "CPU worker persistente       : SI\n"
        << "GPU preparacion asincrona    : SI\n"
        << "Double buffering CPU         : SI\n"
        << "Double buffering GPU         : SI\n"
        << "Navegacion GPU               : NO\n"
        << "Heaps GPU                    : NO\n"
        << "Bridges GPU                  : NO\n"
        << "Semantica global identica    : "
        << yes_no(
            result
                .all_cases_semantically_identical
        )
        << "\n\n";

    if (
        !result.valid
    ) {

        std::cerr
            << "[G-UMA5C] FAIL\n";

        return 1;
    }

    std::cout
        << "[G-UMA5C] PASS\n";

    return 0;
}
