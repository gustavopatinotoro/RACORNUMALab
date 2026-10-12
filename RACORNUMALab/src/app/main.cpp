//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar G-UMA5B y comparar end-to-end:
//
//      CPU-Lazy
//      CPU-Bitset
//      GPU-Bitset
//
//  sobre búsquedas RACORN reales.
//
//  Gate:
//  G-UMA5B — RACORN End-to-End CPU/GPU.
//
//  Condición PASS:
//  Las tres estrategias deben conservar exactamente la semántica
//  RACORN antes de interpretar cualquier diferencia de rendimiento.
//
//  Importante:
//  Ejecutar este Gate exclusivamente en configuración Release.
//

#include "core/BuildInfo.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/RacornEndToEndBenchmark.hpp"
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
        RacornEndToEndBenchmark;

    using racorn_uma::platform::apple::
        SystemInfo;

    const auto result =
        RacornEndToEndBenchmark::run();

    std::cout
        << "========================================\n"
        << " RACORNUMALab\n"
        << " Gate G-UMA5B - RACORN End-to-End\n"
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
        << "bridge_budget_ratio         : "
        << result.bridge_budget_ratio
        << "\n\n";

    std::cout
        << "[Benchmark]\n"
        << "Warmup rounds               : "
        << result.warmup_rounds
        << '\n'
        << "Measured rounds             : "
        << result.measured_rounds
        << '\n'
        << "Orden estrategias           : ROTADO\n"
        << "Indice dentro del timing    : NO\n"
        << "Metadata allocation timing  : NO\n"
        << "Pipeline creation timing    : NO\n\n";

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
        << "Threads/group max           : "
        << result
            .max_total_threads_per_threadgroup
        << '\n'
        << "Threads/group usados        : "
        << result.threads_per_threadgroup
        << "\n\n";

    std::cout
        << "[Latencia End-to-End p50/p95 ms]\n\n";

    std::cout
        << std::left
        << std::setw(10)
        << "N"

        << std::right
        << std::setw(10)
        << "Sel.%"

        << std::setw(12)
        << "Lazy50"

        << std::setw(12)
        << "Lazy95"

        << std::setw(12)
        << "CPUbit50"

        << std::setw(12)
        << "CPUbit95"

        << std::setw(12)
        << "GPUbit50"

        << std::setw(12)
        << "GPUbit95"

        << std::setw(15)
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
            << std::setw(10)
            << std::fixed
            << std::setprecision(4)
            << (
                item.mean_selectivity *
                100.0
            )

            << std::setw(12)
            << item
                .cpu_lazy
                .total_p50_ms

            << std::setw(12)
            << item
                .cpu_lazy
                .total_p95_ms

            << std::setw(12)
            << item
                .cpu_bitset
                .total_p50_ms

            << std::setw(12)
            << item
                .cpu_bitset
                .total_p95_ms

            << std::setw(12)
            << item
                .gpu_bitset
                .total_p50_ms

            << std::setw(12)
            << item
                .gpu_bitset
                .total_p95_ms

            << std::setw(15)
            << item.p50_winner

            << '\n';
    }

    std::cout
        << "\n[Descomposicion p50 ms]\n\n";

    std::cout
        << std::left
        << std::setw(10)
        << "N"

        << std::right
        << std::setw(12)
        << "CPUprep"

        << std::setw(12)
        << "CPUsearch"

        << std::setw(12)
        << "GPUprep"

        << std::setw(12)
        << "GPUdev"

        << std::setw(12)
        << "GPUsearch"

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
            << std::setw(12)
            << std::fixed
            << std::setprecision(4)
            << item
                .cpu_bitset
                .prepare_p50_ms

            << std::setw(12)
            << item
                .cpu_bitset
                .search_p50_ms

            << std::setw(12)
            << item
                .gpu_bitset
                .prepare_p50_ms;

        if (
            item
                .gpu_bitset
                .gpu_device_timing_available
        ) {

            std::cout
                << std::setw(12)
                << item
                    .gpu_bitset
                    .gpu_device_p50_ms;

        } else {

            std::cout
                << std::setw(12)
                << "N/A";
        }

        std::cout
            << std::setw(12)
            << item
                .gpu_bitset
                .search_p50_ms

            << '\n';
    }

    std::cout
        << "\n[Speedup vs CPU-Lazy p50]\n";

    for (
        auto const& item :
        result.cases
    ) {

        std::cout
            << "N="
            << item.vector_count
            << " | CPU-Bitset="
            << std::fixed
            << std::setprecision(3)
            << item
                .cpu_bitset_speedup_vs_lazy
            << "x | GPU-Bitset="
            << item
                .gpu_bitset_speedup_vs_lazy
            << "x\n";
    }

    std::cout
        << "\n[Calidad y trabajo RACORN]\n";

    for (
        auto const& item :
        result.cases
    ) {

        std::cout
            << "N="
            << item.vector_count
            << " | elegibles="
            << item.selected_min
            << ".."
            << item.selected_max
            << " | Recall@"
            << result.k
            << "="
            << std::fixed
            << std::setprecision(4)
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

            << "  Bitset word mismatches      : "
            << item.bitset_word_mismatches
            << '\n'

            << "  Predicate/bitset mismatches : "
            << item
                .predicate_representation_mismatches
            << '\n'

            << "  Top-k mismatch queries      : "
            << item.topk_mismatch_queries
            << '\n'

            << "  Recall mismatch queries     : "
            << item.recall_mismatch_queries
            << '\n'

            << "  Dist mismatch queries       : "
            << item
                .computed_distances_mismatch_queries
            << '\n'

            << "  Visited mismatch queries    : "
            << item
                .visited_members_mismatch_queries
            << '\n'

            << "  Pred-eval mismatch queries  : "
            << item
                .predicate_evaluations_mismatch_queries
            << '\n'

            << "  Fallback mismatch queries   : "
            << item
                .fallback_count_mismatch_queries
            << '\n'

            << "  Ground truth completo       : "
            << yes_no(
                item.exact_ground_truth_complete
            )
            << '\n'

            << "  Resultados filtro validos   : "
            << yes_no(
                item.filter_results_valid
            )
            << '\n'

            << "  Semantica identica          : "
            << yes_no(
                item.navigation_semantics_identical
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
        << "CPU-Lazy disponible          : SI\n"
        << "CPU-Bitset disponible        : SI\n"
        << "GPU-Bitset disponible        : SI\n"
        << "Navegacion GPU               : NO\n"
        << "Heaps GPU                    : NO\n"
        << "Bridges GPU                  : NO\n"
        << "Copia explicita a VRAM       : NO\n"
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
            << "[G-UMA5B] FAIL\n";

        return 1;
    }

    std::cout
        << "[G-UMA5B] PASS\n";

    return 0;
}
