//
//  main.cpp
//  RACORNUMALab
//
//  Propósito:
//  Ejecutar G-UMA5D.
//
//  Objetivo:
//  Identificar el plan heterogéneo más rápido únicamente en regiones
//  donde Recall@10 medio >= 0.99.
//
//  Gate:
//  G-UMA5D — Quality-Constrained Heterogeneous Crossover.
//
//  Ejecutar exclusivamente en Release.
//

#include "core/BuildInfo.hpp"
#include "integration/RacornProbe.hpp"
#include "platform/apple/RacornQualityCrossoverBenchmark.hpp"
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
        RacornQualityCrossoverBenchmark;

    using racorn_uma::platform::apple::
        SystemInfo;

    const auto result =
        RacornQualityCrossoverBenchmark::run();

    std::cout
        << "\n========================================\n"
        << " RACORNUMALab\n"
        << " Gate G-UMA5D - Quality Constrained\n"
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
        << "Recall target               : "
        << std::fixed
        << std::setprecision(4)
        << result.recall_target
        << '\n'
        << "Warmup batches              : "
        << result.warmup_batches
        << '\n'
        << "Measured batches            : "
        << result.measured_batches
        << '\n'
        << "two_hop_threshold           : "
        << result.two_hop_threshold
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
        << "[Frontera calidad/rendimiento]\n\n";

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

        << std::setw(11)
        << "Lazy50"

        << std::setw(11)
        << "CPU50"

        << std::setw(11)
        << "GPU50"

        << std::setw(10)
        << "Quality"

        << std::setw(24)
        << "Admissible winner"

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

            << std::setw(11)
            << item
                .cpu_lazy
                .batch_p50_ms

            << std::setw(11)
            << item
                .cpu_bitset_pipeline
                .batch_p50_ms

            << std::setw(11)
            << item
                .gpu_bitset_pipeline
                .batch_p50_ms

            << std::setw(10)
            << (
                item.quality_admissible
                    ? "PASS"
                    : "FAIL"
            )

            << std::setw(24)
            << item.admissible_winner

            << '\n';
    }

    std::cout
        << "\n[QPS p50]\n";

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
            << '\n'

            << "  CPU-Lazy            : "
            << std::setprecision(2)
            << item.cpu_lazy.qps_p50
            << " QPS\n"

            << "  CPU-Bitset-Pipeline : "
            << item
                .cpu_bitset_pipeline
                .qps_p50
            << " QPS\n"

            << "  GPU-Bitset-Pipeline : "
            << item
                .gpu_bitset_pipeline
                .qps_p50
            << " QPS\n";
    }

    std::cout
        << "\n[Trabajo RACORN]\n";

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

            << " | Recall="
            << item.mean_recall_at_k

            << " | min="
            << item.min_recall_at_k

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
        << "\n[Pipeline y speedup]\n";

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

            << " | CPU/Lazy="
            << std::setprecision(3)
            << item
                .cpu_pipeline_speedup_vs_lazy
            << "x"

            << " | GPU/Lazy="
            << item
                .gpu_pipeline_speedup_vs_lazy
            << "x"

            << " | GPU/CPU="
            << item
                .gpu_speedup_vs_cpu_pipeline
            << "x"

            << " | CPU wait="
            << std::setprecision(4)
            << item
                .cpu_bitset_pipeline
                .exposed_wait_p50_ms
            << " ms"

            << " | GPU wait="
            << item
                .gpu_bitset_pipeline
                .exposed_wait_p50_ms
            << " ms\n";
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
            << " BR="
            << std::fixed
            << std::setprecision(2)
            << item.bridge_budget_ratio
            << '\n'

            << "  Bitset mismatches             : "
            << item.bitset_word_mismatches
            << '\n'

            << "  Predicate mismatches          : "
            << item
                .predicate_representation_mismatches
            << '\n'

            << "  CPU top-k mismatch            : "
            << item
                .cpu_pipeline_topk_mismatch_queries
            << '\n'

            << "  CPU counter mismatch          : "
            << item
                .cpu_pipeline_counter_mismatch_queries
            << '\n'

            << "  GPU top-k mismatch            : "
            << item
                .gpu_pipeline_topk_mismatch_queries
            << '\n'

            << "  GPU counter mismatch          : "
            << item
                .gpu_pipeline_counter_mismatch_queries
            << '\n'

            << "  Ground truth completo         : "
            << yes_no(
                item.exact_ground_truth_complete
            )
            << '\n'

            << "  CPU semantica                 : "
            << yes_no(
                item.cpu_pipeline_semantics_identical
            )
            << '\n'

            << "  GPU semantica                 : "
            << yes_no(
                item.gpu_pipeline_semantics_identical
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
        << '\n'
        << "Existe caso Recall admisible : "
        << yes_no(
            result
                .at_least_one_admissible_case
        )
        << "\n\n";

    if (
        !result.valid
    ) {

        std::cerr
            << "[G-UMA5D] FAIL\n";

        return 1;
    }

    std::cout
        << "[G-UMA5D] PASS\n";

    return 0;
}
