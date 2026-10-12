//
//  RacornQualityCrossoverBenchmark.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir G-UMA5D: búsqueda de la región de ejecución heterogénea
//  admisible bajo una restricción explícita de calidad.
//
//  Problema experimental:
//
//      minimizar latencia/throughput
//
//  sujeto a:
//
//      Recall@10 medio >= 0.99
//
//  Además se registra el Recall mínimo por consulta para detectar
//  degradaciones que puedan quedar ocultas por la media.
//
//  Dimensiones exploradas:
//
//      N  ∈ {250000, 500000}
//      BR ∈ {0.50, 0.75, 1.00}
//
//  Planes:
//
//      CPU-Lazy
//      CPU-Bitset-Pipeline
//      GPU-Bitset-Pipeline
//
//  Invariante principal:
//
//      RACORN navega exclusivamente en CPU.
//
//  La GPU únicamente materializa el bitset de elegibilidad.
//
//  Condición metodológica:
//
//      El ganador de rendimiento solo se declara cuando la configuración
//      (N, BR) satisface la restricción de Recall.
//
//  Seguridad:
//  - Sin red.
//  - Sin persistencia.
//  - Sin modificación de vendor/racorn.
//

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace racorn_uma::platform::apple {

struct QualityPlanStats final {
    double batch_p50_ms{};
    double batch_p95_ms{};

    double ms_per_query_p50{};
    double qps_p50{};

    double exposed_wait_p50_ms{};
    double exposed_wait_p95_ms{};
};

struct QualityCrossoverCase final {
    std::size_t vector_count{};
    double bridge_budget_ratio{};

    std::size_t selected_min{};
    std::size_t selected_max{};
    double mean_selectivity{};

    double mean_recall_at_k{};
    double min_recall_at_k{};

    double mean_computed_distances{};
    double mean_visited_members{};
    double mean_predicate_evaluations{};
    double mean_fallback_count{};

    QualityPlanStats cpu_lazy;
    QualityPlanStats cpu_bitset_pipeline;
    QualityPlanStats gpu_bitset_pipeline;

    double cpu_pipeline_speedup_vs_lazy{};
    double gpu_pipeline_speedup_vs_lazy{};
    double gpu_speedup_vs_cpu_pipeline{};

    bool quality_admissible{};

    std::string fastest_plan;
    std::string admissible_winner;

    std::size_t bitset_word_mismatches{};
    std::size_t predicate_representation_mismatches{};

    std::size_t cpu_pipeline_topk_mismatch_queries{};
    std::size_t cpu_pipeline_counter_mismatch_queries{};

    std::size_t gpu_pipeline_topk_mismatch_queries{};
    std::size_t gpu_pipeline_counter_mismatch_queries{};

    bool exact_ground_truth_complete{};
    bool filter_results_valid{};

    bool cpu_pipeline_semantics_identical{};
    bool gpu_pipeline_semantics_identical{};

    bool valid{};
};

struct RacornQualityCrossoverBenchmarkResult final {
    bool metal_available{};
    bool unified_memory{};

    bool library_available{};
    bool kernel_available{};
    bool pipeline_available{};
    bool command_queue_available{};

    std::string device_name;
    std::string kernel_name;
    std::string metal_error;

    std::size_t dimensions{};
    std::size_t connectivity{};

    std::size_t ef_construction{};
    std::size_t ef_search{};

    std::size_t k{};
    std::size_t query_count{};

    double two_hop_threshold{};
    double recall_target{};

    std::size_t warmup_batches{};
    std::size_t measured_batches{};

    std::size_t thread_execution_width{};
    std::size_t max_total_threads_per_threadgroup{};
    std::size_t threads_per_threadgroup{};

    std::size_t benchmark_sink{};

    std::vector<QualityCrossoverCase> cases;

    bool all_cases_semantically_identical{};
    bool at_least_one_admissible_case{};

    bool racorn_modified{};

    bool valid{};
};

class RacornQualityCrossoverBenchmark final {

public:

    [[nodiscard]]
    static RacornQualityCrossoverBenchmarkResult run();
};

} // namespace racorn_uma::platform::apple
