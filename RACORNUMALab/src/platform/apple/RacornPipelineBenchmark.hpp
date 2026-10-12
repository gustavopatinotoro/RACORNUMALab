//
//  RacornPipelineBenchmark.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir G-UMA5C, benchmark de ejecución solapada para RACORN.
//
//  Planes evaluados:
//
//      CPU-Lazy
//      CPU-Bitset Serial
//      CPU-Bitset Pipeline
//      GPU-Bitset Pipeline
//
//  El objetivo es determinar si la preparación del filtro de q(i+1)
//  puede ocultarse mientras RACORN procesa q(i).
//
//  Invariante principal:
//
//      RACORN siempre navega en CPU.
//
//  La GPU únicamente materializa elegibilidad.
//
//  El pipeline utiliza doble buffering para impedir que un productor
//  modifique el bitset que RACORN está consumiendo.
//
//  Métrica principal:
//  Throughput de un lote de 16 consultas.
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

struct PipelinePlanStats final {

    double batch_p50_ms{};
    double batch_p95_ms{};

    double ms_per_query_p50{};
    double qps_p50{};

    double exposed_wait_p50_ms{};
    double exposed_wait_p95_ms{};
};

struct RacornPipelineCase final {

    std::size_t vector_count{};

    std::size_t selected_min{};
    std::size_t selected_max{};

    double mean_selectivity{};

    double mean_recall_at_k{};

    double mean_computed_distances{};
    double mean_visited_members{};
    double mean_predicate_evaluations{};
    double mean_fallback_count{};

    PipelinePlanStats cpu_lazy;
    PipelinePlanStats cpu_bitset_serial;
    PipelinePlanStats cpu_bitset_pipeline;
    PipelinePlanStats gpu_bitset_pipeline;

    double cpu_pipeline_speedup_vs_serial{};
    double gpu_pipeline_speedup_vs_serial{};

    double gpu_pipeline_speedup_vs_lazy{};

    std::string throughput_winner;

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

struct RacornPipelineBenchmarkResult final {

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
    double bridge_budget_ratio{};

    std::size_t warmup_batches{};
    std::size_t measured_batches{};

    std::size_t thread_execution_width{};
    std::size_t max_total_threads_per_threadgroup{};
    std::size_t threads_per_threadgroup{};

    std::size_t benchmark_sink{};

    std::vector<RacornPipelineCase> cases;

    bool all_cases_semantically_identical{};

    bool racorn_modified{};

    bool valid{};
};

class RacornPipelineBenchmark final {

public:

    [[nodiscard]]
    static RacornPipelineBenchmarkResult run();
};

} // namespace racorn_uma::platform::apple
