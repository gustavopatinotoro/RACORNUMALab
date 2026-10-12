//
//  RacornEndToEndBenchmark.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir el benchmark end-to-end que compara tres estrategias de
//  evaluación de filtros dentro de una búsqueda RACORN:
//
//      A. CPU-Lazy:
//         RACORN evalúa directamente el predicado sobre metadata.
//
//      B. CPU-Bitset:
//         CPU materializa previamente un bitset Shared.
//         RACORN consume el bitset.
//
//      C. GPU-Bitset:
//         GPU Metal materializa previamente el mismo bitset Shared.
//         RACORN consume el bitset.
//
//  Gate:
//  G-UMA5B — RACORN End-to-End CPU/GPU.
//
//  Restricción arquitectónica:
//  RACORN permanece completamente inalterado y navega exclusivamente
//  en CPU.
//
//  GPU:
//  - Evalúa metadata.
//  - Produce bitset.
//  - No navega.
//  - No mantiene heaps.
//  - No selecciona bridges.
//  - No ejecuta lógica HNSW/RACORN.
//
//  Métricas:
//  - Latencia p50/p95 total.
//  - Preparación p50/p95.
//  - Navegación RACORN p50/p95.
//  - Tiempo device GPU p50/p95.
//  - Recall@k.
//  - Contadores RACORN.
//  - Equivalencia semántica.
//
//  Seguridad:
//  - Sin red.
//  - Sin archivos externos.
//  - Sin modificación de vendor/racorn.
//

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace racorn_uma::platform::apple {

struct EndToEndLatencyStats final {

    double total_p50_ms{};
    double total_p95_ms{};

    double prepare_p50_ms{};
    double prepare_p95_ms{};

    double search_p50_ms{};
    double search_p95_ms{};

    double gpu_device_p50_ms{};
    double gpu_device_p95_ms{};

    bool gpu_device_timing_available{};
};

struct RacornEndToEndCase final {

    std::size_t vector_count{};

    std::size_t selected_min{};
    std::size_t selected_max{};

    double mean_selectivity{};

    double mean_recall_at_k{};

    double mean_computed_distances{};
    double mean_visited_members{};
    double mean_predicate_evaluations{};
    double mean_fallback_count{};

    EndToEndLatencyStats cpu_lazy;
    EndToEndLatencyStats cpu_bitset;
    EndToEndLatencyStats gpu_bitset;

    double cpu_bitset_speedup_vs_lazy{};
    double gpu_bitset_speedup_vs_lazy{};

    std::string p50_winner;

    std::size_t bitset_word_mismatches{};
    std::size_t predicate_representation_mismatches{};

    std::size_t topk_mismatch_queries{};
    std::size_t recall_mismatch_queries{};
    std::size_t computed_distances_mismatch_queries{};
    std::size_t visited_members_mismatch_queries{};
    std::size_t predicate_evaluations_mismatch_queries{};
    std::size_t fallback_count_mismatch_queries{};

    bool exact_ground_truth_complete{};

    bool filter_results_valid{};

    bool navigation_semantics_identical{};

    bool valid{};
};

struct RacornEndToEndBenchmarkResult final {

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

    std::size_t warmup_rounds{};
    std::size_t measured_rounds{};

    std::size_t thread_execution_width{};
    std::size_t max_total_threads_per_threadgroup{};
    std::size_t threads_per_threadgroup{};

    std::size_t benchmark_sink{};

    std::vector<RacornEndToEndCase> cases;

    bool all_cases_semantically_identical{};

    bool racorn_modified{};

    bool valid{};
};

class RacornEndToEndBenchmark final {

public:

    [[nodiscard]]
    static RacornEndToEndBenchmarkResult run();
};

} // namespace racorn_uma::platform::apple
