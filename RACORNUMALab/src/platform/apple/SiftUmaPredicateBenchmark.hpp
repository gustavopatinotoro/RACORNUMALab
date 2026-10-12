//
//  SiftUmaPredicateBenchmark.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir G-DATA3A, benchmark end-to-end de ejecución de predicados
//  sobre SIFT100K real usando tres estrategias:
//
//      1. CPU-Lazy.
//      2. CPU-Bitset en memoria Shared.
//      3. GPU-Bitset Metal en memoria Shared.
//
//  Principio arquitectónico:
//      GPU calcula información.
//      CPU decide navegación.
//
//  La GPU no participa en:
//      - navegación HNSW;
//      - heaps;
//      - visited set;
//      - ACORN/RACORN expansion;
//      - selección ASF;
//      - terminación.
//
//  Configuración experimental:
//
//      Dataset       : SIFT1M train[0:100000]
//      N             : 100000
//      D             : 128
//      Q semántica   : 1000
//      Q benchmark   : 256
//      M             : 16
//      efC           : 100
//      efS           : 200
//      k             : 10
//      BR            : 1.0
//      TT            : 0.30
//      exact fallback: desactivado
//
//  Invariantes:
//  - RACORN upstream intacto.
//  - Metadata SoA externa.
//  - CPU y GPU observan la misma memoria Shared.
//  - Ground truth filtrado exacto independiente.
//  - Underfill es fallo de calidad, no invalidez experimental.
//

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace racorn_uma::platform::apple {

struct SiftUmaLatencyStats final {

    double total_p50_ms{};
    double total_p95_ms{};
    double total_p99_ms{};

    double prepare_p50_ms{};
    double prepare_p95_ms{};
    double prepare_p99_ms{};

    double search_p50_ms{};
    double search_p95_ms{};
    double search_p99_ms{};

    bool gpu_device_timing_available{};

    double gpu_device_p50_ms{};
    double gpu_device_p95_ms{};
    double gpu_device_p99_ms{};

    double qps_from_p50{};
};

struct SiftUmaPredicateBenchmarkResult final {

    std::filesystem::path data_directory;

    std::size_t vector_count{};
    std::size_t dimensions{};

    std::size_t semantic_query_count{};
    std::size_t benchmark_query_count{};

    std::size_t connectivity{};
    std::size_t ef_construction{};
    std::size_t ef_search{};
    std::size_t k{};

    double bridge_budget_ratio{};
    double two_hop_threshold{};

    std::size_t warmup_rounds{};
    std::size_t measured_rounds{};

    bool base_file_valid{};
    bool query_file_valid{};
    bool metadata_file_valid{};

    bool metal_available{};
    bool unified_memory{};
    bool library_available{};
    bool kernel_available{};
    bool pipeline_available{};
    bool command_queue_available{};

    std::string device_name;
    std::string kernel_name;
    std::string metal_error;

    std::size_t thread_execution_width{};
    std::size_t max_total_threads_per_threadgroup{};
    std::size_t threads_per_threadgroup{};

    std::size_t selected_min{};
    std::size_t selected_max{};
    double mean_selectivity{};

    double mean_recall_at_k{};
    double min_recall_at_k{};

    std::size_t perfect_queries{};
    std::size_t imperfect_queries{};
    std::size_t underfilled_queries{};

    double mean_computed_distances{};
    double mean_visited_members{};
    double mean_predicate_evaluations{};
    double mean_fallback_count{};

    std::size_t bitset_word_mismatches{};
    std::size_t predicate_representation_mismatches{};

    std::size_t topk_mismatch_queries{};
    std::size_t result_count_mismatch_queries{};
    std::size_t recall_mismatch_queries{};
    std::size_t computed_distances_mismatch_queries{};
    std::size_t visited_members_mismatch_queries{};
    std::size_t predicate_evaluations_mismatch_queries{};
    std::size_t fallback_count_mismatch_queries{};

    bool exact_ground_truth_complete{};
    bool filter_results_valid{};
    bool navigation_semantics_identical{};

    SiftUmaLatencyStats cpu_lazy;
    SiftUmaLatencyStats cpu_bitset;
    SiftUmaLatencyStats gpu_bitset;

    double cpu_bitset_speedup_vs_lazy{};
    double gpu_bitset_speedup_vs_lazy{};
    double gpu_vs_cpu_bitset{};

    std::string p50_winner;

    std::size_t benchmark_sink{};

    bool racorn_modified{};
    bool valid{};
};

class SiftUmaPredicateBenchmark final {

public:

    [[nodiscard]]
    static SiftUmaPredicateBenchmarkResult run(
        std::filesystem::path const& data_directory
    );
};

} // namespace racorn_uma::platform::apple
