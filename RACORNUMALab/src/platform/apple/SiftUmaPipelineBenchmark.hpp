//
//  SiftUmaPipelineBenchmark.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir G-DATA3B, experimento final de pipeline CPU/GPU sobre SIFT100K.
//
//  Planes:
//
//      CPU-Lazy
//          búsqueda secuencial con predicado directo.
//
//      CPU-Bitset-Pipeline
//          worker CPU persistente prepara máscara(q[i+1])
//          mientras el hilo principal navega q[i].
//
//      GPU-Bitset-Pipeline
//          Metal prepara máscara(q[i+1])
//          mientras CPU navega q[i].
//
//  Criterio previamente fijado para continuar investigando Metal:
//
//      GPU_vs_CPU_p50 >= 1.02
//
//      Y
//
//      GPU_p95 <= CPU_p95 * 1.02
//
//  Si no se satisfacen ambas condiciones:
//      cerrar la línea Metal-predicate.
//
//  PASS experimental:
//      significa que el experimento fue válido y semánticamente idéntico.
//      No implica que GPU haya ganado.
//

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

namespace racorn_uma::platform::apple {

struct PipelineLatencyStats final {

    double batch_p50_ms{};
    double batch_p95_ms{};
    double batch_p99_ms{};

    double exposed_wait_p50_ms{};
    double exposed_wait_p95_ms{};

    double prepare_work_p50_ms{};
    double prepare_work_p95_ms{};

    double qps_p50{};
};

struct SiftUmaPipelineBenchmarkResult final {

    std::filesystem::path data_directory;

    std::size_t vector_count{};
    std::size_t dimensions{};
    std::size_t query_count{};

    std::size_t connectivity{};
    std::size_t ef_construction{};
    std::size_t ef_search{};
    std::size_t k{};

    double bridge_budget_ratio{};
    double two_hop_threshold{};

    std::size_t warmup_batches{};
    std::size_t measured_batches{};

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
    std::size_t threads_per_threadgroup{};

    std::size_t bitset_word_mismatches{};

    std::size_t cpu_topk_mismatches{};
    std::size_t cpu_counter_mismatches{};

    std::size_t gpu_topk_mismatches{};
    std::size_t gpu_counter_mismatches{};

    bool ground_truth_complete{};
    bool filter_results_valid{};
    bool semantic_identity{};

    double mean_recall_at_k{};
    double min_recall_at_k{};

    std::size_t perfect_queries{};
    std::size_t imperfect_queries{};
    std::size_t underfilled_queries{};

    PipelineLatencyStats cpu_lazy;
    PipelineLatencyStats cpu_pipeline;
    PipelineLatencyStats gpu_pipeline;

    double cpu_speedup_vs_lazy{};
    double gpu_speedup_vs_lazy{};
    double gpu_speedup_vs_cpu{};

    double gpu_p95_ratio_vs_cpu{};

    bool gpu_p50_margin_pass{};
    bool gpu_p95_guard_pass{};

    bool gpu_pipeline_worthwhile{};

    std::string decision;

    std::size_t benchmark_sink{};

    bool racorn_modified{};
    bool valid{};
};

class SiftUmaPipelineBenchmark final {

public:

    [[nodiscard]]
    static SiftUmaPipelineBenchmarkResult run(
        std::filesystem::path const& data_directory
    );
};

} // namespace racorn_uma::platform::apple
