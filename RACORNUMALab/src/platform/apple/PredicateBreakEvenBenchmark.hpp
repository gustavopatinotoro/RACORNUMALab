//
//  PredicateBreakEvenBenchmark.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir el benchmark reproducible utilizado para localizar el punto
//  de equilibrio entre CPU y GPU en la materialización de un bitset de
//  elegibilidad sobre Apple Silicon con memoria unificada.
//
//  Gate:
//  G-UMA5A — Punto de equilibrio CPU/GPU del Predicate Engine.
//
//  Importante:
//  Este Gate NO modifica ni ejecuta RACORN.
//  Su función es caracterizar exclusivamente el trabajo regular que
//  potencialmente será asignado dinámicamente a CPU o GPU.
//
//  Caminos comparados:
//
//  CPU:
//      metadata Shared
//          ↓
//      evaluación secuencial/SIMD del predicado
//          ↓
//      bitset CPU
//
//  GPU:
//      metadata Shared
//          ↓
//      Metal compute
//          ↓
//      bitset Shared
//
//  La GPU utiliza la implementación fusionada:
//
//      predicate + bitset packing
//
//  ya validada semánticamente en G-UMA4.
//
//  Tamaños evaluados:
//      20.000
//      100.000
//      250.000
//      500.000
//      1.048.576
//
//  Complejidades:
//
//      P1:
//          tenant == 7
//
//      P3:
//          tenant == 7
//          AND year >= 2022
//          AND language == 2
//
//      P5:
//          P3
//          AND category IN {3,7,9}
//          AND security_level <= 2
//
//  Métricas:
//
//  CPU:
//      p50 wall-clock
//      p95 wall-clock
//
//  GPU:
//      p50 end-to-end host
//      p95 end-to-end host
//      p50 device execution
//      p95 device execution
//
//  La medición GPU end-to-end incluye:
//      creación de command buffer;
//      encoding;
//      commit;
//      sincronización;
//      disponibilidad del bitset para CPU.
//
//  No incluye:
//      creación del dispositivo;
//      creación del pipeline;
//      asignación de buffers;
//      generación de metadata.
//
//  Invariantes:
//  - Metadata preasignada.
//  - Pipeline precompilado antes de medir.
//  - Buffers preasignados.
//  - Sin allocation dentro del camino CPU medido.
//  - Exactitud CPU/GPU validada para cada caso.
//  - CPU y GPU utilizan exactamente la misma metadata.
//  - RACORN permanece intacto.
//
//  Seguridad:
//  - Sin red.
//  - Sin archivos externos.
//  - Sin persistencia.
//

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace racorn_uma::platform::apple {

struct PredicateBreakEvenCase final {

    std::size_t element_count{};

    std::size_t predicate_mode{};

    std::string predicate_name;

    std::size_t selected_count{};

    double observed_selectivity{};

    double cpu_p50_ms{};
    double cpu_p95_ms{};

    double gpu_wall_p50_ms{};
    double gpu_wall_p95_ms{};

    double gpu_device_p50_ms{};
    double gpu_device_p95_ms{};

    bool gpu_device_timing_available{};

    double gpu_wall_speedup_vs_cpu{};

    bool gpu_faster_at_p50{};

    std::size_t bitset_word_count{};

    std::size_t mismatch_word_count{};

    bool exact_match{};
};

struct PredicateBreakEvenBenchmarkResult final {

    bool metal_available{};
    bool unified_memory{};

    bool library_available{};
    bool kernel_available{};
    bool pipeline_available{};
    bool command_queue_available{};

    std::string device_name;
    std::string kernel_name;
    std::string metal_error;

    std::size_t max_element_count{};

    std::size_t metadata_column_count{};

    std::size_t warmup_iterations{};
    std::size_t measured_iterations{};

    std::size_t thread_execution_width{};
    std::size_t max_total_threads_per_threadgroup{};
    std::size_t threads_per_threadgroup{};

    std::vector<PredicateBreakEvenCase> cases;

    bool all_cases_exact{};
    bool valid{};
};

class PredicateBreakEvenBenchmark final {

public:

    [[nodiscard]]
    static PredicateBreakEvenBenchmarkResult run();
};

} // namespace racorn_uma::platform::apple
