//
//  PredicateEngineProbe.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir la primera prueba funcional de evaluación de predicados de
//  metadatos mediante GPU Metal sobre memoria unificada.
//
//  Gate:
//  G-UMA2 — Predicate Engine Metal.
//
//  Objetivo:
//  Ejecutar exactamente el mismo predicado lógico mediante:
//
//      1. CPU: oráculo de referencia.
//      2. GPU Metal: ejecución masivamente paralela.
//
//  El resultado GPU se materializa como una máscara de elegibilidad:
//
//      mask[i] = 1  -> vector i satisface el filtro.
//      mask[i] = 0  -> vector i no satisface el filtro.
//
//  Decisión de diseño:
//  G-UMA2 utiliza un byte por vector, no un bitset compacto.
//
//  Motivo:
//  - Un hilo GPU procesa exactamente un registro.
//  - Escrituras lineales.
//  - Sin atomics.
//  - Sin coordinación entre hilos.
//  - Sin mezclar la semántica del predicado con la compactación.
//
//  La compactación a bitset se estudiará independientemente en G-UMA3.
//
//  Invariantes:
//  - RACORN permanece completamente intacto.
//  - HNSW no se ejecuta.
//  - Un único dispatch GPU.
//  - Una única sincronización CPU/GPU.
//  - Los metadatos permanecen en MTLStorageModeShared.
//  - La máscara permanece en MTLStorageModeShared.
//  - No existen copias explícitas CPU→VRAM→CPU.
//  - La equivalencia debe ser exacta para todos los registros.
//
//  Seguridad:
//  - Sin red.
//  - Sin archivos externos.
//  - Sin persistencia.
//  - Objective-C/Metal permanece confinado a la capa Apple.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace racorn_uma::platform::apple {

struct PredicateEngineProbeResult final {

    bool metal_available{};
    bool unified_memory{};

    bool library_available{};
    bool kernel_available{};
    bool pipeline_available{};

    bool command_queue_available{};

    bool metadata_buffer_available{};
    bool mask_buffer_available{};

    bool command_buffer_available{};
    bool encoder_available{};
    bool command_completed{};

    bool exact_match{};

    std::string device_name;
    std::string kernel_name;
    std::string metal_error;

    std::size_t element_count{};
    std::size_t metadata_column_count{};

    std::uint64_t metadata_bytes{};
    std::uint64_t mask_bytes{};

    std::size_t thread_execution_width{};
    std::size_t max_total_threads_per_threadgroup{};
    std::size_t threads_per_threadgroup{};

    std::size_t cpu_selected_count{};
    std::size_t gpu_selected_count{};

    double observed_selectivity{};

    std::size_t mismatch_count{};
    std::size_t invalid_mask_value_count{};

    std::size_t first_mismatch_index{};

    std::uint8_t first_mismatch_expected{};
    std::uint8_t first_mismatch_observed{};

    bool valid{};
};

class PredicateEngineProbe final {

public:

    [[nodiscard]]
    static PredicateEngineProbeResult run();
};

} // namespace racorn_uma::platform::apple
