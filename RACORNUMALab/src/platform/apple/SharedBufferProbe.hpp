//
//  SharedBufferProbe.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir una prueba funcional CPU↔GPU sobre memoria unificada
//  utilizando un MTLBuffer con MTLStorageModeShared.
//
//  Gate:
//  G-UMA1 — Primer flujo real CPU → GPU → CPU sobre UMA.
//
//  Flujo:
//      CPU inicializa un buffer Shared.
//              ↓
//      GPU Metal transforma el mismo buffer.
//              ↓
//      CPU espera la finalización del command buffer.
//              ↓
//      CPU verifica bit a bit el mismo recurso.
//
//  Objetivo:
//  Demostrar que el laboratorio puede utilizar memoria compartida
//  CPU/GPU sin introducir una copia explícita CPU→VRAM→CPU.
//
//  Invariantes:
//  - RACORN permanece completamente intacto.
//  - HNSW no se ejecuta.
//  - Un único kernel.
//  - Un único dispatch.
//  - Una única sincronización después del dispatch.
//  - No se mide rendimiento en este Gate.
//  - El resultado debe coincidir exactamente con el oráculo CPU.
//
//  Seguridad:
//  - Sin red.
//  - Sin archivos externos.
//  - Sin estado persistente.
//  - Los tipos Objective-C/Metal no salen de la capa Apple.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace racorn_uma::platform::apple {

struct SharedBufferProbeResult final {

    bool metal_available{};

    bool unified_memory{};

    bool library_available{};

    bool kernel_available{};

    bool pipeline_available{};

    bool command_queue_available{};

    bool shared_buffer_available{};

    bool command_buffer_available{};

    bool encoder_available{};

    bool command_completed{};

    bool exact_match{};

    std::string device_name;

    std::string kernel_name;

    std::string metal_error;

    std::size_t element_count{};

    std::uint64_t buffer_bytes{};

    std::size_t thread_execution_width{};

    std::size_t max_total_threads_per_threadgroup{};

    std::size_t threads_per_threadgroup{};

    std::size_t mismatch_count{};

    std::size_t first_mismatch_index{};

    std::uint32_t first_mismatch_expected{};

    std::uint32_t first_mismatch_observed{};

    bool valid{};
};

class SharedBufferProbe final {

public:

    [[nodiscard]]
    static SharedBufferProbeResult run();
};

} // namespace racorn_uma::platform::apple
