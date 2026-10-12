//
//  BitsetCompactionProbe.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir una prueba funcional de compactación GPU de una máscara
//  de elegibilidad de 1 byte por vector hacia un bitset compacto
//  de 1 bit por vector.
//
//  Gate:
//  G-UMA3 — Compactación GPU a bitset UMA.
//
//  Entrada:
//      eligibility[i] ∈ {0,1}
//
//  Salida:
//      bitset[word] = 32 decisiones de elegibilidad.
//
//  Convención:
//      word = index / 32
//      bit  = index % 32
//
//      eligible(index) =
//          (bitset[word] >> bit) & 1
//
//  Diseño GPU:
//  - Un hilo procesa exactamente una palabra uint32_t.
//  - Cada hilo lee hasta 32 bytes consecutivos.
//  - Cada hilo escribe exactamente una palabra uint32_t.
//  - Sin atomics.
//  - Sin barreras.
//  - Sin memoria threadgroup.
//  - Acceso determinista.
//
//  Objetivo:
//  Probar que GPU y CPU producen exactamente la misma representación
//  compacta antes de permitir que RACORN la utilice.
//
//  Invariantes:
//  - RACORN permanece intacto.
//  - HNSW no se ejecuta.
//  - El predicado GPU de G-UMA2 no se ejecuta.
//  - Un único dispatch GPU.
//  - Una única sincronización GPU→CPU.
//  - Entrada y salida utilizan MTLStorageModeShared.
//  - Sin copias explícitas CPU↔VRAM.
//
//  Seguridad:
//  - Sin red.
//  - Sin archivos externos.
//  - Sin persistencia.
//  - Objective-C/Metal confinado a platform/apple.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace racorn_uma::platform::apple {

struct BitsetCompactionProbeResult final {

    bool metal_available{};
    bool unified_memory{};

    bool library_available{};
    bool kernel_available{};
    bool pipeline_available{};

    bool command_queue_available{};

    bool mask_buffer_available{};
    bool bitset_buffer_available{};

    bool command_buffer_available{};
    bool encoder_available{};
    bool command_completed{};

    bool exact_word_match{};
    bool exact_bit_match{};

    std::string device_name;
    std::string kernel_name;
    std::string metal_error;

    std::size_t element_count{};

    std::size_t bitset_word_count{};

    std::uint64_t mask_bytes{};
    std::uint64_t bitset_bytes{};

    double compression_ratio{};
    double storage_reduction_fraction{};

    std::size_t cpu_selected_count{};
    std::size_t bitset_selected_count{};

    std::size_t thread_execution_width{};
    std::size_t max_total_threads_per_threadgroup{};
    std::size_t threads_per_threadgroup{};

    std::size_t word_mismatch_count{};
    std::size_t bit_mismatch_count{};

    std::size_t first_mismatch_word{};
    std::uint32_t first_mismatch_expected_word{};
    std::uint32_t first_mismatch_observed_word{};

    std::size_t first_mismatch_index{};
    std::uint8_t first_mismatch_expected_bit{};
    std::uint8_t first_mismatch_observed_bit{};

    bool valid{};
};

class BitsetCompactionProbe final {

public:

    [[nodiscard]]
    static BitsetCompactionProbeResult run();
};

} // namespace racorn_uma::platform::apple
