//
//  RacornUmaSemanticProbe.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir la primera integración funcional entre RACORN ejecutado
//  íntegramente en CPU y un bitset de elegibilidad producido mediante
//  GPU Metal sobre memoria unificada.
//
//  Gate:
//  G-UMA4 — RACORN CPU consumiendo bitset GPU/UMA.
//
//  Principio arquitectónico:
//  La GPU únicamente calcula elegibilidad.
//
//  La GPU NO:
//  - navega HNSW;
//  - mantiene heaps;
//  - selecciona bridges;
//  - ejecuta ACORN 2-hop;
//  - decide terminación;
//  - modifica RACORN.
//
//  RACORN permanece como autoridad exclusiva de navegación.
//
//  Comparación:
//      Ruta A:
//          Metadata Shared
//              ↓
//          Predicate CPU
//              ↓
//          RACORN CPU
//
//      Ruta B:
//          Metadata Shared
//              ↓
//          GPU Metal
//              ↓
//          Bitset Shared
//              ↓
//          Predicate O(1) CPU
//              ↓
//          RACORN CPU
//
//  Criterio fuerte de equivalencia:
//  Para cada consulta deben coincidir:
//
//  - Top-k ordenado.
//  - Recall@k.
//  - computed_distances.
//  - visited_members.
//  - predicate_evaluations.
//  - fallback_count.
//
//  Además:
//  - CPU predicate y bitset deben coincidir para todos los IDs.
//  - Todos los resultados deben satisfacer el filtro.
//  - Ground truth exacto filtrado debe contener k elementos.
//
//  Seguridad:
//  - Sin red.
//  - Sin archivos externos.
//  - Sin modificación de vendor/racorn.
//  - Objective-C/Metal confinado a platform/apple.
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace racorn_uma::platform::apple {

struct RacornUmaSemanticProbeResult final {

    bool metal_available{};
    bool unified_memory{};

    bool library_available{};
    bool kernel_available{};
    bool pipeline_available{};

    bool command_queue_available{};
    bool metadata_buffer_available{};
    bool bitset_buffer_available{};
    bool command_completed{};

    std::string device_name;
    std::string kernel_name;
    std::string metal_error;

    std::size_t vector_count{};
    std::size_t dimensions{};
    std::size_t connectivity{};

    std::size_t ef_construction{};
    std::size_t ef_search{};

    std::size_t k{};
    std::size_t query_count{};

    double two_hop_threshold{};
    double bridge_budget_ratio{};

    std::size_t metadata_column_count{};

    std::uint64_t metadata_bytes{};
    std::uint64_t bitset_bytes{};

    std::size_t bitset_word_count{};

    std::size_t cpu_selected_count{};
    std::size_t gpu_selected_count{};

    double observed_selectivity{};

    std::size_t predicate_bit_mismatch_count{};

    bool predicate_exact_match{};

    bool exact_ground_truth_complete{};

    double cpu_mean_recall_at_k{};
    double gpu_bitset_mean_recall_at_k{};

    double cpu_mean_computed_distances{};
    double gpu_mean_computed_distances{};

    double cpu_mean_visited_members{};
    double gpu_mean_visited_members{};

    double cpu_mean_predicate_evaluations{};
    double gpu_mean_predicate_evaluations{};

    double cpu_mean_fallback_count{};
    double gpu_mean_fallback_count{};

    std::size_t topk_mismatch_queries{};
    std::size_t recall_mismatch_queries{};

    std::size_t computed_distances_mismatch_queries{};
    std::size_t visited_members_mismatch_queries{};
    std::size_t predicate_evaluations_mismatch_queries{};
    std::size_t fallback_count_mismatch_queries{};

    bool cpu_results_filter_valid{};
    bool gpu_results_filter_valid{};

    bool navigation_semantics_identical{};

    bool racorn_modified{};

    bool valid{};
};

class RacornUmaSemanticProbe final {

public:

    [[nodiscard]]
    static RacornUmaSemanticProbeResult run();
};

} // namespace racorn_uma::platform::apple
