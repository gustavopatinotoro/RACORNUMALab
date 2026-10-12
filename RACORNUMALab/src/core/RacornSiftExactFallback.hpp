//
//  RacornSiftExactFallback.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir G-DATA2, comparación controlada entre RACORN-1 y RACORN-1+
//  sobre SIFT100K.
//
//  Variables experimentales:
//
//      bridge_budget_ratio ∈ {
//          0.75,
//          1.00,
//          1.25
//      }
//
//      fallback_exact_threshold ∈ {
//          0.000,
//          0.001,
//          0.002,
//          0.003,
//          0.005
//      }
//
//  fallback_exact_threshold == 0:
//      RACORN-1.
//
//  fallback_exact_threshold > 0:
//      RACORN-1+.
//
//  Objetivo:
//  Determinar si Adaptive Exact Fallback elimina la cola residual de
//  errores observada en G-DATA1A sin convertir todas las consultas en
//  búsquedas exactas.
//
//  Métricas:
//  - Recall@10 medio.
//  - Recall mínimo.
//  - P01 / P05.
//  - Queries perfectas.
//  - Underfill.
//  - Full-but-inexact.
//  - Queries que activaron exact fallback.
//  - Fracción de exact fallback.
//  - Selectividad observada por RACORN en fallback.
//  - Distancias.
//  - Nodos visitados.
//  - Evaluaciones de predicado.
//  - fallback_count.
//
//  Invariantes:
//  - RACORN upstream permanece intacto.
//  - Navegación RACORN exclusivamente CPU.
//  - Sin Metal.
//  - Sin GPU.
//  - Sin bitsets.
//  - Ground truth filtrado exacto independiente.
//  - Underfill o Recall bajo son resultados algorítmicos, no invalidez
//    experimental.
//

#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace racorn_uma::core {

struct SiftExactFallbackCase final {

    double bridge_budget_ratio{};
    double fallback_exact_threshold{};

    bool racorn_one_plus{};

    double mean_recall_at_k{};
    double min_recall_at_k{};

    double recall_p01{};
    double recall_p05{};

    std::size_t perfect_queries{};
    std::size_t imperfect_queries{};

    std::size_t underfilled_queries{};
    std::size_t full_but_inexact_queries{};

    std::size_t exact_fallback_queries{};
    double exact_fallback_fraction{};

    std::size_t fallback_selectivity_samples{};
    double mean_fallback_selectivity{};

    double mean_computed_distances{};
    double mean_visited_members{};
    double mean_predicate_evaluations{};
    double mean_fallback_count{};

    std::size_t result_count_min{};
    std::size_t result_count_max{};

    bool mean_recall_pass{};
    bool min_recall_pass{};
    bool robust_quality_pass{};

    bool ground_truth_complete{};
    bool returned_results_filter_valid{};
    bool cardinality_valid{};

    bool valid{};
};

struct RacornSiftExactFallbackResult final {

    std::filesystem::path data_directory;

    std::size_t vector_count{};
    std::size_t dimensions{};
    std::size_t query_count{};

    std::size_t connectivity{};
    std::size_t ef_construction{};
    std::size_t ef_search{};
    std::size_t k{};

    double two_hop_threshold{};

    double mean_recall_target{};
    double min_recall_target{};

    double mean_selectivity{};

    std::size_t selected_min{};
    std::size_t selected_max{};

    bool base_file_valid{};
    bool query_file_valid{};
    bool metadata_file_valid{};
    bool ground_truth_complete{};

    bool racorn_modified{};

    std::vector<SiftExactFallbackCase> cases;

    bool all_cases_valid{};
    bool robust_configuration_found{};

    bool valid{};
};

class RacornSiftExactFallback final {

public:

    [[nodiscard]]
    static RacornSiftExactFallbackResult run(
        std::filesystem::path const& data_directory
    );
};

} // namespace racorn_uma::core
