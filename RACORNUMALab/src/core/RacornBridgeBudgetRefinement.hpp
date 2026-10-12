//
//  RacornBridgeBudgetRefinement.hpp
//  RACORNUMALab
//
//  Propósito:
//  Definir G-UMA5E, experimento de refinamiento del presupuesto de
//  bridges de RACORN bajo una restricción robusta de calidad.
//
//  Pregunta:
//      ¿Cuál es el menor bridge_budget_ratio que mantiene:
//
//          Recall@10 medio >= 0.99
//          Recall@10 mínimo por query >= 0.99
//
//      para cada tamaño de índice evaluado?
//
//  Dimensiones:
//
//      N  ∈ {250000, 500000}
//      BR ∈ {0.50, 0.55, 0.60, 0.65, 0.70, 0.75}
//
//  Métricas:
//  - Recall@10 medio.
//  - Recall@10 mínimo.
//  - Query de peor Recall.
//  - computed_distances.
//  - visited_members.
//  - predicate_evaluations.
//  - fallback_count.
//  - selectividad observada.
//
//  Invariantes:
//  - RACORN no se modifica.
//  - Sin Metal.
//  - Sin GPU.
//  - Sin bitsets.
//  - Predicado directo CPU.
//  - Ground truth filtrado independiente.
//

#pragma once

#include <cstddef>
#include <vector>

namespace racorn_uma::core {

struct BridgeBudgetRefinementCase final {

    std::size_t vector_count{};

    double bridge_budget_ratio{};

    double mean_selectivity{};

    std::size_t selected_min{};
    std::size_t selected_max{};

    double mean_recall_at_k{};
    double min_recall_at_k{};

    std::size_t worst_query_index{};

    double mean_computed_distances{};
    double mean_visited_members{};
    double mean_predicate_evaluations{};
    double mean_fallback_count{};

    std::size_t result_count_min{};
    std::size_t result_count_max{};

    bool exact_ground_truth_complete{};

    bool returned_results_filter_valid{};

    bool mean_recall_pass{};
    bool min_recall_pass{};

    bool robust_quality_pass{};

    bool valid{};
};

struct BridgeBudgetOptimalPoint final {

    std::size_t vector_count{};

    bool found{};

    double minimum_robust_bridge_budget_ratio{};

    double mean_recall_at_k{};
    double min_recall_at_k{};

    double mean_computed_distances{};
    double mean_predicate_evaluations{};
};

struct RacornBridgeBudgetRefinementResult final {

    std::size_t dimensions{};
    std::size_t connectivity{};

    std::size_t ef_construction{};
    std::size_t ef_search{};

    std::size_t k{};
    std::size_t query_count{};

    double two_hop_threshold{};

    double mean_recall_target{};
    double min_recall_target{};

    bool racorn_modified{};

    std::vector<BridgeBudgetRefinementCase> cases;

    std::vector<BridgeBudgetOptimalPoint> optimal_points;

    bool all_cases_valid{};

    bool robust_point_found_for_all_sizes{};

    bool valid{};
};

class RacornBridgeBudgetRefinement final {

public:

    [[nodiscard]]
    static RacornBridgeBudgetRefinementResult run();
};

} // namespace racorn_uma::core
